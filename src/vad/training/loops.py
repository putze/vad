from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.vad.training.callbacks import (
    BestModelTracker,
    EarlyStopping,
)
from src.vad.training.formatting import format_metrics
from src.vad.training.logger import TensorBoardLogger
from src.vad.training.metrics import BinaryClassificationMetrics, VADMetricsTracker


def make_padding_mask(lengths: Tensor, max_len: int) -> Tensor:
    """
    Create a boolean mask for variable-length sequences.

    Args:
        lengths (Tensor): Tensor [B] containing valid lengths.
        max_len (int): Maximum sequence length.

    Returns:
        Tensor: Boolean mask [B, T].
    """
    time_index = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return time_index < lengths.unsqueeze(1)


def extract_logits_for_loss(logits: Tensor) -> Tensor:
    """
    Normalize logits to shape [B, T].

    Args:
        logits (Tensor): Tensor [B, T] or [B, 1, T].

    Returns:
        Tensor: [B, T].
    """
    if logits.ndim == 2:
        return logits

    if logits.ndim == 3 and logits.shape[1] == 1:
        return logits[:, 0, :]

    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


def masked_bce_with_logits_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Compute BCE loss only over valid frames.

    Args:
        logits (Tensor): [B, T]
        targets (Tensor): [B, T]
        mask (Tensor): [B, T]

    Returns:
        Scalar loss.
    """
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
    )

    loss = loss[mask]

    if loss.numel() == 0:
        return logits.new_tensor(0.0)

    return loss.mean()


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    num_epochs: int | None = None,
    show_progress: bool = False,
) -> BinaryClassificationMetrics:
    """
    Run one epoch.

    Args:
        model: Model.
        dataloader: DataLoader.
        device: Device.
        optimizer: If provided, training mode.
        epoch: Current epoch number (1-based).
        num_epochs: Total number of epochs.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        BinaryClassificationMetrics
    """
    is_training = optimizer is not None
    model.train(is_training)

    tracker = VADMetricsTracker()

    progress_bar: tqdm | None = None
    batch_iter: Iterator[tuple[Tensor, Tensor, Tensor]]

    if show_progress:
        split = "train" if is_training else "val"
        desc = (
            f"Epoch {epoch}/{num_epochs} [{split}]"
            if epoch is not None and num_epochs is not None
            else split
        )
        progress_bar = tqdm(dataloader, desc=desc, leave=is_training)
        batch_iter = iter(progress_bar)
    else:
        batch_iter = iter(dataloader)

    for batch_index, (features, labels, lengths) in enumerate(batch_iter, start=1):
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        mask = make_padding_mask(lengths, labels.shape[1])

        with torch.set_grad_enabled(is_training):
            logits = model(features)
            logits_bt = extract_logits_for_loss(logits)

            loss = masked_bce_with_logits_loss(
                logits=logits_bt,
                targets=labels,
                mask=mask,
            )

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        tracker.update(
            logits=logits,
            targets=labels,
            loss=loss,
            mask=mask,
        )

        if progress_bar is not None and batch_index % 10 == 0:
            metrics = tracker.compute()
            progress_bar.set_postfix(
                loss=f"{metrics.loss:.4f}",
                f1=f"{metrics.f1:.4f}",
                acc=f"{metrics.accuracy:.4f}",
            )

    return tracker.compute()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    log_dir: str | Path = "runs/vad",
    checkpoint_path: str | Path = "best_model.pt",
) -> None:
    """
    Full training loop with TensorBoard logging.

    Args:
        model: Model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        device: Device.
        num_epochs: Number of epochs.
        log_dir: TensorBoard directory.
        checkpoint_path: Where to save best model.
    """
    model.to(device)

    logger = TensorBoardLogger(log_dir)
    best_model = BestModelTracker(mode="max")
    early_stopping = EarlyStopping(patience=5, mode="min")

    checkpoint_path = Path(checkpoint_path)

    try:
        for epoch in range(1, num_epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                epoch=epoch,
                num_epochs=num_epochs,
                show_progress=True,
            )

            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                epoch=epoch,
                num_epochs=num_epochs,
                show_progress=True,
            )

            # TensorBoard logging
            logger.log_epoch(epoch, train_metrics, val_metrics)

            # Console logging
            print(f"Epoch {epoch:03d}")
            print(format_metrics("train", train_metrics))
            print(format_metrics("val  ", val_metrics))

            # Best model (based on F1)
            if best_model.update(epoch, val_metrics.f1):
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved best model at epoch {epoch}")

            # Early stopping (based on loss)
            if early_stopping.step(val_metrics.loss):
                print(f"Early stopping at epoch {epoch}")
                break

        # Log final hparams
        logger.log_hparams(
            hparams={
                "optimizer": optimizer.__class__.__name__,
                "epochs": num_epochs,
            },
            final_metrics={
                "hparam/val_loss": val_metrics.loss,
                "hparam/val_f1": val_metrics.f1,
            },
        )

    finally:
        logger.close()

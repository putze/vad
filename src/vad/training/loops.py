from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from vad.config import AudioConfig, TrainingConfig
from vad.training.callbacks import EarlyStopping
from vad.training.checkpoint_manager import CheckpointManager
from vad.training.formatting import format_metrics
from vad.training.logger import TensorBoardLogger
from vad.training.metrics import BinaryClassificationMetrics, VADMetricsTracker
from vad.training.run_config import ExperimentPaths


def extract_logits_for_loss(logits: Tensor) -> Tensor:
    """
    Normalize model outputs to shape [B, T] for loss computation.

    Args:
        logits: Model output tensor with shape [B, T] or [B, 1, T].

    Returns:
        Tensor with shape [B, T].

    Raises:
        ValueError: If the tensor shape is unsupported.
    """
    if logits.ndim == 2:
        return logits

    if logits.ndim == 3 and logits.shape[1] == 1:
        return logits[:, 0, :]

    raise ValueError(f"Expected logits with shape [B, T] or [B, 1, T], got {tuple(logits.shape)}")


def masked_bce_with_logits_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Compute binary cross-entropy loss over valid frames only.

    The loss is averaged over positions where ``mask`` is True. Padded or
    otherwise invalid frames do not contribute to the loss.

    Args:
        logits: Logits with shape [B, T].
        targets: Binary targets with shape [B, T].
        mask: Boolean validity mask with shape [B, T].

    Returns:
        Scalar tensor containing the mean loss over valid frames.

    Raises:
        ValueError: If input shapes do not match.
    """
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError(
            "logits, targets, and mask must have the same shape, got "
            f"{tuple(logits.shape)}, {tuple(targets.shape)}, and {tuple(mask.shape)}"
        )

    valid_count = mask.sum()
    if valid_count.item() == 0:
        return logits.new_tensor(0.0)

    per_frame_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
    )

    masked_loss = per_frame_loss * mask.to(dtype=per_frame_loss.dtype)
    return masked_loss.sum() / valid_count


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
    Run one full training or validation epoch.

    If an optimizer is provided, the model is run in training mode and
    parameters are updated. Otherwise, the epoch is run in evaluation mode.

    Args:
        model: Model to evaluate or train.
        dataloader: DataLoader yielding ``(features, labels, lengths, frame_masks)``.
        device: Device on which computation is performed.
        optimizer: Optimizer used for training. If None, no gradients are
            computed and no parameter updates are performed.
        epoch: Current epoch number, used only for progress display.
        num_epochs: Total number of epochs, used only for progress display.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        Aggregated frame-level metrics for the epoch.
    """
    is_training = optimizer is not None
    model.train(is_training)

    tracker = VADMetricsTracker()

    progress_bar: tqdm | None = None
    batch_iter: Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]

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

    for batch_index, (features, labels, lengths, frame_masks) in enumerate(batch_iter, start=1):
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        frame_masks = frame_masks.to(device).bool()

        with torch.set_grad_enabled(is_training):
            logits = model(features)
            logits_bt = extract_logits_for_loss(logits)

            loss = masked_bce_with_logits_loss(
                logits=logits_bt,
                targets=labels,
                mask=frame_masks,
            )

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        tracker.update_from_logits(
            logits=logits_bt,
            targets=labels,
            loss=loss,
            mask=frame_masks,
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
    experiment_name: str = "causal_conv",
    checkpoint_path: str | Path = "checkpoints",
    audio_config: AudioConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> None:
    """
    Train a model and evaluate it on a validation set each epoch.

    This function handles:
        - experiment directory creation
        - TensorBoard logging
        - console logging
        - best/last checkpoint saving
        - early stopping

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer used for training.
        device: Device on which training runs.
        num_epochs: Maximum number of training epochs.
        log_dir: Root directory for TensorBoard logs.
        experiment_name: Name used to group runs of the same experiment.
        checkpoint_path: Root directory for checkpoints.
        audio_config: Optional audio/preprocessing config to store in checkpoints.
        training_config: Optional training config to store in checkpoints.
    """
    model.to(device)

    experiment = ExperimentPaths.create(
        log_root=log_dir,
        experiment_name=experiment_name,
        checkpoint_root=checkpoint_path,
    )
    logger = TensorBoardLogger(experiment.log_dir)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=experiment.checkpoint_dir,
        monitor="val_f1",
        mode="max",
    )
    early_stopping = EarlyStopping(patience=5, mode="min")

    try:
        for epoch in range(1, num_epochs + 1):
            train_metrics = run_epoch(
                model=model,
                dataloader=train_loader,
                device=device,
                optimizer=optimizer,
                epoch=epoch,
                num_epochs=num_epochs,
                show_progress=True,
            )

            val_metrics = run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                optimizer=None,
                epoch=epoch,
                num_epochs=num_epochs,
                show_progress=True,
            )

            lr = optimizer.param_groups[0]["lr"]
            logger.log_epoch(epoch, train_metrics, val_metrics, lr)

            print(f"Epoch {epoch:03d}")
            print(format_metrics("train", train_metrics))
            print(format_metrics("val", val_metrics))

            metrics = {
                "train_loss": train_metrics.loss,
                "train_f1": train_metrics.f1,
                "val_loss": val_metrics.loss,
                "val_f1": val_metrics.f1,
                "val_accuracy": val_metrics.accuracy,
            }

            extra_state: dict[str, object] = {
                "optimizer_name": optimizer.__class__.__name__,
                "device": str(device),
            }
            if audio_config is not None:
                extra_state["audio_config"] = audio_config
            if training_config is not None:
                extra_state["training_config"] = training_config

            improved = checkpoint_manager.step(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                extra_state=extra_state,
            )

            if improved:
                print(
                    f"Saved new best checkpoint with "
                    f"{checkpoint_manager.monitor}={metrics[checkpoint_manager.monitor]:.4f}"
                )

            if early_stopping.step(val_metrics.loss):
                print(f"Early stopping at epoch {epoch}")
                break

        logger.log_hparams(
            hparams={
                "optimizer": optimizer.__class__.__name__,
                "epochs": num_epochs,
                "audio_sample_rate": audio_config.sample_rate if audio_config is not None else None,
                "audio_n_mels": audio_config.n_mels if audio_config is not None else None,
                "audio_frame_length_ms": (
                    audio_config.frame_length_ms if audio_config is not None else None
                ),
                "audio_frame_shift_ms": (
                    audio_config.frame_shift_ms if audio_config is not None else None
                ),
            },
            final_metrics={
                "hparam/val_loss": val_metrics.loss,
                "hparam/val_f1": val_metrics.f1,
            },
        )

    finally:
        logger.close()

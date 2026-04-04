from torch.utils.data import DataLoader

from src.vad.data.base import ProcessedVADDataset
from src.vad.data.collate import pad_collate_fn


def build_dataloader(
    raw_dataset,
    processor,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ProcessedVADDataset(
        base_dataset=raw_dataset,
        processor=processor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pad_collate_fn,
        num_workers=num_workers,
    )


def build_train_loader(
    raw_dataset,
    processor,
    batch_size: int = 16,
    num_workers: int = 4,
) -> DataLoader:
    return build_dataloader(
        raw_dataset=raw_dataset,
        processor=processor,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def build_eval_loader(
    raw_dataset,
    processor,
    batch_size: int = 16,
    num_workers: int = 4,
) -> DataLoader:
    return build_dataloader(
        raw_dataset=raw_dataset,
        processor=processor,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

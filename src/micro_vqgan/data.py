import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2


_train_transform = v2.Compose(
    [
        v2.Resize(256),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

_val_transform = v2.Compose(
    [
        v2.Resize(256),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def _apply_train_transforms(examples):
    return {"pixel_values": [_train_transform(img) for img in examples["image"]]}


def _apply_val_transforms(examples):
    return {"pixel_values": [_val_transform(img) for img in examples["image"]]}


class CelebADataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def setup(self, stage: str = ""):
        dataset = load_dataset("mattymchen/celeba-hq")
        self.train_dataset = dataset["train"].with_transform(_apply_train_transforms)
        self.val_dataset = dataset["validation"].with_transform(_apply_val_transforms)
        self.predict_dataset = dataset["train"].with_transform(_apply_val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 1,
            shuffle=True,
            drop_last=self.drop_last,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 1,
            shuffle=False,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 1,
            shuffle=False,
            drop_last=False,
        )

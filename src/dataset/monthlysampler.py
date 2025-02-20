import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomGeoSampler, Units


class CustomGeoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_img_path: Path,
        train_mask_path: Path,
        val_img_path: Path,
        val_mask_path: Path,
        batch_size: int,
        patch_size: int,
        length_train: int,
        length_validate: int,
        num_workers: int = 4,
        train_transform=None,
        val_transform=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_transform", "val_transform"])

        # Store parameters
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.length_train = length_train
        self.length_validate = length_validate
        self.num_workers = num_workers

        self.train_img_path = train_img_path
        self.train_mask_path = train_mask_path
        self.val_img_path = val_img_path
        self.val_mask_path = val_mask_path
        self.train_transform = train_transform
        self.val_transform = val_transform

        # Initialize datasets and samplers to None
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
        self.val_sampler = None

    def prepare_data(self):
        """Called only once and on one GPU."""
        # Nothing to download or prepare in this case
        pass

    def setup(self, stage: str = None):
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            # Setup training data
            train_imgs = RasterDataset(
                paths=str(self.train_img_path), crs="epsg:32633", res=10
            )
            train_masks = RasterDataset(
                paths=str(self.train_mask_path), crs="epsg:32633", res=10
            )
            train_masks.is_image = False
            self.train_dataset = train_imgs & train_masks

            if len(self.train_dataset) == 0:
                raise ValueError("Train dataset is empty!")

            self.train_sampler = RandomGeoSampler(
                dataset=self.train_dataset,
                size=self.patch_size,
                length=self.length_train,
                units=Units.PIXELS,
            )

        if stage == "validate" or stage is None:
            # Setup validation data
            val_imgs = RasterDataset(
                paths=str(self.val_img_path), crs="epsg:32633", res=10
            )
            val_masks = RasterDataset(
                paths=str(self.val_mask_path), crs="epsg:32633", res=10
            )
            val_masks.is_image = False
            self.val_dataset = val_imgs & val_masks

            if len(self.val_dataset) == 0:
                raise ValueError("Validation dataset is empty!")

            self.val_sampler = RandomGeoSampler(
                dataset=self.val_dataset,
                size=self.patch_size,
                length=self.length_validate,
                units=Units.PIXELS,
            )

    def train_dataloader(self):
        """Get train dataloader."""
        if self.train_dataset is None:
            raise ValueError("Training dataset is not initialized. Call setup() first")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            collate_fn=self._collate_fn_train,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        if self.val_dataset is None:
            raise ValueError(
                "Validation dataset is not initialized. Call setup() first"
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            collate_fn=self._collate_fn_val,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _collate_fn_train(self, batch):
        """Collate function with train transform applied."""
        return self._apply_transform(batch, self.train_transform)

    def _collate_fn_val(self, batch):
        """Collate function with val transform applied."""
        return self._apply_transform(batch, self.val_transform)

    def _apply_transform(self, batch, transform):
        """Apply the provided transformation to the batch."""
        num_months, bands_per_month = 5, 6
        augmented_samples = []

        for sample in batch:
            image = sample["image"].clone().detach()
            selected_month = random.randint(0, num_months - 1)  # noqa: S311
            band_start, band_end = (
                selected_month * bands_per_month,
                (selected_month + 1) * bands_per_month,
            )
            sample["image"] = image[band_start:band_end]

            # convert tensors to numpy
            image_np = sample["image"].numpy().transpose(1, 2, 0)
            mask_np = (
                sample["mask"].numpy().astype(np.uint8).transpose(1, 2, 0)
                if "mask" in sample
                else None
            )

            # apply augmentation if transform is provided
            if transform:
                augmented = (
                    transform(image=image_np, mask=mask_np)
                    if mask_np is not None
                    else transform(image=image_np)
                )
                sample["image"] = augmented["image"].clone().detach()

                if "mask" in sample:
                    sample["mask"] = augmented["mask"].clone().detach().permute(2, 0, 1)
            else:
                sample["image"] = (
                    torch.tensor(image_np.transpose(2, 0, 1)).clone().detach()
                )
                sample["mask"] = torch.tensor(mask_np).clone().detach().permute(2, 0, 1)

            for attr in ("crs", "bounds"):
                sample.pop(attr, None)

            augmented_samples.append(sample)

        return stack_samples(augmented_samples)

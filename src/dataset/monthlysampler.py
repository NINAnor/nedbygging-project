import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomGeoSampler, Units

from .normalizers import (
    adaptive_histogram_equalization,
    dynamic_world_normalization,
    normalize_min_max_image,
    normalize_percentile,
    normalize_percentile_per_channel,
)


class NormalizedRasterDataset(Dataset):
    """Dataset wrapper that applies normalization on-the-fly before sampling."""

    def __init__(self, dataset, normalize_conf):
        self.dataset = dataset
        self.normalize_conf = normalize_conf

        # Forward the spatial index and other attributes needed by RandomGeoSampler
        self.index = dataset.index if hasattr(dataset, "index") else None
        self.crs = dataset.crs if hasattr(dataset, "crs") else None
        self.res = dataset.res if hasattr(dataset, "res") else None
        self.is_image = dataset.is_image if hasattr(dataset, "is_image") else None

        # Forward any other attributes that might be needed by the sampler
        self._bounds = dataset._bounds if hasattr(dataset, "_bounds") else None

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.normalize_conf and "image" in sample:
            sample["image"] = self._normalize_sample(sample["image"])
        return sample

    def __len__(self):
        return len(self.dataset)

    def _normalize_sample(self, image):
        """Normalize a single sample."""
        normalize_functions = {
            "clahe": adaptive_histogram_equalization,
            "dynamic-world": dynamic_world_normalization,
            "min-max": normalize_min_max_image,
            "percentile": normalize_percentile,
            "percentile-per-channel": normalize_percentile_per_channel,
        }

        if self.normalize_conf in normalize_functions:
            return normalize_functions[self.normalize_conf](image)


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
        normalize_conf: str,
        num_workers: int = 4,
        train_transform=None,
        val_transform=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_transform", "val_transform"])
        self.normalize_conf = normalize_conf

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

            combined_dataset = train_imgs & train_masks

            if self.normalize_conf:
                self.train_dataset = NormalizedRasterDataset(
                    combined_dataset, normalize_conf=self.normalize_conf
                )
            else:
                self.train_dataset = combined_dataset

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
            combined_dataset = val_imgs & val_masks

            if self.normalize_conf:
                self.val_dataset = NormalizedRasterDataset(
                    combined_dataset, normalize_conf=self.normalize_conf
                )
            else:
                self.val_dataset = combined_dataset

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
        num_months = 5
        bands_per_month = 6

        augmented_samples = []

        for sample in batch:
            image = sample["image"].clone().detach()
            # mask = sample["mask"].clone().detach()

            # Select a random month
            selected_month = random.randint(0, num_months - 1)  # noqa: S311
            band_start = selected_month * bands_per_month
            band_end = band_start + bands_per_month

            sample["image"] = image[band_start:band_end, :, :]

            # Convert tensors to numpy for augmentation
            image_np = sample["image"].numpy().transpose(1, 2, 0)
            mask_np = sample["mask"].numpy().astype(np.uint8).transpose(1, 2, 0)

            # Apply augmentation if transform is provided
            if transform is not None:
                augmented = transform(image=image_np, mask=mask_np)

                # Ensure 'augmented' outputs are numpy arrays before conversion
                if isinstance(augmented["image"], np.ndarray):
                    sample["image"] = (
                        torch.from_numpy(augmented["image"]).clone().detach()
                    )
                else:
                    sample["image"] = augmented["image"].clone().detach()

                if isinstance(augmented["mask"], np.ndarray):
                    sample["mask"] = (
                        torch.from_numpy(augmented["mask"])
                        .clone()
                        .detach()
                        .permute(2, 0, 1)
                    )
                else:
                    sample["mask"] = augmented["mask"].clone().detach().permute(2, 0, 1)
            else:
                sample["image"] = (
                    torch.tensor(image_np.transpose(2, 0, 1)).clone().detach()
                )
                sample["mask"] = torch.tensor(mask_np).clone().detach().permute(2, 0, 1)

            augmented_samples.append(sample)

        # Remove frozen attributes
        for sample in augmented_samples:
            sample.pop("crs", None)
            sample.pop("bounds", None)

        stacked_batch = stack_samples(augmented_samples)

        return stacked_batch

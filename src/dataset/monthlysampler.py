from pathlib import Path
from torchgeo.datasets import RasterDataset, unbind_samples, stack_samples
from torchgeo.samplers import RandomGeoSampler, Units
from torch.utils.data import DataLoader
from typing import Union, Callable
import random
import torch
import numpy as np


def select_random_month_and_augment(batch, transform):
    num_months = 5
    bands_per_month = 6

    augmented_samples = []

    for sample in batch:
        image = sample['image']
        mask = sample['mask']

        # Randomly select a month
        selected_month = random.randint(0, num_months - 1)
        band_start = selected_month * bands_per_month
        band_end = band_start + bands_per_month

        # Select the bands for the chosen month (6 bands)
        sample['image'] = image[band_start:band_end, :, :]

        # Convert tensors to NumPy for Albumentations processing
        image_np = sample['image'].numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        mask_np = np.transpose(sample['mask'].numpy(), (1, 2, 0)).astype(np.uint8)

        # Apply augmentation (only image is normalized)
        augmented = transform(image=image_np, mask=mask_np)

        # Convert back to PyTorch tensors
        sample['image'] = torch.tensor(augmented['image']).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        sample['mask'] = torch.tensor(augmented['mask']).permute(2, 0, 1)  # Back to (1, H, W)

        sample['selected_month'] = selected_month  # Store the selected month
        augmented_samples.append(sample)

    return stack_samples(augmented_samples)


def loader(
    path_imgs: Union[str, Path], 
    path_masks: Union[str, Path], 
    size: int, 
    length: int, 
    batch_size: int, 
    transform: Callable = None,  # Transformation function as a parameter
) -> DataLoader:

    # Ensure paths are Path objects for better handling
    path_imgs = Path(path_imgs)
    path_masks = Path(path_masks)

    # Load Raster datasets for images and masks
    imgs = RasterDataset(paths=path_imgs.as_posix(), crs='epsg:32633', res=10)
    masks = RasterDataset(paths=path_masks.as_posix(), crs='epsg:32633', res=10)
    masks.is_image = False  # Ensure masks are not treated as images

    # Set up the spatial sampler
    sampler = RandomGeoSampler(dataset=imgs, size=size, length=length, units=Units.PIXELS)

    # Combine images and masks datasets
    combined_dataset = imgs & masks

    collate_fn = lambda batch: select_random_month_and_augment(batch, transform)

    # Create the DataLoader
    dataloader = DataLoader(
        dataset=combined_dataset, 
        sampler=sampler, 
        batch_size=batch_size, 
        collate_fn=collate_fn
    )

    return dataloader




import logging

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from scipy.special import expit as sigmoid


def normalize_percentile(img_array, percentiles=(2, 98)):
    """Normalize image using percentile clipping to enhance contrast."""
    min_val, max_val = np.percentile(img_array, percentiles)
    return np.clip((img_array - min_val) / (max_val - min_val), 0, 1)


def normalize_image(image):
    """Min-max normalization per channel."""
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)


def adaptive_histogram_equalization(image):
    """Apply CLAHE per channel separately."""
    image = (image * 255).astype(np.uint8)  # convert to uint8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for i in range(image.shape[-1]):  # apply CLAHE per channel
        image[..., i] = clahe.apply(image[..., i])

    return image / 255.0  # scale back to [0,1]


def dynamic_world_normalization(image, p_low=30, p_high=70, log_transform=True):
    """
    Apply the Dynamic World normalization scheme to an image.

    Parameters:
        image (np.ndarray): Input image array (H, W, C).
        p_low (int): Lower percentile for remapping.
        p_high (int): Upper percentile for remapping.
        log_transform (bool): Whether to apply log-transform.

    Returns:
        np.ndarray: Normalized image.
    """
    image = image.astype(np.float32)

    if log_transform:
        image = np.log1p(image)  # log(1 + x) to prevent log(0) issues

    norm_image = np.zeros_like(image)

    for c in range(image.shape[2]):  # Process each channel separately
        low, high = np.percentile(image[..., c], [p_low, p_high])
        if low == high:
            norm_image[..., c] = 0.5  # Assign a neutral value if no variance
        else:
            norm_image[..., c] = sigmoid((image[..., c] - low) / (high - low))

    return norm_image


def preprocess_image(image, **kwargs):
    """
    Preprocess image for model input
    Used for testing the methods in this file.

    Example:
    train_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Lambda(image=preprocess_image), # <----here
            ToTensorV2(),
        ]

    """
    image = normalize_percentile(image)
    return image.astype(np.float32)


def build_transform(mode="train"):
    logger = logging.getLogger(__name__)

    if mode == "train":
        train_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Lambda(name="normalize_percentile", image=preprocess_image),
            ToTensorV2(),
        ]

        transform = A.Compose(
            train_transforms,
            additional_targets={"mask": "mask"},
        )
        logger.info(f"Training transforms: {train_transforms}")

    elif mode == "val" or mode == "test":
        print("Applying validation/test transforms.")
        val_transforms = [
            A.Lambda(name="normalize_percentile", image=preprocess_image),
            ToTensorV2(),
        ]

        transform = A.Compose(
            val_transforms,
            additional_targets={"mask": "mask"},
        )
        logger.info(f"Validation/test transforms: {val_transforms}")
    return transform

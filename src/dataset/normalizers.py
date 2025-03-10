import albumentations as A
import cv2
import numpy as np
from scipy.special import expit as sigmoid


def batch_normalize(batch, axis=(2, 3), c=1e-8):
    """
    Normalize a batch to zero mean and unit standard deviation.
    Args:
        batch (torch.Tensor): (N, C, H, W) shape
        axis (tuple): Axes over which to normalize (height, width)
        c (float): Small constant to avoid division by zero
    """
    mean = batch.mean(dim=axis, keepdim=True)
    std = batch.std(dim=axis, keepdim=True) + c
    return (batch - mean) / std


def normalize_percentile_per_channel(image, percentiles=(2, 98), **kwargs):
    """Normalize each band independently using percentile clipping."""
    normalized_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):  # Assuming (channels, height, width) format
        min_val, max_val = np.percentile(image[i], percentiles)
        normalized_image[i] = np.clip((image[i] - min_val) / (max_val - min_val), 0, 1)

    return normalized_image.astype(np.float32)


def normalize_percentile(image, percentiles=(2, 98), **kwargs):
    """Normalize image using percentile clipping to enhance contrast."""
    min_val, max_val = np.percentile(image, percentiles)
    return np.clip((image - min_val) / (max_val - min_val), 0, 1).astype(np.float32)


def normalize_min_max_image(image, **kwargs):
    """Min-max normalization per channel."""
    return ((image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)).astype(
        np.float32
    )


def adaptive_histogram_equalization(image, **kwargs):
    """Apply CLAHE per channel separately."""
    image = normalize_min_max_image(image)  # Normalize to [0,1]
    image = (image * 255).astype(np.uint8)  # Convert to uint8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(image.shape[-1]):  # Apply CLAHE per channel
        image[..., i] = clahe.apply(image[..., i])
    return (image / 255.0).astype(np.float32)  # Scale back to [0,1]


def dynamic_world_normalization(
    image, p_low=30, p_high=70, log_transform=True, **kwargs
):
    """Apply the Dynamic World normalization scheme to an image."""
    image = image.astype(np.float32)
    image = normalize_min_max_image(image)

    if log_transform:
        image = np.log1p(image)  # log(1 + x) to prevent log(0) issues

    norm_image = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):  # Process each channel separately
        low, high = np.percentile(image[..., c], [p_low, p_high])
        if low == high:
            norm_image[..., c] = 0.5  # Assign a neutral value if no variance
        else:
            norm_image[..., c] = sigmoid((image[..., c] - low) / (high - low))

    return norm_image.astype(np.float32)


def get_normalize_technique(transforms, normalize_conf):
    normalize_functions = {
        "clahe": adaptive_histogram_equalization,
        "dynamic-world": dynamic_world_normalization,
        "min-max": normalize_min_max_image,
        "percentile": normalize_percentile,
        "percentile-per-channel": normalize_percentile_per_channel,
    }

    if normalize_conf in normalize_functions:
        transforms.append(
            A.Lambda(
                name=normalize_conf,
                image=lambda img, **kwargs: normalize_functions[normalize_conf](
                    img, **kwargs
                ),
            )
        )
    return transforms

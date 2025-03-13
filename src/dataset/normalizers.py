import logging

import cv2
import numpy as np
import torch
from scipy.special import expit as sigmoid


def normalize_percentile_per_channel(image, percentiles=(2, 98), **kwargs):
    """Normalize each band independently using percentile clipping."""
    # check if input is a torch tensor and convert if needed
    logger = logging.getLogger(__name__)
    logger.info("Normalizing image using percentile clipping per channel")
    logger.info(f"Percentiles: {percentiles}")
    is_torch_tensor = isinstance(image, torch.Tensor)
    if is_torch_tensor:
        device = image.device
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image

    normalized_image = np.zeros_like(image_np, dtype=np.float32)

    for i in range(image_np.shape[0]):  # Assuming (channels, height, width) format
        min_val, max_val = np.percentile(image_np[i], percentiles)
        normalized_image[i] = np.clip(
            (image_np[i] - min_val) / (max_val - min_val + 1e-8), 0, 1
        )

    # Convert back to tensor if input was tensor
    if is_torch_tensor:
        return torch.from_numpy(normalized_image).to(device)
    return normalized_image.astype(np.float32)


def normalize_percentile(image, percentiles=(2, 98), **kwargs):
    """Normalize image using percentile clipping to enhance contrast."""
    # check if input is a torch tensor and convert if needed
    logger = logging.getLogger(__name__)
    logger.info("Normalizing image using percentile clipping")
    logger.info(f"Percentiles: {percentiles}")
    is_torch_tensor = isinstance(image, torch.Tensor)
    if is_torch_tensor:
        device = image.device
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image

    min_val, max_val = np.percentile(image_np, percentiles)
    normalized = np.clip(
        (image_np - min_val) / (max_val - min_val + 1e-8), 0, 1
    ).astype(np.float32)

    # convert back to tensor if input was tensor
    if is_torch_tensor:
        return torch.from_numpy(normalized).to(device)
    return normalized


def normalize_min_max_image(image, **kwargs):
    """Min-max normalization per channel."""
    # check if input is a torch tensor and convert if needed
    is_torch_tensor = isinstance(image, torch.Tensor)
    if is_torch_tensor:
        device = image.device
        min_val = torch.min(image)
        max_val = torch.max(image)
        return ((image - min_val) / (max_val - min_val + 1e-8)).to(device)
    else:
        return (
            (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        ).astype(np.float32)


def adaptive_histogram_equalization(image, **kwargs):
    """Apply CLAHE per channel separately."""
    # check if input is a torch tensor and convert if needed
    is_torch_tensor = isinstance(image, torch.Tensor)
    if is_torch_tensor:
        device = image.device
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image

    # CLAHE needs numpy array processing
    image_np = normalize_min_max_image(image_np)  # Normalize to [0,1]
    image_np = (image_np * 255).astype(np.uint8)  # Convert to uint8

    # handle channel dimension correctly
    is_channels_first = (
        image_np.shape[0] < image_np.shape[1] and image_np.shape[0] < image_np.shape[2]
    )

    if is_channels_first:
        # Convert from (C, H, W) to (H, W, C) for OpenCV
        image_np = np.transpose(image_np, (1, 2, 0))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(image_np.shape[-1]):  # Apply CLAHE per channel
        image_np[..., i] = clahe.apply(image_np[..., i])

    # convert back to (C, H, W) if needed
    if is_channels_first:
        image_np = np.transpose(image_np, (2, 0, 1))

    normalized = (image_np / 255.0).astype(np.float32)  # Scale back to [0,1]

    # convert back to tensor if input was tensor
    if is_torch_tensor:
        return torch.from_numpy(normalized).to(device)
    return normalized


def dynamic_world_normalization(
    image, p_low=30, p_high=70, log_transform=True, **kwargs
):
    """Apply the Dynamic World normalization scheme to an image."""

    logger = logging.getLogger(__name__)
    logger.info("Normalizing image using Dynamic World Normalization")
    logger.info(f"Percentiles: {p_low} - {p_high}")
    # Check if input is a torch tensor and convert if needed
    is_torch_tensor = isinstance(image, torch.Tensor)
    if is_torch_tensor:
        device = image.device
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image

    # check if channels are first or last
    is_channels_first = (
        image_np.shape[0] < image_np.shape[1] and image_np.shape[0] < image_np.shape[2]
    )

    # convert to (H, W, C) for processing if needed
    if is_channels_first:
        image_np = np.transpose(image_np, (1, 2, 0))

    image_np = image_np.astype(np.float32)
    image_np = normalize_min_max_image(image_np)

    if log_transform:
        image_np = np.log1p(image_np)  # prevent log(0) issues

    norm_image = np.zeros_like(image_np, dtype=np.float32)
    for c in range(image_np.shape[2]):  # Process each channel separately
        low, high = np.percentile(image_np[..., c], [p_low, p_high])
        if low == high:
            norm_image[..., c] = 0.5  # assign a neutral value if no variance
        else:
            norm_image[..., c] = sigmoid((image_np[..., c] - low) / (high - low))

    # convert back to (C, H, W) if needed
    if is_channels_first:
        norm_image = np.transpose(norm_image, (2, 0, 1))

    # convert back to tensor if input was tensor
    if is_torch_tensor:
        return torch.from_numpy(norm_image).to(device)
    return norm_image.astype(np.float32)

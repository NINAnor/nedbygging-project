import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


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


def preprocess_image(image):
    """Applies both CLAHE and normalization before Albumentations."""
    image = normalize_image(image)
    image = adaptive_histogram_equalization(image)
    return image


def preprocess_fn(image, **kwargs):
    """Wrapper function for Albumentations Lambda, ensures NumPy array output."""
    return preprocess_image(image).astype(np.float32)


def build_transform(mode="train", plot_batch=False):
    if mode == "train":
        train_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.Normalize(),  # OBS the lambda function below does the normalization
            A.Lambda(image=preprocess_fn),
            A.RandomBrightnessContrast(p=1),
        ]

        if not plot_batch:
            train_transforms.append(
                ToTensorV2(),
            )

        print(train_transforms)
        transform = A.Compose(
            train_transforms,
            additional_targets={"mask": "mask"},
        )

    elif mode == "val":
        val_transforms = [
            A.Normalize(),  # OBS the lambda function below does the normalization
            A.Lambda(image=preprocess_fn),
        ]

        if not plot_batch:
            val_transforms.append(ToTensorV2())

        transform = A.Compose(
            val_transforms,
            additional_targets={"mask": "mask"},
        )
        print(val_transforms)

    return transform

import logging

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .normalizers import get_normalize_technique


def build_transform(mode="train", normalize_conf="min-max"):
    """Builds the appropriate transform for the given mode."""
    logger = logging.getLogger(__name__)

    if mode == "train":
        transform_list = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]

        transform_list = get_normalize_technique(
            transform_list, normalize_conf=normalize_conf
        )
        transform_list.append(ToTensorV2())

        transforms = A.Compose(
            transforms=transform_list,
            additional_targets={"mask": "mask"},
        )
        logger.info(f"Training transforms: {transform_list}")

    elif mode == "val" or mode == "test":
        print("Applying validation/test transforms.")
        transform_list = get_normalize_technique([], normalize_conf=normalize_conf)
        transform_list.append(ToTensorV2())

        transforms = A.Compose(
            transforms=transform_list,
            additional_targets={"mask": "mask"},
        )
        logger.info(f"Validation/test transforms: {transform_list}")
    return transforms

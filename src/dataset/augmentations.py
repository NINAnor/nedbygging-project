import logging

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transform(mode="train"):
    """
    Builds the appropriate transform for the given mode.
    The images are normalized before they reach this point,
    so we normalize the entire image, not only the 256x256 patches.
    """
    logger = logging.getLogger(__name__)

    if mode == "train":
        transform_list = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ]

        transforms = A.Compose(
            transforms=transform_list,
            additional_targets={"mask": "mask"},
        )
        logger.info(f"Training transforms: {transform_list}")

    elif mode == "val" or mode == "test":
        logger.info("Applying validation/test transforms.")

        transforms = A.Compose(
            transforms=[ToTensorV2()],
            additional_targets={"mask": "mask"},
        )
        logger.info(f"Validation/test transforms: {transforms}")
    return transforms

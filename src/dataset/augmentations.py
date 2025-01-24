import torch
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transform(mode='train', plot_batch=False):

    if mode == 'train':

        if plot_batch:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                #A.OneOf([
                #    A.RandomBrightnessContrast(p=0.5),
                #    A.GaussNoise(p=0.5),
                #], p=0.8),

                A.Normalize(normalization="min_max_per_channel"),

            ], additional_targets={'mask': 'mask'})
             
        else:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.GaussNoise(p=0.5),
                ], p=0.8),

                A.Normalize(normalization="min_max_per_channel"),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})


    elif mode == 'val':
        if plot_batch:
            transform = A.Compose([
                A.Normalize(normalization="min_max_per_channel"),
            ], additional_targets={'mask': 'mask'})

        else:
            transform = A.Compose([
                A.Normalize(normalization="min_max_per_channel"),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})

    return transform



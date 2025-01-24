import logging
import pathlib

import hydra

from pathlib import Path

from dataset.monthlysampler import loader
from dataset.augmentations import build_transform

BASE_DIR = pathlib.Path(__file__).parent.parent

logging.basicConfig(level=(logging.INFO))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):

    root = Path(cfg.paths.ROOT_PATH)

    train_path_imgs = root/'tra_scene'
    train_path_masks = root/'tra_truth'
    val_path_imgs = root/'val_scene'
    val_path_masks = root/'val_truth'

    # instantiate the transform
    train_transform = build_transform(mode='train')
    val_transform = build_transform(mode='val')

    # instantiate the training dataloader

    train_loader = loader(train_path_imgs, 
                               train_path_masks, 
                               cfg.training.SIZE, 
                               cfg.training.LENGTH, 
                               cfg.training.BATCH_SIZE_FOR_TRAINING, 
                               transform=train_transform)
    
    val_loader = loader(val_path_imgs, 
                               val_path_masks, 
                               cfg.training.SIZE, 
                               cfg.training.LENGTH, 
                               cfg.training.BATCH_SIZE_FOR_TRAINING, 
                               transform=val_transform)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("Train Image Shape:", train_batch['image'].shape)  # expect: (batch_size, 6, H, W)
    print("Val Image Shape:", val_batch['image'].shape)      # expect: (batch_size, 6, H, W)

    # Checking normalization values
    print("Train Image Min/Max:", train_batch['image'].min(), train_batch['image'].max())
    print("Val Image Min/Max:", val_batch['image'].min(), val_batch['image'].max())






if __name__ == "__main__":
    main()

import logging
import pathlib

import hydra

from pathlib import Path

import matplotlib.pyplot as plt

from dataset.monthlysampler import loader
from dataset.augmentations import build_transform
from dataset.visualisation import plot_batch

BASE_DIR = pathlib.Path(__file__).parent.parent

logging.basicConfig(level=(logging.INFO))

def plotBatch(train_loader, val_loader):

    print("PLOTTING A BATCH OF IMAGE AND MASKS")

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    plot_batch(train_batch, bright=3., cols=4, width=5, chnls=[0, 1, 2])
    plt.suptitle('Training Batch')
    plt.savefig("training_batch.png")

    plot_batch(val_batch, bright=3., cols=4, width=5, chnls=[0, 1, 2])
    plt.suptitle('Validation Batch')
    plt.savefig("validation_batch.png")

    exit()



@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):

    root = Path(cfg.paths.ROOT_PATH)

    train_path_imgs = root/'tra_scene'
    train_path_masks = root/'tra_truth'
    val_path_imgs = root/'val_scene'
    val_path_masks = root/'val_truth'

    # instantiate the transform
    train_transform = build_transform(mode='train', plot_batch=cfg.training.PLOT_BATCH)
    val_transform = build_transform(mode='val', plot_batch=cfg.training.PLOT_BATCH)

    # instantiate the training dataloader

    train_loader = loader(train_path_imgs, 
                               train_path_masks, 
                               cfg.training.SIZE, 
                               cfg.training.LENGTH, 
                               cfg.training.BATCH_SIZE, 
                               transform=train_transform)
    
    val_loader = loader(val_path_imgs, 
                               val_path_masks, 
                               cfg.training.SIZE, 
                               cfg.training.LENGTH, 
                               cfg.training.BATCH_SIZE, 
                               transform=val_transform)

    if cfg.training.PLOT_BATCH:
        plotBatch(train_loader, val_loader)

if __name__ == "__main__":
    main()

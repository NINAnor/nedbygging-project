import logging
from pathlib import Path

import hydra
import torch
import torch.utils
import torch.utils.data
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataset.augmentations import build_transform
from dataset.monthlysampler import CustomGeoDataModule
from dataset.visualisation import plot_batch
from model import SegmentationModel

torch.backends.cudnn.benchmark = True


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    root = Path(cfg.paths.ROOT_PATH)
    output_dir = HydraConfig.get().run.dir
    logging.basicConfig(
        filename=f"{output_dir}/training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Files will be saved to: {output_dir}")

    train_path_imgs = root / "new_train_val_scene" / "train"
    train_path_masks = root / "new_train_val_truth" / "train"
    val_path_imgs = root / "new_train_val_scene" / "val"
    val_path_masks = root / "new_train_val_truth" / "val"

    # instantiate the transform
    train_transform = build_transform(mode="train", normalize_conf=cfg.data.NORMALIZE)
    val_transform = build_transform(mode="val", normalize_conf=cfg.data.NORMALIZE)

    data_module = CustomGeoDataModule(
        train_img_path=train_path_imgs,
        train_mask_path=train_path_masks,
        val_img_path=val_path_imgs,
        val_mask_path=val_path_masks,
        batch_size=cfg.training.BATCH_SIZE,
        patch_size=cfg.training.PATCH_SIZE,
        length_train=cfg.training.LENGTH_TRAIN,
        length_validate=cfg.training.LENGTH_VALIDATE,
        train_transform=train_transform,
        val_transform=val_transform,
        standardize=cfg.data.STANDARDIZE,
    )
    data_module.setup("fit")
    data_module.setup("validate")

    if cfg.training.PLOT_BATCH:
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        plot_batch(output_dir, train_loader, val_loader)

    num_classes = cfg.training.NUM_CLASSES
    model = SegmentationModel(
        num_classes=num_classes,
        lr=cfg.training.LR,
        criterion=cfg.training.LOSS,
        model_cfg=cfg.training.MODEL,
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.training.PATIENCE,
        verbose=True,
        mode="min",
    )

    trainer = Trainer(
        log_every_n_steps=3,
        default_root_dir=output_dir,
        max_epochs=cfg.training.NUM_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)


if __name__ == "__main__":
    main()

import json
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
    # Convert output_dir to a Path object
    output_dir = Path(HydraConfig.get().run.dir)

    logging.basicConfig(
        filename=str(
            output_dir / "training.log"
        ),  # Convert Path to string for filename
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
    train_transform = build_transform(mode="train")
    val_transform = build_transform(mode="val")

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
        normalize_conf=cfg.data.NORMALIZE,
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
    results = trainer.validate(model, datamodule=data_module)

    metrics_txt_path = output_dir / "validation_metrics.txt"
    with Path.open(metrics_txt_path, "w") as f:
        f.write("Validation Metrics\n")
        f.write("=" * 80 + "\n\n")

        # Format the results in a readable table
        for key, value in results[0].items():
            f.write(f"{key:<40} {value:.6f}\n")

    # Also save as JSON for programmatic access
    metrics_json_path = output_dir / "validation_metrics.json"
    with Path.open(metrics_json_path, "w") as f:
        json.dump(results[0], f, indent=4)

    logger.info(f"Validation metrics saved to {metrics_txt_path}")
    logger.info("Training finished.")


if __name__ == "__main__":
    main()

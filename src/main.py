import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchmetrics
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import MulticlassJaccardIndex

from dataset.augmentations import build_transform
from dataset.monthlysampler import CustomGeoDataModule
from dataset.visualisation import plotBatch

logging.basicConfig(level=(logging.INFO))

torch.backends.cudnn.benchmark = True


def get_deeplabv3_model(num_classes):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=6,
        classes=num_classes,
    )

    # model = models.deeplabv3_resnet50(weights="COCO_WITH_VOC_LABELS_V1")
    # model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    return model


class SegmentationModel(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-4):
        super().__init__()
        self.model = get_deeplabv3_model(num_classes)
        self.lr = lr
        self.num_classes = num_classes

        # Metrics from torchmetrics
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=-1
        )
        self.train_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1
        )
        self.train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1
        )
        self.train_iou = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=-1
        )

        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=-1
        )
        self.val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1
        )
        self.val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1
        )
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=-1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"].squeeze(1).long()

        outputs = self(images)

        loss = F.cross_entropy(outputs, masks, ignore_index=-1)

        pred = torch.argmax(outputs, dim=1)

        # Log metrics using torchmetrics
        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy(pred, masks))
        self.log("train_precision", self.train_precision(pred, masks))
        self.log("train_recall", self.train_recall(pred, masks))
        self.log("train_iou", self.train_iou(pred, masks))

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"].squeeze(1).long()
        outputs = self(images)

        val_loss = F.cross_entropy(outputs, masks, ignore_index=-1)

        pred = torch.argmax(outputs, dim=1)

        # Log validation metrics
        self.log("val_loss", val_loss)
        self.log("val_accuracy", self.val_accuracy(pred, masks))
        self.log("val_precision", self.val_precision(pred, masks))
        self.log("val_recall", self.val_recall(pred, masks))
        self.log("val_iou", self.val_iou(pred, masks))

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    root = Path(cfg.paths.ROOT_PATH)
    output_dir = HydraConfig.get().run.dir

    logging.basicConfig(
        filename=f"{output_dir}/training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    train_path_imgs = root / "tra_scene"
    train_path_masks = root / "tra_truth"
    val_path_imgs = root / "val_scene"
    val_path_masks = root / "val_truth"

    # instantiate the transform
    train_transform = build_transform(mode="train", plot_batch=cfg.training.PLOT_BATCH)
    val_transform = build_transform(mode="val", plot_batch=cfg.training.PLOT_BATCH)

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
    )

    if cfg.training.PLOT_BATCH:
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        plotBatch(train_loader, val_loader)

    num_classes = cfg.training.NUM_CLASSES
    model = SegmentationModel(num_classes=num_classes, lr=cfg.training.LR)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.training.PATIENCE,
        verbose=True,
        mode="min",
    )

    data_module.setup("fit")
    data_module.setup("validate")

    trainer = Trainer(
        default_root_dir=output_dir,
        max_epochs=cfg.training.NUM_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)


if __name__ == "__main__":
    main()

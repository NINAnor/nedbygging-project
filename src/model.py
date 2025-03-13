import logging

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)

from utils import CLASS_LABELS


def dice_loss(pred, target, num_classes, smooth=1.0):
    """Computes Dice Loss for multi-class segmentation."""
    pred = F.softmax(pred, dim=1)  # apply softmax to get class probabilities
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()  # dice Loss


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Computes Focal Loss for multi-class segmentation."""
    ce_loss = F.cross_entropy(pred, target, reduction="none")
    pt = torch.exp(-ce_loss)  # probability of correct class
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()


def get_model(model_cfg, num_classes):
    if model_cfg == "deeplabv3":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=6,
            classes=num_classes,
        )

    elif model_cfg == "unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=6,
            classes=num_classes,
        )

    return model


class SegmentationModel(pl.LightningModule):
    def __init__(self, num_classes, criterion, model_cfg, lr=1e-4):
        super().__init__()
        self.model = get_model(model_cfg, num_classes)
        self.lr = lr
        self.num_classes = num_classes
        self.criterion = criterion

        # overall metrics
        self.train_accuracy = MulticlassAccuracy(
            num_classes=num_classes, average="macro"
        )
        self.train_precision = MulticlassPrecision(
            num_classes=num_classes, average="macro"
        )
        self.train_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.train_iou = MulticlassJaccardIndex(
            num_classes=num_classes, average="macro"
        )

        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_precision = MulticlassPrecision(
            num_classes=num_classes, average="macro"
        )
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, average="macro")

        # class-wise metrics
        self.train_class_accuracy = MulticlassAccuracy(
            num_classes=num_classes, average=None
        )
        self.val_class_accuracy = MulticlassAccuracy(
            num_classes=num_classes, average=None
        )

        self.train_class_precision = MulticlassPrecision(
            num_classes=num_classes, average=None
        )
        self.val_class_precision = MulticlassPrecision(
            num_classes=num_classes, average=None
        )

        self.train_class_recall = MulticlassRecall(
            num_classes=num_classes, average=None
        )
        self.val_class_recall = MulticlassRecall(num_classes=num_classes, average=None)

        self.train_class_iou = MulticlassJaccardIndex(
            num_classes=num_classes, average=None
        )
        self.val_class_iou = MulticlassJaccardIndex(
            num_classes=num_classes, average=None
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"].squeeze(1).long()

        outputs = self(images)

        # Compute losses
        if self.criterion == "dice":
            dice = dice_loss(outputs, masks, num_classes=self.num_classes)
            focal = focal_loss(outputs, masks)
            self.log("train_dice_loss", dice, prog_bar=True, on_epoch=True)
            self.log("train_focal_loss", focal, prog_bar=True, on_epoch=True)
            train_loss = 0.5 * dice + 0.5 * focal
        elif self.criterion == "cross_entropy":
            train_loss = F.cross_entropy(outputs, masks)

        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True)

        pred = torch.argmax(outputs, dim=1)

        # Log overall metrics
        self.log("train_accuracy", self.train_accuracy(pred, masks))
        self.log("train_precision", self.train_precision(pred, masks))
        self.log("train_recall", self.train_recall(pred, masks))
        self.log("train_iou", self.train_iou(pred, masks))

        # Log class-wise metrics
        class_accuracy = self.train_class_accuracy(pred, masks)
        class_precision = self.train_class_precision(pred, masks)
        class_recall = self.train_class_recall(pred, masks)
        class_iou = self.train_class_iou(pred, masks)

        for label, index in CLASS_LABELS.items():
            classname = (
                f"train_class_{label}"  # Using the string label from the dictionary
            )
            self.log(f"{classname}_accuracy", class_accuracy[index])
            self.log(f"{classname}_precision", class_precision[index])
            self.log(f"{classname}_recall", class_recall[index])
            self.log(f"{classname}_iou", class_iou[index])

        return train_loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"].squeeze(1).long()
        outputs = self(images)

        # Compute losses
        if self.criterion == "dice":
            dice = dice_loss(outputs, masks, num_classes=self.num_classes)
            focal = focal_loss(outputs, masks)
            self.log("val_dice_loss", dice, prog_bar=True, on_epoch=True)
            self.log("val_focal_loss", focal, prog_bar=True, on_epoch=True)
            val_loss = 0.5 * dice + 0.5 * focal
        elif self.criterion == "cross_entropy":
            val_loss = F.cross_entropy(outputs, masks)

        pred = torch.argmax(outputs, dim=1)

        # Log overall losses
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)

        # Log overall metrics
        self.log("val_accuracy", self.val_accuracy(pred, masks))
        self.log("val_precision", self.val_precision(pred, masks))
        self.log("val_recall", self.val_recall(pred, masks))
        self.log("val_iou", self.val_iou(pred, masks))

        # Log class-wise metrics
        class_accuracy = self.val_class_accuracy(pred, masks)
        class_precision = self.val_class_precision(pred, masks)
        class_recall = self.val_class_recall(pred, masks)
        class_iou = self.val_class_iou(pred, masks)

        for label, index in CLASS_LABELS.items():
            classname = (
                f"val_class_{label}"  # Using the string label from the dictionary
            )
            self.log(f"{classname}_accuracy", class_accuracy[index])
            self.log(f"{classname}_precision", class_precision[index])
            self.log(f"{classname}_recall", class_recall[index])
            self.log(f"{classname}_iou", class_iou[index])

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def load_model(checkpoint_path, num_classes, device):
    model = get_model(num_classes)
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from {checkpoint_path}")
    # Load the saved state dict
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Remove the 'model.' prefix from the state_dict keys
    state_dict = {
        k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()
    }

    # Filter out the auxiliary classifier keys
    state_dict = {k: v for k, v in state_dict.items() if "aux_classifier" not in k}

    # Load the modified state_dict into the model
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

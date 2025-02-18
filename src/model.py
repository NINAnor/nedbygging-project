import logging

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassJaccardIndex


def get_deeplabv3_model(num_classes):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=6,
        classes=num_classes,
    )

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


def load_model(checkpoint_path, num_classes, device):
    model = get_deeplabv3_model(num_classes)
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

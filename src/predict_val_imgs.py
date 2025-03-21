import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from dataset.augmentations import build_transform
from dataset.monthlysampler import CustomGeoDataModule
from dataset.visualisation import get_color_map_and_legend
from model import load_model


@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    root = Path(cfg.paths.ROOT_PATH)
    output_path = Path(cfg.paths.OUTPUT_PATH) / "val_preds"
    output_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(output_path / "pred.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    val_path_imgs = root / "new_train_val_scene" / "val"
    val_path_mask = root / "new_train_val_truth" / "val"

    test_transform = build_transform(mode="test")

    data_module = CustomGeoDataModule(
        test_img_path=val_path_imgs,
        test_mask_path=val_path_mask,
        test_transform=test_transform,
        normalize_conf=cfg.data.NORMALIZE,
        batch_size=cfg.training.BATCH_SIZE,
        patch_size=cfg.training.PATCH_SIZE,
        length_test=cfg.training.LENGTH_VALIDATE,
    )
    data_module.setup("test_on_val")

    test_dataloader = data_module.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = load_model(
        model_cfg=cfg.training.MODEL,
        checkpoint_path=cfg.paths.CHECKPOINT_PATH,
        num_classes=cfg.training.NUM_CLASSES,
        device=device,
    )

    logger.info(f"The predictions are stored at {output_path}")
    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(test_dataloader),
            desc="Generating predictions",
            total=len(test_dataloader),
            unit="batch",
        ):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            # save predictions
            batch_size = images.shape[0]
            for i in range(batch_size):
                # get current image and prediction
                img = images[i]
                pred = preds[i]
                mask = masks[i]

                # save prediction as numpy array
                save_path = output_path / f"prediction_patch_{batch_idx}_{i}.npy"
                np.save(save_path, pred.cpu().numpy())

                # create visualization with custom plotting functions
                fig, axs = plt.subplots(1, 3, figsize=(15, 6))

                img = img.float()
                img = img.permute([1, 2, 0])
                img = img[:, :, :3]
                img = img.cpu().numpy()

                axs[0].imshow(img)
                axs[0].axis("off")
                axs[0].set_title("Input Image")
                class_colors, legend_elements = get_color_map_and_legend()

                fig.legend(
                    handles=legend_elements,
                    loc="lower center",
                    title="Classes",
                    ncol=len(legend_elements),  # Arrange legend elements horizontally
                    bbox_to_anchor=(0.5, -0.05),  # Move legend slightly below the plot
                )

                # plot the ground truth mask
                axs[1].imshow(
                    mask.squeeze().cpu().numpy(), cmap=ListedColormap(class_colors)
                )
                axs[1].axis("off")
                axs[1].set_title("Ground truth mask")

                # plot the predicted mask
                axs[2].imshow(
                    pred.squeeze().cpu().numpy(), cmap=ListedColormap(class_colors)
                )
                axs[2].axis("off")
                axs[2].set_title("Predicted mask")

                # add overall title
                plt.suptitle("Prediction", fontsize=16)
                plt.tight_layout()

                plt_path = output_path / f"prediction_patch_{batch_idx}_{i}.png"
                plt.savefig(plt_path, bbox_inches="tight", dpi=150)
                plt.close(fig)

    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()

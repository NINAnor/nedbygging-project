# import logging
# from pathlib import Path

# import hydra
# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# from dataset.augmentations import build_transform
# from dataset.monthlysampler import CustomGeoDataModule
# from model import load_model


# @hydra.main(config_path="../configs", config_name="config")
# def main(cfg):
#     root = Path(cfg.paths.ROOT_PATH)
#     output_path = Path(cfg.paths.OUTPUT_PATH)
#     output_path.mkdir(parents=True, exist_ok=True)

#     logging.basicConfig(
#         filename=f"{output_path}/pred.log",
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#     )

#     logger = logging.getLogger(__name__)
#     pred_path_imgs = root / "new_to_predict"

#     # Load the model
#     test_transform = build_transform(mode="test", plot_batch=cfg.training.PLOT_BATCH)

#     data_module = CustomGeoDataModule(
#         test_img_path=pred_path_imgs,
#         test_transform=test_transform,
#         batch_size=cfg.training.BATCH_SIZE,
#         patch_size=cfg.training.PATCH_SIZE,
#     )
#     data_module.setup("test")

#     test_dataloader = data_module.test_dataloader()

#     num_classes = cfg.training.NUM_CLASSES
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     model = load_model(cfg.paths.CHECKPOINT_PATH, num_classes, device)

#     predictions = []
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(test_dataloader):
#             images = batch["image"].to(device)
#             outputs = model(images)

#             print(f"Batch {batch_idx} Output Shape: {outputs.shape}")

#             preds = torch.argmax(outputs, dim=1)  # Shape: [B, H, W]

#             # Save predictions
#             for i, (img, pred) in enumerate(zip(images[:1], preds[:1], strict=False)):
#                 save_path = output_path / f"prediction_{batch_idx}_{i}.npy"
#                 np.save(save_path, pred.cpu().numpy())  # Save as numpy array
#                 print(f"Saved: {save_path}")

#                 # Convert prediction to numpy
#                 pred_np = pred.cpu().numpy()
#                 print(f"Image Shape: {img.shape}")
#                 # Create a side-by-side plot
#                 fig, ax = plt.subplots(1, 2, figsize=(10, 5))

#                 # Left: Input Image
#                 ax[0].imshow(img_np)
#                 ax[0].set_title("Input Image")
#                 ax[0].axis("off")

#                 # Right: Predicted Segmentation Mask
#                 ax[1].imshow(pred_np, cmap="jet")
#                 ax[1].set_title("Predicted Mask")
#                 ax[1].axis("off")
#                 fig.colorbar(ax[1].imshow(pred_np, cmap="jet"), ax=ax[1])

#                 # Save figure
#                 plt.savefig(
#                     output_path / f"prediction_{batch_idx}_{i}.png",
#                     bbox_inches="tight"
#                 )
#                 plt.close()

#     print(f"Predictions saved to {output_path}")


# if __name__ == "__main__":
#     main()

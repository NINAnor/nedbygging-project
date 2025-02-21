from collections.abc import Iterable

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from torchgeo.datasets import unbind_samples

from utils import CLASS_LABELS


def get_color_map_and_legend():
    class_colors = [
        "black",
        "red",
        "yellow",
        "lime",
        "green",
        "purple",
        "cyan",
        "magenta",
        "blue",
    ]

    legend_elements = [
        Patch(facecolor=class_colors[i], edgecolor="black", label=label)
        for i, label in enumerate(CLASS_LABELS.keys())
    ]
    return class_colors, legend_elements


def plot_imgs(
    images: Iterable, axs: Iterable, chnls: list[int] | None = None, bright: float = 0.0
):
    for img, ax in zip(images, axs, strict=False):
        img = img.float()
        img = img.permute(chnls)
        img = img[:, :, :3]
        if bright != 0.0:
            img = torch.clamp(bright * img, min=0, max=1).numpy()
        img = img.numpy()

        ax.imshow(img)
        ax.axis("off")


def plot_msks(masks: Iterable, axs: Iterable):
    class_colors, legend_elements = get_color_map_and_legend()
    for mask, ax in zip(masks, axs, strict=False):
        ax.imshow(mask.squeeze().numpy(), cmap=ListedColormap(class_colors))
        ax.axis("off")

    axs[0].legend(handles=legend_elements, loc="upper right", title="Classes")


def plot_batch(
    batch: dict,
    bright: float = 0.0,
    cols: int = 4,
    width: int = 5,
    chnls: list[int] | None = None,
):
    # for some reason ruff is complaining about using a "mutable default argument"?
    if chnls is None:  # Initialize the default inside the function
        chnls = [0, 1, 2]
    # Get the samples and the number of items in the batch
    samples = unbind_samples(batch.copy())

    # if batch contains images and masks, the number of images will be doubled
    n = 2 * len(samples) if ("image" in batch) and ("mask" in batch) else len(samples)

    # calculate the number of rows in the grid
    rows = n // cols + (1 if n % cols != 0 else 0)

    # create a grid
    _, axs = plt.subplots(rows, cols, figsize=(cols * width, rows * width))

    if ("image" in batch) and ("mask" in batch):
        # plot the images on the even axis
        plot_imgs(
            images=map(lambda x: x["image"], samples),
            axs=axs.reshape(-1)[::2],
            chnls=chnls,
            bright=bright,
        )

        # plot the masks on the odd axis
        plot_msks(masks=map(lambda x: x["mask"], samples), axs=axs.reshape(-1)[1::2])

    else:
        if "image" in batch:
            plot_imgs(
                images=map(lambda x: x["image"], samples),
                axs=axs.reshape(-1),
                chnls=chnls,
                bright=bright,
            )

        elif "mask" in batch:
            plot_msks(masks=map(lambda x: x["mask"], samples), axs=axs.reshape(-1))


def plotBatch(output_folder, train_loader, val_loader):
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    plot_batch(train_batch, cols=4, width=5, chnls=[2, 1, 0])
    plt.suptitle("Training Batch")
    plt.savefig(f"{output_folder}/training_batch.png")

    plot_batch(val_batch, cols=4, width=5, chnls=[2, 1, 0])
    plt.suptitle("Validation Batch")
    plt.savefig(f"{output_folder}/validation_batch.png")

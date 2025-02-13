from collections.abc import Iterable

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from torchgeo.datasets import unbind_samples


def plot_imgs(
    images: Iterable, axs: Iterable, chnls: list[int] | None = None, bright: float = 0.0
):
    # for some reason ruff is complaining about using a "mutable default argument"?
    if chnls is None:  # Initialize the default inside the function
        chnls = [0, 1, 2]

    for img, ax in zip(images, axs, strict=False):
        img = img[:, :, chnls]  # take only the 3 first channels (RGB for May)
        img = img.float()
        img = img

        # Normalize the image to the [0, 1] range using min-max normalization
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-8)

        if bright != 0.0:
            img_arr = torch.clamp(bright * img, min=0, max=1).numpy()
        else:
            img_arr = img.numpy()
        rgb = img_arr.transpose(0, 1, 2)

        ax.imshow(rgb)
        ax.axis("off")


def plot_msks(masks: Iterable, axs: Iterable):
    class_labels = ["urban", "cropland", "grass", "forest", "wetland", "water"]
    class_colors = ["red", "yellow", "lime", "green", "purple", "blue"]

    legend_elements = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for color, label in zip(class_colors, class_labels, strict=False)
    ]
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


def plotBatch(train_loader, val_loader):
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    plot_batch(train_batch, cols=4, width=5, chnls=[2, 1, 0])
    plt.suptitle("Training Batch")
    plt.savefig("training_batch.png")

    plot_batch(val_batch, cols=4, width=5, chnls=[2, 1, 0])
    plt.suptitle("Validation Batch")
    plt.savefig("validation_batch.png")
    exit()

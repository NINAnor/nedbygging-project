# Nedbygging project

In this repository we are training a segmentation model, DeepLabV3+ (using the [segmentation models](https://github.com/qubvel-org/segmentation_models.pytorch) library) to identify different landscape classes (listed in `src/utils.py`) on Sentinell 2 imagery.

## Setup

To set up the python environment, run the following command:

```bash
poetry install
```

Finally, set up the paths in `configs/paths/default.yaml`. For instance:

`ROOT_PATH: /data/P/Project/`

## Train the model

NOTE: You have to change the root path of the dataset in `configs/paths/default.yaml`

To train the model:

```bash
poetry run python src/main.py
```

The outputs of the training script (including the `pytorch-lightning` outputs) should be stored in an `output` directory.

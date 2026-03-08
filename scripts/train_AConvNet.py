"""Training script for AConvNet model on MSTAR dataset.

This script performs end-to-end training of the AConvNet neural network model
on the MSTAR (Moving and Stationary Target Recognition) dataset. It loads
configuration parameters from JSON files, preprocesses the dataset with
random/center cropping and normalization, initializes the AConvNet model,
and runs the training loop with validation.

The script uses command-line flags to specify:
    - experiments_path: Path to experiments directory
    - config_name: Path to configuration JSON file (relative to experiments_path)

Usage:
    python train_AConvNet.py --config_name=AConvNet/config/AConvNet-SOC.json
"""

import os
import sys

import torch
import torchvision
from absl import app, flags, logging

# Get the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add src/ to path
sys.path.append(os.path.join(project_root, "src"))

# modules in src — imported after sys.path is configured
from data.MSTAR import (  # noqa: E402
    load,  # type: ignore
    preprocess,  # type: ignore
)
from models._base import Model  # noqa: E402  # type: ignore
from models.AConvNet.network import AConvNet  # noqa: E402  # type: ignore
from utils import common  # noqa: E402  # type: ignore

DATA_PATH = os.path.join(project_root, "datasets/MSTAR/MSTAR_IMG_JSON")

# DATA_PATH = 'datasets/MSTAR/mstar_data_paper_AConvNet/'

model_str = "AConvNet"
flags.DEFINE_string(
    "experiments_path", os.path.join(common.project_root, "experiments"), help=""
)
flags.DEFINE_string("config_name", f"{model_str}/config/AConvNet-SOC.json", help="")
FLAGS = flags.FLAGS


common.set_random_seed(42)


def main(_):
    logging.info("Start")
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    config = common.load_config(os.path.join(experiments_path, config_name))
    logging.info(config)

    dataset = config["dataset"]
    classes = config["num_classes"]
    channels = config["channels"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    proportion = config.get("proportion", None)

    lr = config["lr"]
    lr_step = config["lr_step"]
    lr_decay = config["lr_decay"]

    weight_decay = config["weight_decay"]
    dropout_rate = config["dropout_rate"]

    experience_name = config["experience_name"]

    # augment = config['use_augment']
    # normalize = config['normalize']
    # lee_window_size = config['lee_window_size']
    # lee_noise_variance = config['lee_noise_variance']

    # define your preprocessing functions
    transform = [
        torchvision.transforms.Compose(
            [
                # preprocess.LeeFilterTransform(window_size=5, noise_variance=0),
                preprocess.RandomCrop(94),
                torchvision.transforms.Lambda(lambda x: x / 255.0),
            ]
        ),
        torchvision.transforms.Compose(
            [
                # preprocess.LeeFilterTransform(window_size=5, noise_variance=0),
                preprocess.CenterCrop(94),
                torchvision.transforms.Lambda(lambda x: x / 255.0),
            ]
        ),
    ]

    train_set, val_set = load.load_dataset(
        data_path=DATA_PATH,
        is_train=True,
        transform=transform,
        name=dataset,
        batch_size=batch_size,
        augment=False,  # augment
        proportion=proportion,
    )

    net = AConvNet(classes=classes, channels=channels, dropout_rate=dropout_rate)

    m = Model(
        net=net,
        lr=lr,
        lr_step=lr_step,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        criterion=torch.nn.CrossEntropyLoss(),
    )

    m.run(
        train_set,
        val_set,
        epochs,
        experience_name,
        experiments_path,
        debug=False,
    )

    logging.info("Finish")


if __name__ == "__main__":
    app.run(main)

"""Training script for AlexNet model on MSTAR dataset.

This script performs end-to-end training of the AlexNet convolutional neural
network model on the MSTAR (Moving and Stationary Target Recognition) dataset.
It loads configuration parameters from JSON files, preprocesses grayscale SAR
images to AlexNet's standard input size (227x227), and runs the training loop
with validation.

The script uses command-line flags to specify:
    - experiments_path: Path to experiments directory
    - config_name: Path to configuration JSON file (relative to experiments_path)

Usage:
    python train_alexnet.py --config_name=alexnet/config/alexnet-SOC.json
"""

import os
import sys

import torch
from absl import app, flags, logging
from torchvision import transforms

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

# modules in src — imported after sys.path is configured
from data.MSTAR import load  # noqa: E402
from models._base import Model  # noqa: E402
from models.alexnet.network import AlexNet  # noqa: E402
from utils import common  # noqa: E402

DATA_PATH = os.path.join(project_root, "datasets/MSTAR/MSTAR_IMG_JSON")
model_str = "alexnet"
flags.DEFINE_string(
    "experiments_path", os.path.join(common.project_root, "experiments"), help=""
)
flags.DEFINE_string("config_name", f"{model_str}/config/{model_str}-SOC.json", help="")
FLAGS = flags.FLAGS

common.set_random_seed(42)


def main(_):
    logging.info("Start")
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    config = common.load_config(os.path.join(experiments_path, config_name))

    dataset = config["dataset"]
    classes = config["num_classes"]
    channels = config["channels"]  # noqa: F841  # read for completeness, AlexNet uses 1 channel
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    proportion = config.get("proportion", None)

    lr = config["lr"]
    lr_step = config["lr_step"]
    lr_decay = config["lr_decay"]

    weight_decay = config["weight_decay"]
    dropout_rate = config["dropout_rate"]

    experience_name = config["experience_name"]

    # Standard AlexNet input size
    input_size = (227, 227)

    # Note: We include Grayscale(1) because ImageFolder loads as RGB by default
    shared_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(input_size),
            transforms.Lambda(lambda x: x / 255.0),
        ]
    )

    train_set, val_set = load.load_dataset(
        data_path=DATA_PATH,
        transform=shared_transforms,
        batch_size=batch_size,
        is_train=True,
        name=dataset,
        proportion=proportion,
        augment=True,
    )

    net = AlexNet(classes=classes, dropout_rate=dropout_rate)

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

from absl import logging
from absl import flags
from absl import app

from tqdm import tqdm

from torch.utils import tensorboard
from torch.utils.data import DataLoader, random_split

import torchvision
import torch

import numpy as np

import json

import sys
import os

# Get the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add src/ to path
sys.path.append(os.path.join(project_root, "src"))

# modules in src
from data.MSTAR import preprocess # type: ignore
from data.MSTAR import dataset # type: ignore
from utils import common # type: ignore
from models.AConvNet.network import AConvNet # type: ignore
from models._base import Model # type: ignore
from data.MSTAR import load # type: ignore

DATA_PATH = os.path.join(project_root, 'datasets/MSTAR/MSTAR_IMG_JSON')

# DATA_PATH = 'datasets/MSTAR/mstar_data_paper_AConvNet/'

model_str = 'AConvNet'
flags.DEFINE_string('experiments_path', os.path.join(common.project_root, 'experiments'), help='')
flags.DEFINE_string('config_name', f'{model_str}/config/AConvNet-SOC.json', help='')
FLAGS = flags.FLAGS


common.set_random_seed(42)

def main(_):
    logging.info('Start')
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    config = common.load_config(os.path.join(experiments_path, config_name))
    logging.info(config)

    dataset = config['dataset']
    classes = config['num_classes']
    channels = config['channels']
    epochs = config['epochs']
    batch_size = config['batch_size']
    proportion = config.get("proportion", None)

    lr = config['lr']
    lr_step = config['lr_step']
    lr_decay = config['lr_decay']

    weight_decay = config['weight_decay']
    dropout_rate = config['dropout_rate']

    experience_name = config['experience_name']

    # define your preprocessing functions
    transform = torchvision.transforms.Compose([
        preprocess.RandomCrop(94), 
        torchvision.transforms.Lambda(lambda x: x / 255.0)
    ])

    train_set, val_set = load.load_dataset(
        data_path=DATA_PATH, 
        is_train=True, 
        transform=transform, 
        name=dataset, 
        batch_size=batch_size, 
        augment=True, 
        proportion=proportion
    )

    net = AConvNet(
        classes=classes,
        channels=channels,
        dropout_rate=dropout_rate
    )

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

    logging.info('Finish')


if __name__ == '__main__':
    app.run(main)
    
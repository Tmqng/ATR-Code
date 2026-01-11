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
from data.MSTAR.paper_AConvNet import preprocess
from data.MSTAR.paper_AConvNet import loader
from utils import common
from models import AConvNet

DATA_PATH = 'datasets/MSTAR/MSTAR_IMG_JSON'

# DATA_PATH = 'datasets/MSTAR/mstar_data_paper_AConvNet/'

model_str = 'AConvNet'


flags.DEFINE_string('experiments_path', os.path.join(common.project_root, 'experiments'), help='')
flags.DEFINE_string('config_name', f'{model_str}/config/AConvNet-SOC.json', help='')
FLAGS = flags.FLAGS


common.set_random_seed(12321)

def load_dataset(path, is_train, name, batch_size):
    """
    Docstring for load_dataset
    
    :param path: Description
    :param is_train: Description
    :param name: Description
    :param batch_size: Description

    Load train, val or test dataset and apply transformations.
    """

    val_transform = torchvision.transforms.Compose([preprocess.CenterCrop(94)])

    train_transform = torchvision.transforms.Compose([preprocess.RandomCrop(94)])

    _dataset = loader.Dataset(
        path, name=name, is_train=is_train,
        transform=None
    )

    if is_train:

        # TODO Data_augmentation (in preprocess file)
        print(f"Augmenting training data with patches...")
        # Extract patches from training data
        augmented_samples = preprocess.augment_dataset_with_patches(
            _dataset,
            # patch_size=patch_size,
            # stride=stride,
            # chip_size=chip_size,
            desc="Train augmentation"
        )

        print(f"\nRésultats augmentation :")
        print(f"  Train : {len(_dataset)} images → {len(augmented_samples)} patches")
        print(f"  Facteur : ~{len(augmented_samples) / len(_dataset):.0f}x (13x13 = 169 patches/image)")

        augmented_dataset = preprocess.AugmentedDataset(augmented_samples)

        # Split into train (80%) and validation (20%)
        train_size = int(0.8 * len(augmented_dataset))
        val_size = len(augmented_dataset) - train_size

        train_dataset, val_dataset = random_split(augmented_dataset, [train_size, val_size])

        # CenterCrop for val and RandomCrop for train
        train_dataset_transformed = preprocess.TransformWrapper(train_dataset, train_transform)
        val_dataset_transformed = preprocess.TransformWrapper(val_dataset, val_transform)

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset_transformed, batch_size=batch_size, shuffle=is_train, num_workers=1
        )

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset_transformed, batch_size=batch_size, shuffle=False, num_workers=1
        )

        return train_data_loader, val_data_loader


    else:
        test_dataset_transformed = preprocess.TransformWrapper(_dataset, val_transform)
        data_loader = torch.utils.data.DataLoader(
            test_dataset_transformed, batch_size=batch_size, shuffle=is_train, num_workers=1
        )
        return data_loader


@torch.no_grad()
def validation(m, ds):
    num_data = 0
    corrects = 0

    # Test loop
    m.net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    for i, data in enumerate(tqdm(ds)):
        images, labels, _ = data

        images = images.to(m.device)
        labels = labels.to(m.device).type(torch.LongTensor)

        predictions = m.inference(images)
        predictions = _softmax(predictions)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(m.device)).sum().item()

    accuracy = 100 * corrects / num_data
    return accuracy


def run(epochs, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        model_name, experiments_path=None):
    train_set, val_set = load_dataset(DATA_PATH, True, dataset, batch_size)
    # test_set = load_dataset(DATA_PATH, False, dataset, batch_size)

    m = AConvNet.Model(
        classes=classes, dropout_rate=dropout_rate, channels=channels,
        lr=lr, lr_step=lr_step, lr_decay=lr_decay,
        weight_decay=weight_decay
    )

    model_path = os.path.join(experiments_path, f'{model_str}/models/{model_name}')
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    history_path = os.path.join(experiments_path, f'{model_str}/history')
    if not os.path.exists(history_path):
        os.makedirs(history_path, exist_ok=True)

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(epochs):
        _loss = []

        m.net.train()
        for i, data in enumerate(tqdm(train_set)):
            images, labels, _ = data
            _loss.append(m.optimize(images, labels))

        if m.lr_scheduler:
            lr = m.lr_scheduler.get_last_lr()[0]
            m.lr_scheduler.step()

        train_accuracy = validation(m, train_set)
        val_accuracy = validation(m, val_set)

        logging.info(
            f'Epoch: {epoch + 1:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | lr={lr} | Train accuracy={train_accuracy:.2f} | Validation accuracy={val_accuracy:.2f}'
        )

        history['train_loss'].append(np.mean(_loss))
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        if experiments_path:
            m.save(os.path.join(model_path, f'model-{epoch + 1:03d}.pth'))

        with open(os.path.join(history_path, f'history-{model_name}.json'), mode='w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=True, indent=2)


def main(_):
    logging.info('Start')
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    config = common.load_config(os.path.join(experiments_path, config_name))

    dataset = config['dataset']
    classes = config['num_classes']
    channels = config['channels']
    epochs = config['epochs']
    batch_size = config['batch_size']

    lr = config['lr']
    lr_step = config['lr_step']
    lr_decay = config['lr_decay']

    weight_decay = config['weight_decay']
    dropout_rate = config['dropout_rate']

    model_name = config['model_name']

    run(epochs, dataset, classes, channels, batch_size,
        lr, lr_step, lr_decay, weight_decay, dropout_rate,
        model_name, experiments_path)

    logging.info('Finish')


if __name__ == '__main__':
    app.run(main)
    
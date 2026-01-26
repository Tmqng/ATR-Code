import json
import os
import sys

import numpy as np
import torch
from torchvision import datasets, transforms
from absl import app, flags, logging
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

from models._base import Model
from models.alexnet.network import AlexNet
from utils import common
from data.MSTAR.paper_AConvNet import preprocess
from data.MSTAR.paper_AConvNet import loader


DATA_PATH = os.path.join(project_root, "datasets/MSTAR/MSTAR_IMG_JSON")
model_str = "alexnet"
flags.DEFINE_string(
    "experiments_path", os.path.join(common.project_root, "experiments"), help=""
)
flags.DEFINE_string("config_name", f"{model_str}/config/{model_str}-SOC.json", help="")
FLAGS = flags.FLAGS

common.set_random_seed(42)


# TODO: Load dataset
# TODO: Train model
# TODO:


def load_dataset(
    data_path: str, name: str, is_train: bool, batch_size: int = 32, proportion: float = None, augment: bool = True
) -> tuple[DataLoader, DataLoader]:
    """
    Load MSTAR data using native PyTorch tools for AlexNet.

    Args:
        data_path: Path to the 'train' folder of the dataset.
        batch_size: Number of samples per batch.
        train_split: Proportion of data to use for training.
    """

    # Standard AlexNet input size
    input_size = (227, 227)

    # Note: We include Grayscale(1) because ImageFolder loads as RGB by default
    shared_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(input_size),
            # transforms.ToTensor(),
        ]
    )

    # Load the entire training directory
    # full_dataset = datasets.ImageFolder(root=data_path, transform=shared_transforms)
    full_dataset = loader.Dataset(
        data_path, name=name, is_train=is_train,
        transform=None, proportion=proportion
    )

    if is_train:

        if augment:
            # Data_augmentation (in preprocess file)
            print(f"Augmenting training data with patches...")
            # Extract patches from training data
            augmented_samples = preprocess.augment_dataset_with_patches(
                full_dataset,
                # patch_size=patch_size,
                # stride=stride,
                # chip_size=chip_size,
                desc="Train augmentation"
            )

            print(f"\nRésultats augmentation :")
            print(f"  Train : {len(full_dataset)} images → {len(augmented_samples)} patches")
            print(f"  Facteur : ~{len(augmented_samples) / len(full_dataset):.0f}x (13x13 = 169 patches/image)")

            augmented_dataset = preprocess.AugmentedDataset(augmented_samples)
        else:
            augmented_dataset = full_dataset

        # Split into train (80%) and validation (20%)
        train_size = int(0.8 * len(augmented_dataset))
        val_size = len(augmented_dataset) - train_size

        train_dataset, val_dataset = random_split(augmented_dataset, [train_size, val_size])

        # CenterCrop for val and RandomCrop for train
        train_dataset_transformed = preprocess.TransformWrapper(train_dataset, shared_transforms)
        val_dataset_transformed = preprocess.TransformWrapper(val_dataset, shared_transforms)
        
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset_transformed, batch_size=batch_size, shuffle=is_train, num_workers=1
        )

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset_transformed, batch_size=batch_size, shuffle=False, num_workers=1
        )

        # Check first batch
        for images, labels, _ in train_data_loader:
            print(f"\nFirst batch shapes:")
            print(f"  Images: {images.shape}, dtype: {images.dtype}")
            print(f"  Labels: {labels.shape}, dtype: {labels.dtype}")
            print(f"  Labels values: {labels.tolist()[:10]}")
            print(f"  Unique labels: {torch.unique(labels).tolist()}")
            break

        return train_data_loader, val_data_loader
    
    else:
        print('is_train must be True ;)')


@torch.no_grad()
def validation(m, ds, debug=False):
    num_data = 0
    corrects = 0

    # Test loop
    m.net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    for i, data in enumerate(tqdm(ds)):
        images, labels, _ = data

        images = images.to(m.device)
        labels = labels.to(m.device)

        predictions = m.inference(images)
        predictions = predictions.to(m.device)
        predictions = _softmax(predictions)

        _, predictions = torch.max(predictions.data, 1)

        # DEBUG: Check predictions
        if debug and i == 0:
            logging.info(f"Predicted classes: {predictions[:10]}")
            logging.info(f"True labels: {labels[:10]}")
            logging.info(f"Matches: {(predictions == labels)[:10]}")

        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(m.device)).sum().item()

    accuracy = 100 * corrects / num_data
    return accuracy


def run(
    epochs,
    dataset,
    classes,
    channels,
    batch_size,
    lr,
    lr_step,
    lr_decay,
    weight_decay,
    dropout_rate,
    model_name,
    proportion=None,
    experiments_path=None,
    debug=False,
):
    # data_path = os.path.join(DATA_PATH, dataset)
    data_path = DATA_PATH

    train_set, val_set = load_dataset(data_path=data_path, batch_size=batch_size, is_train=True, name=dataset, proportion=proportion)
    # test_set = load_dataset(DATA_PATH, False, dataset, batch_size)

    net = AlexNet(classes=classes, dropout_rate=dropout_rate)

    m = Model(
        net=net,
        lr=lr,
        lr_step=lr_step,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        criterion=torch.nn.CrossEntropyLoss(),
    )

    model_path = os.path.join(experiments_path, f"{model_str}/models/{model_name}")
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    history_path = os.path.join(experiments_path, f"{model_str}/history")
    if not os.path.exists(history_path):
        os.makedirs(history_path, exist_ok=True)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        _loss = []

        m.net.train()
        for i, data in enumerate(tqdm(train_set)):
            images, labels, _ = data

            images = images.to(m.device)
            labels = labels.to(m.device)

            _loss.append(m.optimize(images, labels))

            # DEBUG: Check predictions
            if debug and i == 0:
                logging.info(f"Labels: {labels[:10]}")
                logging.info(f"Loss: {_loss[:10]}")

        if m.lr_scheduler:
            lr = m.lr_scheduler.get_last_lr()[0]
            m.lr_scheduler.step()

        train_accuracy = validation(m, train_set, debug=debug)
        val_accuracy = validation(m, val_set, debug=debug)

        logging.info(
            f"Epoch: {epoch + 1:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | lr={lr} | Train accuracy={train_accuracy:.2f} | Validation accuracy={val_accuracy:.2f}"
        )

        history["train_loss"].append(np.mean(_loss))
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if experiments_path:
            m.save(os.path.join(model_path, f"model-{epoch + 1:03d}.pth"))

        with open(
            os.path.join(history_path, f"history-{model_name}.json"),
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(history, f, ensure_ascii=True, indent=2)


def main(_):
    logging.info("Start")
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    config = common.load_config(os.path.join(experiments_path, config_name))

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

    model_name = config["model_name"]

    run(
        epochs,
        dataset,
        classes,
        channels,
        batch_size,
        lr,
        lr_step,
        lr_decay,
        weight_decay,
        dropout_rate,
        model_name,
        proportion,
        experiments_path,
    )

    logging.info("Finish")


if __name__ == "__main__":
    app.run(main)

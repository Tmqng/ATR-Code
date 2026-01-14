
from typing import Dict

import torch
import torch.nn.functional as F
from absl import logging
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np


def extract_patches_from_tensor(image_tensor, patch_size, stride, chip_size):
    """
    Extract MSTAR patches with central cropping.

    Args:
        image_tensor: Tensor (C, H, W) ou (1, C, H, W)
        patch_size: Patch size (94)
        stride: Stride (1)
        chip_size: After central cropping size (100)

    Returns:
        patches: Tensor (N_patches, C, patch_size, patch_size)
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Crop
    if image_tensor.size(2) > chip_size:
        start = (image_tensor.size(2) - chip_size) // 2
        image_tensor = image_tensor[:, :, start : start + chip_size, start : start + chip_size]

    # Extract with unfold
    patches_unfold = F.unfold(image_tensor, kernel_size=patch_size, stride=stride)

    # Reshape
    C = image_tensor.size(1)
    patches = patches_unfold.transpose(1, 2)
    patches = patches.reshape(-1, C, patch_size, patch_size)

    return patches

class AugmentedDataset(Dataset):
    """Dataset for the augmented patches."""

    def __init__(self, augmented_data):
        self.data = augmented_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_augmented_dataset(dataset, config):
    """
    Applies patch augmentation to a dataset.

    Args:
        dataset: Dataset PyTorch
        config: Configuration

    Returns:
        AugmentedDataset with all patches.
    """
    logging(f" Extraction des patches (patch_size={config['patch_size']}, "
          f"stride={config['stride']})...")

    augmented_samples = []
    temp_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for images, labels in tqdm(temp_loader, desc="Patches"):
        images_squeeze = images.squeeze(0)
        all_patches = extract_patches_from_tensor(
            images_squeeze,
            patch_size=config["patch_size"],
            stride=config["stride"],
            chip_size=config["chip_size"],
        )

        label = labels.item()
        for patch in all_patches:
            augmented_samples.append((patch, label))

    logging("Finished extraction:")
    logging(f"   Original images: {len(dataset)}")
    logging(f"   Patches generated: {len(augmented_samples)}")
    logging(f"   Augmentation factor: {len(augmented_samples) / len(dataset):.1f}x")

    return AugmentedDataset(augmented_samples)


def load_dataset(
    data_path: str, config: Dict = None
) -> tuple[DataLoader, DataLoader]:
    """
    Load MSTAR data using native PyTorch tools for Swin Transformer.

    Args:
        data_path: Path to the 'train' folder of the dataset.
        config: Configuration dictionary (not used here but kept for compatibility).
    """

    # Base transformations for training (before patch extraction)
    train_transform_base = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    # Transformations for validation/test (full images)
    test_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(size=100),
            transforms.Resize(config["img_size"]),  # Resize à 96x96 pour Swin
            transforms.ToTensor(),
        ]
    )

    full_train_dataset = datasets.ImageFolder(root=data_path + "/train", transform=train_transform_base)
    test_dataset = datasets.ImageFolder(root=data_path + "/test", transform=test_transform)

    train_size = int((1 - config["val_split"]) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset_base, val_dataset_base = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    # Augmentation by patches on train
    train_dataset_aug = create_augmented_dataset(train_dataset_base, config)

    # Augmentation by patches on val
    val_dataset_aug = create_augmented_dataset(val_dataset_base, config)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset_aug,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset_aug,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    return train_loader, val_loader, test_loader


@torch.no_grad()
def validation(m, ds):
    """


    Args:
        m (_type_): _description_
        ds (_type_): _description_

    Returns:
        _type_: _description_
    """
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
        if i == 0:
            logging(f"Predicted classes: {predictions[:10]}")
            logging(f"True labels: {labels[:10]}")
            logging(f"Matches: {(predictions == labels)[:10]}")

        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(m.device)).sum().item()

    accuracy = 100 * corrects / num_data
    return accuracy



def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, epoch):
    """Entraîne le modèle pour une époque"""
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(cfg.device), labels.to(cfg.device)

        # Padding si nécessaire (94 -> 96)
        if images.size(-1) != cfg.img_size:
            pad = cfg.img_size - images.size(-1)
            images = F.pad(images, (0, pad, 0, pad), mode="constant", value=0)

        # Forward avec mixed precision
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Scheduler step (par batch)
        scheduler.step_update(epoch * len(loader) + batch_idx)

        # Métriques
        acc = (outputs.argmax(1) == labels).float().mean()
        losses.update(loss.item(), images.size(0))
        accs.update(acc.item(), images.size(0))

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{losses.avg:.4f}",
                "acc": f"{accs.avg:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

    return losses.avg, accs.avg


def validate(model, loader, criterion, cfg, desc="Val"):
    """Évalue le modèle"""
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[{desc}]")
        for images, labels in pbar:
            images, labels = images.to(cfg.device), labels.to(cfg.device)

            # Padding si nécessaire
            if images.size(-1) != cfg.img_size:
                pad = cfg.img_size - images.size(-1)
                images = F.pad(images, (0, pad, 0, pad), mode="constant", value=0)

            # Forward
            with torch.amp.autocast(enabled=cfg.use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Métriques
            preds = outputs.argmax(1)
            acc = (preds == labels).float().mean()

            losses.update(loss.item(), images.size(0))
            accs.update(acc.item(), images.size(0))

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{losses.avg:.4f}", "acc": f"{accs.avg:.4f}"})

    return losses.avg, accs.avg, np.array(all_preds), np.array(all_labels)


class AverageMeter:
    """Calcule et stocke la moyenne et la valeur actuelle"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
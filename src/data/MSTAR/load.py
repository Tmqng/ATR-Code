import dataset
import preprocess

import torch
from torch.utils.data import DataLoader, random_split

def load_dataset(path, is_train, transform, name, batch_size, augment, proportion):
    """
    Docstring for load_dataset
    
    :param path: Description
    :param is_train: Description
    :param transform: transform or [train_transform, val_transform]
    :param name: Description
    :param batch_size: Description

    Load train, val or test dataset and apply transformations.
    """

    # Load Dataset from files
    _dataset = dataset.Dataset(
        path,
        name=name, 
        is_train=is_train,
        proportion=proportion
    )

    if is_train:

        if augment:
            # Data_augmentation (in preprocess file)
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
        else:
            augmented_dataset = _dataset

        # Split into train (80%) and validation (20%)
        train_size = int(0.8 * len(augmented_dataset))
        val_size = len(augmented_dataset) - train_size

        train_dataset, val_dataset = random_split(augmented_dataset, [train_size, val_size])

        if isinstance(transform, list):
            train_transform, val_transform= transform
        else:
            train_transform = transform
            val_transform = transform
            
        train_dataset_transformed = preprocess.TransformWrapper(train_dataset, train_transform)
        val_dataset_transformed = preprocess.TransformWrapper(val_dataset, val_transform)
        
        train_data_loader = DataLoader(
            train_dataset_transformed, batch_size=batch_size, shuffle=is_train, num_workers=1
        )

        val_data_loader = DataLoader(
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
        test_dataset_transformed = preprocess.TransformWrapper(_dataset, transform)
        data_loader = torch.utils.data.DataLoader(
            test_dataset_transformed, batch_size=batch_size, shuffle=is_train, num_workers=1
        )
        return data_loader
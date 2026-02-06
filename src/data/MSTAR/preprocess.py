import numpy as np
import tqdm

import torch
import torch.nn.functional as F

from skimage import transform
from torch.utils.data import DataLoader

from . import dataset


class ToTensor(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        _input = sample

        if len(_input.shape) < 3:
            _input = _input.unsqueeze(0)

        # Normalize to [0, 1] range if not already normalized
        if _input.dtype != torch.float32:
            _input = _input.float()
        if _input.max() > 1.0:  # Assuming values are in [0, 255] range
            _input = _input / 255.0

        return _input


class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, sample):
        _input = sample

        if len(_input.shape) < 3:
            _input = np.expand_dims(_input, axis=0)

        _, h, w = _input.shape
        oh, ow = self.size

        dh = h - oh
        dw = w - ow
        y = np.random.randint(0, dh) if dh > 0 else 0
        x = np.random.randint(0, dw) if dw > 0 else 0
        oh = oh if dh > 0 else h
        ow = ow if dw > 0 else w

        return _input[:, y: y + oh, x: x + ow]


class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, sample):
        _input = sample

        if len(_input.shape) < 3:
            _input = np.expand_dims(_input, axis=0)

        _, h, w = _input.shape
        oh, ow = self.size
        y = (h - oh) // 2
        x = (w - ow) // 2

        return _input[:, y: y + oh, x: x + ow]
    

class TransformWrapper(object):
    """
    Wrapper to apply transforms to dataset subsets after random_split.
    
    Since random_split doesn't know about transforms, we need to wrap
    the subsets to apply transforms on-the-fly during data loading.
    """
    
    def __init__(self, dataset_subset, transform):
        self.dataset = dataset_subset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get item from the dataset subset
        image, label, serial_number = self.dataset[idx]
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label, serial_number
    
class AugmentedDataset(dataset.Dataset):
    """Dataset contenant les patches augmentés"""
    def __init__(self, augmented_data):
        self.data = augmented_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def augment_dataset_with_patches(dataset, patch_size=94, stride=1, chip_size=100, desc="Extraction"):
    """
    Applique l'augmentation par patches sur un dataset.
    Chaque image génère 169 patches (13x13).
    """
    augmented_samples = []

    # DataLoader temporaire
    temp_loader = DataLoader(dataset, batch_size=1, shuffle=False) 

    for images, labels, serial_numbers in tqdm.tqdm(temp_loader, desc=desc):
        images_squeeze = images.squeeze(0)
        patches = extract_patches_from_tensor(
            images_squeeze,
            patch_size=patch_size,
            stride=stride,
            chip_size=chip_size
        )

        label = labels.item()
        serial_number = serial_numbers[0]
        for patch in patches:
            augmented_samples.append((patch, label, serial_number))

    return augmented_samples

def extract_patches_from_tensor(image_tensor, patch_size, stride, chip_size):
    """
    Extrait les patches 94x94 avec stride=1 après rognage à 100x100.
    Fidèle au protocole MSTAR.
    """

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.float()

    # Rogne au centre à 100x100 (chip_size)
    if image_tensor.size(2) > chip_size:
        start = (image_tensor.size(2) - chip_size) // 2
        image_tensor = image_tensor[:, :, start:start + chip_size, start:start + chip_size]

    # Extraction des patches avec F.unfold
    patches_unfold = F.unfold(
        image_tensor,
        kernel_size=patch_size,
        stride=stride
    )

    # Remise en forme (N_patches, C, H, W)
    C = image_tensor.size(1)
    patches = patches_unfold.transpose(1, 2)
    patches = patches.reshape(-1, C, patch_size, patch_size)

    return patches

class LeeFilter:
    """
    Lee Filter for speckle noise reduction in SAR images.
    Preserves edges while reducing multiplicative noise.
    """

    def __init__(self, window_size=3, noise_variance=None):
        """
        Initialize Lee Filter.

        Args:
            window_size (int): Size of the filtering window (odd number, e.g., 3, 5, 7)
            noise_variance (float): Known noise variance. If None, estimated from data.
        """
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd")
        self.window_size = window_size
        self.noise_variance = noise_variance
        self.padding = window_size // 2

    def __call__(self, image):
        """
        Apply Lee filter to image.

        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W)

        Returns:
            torch.Tensor: Filtered image
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        if image.dim() != 3:
            raise ValueError("Input must be a 3D tensor (C, H, W)")

        # Ensure image is float
        image = image.float()

        # Apply filter to each channel
        filtered_channels = []
        for c in range(image.shape[0]):
            channel = image[c:c+1]  # Keep channel dimension
            filtered_channel = self._apply_lee_filter(channel)
            filtered_channels.append(filtered_channel)

        return torch.cat(filtered_channels, dim=0)

    def _apply_lee_filter(self, image):
        """
        Apply Lee filter to a single channel image.

        Args:
            image (torch.Tensor): Single channel image (1, H, W)

        Returns:
            torch.Tensor: Filtered image
        """
        # Create padding
        padded = F.pad(image, (self.padding, self.padding, self.padding, self.padding), mode='reflect')

        # Compute local mean using convolution
        kernel = torch.ones(1, 1, self.window_size, self.window_size, device=image.device) / (self.window_size ** 2)
        local_mean = F.conv2d(padded, kernel, padding=0)

        # Compute local variance
        local_var = F.conv2d(padded ** 2, kernel, padding=0) - local_mean ** 2

        # Note: local_mean and local_var are already the correct size (H, W) due to convolution

        # Estimate noise variance if not provided
        if self.noise_variance is None:
            # Use minimum variance as noise estimate (common approach)
            noise_var = torch.clamp(local_var.min(), min=1e-10)
        else:
            noise_var = torch.tensor(self.noise_variance, device=image.device, dtype=image.dtype)

        # Lee filter formula: mean + (var / (var + noise_var)) * (pixel - mean)
        # Avoid division by zero
        local_var = torch.clamp(local_var, min=1e-10)

        weight = local_var / (local_var + noise_var)
        filtered = local_mean + weight * (image - local_mean)

        return filtered


class LeeFilterTransform:
    """
    Transform wrapper for Lee Filter to be used in torchvision transforms.
    """

    def __init__(self, window_size=3, noise_variance=None):
        self.filter = LeeFilter(window_size=window_size, noise_variance=noise_variance)

    def __call__(self, sample):
        return self.filter(sample)

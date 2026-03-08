"""Preprocessing transforms, augmentation, and filtering for MSTAR data."""

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from . import dataset


class ToTensor(object):
    """Convert a raw image tensor to a float32 tensor normalized to [0, 1]."""

    def __init__(self):
        pass

    def __call__(self, sample):
        """Apply the conversion.

        Args:
            sample (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Float32 tensor with shape (C, H, W) in [0, 1].
        """
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
    """Randomly crop a tensor image to the given spatial size.

    Args:
        size (int | tuple[int, int]): Desired output (height, width). If an
            int is given, a square crop is produced.
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, sample):
        """Apply a random crop.

        Args:
            sample (torch.Tensor | np.ndarray): Image with shape (C, H, W).

        Returns:
            Cropped image with shape (C, oh, ow).
        """
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

        return _input[:, y : y + oh, x : x + ow]


class CenterCrop(object):
    """Center-crop a tensor image to the given spatial size.

    Args:
        size (int | tuple[int, int]): Desired output (height, width). If an
            int is given, a square crop is produced.
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, sample):
        """Apply a center crop.

        Args:
            sample (torch.Tensor | np.ndarray): Image with shape (C, H, W).

        Returns:
            Center-cropped image with shape (C, oh, ow).
        """
        _input = sample

        if len(_input.shape) < 3:
            _input = np.expand_dims(_input, axis=0)

        _, h, w = _input.shape
        oh, ow = self.size
        y = (h - oh) // 2
        x = (w - ow) // 2

        return _input[:, y : y + oh, x : x + ow]


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
    """Dataset wrapping pre-extracted augmented patches.

    Args:
        augmented_data (list): List of (patch_tensor, label, serial_number) tuples
            produced by :func:`augment_dataset_with_patches`.
    """

    def __init__(self, augmented_data):
        self.data = augmented_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def augment_dataset_with_patches(
    dataset, patch_size=94, stride=1, chip_size=100, desc="Extraction"
):
    """Apply patch-based augmentation to every image in *dataset*.

    Each image yields 169 patches with default settings (chip 100, patch 94, stride 1).

    Args:
        dataset: A dataset returning (image, label, serial_number) tuples.
        patch_size (int): Side length of each extracted square patch.
        stride (int): Sliding-window stride for patch extraction.
        chip_size (int): Central chip size to crop from each image before
            patch extraction.
        desc (str): Description label for the tqdm progress bar.

    Returns:
        list: Flat list of (patch_tensor, label, serial_number) tuples.
    """
    augmented_samples = []

    # Temporary single-sample DataLoader used for iteration
    temp_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for images, labels, serial_numbers in tqdm.tqdm(temp_loader, desc=desc):
        images_squeeze = images.squeeze(0)
        patches = extract_patches_from_tensor(
            images_squeeze, patch_size=patch_size, stride=stride, chip_size=chip_size
        )

        label = labels.item()
        serial_number = serial_numbers[0]
        for patch in patches:
            augmented_samples.append((patch, label, serial_number))

    return augmented_samples


def extract_patches_from_tensor(image_tensor, patch_size, stride, chip_size):
    """Extract sliding-window patches from a single image tensor.

    Follows the MSTAR protocol: center-crop to *chip_size* then extract
    ``patch_size``-sized patches with the given *stride* via ``F.unfold``.

    Args:
        image_tensor (torch.Tensor): Image of shape (C, H, W) or (1, C, H, W).
        patch_size (int): Side length of each square patch.
        stride (int): Stride of the sliding window.
        chip_size (int): Central chip size to crop to before patch extraction.

    Returns:
        torch.Tensor: Patches of shape (N_patches, C, patch_size, patch_size).
    """

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.float()

    # Center-crop to chip_size x chip_size
    if image_tensor.size(2) > chip_size:
        start = (image_tensor.size(2) - chip_size) // 2
        image_tensor = image_tensor[
            :, :, start : start + chip_size, start : start + chip_size
        ]

    # Extract patches using F.unfold
    patches_unfold = F.unfold(image_tensor, kernel_size=patch_size, stride=stride)

    # Reshape from (1, C*patch_size^2, N) to (N_patches, C, patch_size, patch_size)
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
            channel = image[c : c + 1]  # Keep channel dimension
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
        padded = F.pad(
            image,
            (self.padding, self.padding, self.padding, self.padding),
            mode="reflect",
        )

        # Compute local mean using convolution
        kernel = torch.ones(
            1, 1, self.window_size, self.window_size, device=image.device
        ) / (self.window_size**2)
        local_mean = F.conv2d(padded, kernel, padding=0)

        # Compute local variance
        local_var = F.conv2d(padded**2, kernel, padding=0) - local_mean**2

        # Note: local_mean and local_var are already the correct size (H, W) due to convolution

        # Estimate noise variance if not provided
        if self.noise_variance is None:
            # Use minimum variance as noise estimate (common approach)
            noise_var = torch.clamp(local_var.min(), min=1e-10)
        else:
            noise_var = torch.tensor(
                self.noise_variance, device=image.device, dtype=image.dtype
            )

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

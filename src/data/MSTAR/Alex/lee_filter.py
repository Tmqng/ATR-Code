import torch
import torch.nn.functional as F
import numpy as np


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
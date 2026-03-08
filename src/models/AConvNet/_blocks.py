"""Reusable dense and convolution blocks with initialization hooks."""

import collections

import torch.nn as nn

_activations = {"relu": nn.ReLU, "relu6": nn.ReLU6, "leaky_relu": nn.LeakyReLU}


class BaseBlock(nn.Module):
    """Abstract base block that delegates ``forward`` to ``self._layer``."""

    def __init__(self):
        super(BaseBlock, self).__init__()
        self._layer: nn.Sequential

    def forward(self, x):
        """Pass input through the block's sequential layer stack."""
        return self._layer(x)


class DenseBlock(BaseBlock):
    """Fully-connected block with optional activation and weight initialisation.

    Args:
        shape (tuple[int, int]): ``(in_dims, out_dims)`` for the linear layer.

    Keyword Args:
        activation (str, optional): Activation name (``'relu'``, ``'relu6'``,
            ``'leaky_relu'``).
        w_init (callable, optional): Weight initialiser applied to the linear layer.
        b_init (callable, optional): Bias initialiser applied to the linear layer.
    """

    def __init__(self, shape, **params):
        super(DenseBlock, self).__init__()
        in_dims, out_dims = shape
        _seq = collections.OrderedDict(
            [
                ("dense", nn.Linear(in_dims, out_dims)),
            ]
        )
        _act_name = params.get("activation")
        if _act_name:
            _seq.update({_act_name: _activations[_act_name](inplace=True)})

        self._layer = nn.Sequential(_seq)

        w_init = params.get("w_init", None)
        idx = list(dict(self._layer.named_children()).keys()).index("dense")
        if w_init:
            w_init(self._layer[idx].weight)
        b_init = params.get("b_init", None)
        if b_init:
            b_init(self._layer[idx].bias)


class Conv2DBlock(BaseBlock):
    """Convolutional block with optional BatchNorm, activation, and max-pool.

    Args:
        shape (tuple[int, int, int, int]): ``(H, W, in_channels, out_channels)``
            for the convolution kernel.
        stride (int): Convolution stride.
        padding (str | int): Padding mode/size passed to ``nn.Conv2d`` (default ``'same'``).

    Keyword Args:
        batch_norm (bool, optional): If truthy, add BatchNorm2d after conv.
        activation (str, optional): Activation name (``'relu'``, ``'relu6'``,
            ``'leaky_relu'``).
        max_pool (bool, optional): If truthy, append a MaxPool2d layer.
        max_pool_size (int): Max-pool kernel size (default 2).
        max_pool_stride (int): Max-pool stride (default equals kernel size).
        w_init (callable, optional): Weight initialiser for the conv layer.
        b_init (callable, optional): Bias initialiser for the conv layer.
    """

    def __init__(self, shape, stride, padding="same", **params):
        super(Conv2DBlock, self).__init__()

        h, w, in_channels, out_channels = shape
        _seq = collections.OrderedDict(
            [
                (
                    "conv",
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(h, w),
                        stride=stride,
                        padding=padding,
                    ),
                )
            ]
        )

        _bn = params.get("batch_norm")
        if _bn:
            _seq.update({"bn": nn.BatchNorm2d(out_channels)})

        _act_name = params.get("activation")
        if _act_name:
            _seq.update({_act_name: _activations[_act_name](inplace=True)})

        _max_pool = params.get("max_pool")
        if _max_pool:
            _kernel_size = params.get("max_pool_size", 2)
            _stride = params.get("max_pool_stride", _kernel_size)
            _seq.update(
                {"max_pool": nn.MaxPool2d(kernel_size=_kernel_size, stride=_stride)}
            )

        self._layer = nn.Sequential(_seq)

        w_init = params.get("w_init", None)
        idx = list(dict(self._layer.named_children()).keys()).index("conv")
        if w_init:
            w_init(self._layer[idx].weight)
        b_init = params.get("b_init", None)
        if b_init:
            b_init(self._layer[idx].bias)

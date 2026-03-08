"""AConvNet architecture assembled from reusable convolution blocks."""

import torch.nn as nn

from . import _blocks


class AConvNet(nn.Module):
    """All-Convolutional Network (AConvNet) for SAR ATR.

    Implements the architecture from the original AConvNet paper using
    :class:`~._blocks.Conv2DBlock` building blocks.  Five convolutional stages
    are followed by global average pooling so the model is input-size agnostic.

    Keyword Args:
        classes (int): Number of output classes (default 10).
        channels (int): Number of input channels (default 1 for grayscale SAR).
        dropout_rate (float): Dropout probability applied before the last conv (default 0.5).
        w_init (callable): Weight initialiser (default Kaiming normal for ReLU).
        b_init (callable): Bias initialiser (default constant 0.1).
    """

    def __init__(self, **params):
        super(AConvNet, self).__init__()
        self.model_name = "AConvNet"

        self.dropout_rate = params.get("dropout_rate", 0.5)
        self.classes = params.get("classes", 10)
        self.channels = params.get("channels", 1)

        _w_init = params.get(
            "w_init", lambda x: nn.init.kaiming_normal_(x, nonlinearity="relu")
        )
        _b_init = params.get("b_init", lambda x: nn.init.constant_(x, 0.1))

        self._layer = nn.Sequential(
            _blocks.Conv2DBlock(
                shape=[5, 5, self.channels, 16],
                stride=1,
                padding="valid",
                activation="relu",
                max_pool=True,
                w_init=_w_init,
                b_init=_b_init,
            ),
            _blocks.Conv2DBlock(
                shape=[5, 5, 16, 32],
                stride=1,
                padding="valid",
                activation="relu",
                max_pool=True,
                w_init=_w_init,
                b_init=_b_init,
            ),
            _blocks.Conv2DBlock(
                shape=[6, 6, 32, 64],
                stride=1,
                padding="valid",
                activation="relu",
                max_pool=True,
                w_init=_w_init,
                b_init=_b_init,
            ),
            _blocks.Conv2DBlock(
                shape=[5, 5, 64, 128],
                stride=1,
                padding="valid",
                activation="relu",
                w_init=_w_init,
                b_init=_b_init,
            ),
            nn.Dropout(p=self.dropout_rate),
            _blocks.Conv2DBlock(
                shape=[3, 3, 128, self.classes],
                stride=1,
                padding="valid",
                w_init=_w_init,
                b_init=nn.init.zeros_,
            ),
            nn.AdaptiveAvgPool2d(1),  # added from original paper
            nn.Flatten(),
        )

    def forward(self, x):
        return self._layer(x)

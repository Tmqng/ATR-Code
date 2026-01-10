import torch

from models.AConvNet import network
import torch.nn as nn


class Model(object):
    def __init__(self, **params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device used:", self.device)
        self.net = network.Network(
            classes=params.get('classes', 10),
            channels=params.get('channels', 1),
            dropout_rate=params.get('dropout_rate', 0.5)
        )
        self.net.to(self.device)

        self.lr = params.get('lr', 1e-3)
        self.lr_step = params.get('lr_step', [50])
        self.lr_decay = params.get('lr_decay', 0.1)

        self.lr_scheduler = None

        self.momentum = params.get('momentum', 0.9)
        self.weight_decay = params.get('weight_decay', 4e-3)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        if self.lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=self.lr_step,
                gamma=self.lr_decay
            )

    def optimize(self, x, y):
        p = self.net(x.to(self.device))
        loss = self.criterion(p, y.to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def inference(self, x):
        return self.net(x.to(self.device))

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()



class Simple_AConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(Simple_AConvNet, self).__init__()

        # Conv1: 1 → 16 channels, 5x5, ReLU, MaxPool
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2: 16 → 32 channels, 5x5, ReLU, MaxPool
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv3: 32 → 64 channels, 6x6, ReLU, MaxPool
        self.conv3 = nn.Conv2d(32, 64, kernel_size=6, padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv4: 64 → 128 channels, 5x5, ReLU, Dropout
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=0)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # Conv5: 128 → num_classes channels, 3x3
        self.conv5 = nn.Conv2d(128, num_classes, kernel_size=3, padding=0)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.dropout(self.relu4(self.conv4(x)))
        x = self.conv5(x)
        x = self.gap(x)
        x = self.flatten(x)
        return x

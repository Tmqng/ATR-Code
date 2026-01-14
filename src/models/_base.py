import torch

from models.AConvNet import network
import torch.nn as nn


class Model(object):
    def __init__(self, **params):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used:", self.device)
        self.net = params.get(
            "net",
            network.AConvNet(
                classes=params.get("num_classes", 10),
                channels=params.get("channels", 1),
                dropout_rate=params.get("dropout_rate", 0.5),
            ),
        )
        self.net.to(self.device)

        self.lr = params.get("lr", 1e-3)
        self.lr_step = params.get("lr_step", [50])
        self.lr_decay = params.get("lr_decay", None)

        self.lr_scheduler = None

        self.momentum = params.get("momentum", 0.9)
        self.weight_decay = params.get("weight_decay", 4e-3)

        self.criterion = params.get("criterion", torch.nn.CrossEntropyLoss())
        self.optimizer = params.get("optimizer",
            torch.optim.Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=(self.momentum, 0.999),
                weight_decay=self.weight_decay,
            )
        )

        self.use_amp = params.get("use_amp", False)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        if self.lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer, milestones=self.lr_step, gamma=self.lr_decay
            )

    def optimize(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        
        # On utilise l'autocast ici aussi pour être cohérent
        with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None):
            p = self.net(x)
            loss = self.criterion(p, y)

        self.optimizer.zero_grad(set_to_none=True)
        
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.item(), p

    @torch.no_grad()
    def inference(self, x):
        return self.net(x.to(self.device))

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()

import torch

from models.AConvNet import network
import torch.nn as nn

from absl import logging
from tqdm import tqdm

import numpy as np

import os

import json


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
        self.lr_decay = params.get("lr_decay", 0.1)

        self.lr_scheduler = None

        self.momentum = params.get("momentum", 0.9)
        self.weight_decay = params.get("weight_decay", 4e-3)

        self.criterion = params.get("criterion", torch.nn.CrossEntropyLoss())
        self.optimizer = params.get(
            'optimizer',
            torch.optim.Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=(self.momentum, 0.999),
                weight_decay=self.weight_decay,
            )
        )

        if self.lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer, milestones=self.lr_step, gamma=self.lr_decay
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
    
    @torch.no_grad()
    def validation(self, ds, debug=False):
        # TODO separate into predict (computing predictions) and compute_metrics function
        num_data = 0
        corrects = 0

        # Test loop
        self.net.eval()
        _softmax = torch.nn.Softmax(dim=1)
        for i, data in enumerate(tqdm(ds)):

            try:
                images, labels, _ = data
            except ValueError:
                logging.error('Check your dataset loader')
                raise ValueError

            images = images.to(self.device)
            labels = labels.to(self.device)

            predictions = self.inference(images)
            predictions = predictions.to(self.device)
            predictions = _softmax(predictions)

            _, predictions = torch.max(predictions.data, 1)

            # DEBUG: Check predictions
            if debug and i == 0:
                logging.info(f"Predicted classes: {predictions[:10]}")
                logging.info(f"True labels: {labels[:10]}")
                logging.info(f"Matches: {(predictions == labels)[:10]}")

            labels = labels.type(torch.LongTensor)
            num_data += labels.size(0)
            corrects += (predictions == labels.to(self.device)).sum().item()

        accuracy = 100 * corrects / num_data
        return accuracy
    
    def run(self, train_set, val_set, epochs, experience_name, experiments_path=None, debug=False, early_stopping=False):

        model_path = os.path.join(experiments_path, f'{self.net.model_name}/models/{experience_name}')
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)

        history_path = os.path.join(experiments_path, f'{self.net.model_name}/history')
        if not os.path.exists(history_path):
            os.makedirs(history_path, exist_ok=True)

        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(epochs):
            _loss = []

            self.net.train()
            for i, data in enumerate(tqdm(train_set)):
                images, labels, _ = data
                _loss.append(self.optimize(images, labels))

            if self.lr_scheduler:
                lr = self.lr_scheduler.get_last_lr()[0]
                self.lr_scheduler.step()

            train_accuracy = self.validation(train_set, debug=debug)
            val_accuracy = self.validation(val_set, debug=debug)

            logging.info(
                f'Epoch: {epoch + 1:03d}/{epochs:03d} | loss={np.mean(_loss):.4f} | lr={lr} | Train accuracy={train_accuracy:.2f} | Validation accuracy={val_accuracy:.2f}'
            )

            history['train_loss'].append(np.mean(_loss))
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)

            if experiments_path:
                self.save(os.path.join(model_path, f'model-{epoch + 1:03d}.pth'))

            with open(os.path.join(history_path, f'history-{experience_name}.json'), mode='w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=True, indent=2)

            if early_stopping:
                # choose criterion
                continue

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()

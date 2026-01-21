import os
import logging

import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from scripts.train_alexnet import load_dataset, DATA_PATH
from src.models.alexnet.network import AlexNet

logging.basicConfig(level=logging.INFO)

class AlexNetObjective:
    """
    Objective function for Optuna to optimize AlexNet hyperparameters.
    """

    def __init__(
        self, num_classes: int, data_path: str
    ):
        self.data_path = data_path
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, trial: optuna.Trial) -> float:
        # Hyperparameters to search
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

        logging.info(
            f"Trial with lr={lr}, optimizer={optimizer_name}, batch_size={batch_size}, dropout_rate={dropout_rate}"
        )

        train_loader, val_loader = load_dataset(data_path=self.data_path, batch_size=batch_size)

        # Model setup
        model = self._build_model(dropout_rate=dropout_rate)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Short training (e.g., 2-3 epochs)
        epochs = 5
        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                # Correction du décalage de labels si nécessaire (t-1)
                labels = labels.to(self.device)
                inputs = inputs.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            accuracy = self._validate(model, val_loader)

            # Reporting for pruning (interrompt les essais non prometteurs)
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        logging.info(f"Trial completed with accuracy: {accuracy}")
        return accuracy

    def _build_model(self, dropout_rate: float) -> nn.Module:
        model = AlexNet(classes=self.num_classes, dropout_rate=dropout_rate)
        return model.to(self.device)

    def _validate(self, model: nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total


def run_optimization():
    config_name = "all"
    data_path = os.path.join(DATA_PATH, config_name)
    objective = AlexNetObjective(num_classes=10, data_path=data_path)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    logging.info("Best hyperparameters:", study.best_params)

    with open(f"best_hyperparams_alexnet_{config_name}.json", "w") as f:
        import json

        json.dump(study.best_params, f, indent=4)


if __name__ == "__main__":
    run_optimization()

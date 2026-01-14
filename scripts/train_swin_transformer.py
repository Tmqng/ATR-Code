import json
import os
import time
import sys

import numpy as np
import torch
from absl import app, flags, logging
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

from models._base import Model
from models.swin_transformer.network import create_swin_model
from models.swin_transformer.utils import load_dataset, validation
from utils import common
from models.swin_transformer.utils import train_epoch, validate

DATA_PATH = "/content/ATR-Code/datasets/MSTAR/MSTAR_IMG_JSON/SOC"
model_str = "swin_transformer"
flags.DEFINE_string("experiments_path", os.path.join(common.project_root, "experiments"), help="")
flags.DEFINE_string("config_name", f"{model_str}/config/{model_str}-SOC.json", help="")
FLAGS = flags.FLAGS

common.set_random_seed(42)


def run(
    config,
    experiments_path=None,
):
    train_loader, val_loader, test_loader = load_dataset(DATA_PATH, config=config)

    net = create_swin_model(config)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], betas=(0.9, 0.999)
    )
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    # Scheduler avec warmup
    num_steps = len(train_loader) * config["epochs"]
    warmup_steps = len(train_loader) * config["warmup_epochs"]

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=config["min_lr"],
        warmup_t=warmup_steps,
        warmup_lr_init=1e-6,
        warmup_prefix=True,
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler(enabled=config["use_amp"])

    m = Model(
        net=net,
        lr=config["lr"],
        lr_step=config["lr_step"],
        lr_decay=config["lr_decay"],
        weight_decay=config["weight_decay"],
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
    )

    model_path = os.path.join(experiments_path, f"{model_str}/models/{config['model_name']}")
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    history_path = os.path.join(experiments_path, f"{model_str}/history")
    if not os.path.exists(history_path):
        os.makedirs(history_path, exist_ok=True)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # for epoch in range(config["epochs"]):
    #     _loss = []

    #     m.net.train()
    #     for i, data in enumerate(tqdm(train_loader)):
    #         images, labels, _ = data
    #         images, labels = images.to(config["device"]), labels.to(config["device"])

    #         _loss.append(m.optimize(images, labels))

    #     if m.lr_scheduler:
    #         lr = m.lr_scheduler.get_last_lr()[0]
    #         m.lr_scheduler.step()

    #     train_accuracy = validation(m, train_loader)
    #     val_accuracy = validation(m, val_loader)

    #     logging.info(
    #         f"Epoch: {epoch + 1:03d}/{config['epochs']:03d} | loss={np.mean(_loss):.4f} | lr={lr} |"
    #         f" Train accuracy={train_accuracy:.2f} | Validation accuracy={val_accuracy:.2f}"
    #     )

    #     history["train_loss"].append(np.mean(_loss))
    #     history["train_accuracy"].append(train_accuracy)
    #     history["val_accuracy"].append(val_accuracy)

    #     if experiments_path:
    #         m.save(os.path.join(model_path, f"model-{epoch + 1:03d}.pth"))

    #     with open(
    #         os.path.join(history_path, f"history-{config['model_name']}.json"),
    #         mode="w",
    #         encoding="utf-8",
    #     ) as f:
    #         json.dump(history, f, ensure_ascii=True, indent=2)

    # Historique
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    # Best model tracking
    best_val_acc = 0.0
    best_epoch = 0

    # Timer
    start_time = time.time()

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            m, train_loader, criterion, optimizer, lr_scheduler, scaler, config, epoch
        )

        # Validation
        val_loss, val_acc, _, _ = validate(m, val_loader, criterion, config, "Val")

        # Scheduler epoch step
        lr_scheduler.step(epoch + 1)

        # Sauvegarde historique
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Temps
        epoch_time = time.time() - epoch_start

        # Affichage
        logging.info(f"\n   Epoch {epoch + 1}/{config['epochs']} - {epoch_time:.1f}s")
        logging.info(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logging.info(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        logging.info(f"   LR: {current_lr:.2e}")

        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": m.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config": vars(config),
            }

            torch.save(checkpoint, os.path.join(config["save_dir"], "swin_best.pth"))
            logging.info(f"     Nouveau meilleur modèle sauvegardé! (Val Acc: {val_acc:.4f})")

        # Sauvegarde périodique
        if (epoch + 1) % config["save_freq"] == 0:
            torch.save(checkpoint, os.path.join(config["save_dir"], f"swin_epoch_{epoch + 1}.pth"))
            logging.info(f"   Checkpoint sauvegardé (epoch {epoch + 1})")

    total_time = time.time() - start_time
    logging.info(f"\n  Entraînement terminé en {total_time / 60:.1f} minutes")
    logging.info(f"  Meilleure Val Acc: {best_val_acc:.4f} (epoch {best_epoch})")

def main(_):
    logging.info("Start")
    experiments_path = FLAGS.experiments_path
    config_name = FLAGS.config_name

    config = common.load_config(os.path.join(experiments_path, config_name))

    run(
        config,
        experiments_path,
    )

    logging.info("Finish")


if __name__ == "__main__":
    app.run(main)

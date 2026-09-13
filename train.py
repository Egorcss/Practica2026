# Обучение одной архитектуры. Запуск из корня проекта:
# python train.py --config configs/resnet50.yaml

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import build_dataloaders
from models import build_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # Директория запуска
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("logs") / f"{config['model_name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Логи: {run_dir}")

    # Данные
    train_loader, val_loader, _, class_names = build_dataloaders(config)
    print(f"Классов: {len(class_names)}, train: {len(train_loader.dataset)}, "
          f"val: {len(val_loader.dataset)}")

    # Модель
    model = build_model(config["model_name"], config["num_classes"], config["pretrained"]).to(device)
    print(f"Параметров: {count_parameters(model):,}, размер: {get_model_size_mb(model):.2f} МБ")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=config["lr"],
                      weight_decay=config.get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    patience = config.get("early_stopping_patience", 5)

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        dt = time.time() - t0
        print(f"[{epoch:02d}/{config['epochs']}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = Path("models") / f"{config['model_name']}_best.pth"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": config["model_name"],
                "num_classes": config["num_classes"],
                "input_size": config["input_size"],
                "val_acc": val_acc,
                "epoch": epoch,
                "class_names": class_names,
            }, ckpt_path)
            print(f"  ✓ Сохранена лучшая модель (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹ Early stopping на эпохе {epoch}")
                break

    metrics = {
        "model_name": config["model_name"],
        "best_val_acc": best_val_acc,
        "history": history,
        "num_params": count_parameters(model),
        "model_size_mb": get_model_size_mb(model),
        "config": config,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nГотово. Лучшая val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Путь к YAML-конфигу")
    args = parser.parse_args()
    main(args.config)
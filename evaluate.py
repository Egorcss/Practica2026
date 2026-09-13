# Оценка модели: accuracy, precision/recall/F1, confusion matrix,
# время инференса, размер модели. Запуск из корня:
# python evaluate.py --config configs/resnet50.yaml --ckpt models/resnet50_best.pth

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from dataset import build_dataloaders
from models import build_model
from train import load_config


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


@torch.no_grad()
def measure_inference_time(model, sample, device, n_runs: int = 50) -> float:
    model.eval()
    x = sample.to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000


def evaluate_model(config_path: str, ckpt_path: str, output_dir: str = "reports"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = build_dataloaders(config)

    model = build_model(config["model_name"], config["num_classes"], pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    preds, labels = collect_predictions(model, test_loader, device)

    metrics = {
        "model_name": config["model_name"],
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }

    sample = next(iter(test_loader))[0][:1]
    metrics["inference_time_ms"] = float(measure_inference_time(model, sample, device))
    metrics["model_size_mb"] = float(Path(ckpt_path).stat().st_size / (1024 ** 2))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title(f"Confusion Matrix — {config['model_name']}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{config['model_name']}.png", dpi=150)
    plt.close()

    metrics["per_class_report"] = classification_report(
        labels, preds, target_names=class_names, zero_division=0, output_dict=True
    )

    with open(output_dir / f"eval_{config['model_name']}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"=== {config['model_name']} ===")
    for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
              "inference_time_ms", "model_size_mb"]:
        print(f"  {k}: {metrics[k]:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", default="reports")
    args = parser.parse_args()
    evaluate_model(args.config, args.ckpt, args.output_dir)
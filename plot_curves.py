# Строит кривые обучения (loss и accuracy) для всех обученных моделей.
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ["mobilenetv3", "efficientnet_b0", "resnet50", "densenet121", "vgg16"]


def plot_model(model_name: str, save_dir: str = "reports"):
    # Находим последний запуск для модели
    pattern = f"logs/{model_name}_*/metrics.json"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"✗ {model_name}: не найден metrics.json (паттерн: {pattern})")
        return
    metrics_path = files[-1]

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data["history"]
    best = data.get("best_val_acc", max(h["val_acc"]))
    epochs = len(h["val_acc"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(h["train_loss"], label="train", marker="o")
    axes[0].plot(h["val_loss"], label="val", marker="s")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(h["train_acc"], label="train", marker="o")
    axes[1].plot(h["val_acc"], label="val", marker="s")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{model_name} — обучение ({epochs} эпох, лучшая val_acc = {best:.4f})",
                 fontsize=12)

    Path(save_dir).mkdir(exist_ok=True)
    out_path = Path(save_dir) / f"curves_{model_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ {out_path}  (эпох: {epochs}, лучшая val_acc: {best:.4f})")


if __name__ == "__main__":
    for m in MODELS:
        plot_model(m)
    print("\nВсе графики сохранены в reports/")
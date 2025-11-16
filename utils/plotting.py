import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(history, output_dir, model_name=None):
    """Save training metrics (loss and accuracy) plots and a JSON summary.

    Args:
        history (dict): Training history dictionary with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
        output_dir (str): Directory in which to write outputs.
        model_name (str): Optional model name used in filenames.
    """
    train_losses = history["loss"]
    val_losses = history.get("val_loss", [])
    train_acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    epochs = len(train_losses)
    x = list(range(1, epochs + 1))
    if epochs <= 10:
        ticks = x
    else:
        ticks = sorted(set(np.linspace(1, epochs, num=10, dtype=int).tolist()))
    plt.figure()
    plt.plot(x, train_losses, label="Train Loss")
    plt.plot(x, val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    title_loss = f"{model_name} Training Loss" if model_name else "Model Training Loss"
    plt.title(title_loss)
    plt.xticks(ticks)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_loss.png" if model_name else "loss.png"))
    plt.close()
    plt.figure()
    plt.plot(x, train_acc, label="Train Accuracy")
    plt.plot(x, val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    title_acc = f"{model_name} Training Accuracy" if model_name else "Model Training Accuracy"
    plt.title(title_acc)
    plt.xticks(ticks)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_accuracy.png" if model_name else "accuracy.png"))
    plt.close()
    metrics = {
        "epochs": epochs,
        "train_loss": [float(l) for l in train_losses],
        "val_loss": [float(l) for l in val_losses],
        "train_accuracy": [float(a) for a in train_acc],
        "val_accuracy": [float(a) for a in val_acc],
    }
    metrics_filename = f"{model_name}_training_metrics.json" if model_name else "training_metrics.json"
    with open(os.path.join(output_dir, metrics_filename), "w") as f:
        json.dump(metrics, f, indent=2)

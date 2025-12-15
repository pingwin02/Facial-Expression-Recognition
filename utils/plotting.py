import json
import os

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(y_true, y_pred, output_dir, label_map=None, filename="confusion_matrix.png"):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))

    class_names = "auto"
    if label_map:
        class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_metrics(history, output_dir, model_name=None):
    train_losses = history["loss"]
    val_losses = history.get("val_loss", [])

    acc_key = next((key for key in history.keys() if "accuracy" in key and "val" not in key), "accuracy")
    val_acc_key = f"val_{acc_key}"

    train_acc = history.get(acc_key, [])
    val_acc = history.get(val_acc_key, [])

    epochs = len(train_losses)
    x = list(range(1, epochs + 1))

    if epochs <= 10:
        ticks = x
    else:
        ticks = sorted(set(np.linspace(1, epochs, num=10, dtype=int).tolist()))

    plt.figure()
    plt.plot(x, train_losses, label="Train Loss")
    if val_losses:
        plt.plot(x, val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    title_loss = f"{model_name} Training Loss" if model_name else "Model Training Loss"
    plt.title(title_loss)
    plt.xticks(ticks)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_loss.png" if model_name else "loss.png"))
    plt.close()

    if train_acc:
        plt.figure()
        plt.plot(x, train_acc, label="Train Accuracy")
        if val_acc:
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
        "accuracy_metric_name": acc_key,
    }
    metrics_filename = f"{model_name}_training_metrics.json" if model_name else "training_metrics.json"
    with open(os.path.join(output_dir, metrics_filename), "w") as f:
        json.dump(metrics, f, indent=2)

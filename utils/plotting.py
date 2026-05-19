import json
import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap


def _summarize_frames_by_video(debugs):
    frames_per_video = {}
    frame_ids_per_video = {}

    if not debugs:
        return frames_per_video, frame_ids_per_video

    for idx, debug in enumerate(debugs):
        if not isinstance(debug, dict):
            continue
        video_name = debug.get("video")
        if not video_name:
            continue

        frames_per_video[video_name] = int(frames_per_video.get(video_name, 0) + 1)

        frame_idx = debug.get("frame_idx")
        if frame_idx is None:
            frame_idx = idx
        frame_ids_per_video.setdefault(video_name, []).append(int(frame_idx))

    for video_name in list(frame_ids_per_video.keys()):
        frame_ids_per_video[video_name] = sorted(frame_ids_per_video[video_name])

    return frames_per_video, frame_ids_per_video


def _extract_participants(debugs):
    participants = set()
    if not debugs:
        return sorted(participants)
    for debug in debugs:
        if not isinstance(debug, dict):
            continue
        p = debug.get("participant")
        if p:
            participants.add(str(p))
    return sorted(participants)


def save_confusion_matrix(y_true, y_pred, output_dir, label_map=None, filename="confusion_matrix.png"):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    def _wrap_label(label, width=14):
        normalized = str(label).replace("+", " + ").replace("_", " ")
        chunks = textwrap.wrap(normalized, width=width)
        return "\n".join(chunks) if chunks else str(label)

    class_names = "auto"
    if label_map:
        class_names = [_wrap_label(k) for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    n_classes = cm.shape[0] if hasattr(cm, "shape") else 2
    fig_w = max(10, min(24, 2.0 + 1.5 * n_classes))
    fig_h = max(8, min(20, 2.0 + 1.3 * n_classes))
    plt.figure(figsize=(fig_w, fig_h))

    annot_fontsize = max(14, min(22, 28 - n_classes))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": annot_fontsize, "fontweight": "bold"},
    )
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=13)
    plt.yticks(rotation=0, fontsize=13)
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

    return cm


def plot_metrics(
    history,
    output_dir,
    model_name=None,
    training_debugs=None,
    validation_debugs=None,
    dataset_name=None,
    label_map=None,
    cache_label=None,
    model_summary=None,
):
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

    train_frames_per_video, _ = _summarize_frames_by_video(training_debugs)
    val_frames_per_video, _ = _summarize_frames_by_video(validation_debugs)

    train_participants = _extract_participants(training_debugs)
    val_participants = _extract_participants(validation_debugs)

    metrics["dataset"] = dataset_name
    metrics["model"] = model_name
    if cache_label is not None:
        metrics["cache_label"] = cache_label
    if label_map is not None:
        metrics["label_map"] = {str(k): int(v) for k, v in label_map.items()}
    if model_summary is not None:
        metrics["model_architecture"] = model_summary
    metrics["data_summary"] = {
        "training_videos": int(len(train_frames_per_video)),
        "training_video_names": sorted(train_frames_per_video.keys()),
        "training_frames_total": int(sum(train_frames_per_video.values())),
        "training_participants": len(train_participants),
        "training_participant_ids": train_participants,
        "validation_videos": int(len(val_frames_per_video)),
        "validation_video_names": sorted(val_frames_per_video.keys()),
        "validation_frames_total": int(sum(val_frames_per_video.values())),
        "validation_participants": len(val_participants),
        "validation_participant_ids": val_participants,
    }

    metrics_filename = f"{model_name}_training_metrics.json" if model_name else "training_metrics.json"
    with open(os.path.join(output_dir, metrics_filename), "w") as f:
        json.dump(metrics, f, indent=2)

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.image import save_sample_frames


def save_confusion_matrix(y_true, y_pred, output_dir, label_map=None, filename="confusion_matrix.png"):
    """
    Generates and saves a confusion matrix heatmap.

    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.
        output_dir (str): Directory to save the plot.
        label_map (dict): Optional dictionary mapping label names to integers.
        filename (str): Name of the output file.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))

    class_names = None
    if label_map:
        class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving confusion matrix to {os.path.join(output_dir, filename)}")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def evaluate_model_on_data(
    loaded_model,
    val_tuple,
    output_dir,
    model_name="simple_sample_grid",
    max_samples=10,
    label_map=None,
    dataset_name=None,
):
    """
    Evaluate a loaded model on the full validation set, generate a confusion matrix,
    and save visualization of random sample frames with predictions.

    Args:
        loaded_model: A keras model instance with a .predict method.
        val_tuple: Tuple containing (X_val, y_val, val_debugs).
        output_dir (str): Directory to save sample images and confusion matrix.
        model_name (str): Optional model name for titles and filenames.
        max_samples (int): Maximum number of random samples to include in the visualization grid.
        label_map (dict): Optional mapping from label name->int.
        dataset_name (str): Optional dataset identifier used in the figure title.
    """
    X_val, y_val, val_debugs = val_tuple

    if len(X_val) == 0:
        return

    all_predictions = loaded_model.predict(X_val)
    all_preds_labels = np.argmax(all_predictions, axis=1)

    accuracy = np.mean(all_preds_labels == y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    save_confusion_matrix(
        y_val, all_preds_labels, output_dir, label_map=label_map, filename=f"{model_name}_confusion_matrix.png"
    )

    selected_indices = np.random.choice(len(X_val), size=min(max_samples, len(X_val)), replace=False)

    frames_sample = X_val[selected_indices]
    labels_sample = y_val[selected_indices]
    preds_sample = all_preds_labels[selected_indices]

    class_map = None
    if label_map:
        class_map = {v: k for k, v in label_map.items()}

    debugs_sample = []
    for i, idx in enumerate(selected_indices):
        debug_info = val_debugs[idx] if val_debugs is not None and idx < len(val_debugs) else None

        debug = {"frame_index": int(idx), "predicted_label": int(preds_sample[i]), "true_label": int(labels_sample[i])}

        if debug_info and isinstance(debug_info, dict):
            debug.update({k: debug_info.get(k) for k in ["crop_box", "landmarks"] if k in debug_info})

        if class_map:
            debug["class_map"] = class_map

        debugs_sample.append(debug)

    save_sample_frames(
        frames_sample,
        preds_sample,
        labels_sample,
        debugs_sample,
        output_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        accuracy=accuracy,
        filename=f"{model_name}_samples_with_landmarks.png",
    )

    debugs_no_landmarks = [{k: v for k, v in debug.items() if k != "landmarks"} for debug in debugs_sample]

    save_sample_frames(
        frames_sample,
        preds_sample,
        labels_sample,
        debugs_no_landmarks,
        output_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        accuracy=accuracy,
        filename=f"{model_name}_samples_no_landmarks.png",
    )

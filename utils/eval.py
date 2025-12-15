import numpy as np

from utils.image import save_sample_frames
from utils.plotting import save_confusion_matrix


def evaluate_model_on_data(
        loaded_model,
        val_tuple,
        output_dir,
        model_name="simple_sample_grid",
        max_samples=10,
        label_map=None,
        dataset_name=None,
):
    X_val, y_val, val_debugs = val_tuple

    if len(X_val) == 0:
        return

    all_predictions = loaded_model.predict(X_val, verbose=0)

    is_binary = all_predictions.shape[-1] == 1

    if is_binary:
        all_preds_labels = (all_predictions > 0.5).astype("int32").flatten()

        if label_map and ("happiness" in label_map or "happy" in label_map):
            target_key = "happiness" if "happiness" in label_map else "happy"
            target_idx = label_map[target_key]
            y_val_processed = (np.array(y_val).flatten() == target_idx).astype("int32")
            display_map = {"Not Happiness": 0, "Happiness": 1}
        else:
            y_val_processed = np.array(y_val).flatten()
            display_map = {0: 0, 1: 1}

    else:
        all_preds_labels = np.argmax(all_predictions, axis=1)
        y_val_processed = np.array(y_val).flatten()
        display_map = label_map

    accuracy = np.mean(all_preds_labels == y_val_processed)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    save_confusion_matrix(
        y_val_processed,
        all_preds_labels,
        output_dir,
        label_map=display_map,
        filename=f"{model_name}_confusion_matrix.png",
    )

    selected_indices = np.random.choice(len(X_val), size=min(max_samples, len(X_val)), replace=False)

    frames_sample = X_val[selected_indices]
    labels_sample = y_val_processed[selected_indices]
    preds_sample = all_preds_labels[selected_indices]

    class_map_inv = None
    if display_map:
        class_map_inv = {v: k for k, v in display_map.items()}

    debugs_sample = []
    for i, idx in enumerate(selected_indices):
        debug_info = val_debugs[idx] if val_debugs is not None and idx < len(val_debugs) else None

        debug = {"frame_index": int(idx), "predicted_label": int(preds_sample[i]), "true_label": int(labels_sample[i])}

        if debug_info and isinstance(debug_info, dict):
            debug.update({k: debug_info.get(k) for k in ["crop_box", "landmarks"] if k in debug_info})

        if class_map_inv:
            debug["class_map"] = class_map_inv

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

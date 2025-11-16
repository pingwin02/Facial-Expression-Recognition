import numpy as np

from utils.image import save_sample_frames


def evaluate_model_on_data(
        loaded_model,
        X_val_tuple,
        y_val_tuple,
        output_dir,
        model_name="simple_sample_grid",
        max_samples=10,
        label_map=None,
        dataset_name=None,
):
    """Evaluate a loaded model on a validation set and save sample frames.

    The function accepts either raw arrays or tuples returned by `load_data`.

    Args:
        loaded_model: A keras model instance with a .predict method.
        X_val_tuple: Either np.ndarray of frames or tuple (X_val, y_val, debugs).
        y_val_tuple: Either np.ndarray of labels or the same y_val passed separately.
        output_dir (str): Directory to save sample image.
        model_name (str): Optional model name for titles and filenames.
        max_samples (int): Maximum number of random samples to display.
        label_map (dict): Optional mapping from label name->int to reconstruct names.
        dataset_name (str): Optional dataset identifier used in the figure title.
    """
    if isinstance(X_val_tuple, tuple) or isinstance(X_val_tuple, list):
        X_val = X_val_tuple[0]
        val_debugs = X_val_tuple[2] if len(X_val_tuple) > 2 else [None] * len(X_val_tuple[0])
    else:
        X_val = X_val_tuple
        val_debugs = [None] * len(X_val)

    if isinstance(y_val_tuple, tuple) or isinstance(y_val_tuple, list):
        y_val = y_val_tuple[0]
    else:
        y_val = y_val_tuple

    print(f"Selecting up to {max_samples} samples from the validation set...")
    if len(X_val) == 0:
        print("No validation samples provided.")
        return
    selected_indices = np.random.choice(len(X_val), size=min(max_samples, len(X_val)), replace=False)
    frames = X_val[selected_indices]
    labels = y_val[selected_indices]
    preds = np.argmax(loaded_model.predict(frames), axis=1)

    class_map = None
    if label_map:
        class_map = {v: k for k, v in label_map.items()}

    debugs = []
    for i, frame in enumerate(frames):
        idx = int(selected_indices[i])
        debug_info = None
        if val_debugs and idx < len(val_debugs):
            debug_info = val_debugs[idx]
        debug = {"frame_index": idx, "predicted_label": int(preds[i]), "true_label": int(labels[i])}
        if debug_info and isinstance(debug_info, dict):
            debug.update({k: debug_info.get(k) for k in ["crop_box", "landmarks"] if k in debug_info})
        if class_map:
            debug["class_map"] = class_map
        debugs.append(debug)

    print(f"Generating sample PNG for {len(frames)} samples...")
    save_sample_frames(frames, preds, labels, debugs, output_dir, model_name=model_name, dataset_name=dataset_name)
    print("Sample PNG generation complete.")

    accuracy = np.mean(preds == labels)
    print(f"Validation accuracy: {accuracy:.4f} ({np.sum(preds == labels)}/{len(labels)})")

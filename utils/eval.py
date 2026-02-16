import json
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from utils.image import save_sample_frames
from utils.plotting import save_confusion_matrix
from utils.veatic_visualization import create_veatic_visualizations


def _build_label_names(label_map, labels):
    if not label_map:
        return [str(lbl) for lbl in labels]
    inv = {v: k for k, v in label_map.items()}
    return [str(inv.get(lbl, lbl)) for lbl in labels]


def _format_metrics_block(title, y_true, y_pred, label_map=None):
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    label_names = _build_label_names(label_map, labels)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    precision, recall, f1_vals, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    lines = [f"{title} metrics:"]
    lines.append(f"  Macro-F1: {macro_f1:.4f}")
    lines.append(f"  Weighted-F1: {weighted_f1:.4f}")
    lines.append(f"  Balanced Accuracy: {bal_acc:.4f}")
    lines.append("  Per-class metrics:")

    for idx, class_name in enumerate(label_names):
        lines.append(
            f"    {class_name}: precision={precision[idx]:.4f}, recall={recall[idx]:.4f}, "
            f"f1={f1_vals[idx]:.4f}, support={int(support[idx])}"
        )

    return "\n".join(lines)


def _select_unique_video_indices(val_debugs, max_samples):
    if not val_debugs:
        return []
    selected = []
    seen = set()
    for idx, debug in enumerate(val_debugs):
        if not isinstance(debug, dict):
            continue
        video_name = debug.get("video")
        if not video_name or video_name in seen:
            continue
        selected.append(idx)
        seen.add(video_name)
        if len(selected) >= max_samples:
            break
    return selected


def _select_correct_video_indices(val_debugs, y_true, y_pred, max_samples):
    if not val_debugs:
        return []

    selected = []
    seen = set()

    for idx, debug in enumerate(val_debugs):
        if not isinstance(debug, dict):
            continue
        if idx >= len(y_true) or idx >= len(y_pred):
            continue
        if int(y_true[idx]) != int(y_pred[idx]):
            continue
        video_name = debug.get("video")
        if not video_name or video_name in seen:
            continue
        selected.append(idx)
        seen.add(video_name)
        if len(selected) >= max_samples:
            return selected

    for idx, debug in enumerate(val_debugs):
        if not isinstance(debug, dict):
            continue
        video_name = debug.get("video")
        if not video_name or video_name in seen:
            continue
        selected.append(idx)
        seen.add(video_name)
        if len(selected) >= max_samples:
            break

    return selected


def _collect_metrics_dict(y_true, y_pred, label_map=None):
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    label_names = _build_label_names(label_map, labels)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    precision, recall, f1_vals, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = {}
    for idx, class_name in enumerate(label_names):
        per_class[class_name] = {
            "label_index": int(labels[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1_vals[idx]),
            "support": int(support[idx]),
        }

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(bal_acc),
        "per_class": per_class,
        "labels": [int(lbl) for lbl in labels],
    }


def _evaluate_video_level(predictions, y_labels, debug_infos, is_binary):
    if debug_infos is None or len(debug_infos) != len(y_labels):
        return None

    video_groups = {}
    for idx, debug in enumerate(debug_infos):
        if not isinstance(debug, dict):
            continue
        video_name = debug.get("video")
        if not video_name:
            continue
        video_groups.setdefault(video_name, []).append(idx)

    if not video_groups:
        return None

    y_true_video = []
    y_pred_video = []

    for _, indices in video_groups.items():
        indices = np.array(indices, dtype=int)
        labels_for_video = y_labels[indices]
        values, counts = np.unique(labels_for_video, return_counts=True)
        true_label = int(values[np.argmax(counts)])

        preds_for_video = predictions[indices]
        if is_binary:
            mean_score = float(np.mean(preds_for_video.reshape(-1)))
            pred_label = int(mean_score > 0.5)
        else:
            mean_scores = np.mean(preds_for_video, axis=0)
            pred_label = int(np.argmax(mean_scores))

        y_true_video.append(true_label)
        y_pred_video.append(pred_label)

    return np.array(y_true_video, dtype=np.int32), np.array(y_pred_video, dtype=np.int32)


def evaluate_model_on_data(
    loaded_model,
    val_tuple,
    output_dir,
    model_name="simple_sample_grid",
    max_samples=10,
    label_map=None,
    dataset_name=None,
    dataset_path=None,
):
    X_val, y_val, val_debugs = val_tuple

    if len(X_val) == 0:
        return

    expected_rank = None
    try:
        if hasattr(loaded_model, "input_shape") and loaded_model.input_shape is not None:
            expected_rank = len(loaded_model.input_shape)
    except Exception:
        expected_rank = None

    if expected_rank == 5 and hasattr(X_val, "ndim") and X_val.ndim == 4:
        X_val = np.expand_dims(X_val, axis=1)
    elif expected_rank == 4 and hasattr(X_val, "ndim") and X_val.ndim == 5:
        X_val = X_val[:, X_val.shape[1] // 2]

    all_predictions = loaded_model.predict(X_val, verbose=0)

    is_binary = all_predictions.shape[-1] == 1

    if is_binary:
        all_preds_labels = (all_predictions > 0.5).astype("int32").flatten()

        if label_map and ("neutral" in label_map):
            target_key = "neutral"
            target_idx = label_map[target_key]
            y_val_processed = (np.array(y_val).flatten() == target_idx).astype("int32")
            display_map = {"not neutral": 0, "neutral": 1}
        else:
            y_val_processed = np.array(y_val).flatten()
            display_map = {0: 0, 1: 1}

    else:
        all_preds_labels = np.argmax(all_predictions, axis=1)
        y_val_processed = np.array(y_val).flatten()
        display_map = label_map

    cm_true = y_val_processed
    cm_pred = all_preds_labels
    selected_level = "frame"

    video_eval = _evaluate_video_level(all_predictions, y_val_processed, val_debugs, is_binary=is_binary)
    if video_eval is not None:
        y_video_true, y_video_pred = video_eval
        cm_true = y_video_true
        cm_pred = y_video_pred
        selected_level = "video"

    accuracy = np.mean(cm_pred == cm_true)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    metrics_text = _format_metrics_block(
        "Validation",
        cm_true,
        cm_pred,
        label_map=display_map,
    )
    print(metrics_text)

    metrics_blocks = [metrics_text]
    metrics_json = {
        "dataset": dataset_name,
        "model": model_name,
        "evaluation_level": selected_level,
        "validation": _collect_metrics_dict(cm_true, cm_pred, label_map=display_map),
    }

    save_confusion_matrix(
        cm_true,
        cm_pred,
        output_dir,
        label_map=display_map,
        filename=f"{model_name}_confusion_matrix.png",
    )

    if dataset_name == "veatic":
        selected_indices = _select_correct_video_indices(val_debugs, y_val_processed, all_preds_labels, max_samples)
        if not selected_indices:
            selected_indices = _select_unique_video_indices(val_debugs, max_samples)
        if not selected_indices:
            selected_indices = np.random.choice(len(X_val), size=min(max_samples, len(X_val)), replace=False)
    else:
        selected_indices = np.random.choice(len(X_val), size=min(max_samples, len(X_val)), replace=False)

    frames_sample = X_val[selected_indices]
    if hasattr(frames_sample, "ndim") and frames_sample.ndim == 5:
        center_t = frames_sample.shape[1] // 2
        frames_sample = frames_sample[:, center_t]
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
            debug.update({k: debug_info.get(k) for k in ["crop_box", "landmarks", "video"] if k in debug_info})

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

    if dataset_name == "veatic" and dataset_path is not None:
        print(f"Generating VEATIC visualizations...")
        selected_videos_from_samples = set()
        for idx in selected_indices:
            if idx < len(val_debugs) and isinstance(val_debugs[idx], dict):
                video_name = val_debugs[idx].get("video")
                if video_name:
                    selected_videos_from_samples.add(video_name)

        create_veatic_visualizations(
            val_data=val_tuple,
            predictions=all_preds_labels,
            true_labels=y_val_processed,
            label_map=label_map,
            output_dir=output_dir,
            dataset_path=dataset_path,
            model=loaded_model,
            selected_videos=selected_videos_from_samples,
        )

    metrics_report_path = f"{output_dir}/{model_name}_metrics.txt"
    with open(metrics_report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(metrics_blocks) + "\n")
    print(f"Saved metrics report to {metrics_report_path}")

    metrics_json_path = f"{output_dir}/{model_name}_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics JSON to {metrics_json_path}")


def _format_regression_metrics_block(title, y_true, y_pred, tolerance=0.2):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # Try to compute R² (might fail if no variance)
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = float("nan")

    arousal_true = y_true[:, 0]
    arousal_pred = y_pred[:, 0]
    valence_true = y_true[:, 1]
    valence_pred = y_pred[:, 1]

    arousal_mse = mean_squared_error(arousal_true, arousal_pred)
    arousal_rmse = np.sqrt(arousal_mse)
    arousal_mae = mean_absolute_error(arousal_true, arousal_pred)
    try:
        arousal_r2 = r2_score(arousal_true, arousal_pred)
    except:
        arousal_r2 = float("nan")

    valence_mse = mean_squared_error(valence_true, valence_pred)
    valence_rmse = np.sqrt(valence_mse)
    valence_mae = mean_absolute_error(valence_true, valence_pred)
    try:
        valence_r2 = r2_score(valence_true, valence_pred)
    except:
        valence_r2 = float("nan")

    arousal_corr = np.corrcoef(arousal_true, arousal_pred)[0, 1]
    valence_corr = np.corrcoef(valence_true, valence_pred)[0, 1]

    lines = [f"{title} regression metrics:"]
    lines.append(f"  Overall:")
    lines.append(f"    MSE: {mse:.4f}")
    lines.append(f"    RMSE: {rmse:.4f}")
    lines.append(f"    MAE: {mae:.4f}")
    lines.append(f"    R²: {r2:.4f}")
    lines.append(f"  Arousal:")
    lines.append(f"    MSE: {arousal_mse:.4f}")
    lines.append(f"    RMSE: {arousal_rmse:.4f}")
    lines.append(f"    MAE: {arousal_mae:.4f}")
    lines.append(f"    R²: {arousal_r2:.4f}")
    lines.append(f"    Correlation: {arousal_corr:.4f}")
    lines.append(f"  Valence:")
    lines.append(f"    MSE: {valence_mse:.4f}")
    lines.append(f"    RMSE: {valence_rmse:.4f}")
    lines.append(f"    MAE: {valence_mae:.4f}")
    lines.append(f"    R²: {valence_r2:.4f}")
    lines.append(f"    Correlation: {valence_corr:.4f}")

    return "\n".join(lines)


def _collect_regression_metrics_dict(y_true, y_pred, tolerance=0.2):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except:
        r2 = None

    arousal_true = y_true[:, 0]
    arousal_pred = y_pred[:, 0]
    valence_true = y_true[:, 1]
    valence_pred = y_pred[:, 1]

    arousal_mse = float(mean_squared_error(arousal_true, arousal_pred))
    arousal_mae = float(mean_absolute_error(arousal_true, arousal_pred))
    try:
        arousal_r2 = float(r2_score(arousal_true, arousal_pred))
    except:
        arousal_r2 = None
    arousal_corr = float(np.corrcoef(arousal_true, arousal_pred)[0, 1])
    arousal_accuracy = float(np.mean(np.abs(arousal_true - arousal_pred) <= tolerance))

    valence_mse = float(mean_squared_error(valence_true, valence_pred))
    valence_mae = float(mean_absolute_error(valence_true, valence_pred))
    try:
        valence_r2 = float(r2_score(valence_true, valence_pred))
    except:
        valence_r2 = None
    valence_corr = float(np.corrcoef(valence_true, valence_pred)[0, 1])
    valence_accuracy = float(np.mean(np.abs(valence_true - valence_pred) <= tolerance))

    overall_accuracy = float(
        np.mean((np.abs(arousal_true - arousal_pred) <= tolerance) & (np.abs(valence_true - valence_pred) <= tolerance))
    )

    return {
        "overall": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "accuracy_within_tolerance": overall_accuracy,
            "tolerance": tolerance,
        },
        "arousal": {
            "mse": arousal_mse,
            "rmse": float(np.sqrt(arousal_mse)),
            "mae": arousal_mae,
            "r2": arousal_r2,
            "correlation": arousal_corr,
            "accuracy_within_tolerance": arousal_accuracy,
        },
        "valence": {
            "mse": valence_mse,
            "rmse": float(np.sqrt(valence_mse)),
            "mae": valence_mae,
            "r2": valence_r2,
            "correlation": valence_corr,
            "accuracy_within_tolerance": valence_accuracy,
        },
    }


def evaluate_regression_model_on_data(
    loaded_model,
    val_tuple,
    output_dir,
    model_name="ArousalValenceModel",
    max_samples=10,
    dataset_name=None,
):
    X_val, y_val, val_debugs = val_tuple

    if len(X_val) == 0:
        print("No validation data provided.")
        return

    if hasattr(X_val, "ndim") and X_val.ndim == 4:
        X_val = np.expand_dims(X_val, axis=1)

    print("Generating predictions...")
    predictions = loaded_model.predict(X_val, verbose=0)

    if isinstance(predictions, list):
        arousal_pred = predictions[0].flatten()
        valence_pred = predictions[1].flatten()
        predictions = np.column_stack([arousal_pred, valence_pred])

    y_val = np.asarray(y_val, dtype=np.float32)

    if y_val.ndim == 1:
        print("Warning: y_val is 1D, expected 2D with [arousal, valence]")
        return

    print(f"Predictions shape: {predictions.shape}")
    print(f"Ground truth shape: {y_val.shape}")

    metrics_text = _format_regression_metrics_block("Validation Metrics", y_val, predictions)
    print("\n" + metrics_text)

    metrics_json = {
        "dataset": dataset_name,
        "model": model_name,
        "metrics": _collect_regression_metrics_dict(y_val, predictions),
    }

    metrics_report_path = f"{output_dir}/{model_name}_metrics.txt"
    with open(metrics_report_path, "w", encoding="utf-8") as f:
        f.write(metrics_text + "\n")
    print(f"\nSaved metrics report to {metrics_report_path}")

    metrics_json_path = f"{output_dir}/{model_name}_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics JSON to {metrics_json_path}")

    selected_indices = np.random.choice(len(X_val), size=min(max_samples, len(X_val)), replace=False)

    frames_sample = X_val[selected_indices]
    if hasattr(frames_sample, "ndim") and frames_sample.ndim == 5:
        center_t = frames_sample.shape[1] // 2
        frames_sample = frames_sample[:, center_t]

    y_true_sample = y_val[selected_indices]
    y_pred_sample = predictions[selected_indices]

    debugs_sample = []
    for i, idx in enumerate(selected_indices):
        debug_info = val_debugs[idx] if val_debugs is not None and idx < len(val_debugs) else None

        debug = {
            "frame_index": int(idx),
            "predicted_arousal": float(y_pred_sample[i, 0]),
            "predicted_valence": float(y_pred_sample[i, 1]),
            "true_arousal": float(y_true_sample[i, 0]),
            "true_valence": float(y_true_sample[i, 1]),
        }

        if debug_info and isinstance(debug_info, dict):
            debug.update({k: debug_info.get(k) for k in ["crop_box", "landmarks", "video"] if k in debug_info})

        debugs_sample.append(debug)

    labels_sample = np.arange(len(y_true_sample))
    preds_sample = np.arange(len(y_pred_sample))

    save_sample_frames(
        frames_sample,
        preds_sample,
        labels_sample,
        debugs_sample,
        output_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        accuracy=None,  # Not applicable for regression
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
        accuracy=None,
        filename=f"{model_name}_samples_no_landmarks.png",
    )

    print(f"Evaluation complete. Results saved to {output_dir}")

import json
import os

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
)

from utils.gradcam import save_gradcam_grid
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


def _build_video_to_indices(debugs):
    groups = {}
    if not debugs:
        return groups

    for idx, debug in enumerate(debugs):
        if not isinstance(debug, dict):
            continue
        video_name = debug.get("video")
        if not video_name:
            continue
        groups.setdefault(video_name, []).append(idx)
    return groups


def _expand_indices_per_video(val_debugs, base_video_indices, frames_per_video=10):
    if not base_video_indices:
        return []

    video_to_indices = _build_video_to_indices(val_debugs)
    expanded = []
    seen_videos = set()
    rng = np.random.default_rng()

    for base_idx in base_video_indices:
        if base_idx >= len(val_debugs):
            continue
        debug = val_debugs[base_idx]
        if not isinstance(debug, dict):
            continue
        video_name = debug.get("video")
        if not video_name or video_name in seen_videos:
            continue

        candidate_indices = video_to_indices.get(video_name, [])
        if not candidate_indices:
            continue

        sample_size = min(len(candidate_indices), max(1, int(frames_per_video)))
        selected_for_video = rng.choice(candidate_indices, size=sample_size, replace=False).tolist()
        selected_for_video = sorted(
            selected_for_video,
            key=lambda idx: (
                int(val_debugs[idx].get("frame_idx", 10 ** 9)) if isinstance(val_debugs[idx], dict) else 10 ** 9,
                idx,
            ),
        )
        expanded.extend(selected_for_video)
        seen_videos.add(video_name)

    return expanded


def _read_video_frame_total(dataset_path, video_name):
    if not dataset_path or not video_name:
        return None

    video_id, _ = os.path.splitext(video_name)
    arousal_path = os.path.join(dataset_path, "rating_averaged", f"{video_id}_arousal.csv")

    if not os.path.exists(arousal_path):
        return None

    count = 0
    try:
        with open(arousal_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except OSError:
        return None

    return int(count) if count > 0 else None


def _video_sort_key(video_name):
    if not video_name:
        return (1, "", 10 ** 9)

    base = os.path.splitext(os.path.basename(str(video_name)))[0]
    try:
        return (0, "", int(base))
    except ValueError:
        return (1, base.lower(), 10 ** 9)


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
        if len(np.unique(labels_for_video)) > 1:
            return None

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
        max_samples=9,
        label_map=None,
        dataset_name=None,
        dataset_path=None,
        train_tuple=None,
        cache_label=None,
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

    metrics_json = {
        "dataset": dataset_name,
        "model": model_name,
        "evaluation_level": selected_level,
        "validation": _collect_metrics_dict(cm_true, cm_pred, label_map=display_map),
    }

    cm = save_confusion_matrix(
        cm_true,
        cm_pred,
        output_dir,
        label_map=display_map,
        filename=f"{model_name}_confusion_matrix.png",
    )

    cm_labels = None
    if display_map:
        cm_labels = [str(k) for k, v in sorted(display_map.items(), key=lambda item: item[1])]
    metrics_json["validation"]["confusion_matrix"] = cm.tolist()
    if cm_labels:
        metrics_json["validation"]["confusion_matrix_labels"] = cm_labels
    if cache_label is not None:
        metrics_json["cache_label"] = cache_label
    if label_map is not None:
        metrics_json["label_map"] = {str(k): int(v) for k, v in label_map.items()}

    if dataset_name == "veatic":
        selected_video_anchor_indices = _select_correct_video_indices(
            val_debugs, y_val_processed, all_preds_labels, max_samples
        )
        if not selected_video_anchor_indices:
            selected_video_anchor_indices = _select_unique_video_indices(val_debugs, max_samples)
        if not selected_video_anchor_indices:
            selected_video_anchor_indices = np.random.choice(
                len(X_val), size=min(max_samples, len(X_val)), replace=False
            ).tolist()

        selected_indices = _expand_indices_per_video(
            val_debugs,
            selected_video_anchor_indices,
            frames_per_video=10,
        )
        if not selected_indices:
            selected_indices = selected_video_anchor_indices

        selected_indices = sorted(
            selected_indices,
            key=lambda idx: (
                _video_sort_key(
                    val_debugs[idx].get("video")
                    if idx < len(val_debugs) and isinstance(val_debugs[idx], dict)
                    else None
                ),
                (
                    int(val_debugs[idx].get("frame_idx", 10 ** 9))
                    if idx < len(val_debugs) and isinstance(val_debugs[idx], dict)
                    else 10 ** 9
                ),
                int(idx),
            ),
        )
    else:
        selected_indices = np.random.choice(len(X_val), size=min(max_samples, len(X_val)), replace=False).tolist()

    selected_indices = [int(idx) for idx in selected_indices]

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
    frame_total_cache = {}
    for i, idx in enumerate(selected_indices):
        debug_info = val_debugs[idx] if val_debugs is not None and idx < len(val_debugs) else None

        debug = {"frame_index": int(idx), "predicted_label": int(preds_sample[i]), "true_label": int(labels_sample[i])}

        if debug_info and isinstance(debug_info, dict):
            debug.update({k: debug_info.get(k) for k in ["video", "frame_idx"] if k in debug_info})

            video_name = debug.get("video")
            if video_name:
                if video_name not in frame_total_cache:
                    frame_total_cache[video_name] = _read_video_frame_total(dataset_path, video_name)
                debug["frame_total"] = frame_total_cache[video_name]

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
        filename=f"{model_name}_samples.png",
        highlight_correctness_bg=True,
        cols=3,
    )

    try:
        gradcam_frames = X_val[selected_indices]
        if hasattr(gradcam_frames, "ndim") and gradcam_frames.ndim == 4:
            gradcam_frames = np.expand_dims(gradcam_frames, axis=1)
        save_gradcam_grid(
            loaded_model,
            gradcam_frames,
            preds_sample,
            labels_sample,
            debugs_sample,
            output_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            accuracy=accuracy,
            filename=f"{model_name}_gradcam.png",
            cols=3,
        )
    except Exception as e:
        print(f"Warning: GradCAM generation failed: {e}")

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
            selected_sample_indices=set(selected_indices),
        )

    example_frames_per_video = {}
    example_frame_ids_per_video = {}
    for idx in selected_indices:
        if idx >= len(val_debugs) or not isinstance(val_debugs[idx], dict):
            continue
        debug_item = val_debugs[idx]
        video_name = debug_item.get("video")
        if not video_name:
            continue
        example_frames_per_video[video_name] = int(example_frames_per_video.get(video_name, 0) + 1)
        frame_idx = debug_item.get("frame_idx")
        if frame_idx is not None:
            example_frame_ids_per_video.setdefault(video_name, []).append(int(frame_idx))

    for video_name in list(example_frame_ids_per_video.keys()):
        example_frame_ids_per_video[video_name] = sorted(example_frame_ids_per_video[video_name])

    eval_frames_per_video = {}
    eval_frame_ids_per_video = {}
    for eval_idx, debug_item in enumerate(val_debugs):
        if not isinstance(debug_item, dict):
            continue
        video_name = debug_item.get("video")
        if not video_name:
            continue
        eval_frames_per_video[video_name] = int(eval_frames_per_video.get(video_name, 0) + 1)
        frame_idx = debug_item.get("frame_idx")
        if frame_idx is None:
            frame_idx = eval_idx
        eval_frame_ids_per_video.setdefault(video_name, []).append(int(frame_idx))

    for video_name in list(eval_frame_ids_per_video.keys()):
        eval_frame_ids_per_video[video_name] = sorted(eval_frame_ids_per_video[video_name])

    eval_participants = set()
    for d in val_debugs:
        if isinstance(d, dict):
            p = d.get("participant")
            if p:
                eval_participants.add(str(p))
    eval_participants = sorted(eval_participants)

    test_count = int(len(X_val))
    metrics_json["data_summary"] = {
        "test_samples": test_count,
        "test_participants": len(eval_participants),
        "test_participant_ids": eval_participants,
        "evaluation_videos": int(len(eval_frames_per_video)),
        "evaluation_video_names": sorted(eval_frames_per_video.keys()),
        "evaluation_frames_total": int(sum(eval_frames_per_video.values())),
        "example_videos": int(len(example_frames_per_video)),
        "example_frames_total": int(len(selected_indices)),
    }

    metrics_json_path = f"{output_dir}/{model_name}_evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics JSON to {metrics_json_path}")

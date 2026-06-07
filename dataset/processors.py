import os

os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import cv2
import numpy as np
import pickle
import re
from tqdm import tqdm

from utils.image import get_dlib_detector_predictor, detect_and_crop_face

IMG_HEIGHT = 224
IMG_WIDTH = 224

NUM_SAMPLE_FRAMES = 5
CENTER_IDX = NUM_SAMPLE_FRAMES // 2

TEMPORAL_TINTS = [
    np.array([1.0, 0.1, 0.2], dtype=np.float32),
    np.array([0.2, 1.0, 0.1], dtype=np.float32),
    None,
    np.array([0.1, 0.6, 1.0], dtype=np.float32),
    np.array([0.3, 0.1, 1.0], dtype=np.float32),
]

TEMPORAL_ALPHAS = [1.2, 0.9, 0.0, 0.9, 1.2]


def _get_temporal_tints_alphas(n_frames):
    if n_frames == 5:
        return TEMPORAL_TINTS, TEMPORAL_ALPHAS

    center = n_frames // 2
    tints = []
    alphas = []
    base_colors = [
        np.array([1.0, 0.1, 0.2], dtype=np.float32),
        np.array([0.2, 1.0, 0.1], dtype=np.float32),
        np.array([0.1, 0.6, 1.0], dtype=np.float32),
        np.array([0.3, 0.1, 1.0], dtype=np.float32),
        np.array([0.8, 0.8, 0.1], dtype=np.float32),
        np.array([0.1, 0.9, 0.9], dtype=np.float32),
    ]

    for i in range(n_frames):
        if i == center:
            tints.append(None)
            alphas.append(0.0)
        else:
            color_idx = i % len(base_colors)
            if i > center:
                color_idx = (i - 1) % len(base_colors)
            tints.append(base_colors[color_idx])
            distance = abs(i - center) / max(1, center)
            alphas.append(0.9 + 0.3 * distance)

    return tints, alphas


def normalize_and_create_mask(landmarks, crop_box, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    x1, y1, x2, y2 = crop_box
    crop_w = x2 - x1
    crop_h = y2 - y1

    scale_x = target_size[0] / crop_w
    scale_y = target_size[1] / crop_h

    mask = np.zeros((target_size[1], target_size[0]), dtype=np.float32)

    for lx, ly in landmarks:
        new_x = int((lx - x1) * scale_x)
        new_y = int((ly - y1) * scale_y)

        if 0 <= new_x < target_size[0] and 0 <= new_y < target_size[1]:
            cv2.circle(mask, (new_x, new_y), 1, 1.0, -1)

    if np.any(mask > 0):
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        max_val = np.max(mask)
        if max_val > 0:
            mask = mask / max_val

    return mask


def _frame_quality_score(frame, prev_gray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    motion = 0.0 if prev_gray is None else float(np.mean(cv2.absdiff(gray, prev_gray)))
    brightness = float(np.mean(gray))
    exposure_score = 1.0 - abs(brightness - 127.5) / 127.5

    score = (0.65 * sharpness) + (0.3 * motion) + (0.05 * max(0.0, exposure_score) * 255.0)
    return score, gray


def _has_face_detected(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    return len(faces) >= 1, faces


def _select_diverse_top_indices(scored_indices, target_count, total_frames):
    if not scored_indices:
        return []

    ranked = sorted(scored_indices, key=lambda x: x[1], reverse=True)
    min_gap = max(1, total_frames // max(1, target_count * 3))

    selected = []
    for frame_idx, _ in ranked:
        if all(abs(frame_idx - prev_idx) >= min_gap for prev_idx in selected):
            selected.append(frame_idx)
            if len(selected) >= target_count:
                break

    if len(selected) < target_count:
        for frame_idx, _ in ranked:
            if frame_idx not in selected:
                selected.append(frame_idx)
                if len(selected) >= target_count:
                    break

    return sorted(selected[:target_count])


def _map_video_frame_to_sequence_index(frame_idx, total_frames, sequence_length):
    if sequence_length <= 0:
        return 0
    if total_frames <= 1:
        return 0

    ratio = float(frame_idx) / float(total_frames - 1)
    mapped = int(round(ratio * float(sequence_length - 1)))
    return int(np.clip(mapped, 0, sequence_length - 1))


def _checkpoint_size_tag(size_tag=None):
    if size_tag:
        return re.sub(r"[^a-zA-Z0-9x_-]", "_", str(size_tag))
    return f"{IMG_WIDTH}x{IMG_HEIGHT}"


def _checkpoint_pattern(checkpoint_prefix, size_tag=None):
    safe_size_tag = re.escape(_checkpoint_size_tag(size_tag))
    return re.compile(rf"^{re.escape(checkpoint_prefix)}_{safe_size_tag}_iter(\d+)\.tmp$")


def _checkpoint_path(checkpoint_dir, checkpoint_prefix, iteration, size_tag=None):
    safe_size_tag = _checkpoint_size_tag(size_tag)
    return os.path.join(checkpoint_dir, f"{checkpoint_prefix}_{safe_size_tag}_iter{int(iteration):06d}.tmp")


def _find_latest_checkpoint(checkpoint_dir, checkpoint_prefix, size_tag=None):
    if not checkpoint_dir or not checkpoint_prefix:
        return None
    if not os.path.isdir(checkpoint_dir):
        return None

    pattern = _checkpoint_pattern(checkpoint_prefix, size_tag=size_tag)
    latest_iter = -1
    latest_path = None

    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if not match:
            continue
        iter_idx = int(match.group(1))
        if iter_idx > latest_iter:
            latest_iter = iter_idx
            latest_path = os.path.join(checkpoint_dir, filename)

    return latest_path


def _save_iteration_checkpoint(
        checkpoint_dir,
        checkpoint_prefix,
        next_index,
        X,
        y,
        debugs,
        size_tag=None,
):
    if not checkpoint_dir or not checkpoint_prefix:
        return None

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, checkpoint_prefix, next_index, size_tag=size_tag)
    payload = {
        "next_index": int(next_index),
        "X": X,
        "y": y,
        "debugs": debugs,
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    pattern = _checkpoint_pattern(checkpoint_prefix, size_tag=size_tag)
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if not match:
            continue
        candidate_path = os.path.join(checkpoint_dir, filename)
        if candidate_path == path:
            continue
        try:
            os.remove(candidate_path)
        except OSError:
            pass

    return path


def _load_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, size_tag=None):
    latest_path = _find_latest_checkpoint(checkpoint_dir, checkpoint_prefix, size_tag=size_tag)
    if latest_path is None:
        return 0, [], [], []

    with open(latest_path, "rb") as f:
        payload = pickle.load(f)

    next_index = int(payload.get("next_index", 0))
    X = payload.get("X", [])
    y = payload.get("y", [])
    debugs = payload.get("debugs", [])

    print(f"Resuming from checkpoint: {latest_path} " f"(next_index={next_index}, samples={len(X)})")

    return next_index, X, y, debugs


def _disk_store_base_path(storage_dir, storage_prefix, size_tag=None):
    safe_size_tag = _checkpoint_size_tag(size_tag)
    return os.path.join(storage_dir, f"{storage_prefix}_{safe_size_tag}")


def _disk_store_paths(storage_dir, storage_prefix, size_tag=None):
    base_path = _disk_store_base_path(storage_dir, storage_prefix, size_tag=size_tag)
    return {
        "X": f"{base_path}_X.npy",
        "y": f"{base_path}_y.npy",
        "meta": f"{base_path}_meta.pkl",
        "debugs": f"{base_path}_debugs.pkl",
    }


def _ensure_disk_store_arrays(
        storage_dir,
        storage_prefix,
        max_samples,
        sample_shape,
        sample_dtype=np.uint8,
        label_dtype=np.uint8,
        size_tag=None,
):
    os.makedirs(storage_dir, exist_ok=True)
    paths = _disk_store_paths(storage_dir, storage_prefix, size_tag=size_tag)

    expected_x_shape = (int(max_samples), *sample_shape)
    expected_y_shape = (int(max_samples),)
    sample_dtype = np.dtype(sample_dtype)
    label_dtype = np.dtype(label_dtype)

    if os.path.exists(paths["X"]):
        X_store = np.load(paths["X"], mmap_mode="r+")
        if X_store.shape != expected_x_shape or X_store.dtype != sample_dtype:
            del X_store
            os.remove(paths["X"])
            X_store = np.lib.format.open_memmap(
                paths["X"],
                mode="w+",
                dtype=sample_dtype,
                shape=expected_x_shape,
            )
    else:
        X_store = np.lib.format.open_memmap(
            paths["X"],
            mode="w+",
            dtype=sample_dtype,
            shape=expected_x_shape,
        )

    if os.path.exists(paths["y"]):
        y_store = np.load(paths["y"], mmap_mode="r+")
        if y_store.shape != expected_y_shape or y_store.dtype != label_dtype:
            del y_store
            os.remove(paths["y"])
            y_store = np.lib.format.open_memmap(
                paths["y"],
                mode="w+",
                dtype=label_dtype,
                shape=expected_y_shape,
            )
    else:
        y_store = np.lib.format.open_memmap(
            paths["y"],
            mode="w+",
            dtype=label_dtype,
            shape=expected_y_shape,
        )

    return paths, X_store, y_store


def _convert_sample_for_storage(sample, target_dtype):
    target_dtype = np.dtype(target_dtype)
    if target_dtype == np.uint8:
        return np.clip(np.rint(np.asarray(sample) * 255.0), 0, 255).astype(np.uint8)
    return np.asarray(sample, dtype=target_dtype)


def _save_disk_store_metadata(
        storage_dir,
        storage_prefix,
        sample_count,
        max_samples,
        sample_shape,
        sample_dtype,
        label_dtype,
        size_tag=None,
):
    paths = _disk_store_paths(storage_dir, storage_prefix, size_tag=size_tag)
    payload = {
        "sample_count": int(sample_count),
        "max_samples": int(max_samples),
        "sample_shape": tuple(int(dim) for dim in sample_shape),
        "sample_dtype": np.dtype(sample_dtype).str,
        "label_dtype": np.dtype(label_dtype).str,
    }
    with open(paths["meta"], "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return paths


def _save_disk_backed_iteration_checkpoint(
        checkpoint_dir,
        checkpoint_prefix,
        next_index,
        sample_count,
        debugs,
        size_tag=None,
):
    if not checkpoint_dir or not checkpoint_prefix:
        return None

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, checkpoint_prefix, next_index, size_tag=size_tag)
    payload = {
        "next_index": int(next_index),
        "sample_count": int(sample_count),
        "debugs": debugs,
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    pattern = _checkpoint_pattern(checkpoint_prefix, size_tag=size_tag)
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if not match:
            continue
        candidate_path = os.path.join(checkpoint_dir, filename)
        if candidate_path == path:
            continue
        try:
            os.remove(candidate_path)
        except OSError:
            pass

    return path


def _load_disk_backed_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, size_tag=None):
    latest_path = _find_latest_checkpoint(checkpoint_dir, checkpoint_prefix, size_tag=size_tag)
    if latest_path is None:
        return 0, 0, []

    with open(latest_path, "rb") as f:
        payload = pickle.load(f)

    next_index = int(payload.get("next_index", 0))
    sample_count = int(payload.get("sample_count", 0))
    debugs = payload.get("debugs", [])

    print(
        f"Resuming from checkpoint: {latest_path} "
        f"(next_index={next_index}, samples={sample_count})"
    )

    return next_index, sample_count, debugs


def cleanup_iteration_checkpoints(checkpoint_dir, checkpoint_prefix, size_tag=None):
    if not checkpoint_dir or not checkpoint_prefix:
        return
    if not os.path.isdir(checkpoint_dir):
        return

    pattern = _checkpoint_pattern(checkpoint_prefix, size_tag=size_tag)
    removed_count = 0

    for filename in os.listdir(checkpoint_dir):
        if not pattern.match(filename):
            continue
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        try:
            os.remove(checkpoint_path)
            removed_count += 1
        except OSError:
            pass

    if removed_count > 0:
        print(f"Removed {removed_count} checkpoint tmp file(s) for prefix '{checkpoint_prefix}'.")


def process_video_frames_with_frame_labels(
        df,
        video_dir,
        filename_col,
        label_map,
        frames_per_video=100,
        checkpoint_dir=None,
        checkpoint_prefix=None,
        save_checkpoint_every=1,
        resume_from_checkpoint=True,
        disk_output_dir=None,
        disk_output_prefix=None,
        max_samples=None,
        storage_dtype=np.uint8,
        label_dtype=np.uint8,
):
    use_disk_backed_storage = (
        disk_output_dir is not None and disk_output_prefix is not None and max_samples is not None
    )
    checkpoint_stride = max(1, int(save_checkpoint_every))

    if use_disk_backed_storage:
        max_samples = max(1, int(max_samples))
        if resume_from_checkpoint and checkpoint_dir and checkpoint_prefix:
            start_index, sample_count, debugs = _load_disk_backed_iteration_checkpoint(
                checkpoint_dir,
                checkpoint_prefix,
            )
        else:
            start_index, sample_count, debugs = 0, 0, []

        reset_disk_storage = start_index == 0 and sample_count == 0
        storage_paths = _disk_store_paths(disk_output_dir, disk_output_prefix)
        if not reset_disk_storage and (
                not os.path.exists(storage_paths["X"]) or not os.path.exists(storage_paths["y"])
        ):
            print("Checkpoint found without disk artifacts; restarting this split from scratch.")
            start_index, sample_count, debugs = 0, 0, []
            reset_disk_storage = True

        if reset_disk_storage:
            for path in storage_paths.values():
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        storage_paths, X_store, y_store = _ensure_disk_store_arrays(
            disk_output_dir,
            disk_output_prefix,
            max_samples,
            sample_shape=(IMG_HEIGHT, IMG_WIDTH, 4),
            sample_dtype=storage_dtype,
            label_dtype=label_dtype,
        )
        X, y = None, None
    elif resume_from_checkpoint and checkpoint_dir and checkpoint_prefix:
        start_index, X, y, debugs = _load_iteration_checkpoint(checkpoint_dir, checkpoint_prefix)
    else:
        start_index, X, y, debugs = 0, [], [], []

    def maybe_save_progress(next_index):
        if not checkpoint_dir or not checkpoint_prefix:
            return
        if next_index % checkpoint_stride != 0:
            return

        if use_disk_backed_storage:
            X_store.flush()
            y_store.flush()
            _save_disk_backed_iteration_checkpoint(
                checkpoint_dir,
                checkpoint_prefix,
                next_index,
                sample_count,
                debugs,
            )
        else:
            _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, next_index, X, y, debugs)

    detector_pack = get_dlib_detector_predictor()
    detector, _ = detector_pack

    first_run = True

    for row_idx, (_, row) in enumerate(tqdm(df.iterrows(), desc="Processing videos", total=len(df), unit="video")):
        if row_idx < start_index:
            continue

        video_name = row[filename_col]
        video_path = os.path.join(video_dir, video_name)

        arousal_seq = row.get("full_arousal_sequence", [])
        valence_seq = row.get("full_valence_sequence", [])
        if arousal_seq is None or valence_seq is None:
            maybe_save_progress(row_idx + 1)
            continue

        arousal_seq = list(arousal_seq)
        valence_seq = list(valence_seq)
        sequence_length = min(len(arousal_seq), len(valence_seq))
        if sequence_length <= 0:
            maybe_save_progress(row_idx + 1)
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            maybe_save_progress(row_idx + 1)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        candidate_indices = np.arange(total_frames, dtype=int)
        scored_candidates = []
        face_cache = {}
        prev_gray = None

        for frame_idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            has_face, faces = _has_face_detected(frame, detector)
            if not has_face:
                continue

            face_cache[int(frame_idx)] = faces
            quality, prev_gray = _frame_quality_score(frame, prev_gray)
            scored_candidates.append((int(frame_idx), float(quality)))

        if not scored_candidates:
            cap.release()
            continue

        use_all_frames = frames_per_video is None or int(frames_per_video) <= 0

        if use_all_frames:
            frame_indices = sorted({int(idx) for idx, _ in scored_candidates})
        else:
            target_frames = int(frames_per_video)
            frame_indices = _select_diverse_top_indices(scored_candidates, target_frames, total_frames)
            if not frame_indices:
                frame_indices = [idx for idx, _ in scored_candidates[:target_frames]]

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            cached_faces = face_cache.get(int(frame_idx))
            face, crop_box, landmarks = detect_and_crop_face(frame, *detector_pack, faces=cached_faces)
            if face is None:
                continue

            if face.ndim != 3 or face.shape[-1] < 3:
                continue
            face_rgb = face[:, :, :3]

            if crop_box is not None and landmarks is not None:
                landmark_mask = normalize_and_create_mask(landmarks, crop_box)
            else:
                landmark_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

            combined_img = np.concatenate([face_rgb, landmark_mask[..., np.newaxis]], axis=-1).astype(np.float32)

            if first_run:
                print(f"\nCombined shape: {combined_img.shape}")
                first_run = False

            label_idx = _map_video_frame_to_sequence_index(frame_idx, total_frames, sequence_length)
            frame_labels = row.get("frame_labels", [])
            if label_idx >= len(frame_labels):
                continue
            label_name = frame_labels[label_idx]
            label = label_map.get(label_name)
            if label is None:
                continue

            sample_debug = {
                "video": video_name,
                "frame_idx": int(frame_idx),
                "csv_index": int(label_idx),
                "label_name": label_name,
                "crop_box": crop_box,
                "landmarks": landmarks,
            }

            if use_disk_backed_storage:
                if sample_count >= max_samples:
                    raise RuntimeError(
                        f"Disk-backed storage overflow for '{disk_output_prefix}': "
                        f"sample_count={sample_count}, max_samples={max_samples}."
                    )
                X_store[sample_count] = _convert_sample_for_storage(combined_img, storage_dtype)
                y_store[sample_count] = label
                sample_count += 1
            else:
                X.append(combined_img)
                y.append(label)
            debugs.append(sample_debug)

        cap.release()

        maybe_save_progress(row_idx + 1)

    if use_disk_backed_storage:
        X_store.flush()
        y_store.flush()
        del X_store
        del y_store

        _save_disk_store_metadata(
            disk_output_dir,
            disk_output_prefix,
            sample_count=sample_count,
            max_samples=max_samples,
            sample_shape=(IMG_HEIGHT, IMG_WIDTH, 4),
            sample_dtype=storage_dtype,
            label_dtype=label_dtype,
        )

        with open(storage_paths["debugs"], "wb") as f:
            pickle.dump(debugs, f, protocol=pickle.HIGHEST_PROTOCOL)

        X_disk = np.load(storage_paths["X"], mmap_mode="r")[:sample_count]
        y_disk = np.load(storage_paths["y"], mmap_mode="r")[:sample_count]
        return X_disk, y_disk, debugs

    return np.array(X), np.array(y), debugs


def process_image_directory(
        directory,
        label_map,
        checkpoint_dir=None,
        checkpoint_prefix=None,
        save_checkpoint_every=100,
        resume_from_checkpoint=True,
):
    if resume_from_checkpoint and checkpoint_dir and checkpoint_prefix:
        start_index, X, y, debugs = _load_iteration_checkpoint(checkpoint_dir, checkpoint_prefix)
    else:
        start_index, X, y, debugs = 0, [], [], []

    detector_pack = get_dlib_detector_predictor()

    first_run = True

    entries = []
    for label in sorted(os.listdir(directory)):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue
        if label not in label_map:
            continue
        for image_file in sorted(os.listdir(label_dir)):
            entries.append((label, os.path.join(label_dir, image_file)))

    for sample_idx, (label, image_path) in enumerate(tqdm(entries, desc="Processing images", unit="image")):
        if sample_idx < start_index:
            continue

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is not None:
            face, crop_box, landmarks = detect_and_crop_face(image, *detector_pack)

            if face is not None and crop_box is not None and landmarks is not None:
                if face.ndim != 3 or face.shape[-1] < 3:
                    continue
                face_rgb = face[:, :, :3]

                landmark_mask = normalize_and_create_mask(landmarks, crop_box)
                combined_img = np.concatenate([face_rgb, landmark_mask[..., np.newaxis]], axis=-1).astype(np.float32)

                if first_run:
                    print(f"\nCombined shape: {combined_img.shape}")
                    first_run = False

                X.append(combined_img)
                y.append(label_map[label])
                debugs.append({"crop_box": crop_box, "landmarks": landmarks, "image_path": image_path})

        if checkpoint_dir and checkpoint_prefix and ((sample_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
            _save_iteration_checkpoint(
                checkpoint_dir,
                checkpoint_prefix,
                sample_idx + 1,
                X,
                y,
                debugs,
            )

    return np.array(X), np.array(y), debugs


def _try_detect_face_at(cap, frame_idx, detector_pack):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret or np.var(frame) < 10.0:
        return None
    face, crop_box, _ = detect_and_crop_face(frame, *detector_pack)
    if face is None or face.ndim != 3 or face.shape[-1] < 3 or crop_box is None:
        return None
    return face[:, :, :3].astype(np.float32)


def _read_raw_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret or np.var(frame) < 10.0:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    return (resized / 255.0).astype(np.float32)


_vit_model = None
_vit_processor = None


def _get_vit():
    global _vit_model, _vit_processor
    if _vit_model is None:
        import torch
        from transformers import ViTModel, ViTImageProcessor

        _vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        _vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        _vit_model.eval()
    return _vit_model, _vit_processor


def _transformer_select_frames(cap, candidate_indices, num_select):
    import torch

    frames_rgb = []
    valid_indices = []
    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)
        valid_indices.append(int(idx))

    if len(valid_indices) <= num_select:
        return valid_indices

    model, processor = _get_vit()
    inputs = processor(images=frames_rgb, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    cls_features = out.last_hidden_state[:, 0, :]

    sim = torch.matmul(cls_features, cls_features.T)
    scores = sim.mean(dim=1).numpy()

    top_k = np.argsort(scores)[-num_select:]
    top_k_sorted = sorted(top_k)
    return [valid_indices[i] for i in top_k_sorted]


def _sample_faces_from_video(
        video_path, detector_pack, use_transformer_selection=False, use_random_selection=False,
        use_manual_selection=False
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return None, None

        safe_start = min(5, total_frames - 1) if total_frames > 10 else 0
        safe_end = max(safe_start, total_frames - 2)

        if use_manual_selection:
            from dataset.manual_frame_selector import manual_select_frames

            if use_transformer_selection:
                auto_fill_method = "transformer"
            elif use_random_selection:
                auto_fill_method = "random"
            else:
                auto_fill_method = "uniform"

            manual_indices = manual_select_frames(video_path, NUM_SAMPLE_FRAMES, auto_fill_method=auto_fill_method)
            if manual_indices is not None and len(manual_indices) == NUM_SAMPLE_FRAMES:
                sample_indices = manual_indices
            else:
                already_selected = manual_indices if manual_indices else []
                remaining = NUM_SAMPLE_FRAMES - len(already_selected)
                if use_transformer_selection and remaining > 0:
                    n_candidates = min(total_frames, max(remaining * 6, 30))
                    candidate_indices = [
                        int(i)
                        for i in np.linspace(safe_start, safe_end, n_candidates, dtype=int)
                        if int(i) not in already_selected
                    ]
                    auto_indices = _transformer_select_frames(cap, candidate_indices, remaining)
                elif use_random_selection and remaining > 0:
                    available = [i for i in range(total_frames) if i not in already_selected]
                    auto_indices = sorted(
                        np.random.choice(available, size=min(remaining, len(available)), replace=False).tolist()
                    )
                else:
                    available = [i for i in range(total_frames) if i not in already_selected]
                    auto_indices = np.linspace(safe_start, max(safe_start, len(available) - 1), remaining,
                                               dtype=int).tolist()
                    auto_indices = [available[i] for i in auto_indices]
                sample_indices = sorted(list(already_selected) + auto_indices)[:NUM_SAMPLE_FRAMES]
        elif use_transformer_selection:
            n_candidates = min(total_frames, max(NUM_SAMPLE_FRAMES * 6, 30))
            candidate_indices = np.linspace(safe_start, safe_end, n_candidates, dtype=int).tolist()
            sample_indices = _transformer_select_frames(cap, candidate_indices, NUM_SAMPLE_FRAMES)
            if len(sample_indices) < NUM_SAMPLE_FRAMES:
                sample_indices = np.linspace(safe_start, safe_end, NUM_SAMPLE_FRAMES, dtype=int).tolist()
        elif use_random_selection:
            sample_indices = sorted(
                np.random.choice(
                    range(safe_start, total_frames), size=min(NUM_SAMPLE_FRAMES, total_frames - safe_start),
                    replace=False
                ).tolist()
            )
        else:
            sample_indices = np.linspace(safe_start, safe_end, NUM_SAMPLE_FRAMES, dtype=int).tolist()

        sampled_faces = [_try_detect_face_at(cap, idx, detector_pack) for idx in sample_indices]

        if all(f is None for f in sampled_faces):
            sampled_faces = [_read_raw_frame(cap, idx) for idx in sample_indices]

        fallback = None
        for f in sampled_faces:
            if f is not None:
                fallback = f
                break

        if fallback is None:
            return None, None

        sampled_faces = [f if f is not None else fallback for f in sampled_faces]
        return sample_indices, sampled_faces
    finally:
        cap.release()


def _to_grayscale_rgb(face_rgb):
    gray = np.dot(face_rgb[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    gray = np.clip(gray, 0.0, 1.0).astype(np.float32)
    return np.repeat(gray[..., np.newaxis], 3, axis=-1)


def _compose_temporal_chromatic(sampled_faces, include_preview=False):
    n_frames = len(sampled_faces)
    center_idx = n_frames // 2
    tints, alphas = _get_temporal_tints_alphas(n_frames)

    center_face = sampled_faces[center_idx]
    if center_face is None:
        for offset in range(1, n_frames):
            for candidate in [center_idx - offset, center_idx + offset]:
                if 0 <= candidate < n_frames and sampled_faces[candidate] is not None:
                    center_face = sampled_faces[candidate]
                    break
            if center_face is not None:
                break
    if center_face is None:
        return None

    base_gray = _to_grayscale_rgb(center_face)
    result = base_gray.copy()

    diff_layers = None
    if include_preview:
        diff_layers = [None] * n_frames
        diff_layers[center_idx] = base_gray.copy()

    for i in range(n_frames):
        if tints[i] is None:
            continue
        frame = sampled_faces[i]
        if frame is None:
            continue

        color = tints[i]
        alpha = alphas[i]

        diff = np.abs(frame - center_face)
        diff_magnitude = np.mean(diff, axis=-1, keepdims=True)
        colored_diff = diff_magnitude * color.reshape(1, 1, 3)

        if include_preview:
            diff_layers[i] = np.clip(alpha * colored_diff, 0.0, 1.0)

        result = result + alpha * colored_diff

    result = np.clip(result, 0.0, 1.0).astype(np.float32)

    if include_preview:
        return result, base_gray, diff_layers
    return result


def temporal_encoding_preview_from_video(video_path):
    detector_pack = get_dlib_detector_predictor()
    sample_indices, sampled_faces = _sample_faces_from_video(video_path, detector_pack)
    if sample_indices is None:
        raise RuntimeError(f"Cannot read frames from video: {video_path}")

    pack = _compose_temporal_chromatic(sampled_faces, include_preview=True)
    if pack is None:
        raise RuntimeError("No detectable faces in sampled frames.")

    result, base_gray, diff_layers = pack
    return {
        "encoded_frame": result,
        "base_gray": base_gray,
        "diff_layers": diff_layers,
        "sample_indices": sample_indices,
        "sampled_faces": sampled_faces,
    }


def process_video_temporal_encoding(
        df,
        video_dir,
        filename_col,
        label_map,
        checkpoint_dir=None,
        checkpoint_prefix=None,
        save_checkpoint_every=1,
        resume_from_checkpoint=True,
        use_transformer_selection=False,
        use_random_selection=False,
        use_manual_selection=False,
):
    if resume_from_checkpoint and checkpoint_dir and checkpoint_prefix:
        start_index, X, y, debugs = _load_iteration_checkpoint(checkpoint_dir, checkpoint_prefix)
    else:
        start_index, X, y, debugs = 0, [], [], []

    detector_pack = get_dlib_detector_predictor()

    for row_idx, (_, row) in enumerate(
            tqdm(df.iterrows(), desc="Processing videos (temporal encoding)", total=len(df), unit="video")
    ):
        if row_idx < start_index:
            continue

        video_name = row[filename_col]
        video_path = os.path.join(video_dir, video_name)
        label = row["label"] if label_map is None else label_map.get(row["label"], 0)

        sample_indices, sampled_faces = _sample_faces_from_video(
            video_path, detector_pack, use_transformer_selection, use_random_selection, use_manual_selection
        )
        if sample_indices is None:
            if checkpoint_dir and checkpoint_prefix and ((row_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
                _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, row_idx + 1, X, y, debugs)
            continue

        result = _compose_temporal_chromatic(sampled_faces)
        if result is None:
            if checkpoint_dir and checkpoint_prefix and ((row_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
                _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, row_idx + 1, X, y, debugs)
            continue

        X.append(result)
        y.append(label)

        debug_entry = {
            "video": video_name,
            "sample_indices": sample_indices,
        }
        for pcol in ("participant", "id_examined", "_participant"):
            if pcol in row.index:
                debug_entry["participant"] = str(row[pcol])
                break
        debugs.append(debug_entry)

        if checkpoint_dir and checkpoint_prefix and ((row_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
            _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, row_idx + 1, X, y, debugs)

    return np.array(X), np.array(y), debugs

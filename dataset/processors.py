import cv2
import numpy as np
import os
import pickle
import re
from tqdm import tqdm

from utils.image import get_dlib_detector_predictor, detect_and_crop_face

IMG_HEIGHT = 48
IMG_WIDTH = 48


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


def _build_candidate_indices(total_frames, max_candidates):
    if max_candidates is None:
        return np.arange(total_frames, dtype=int)
    if total_frames <= max_candidates:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, max_candidates, dtype=int)


def _center_window_indices(total_frames, target_count, center_fraction=0.5, window_fraction=0.45):
    if total_frames <= 0:
        return []

    center_idx = int(round((total_frames - 1) * center_fraction))
    half_window = max(1, int(round(total_frames * window_fraction * 0.5)))
    start = max(0, center_idx - half_window)
    end = min(total_frames - 1, center_idx + half_window)

    available = max(1, end - start + 1)
    if available >= target_count:
        return np.linspace(start, end, target_count, dtype=int).tolist()

    return np.linspace(0, total_frames - 1, target_count, dtype=int).tolist()


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
    return len(faces) == 1


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


def _checkpoint_path(checkpoint_dir, checkpoint_prefix, iteration):
    return os.path.join(checkpoint_dir, f"{checkpoint_prefix}_iter{int(iteration):06d}.tmp")


def _find_latest_checkpoint(checkpoint_dir, checkpoint_prefix):
    if not checkpoint_dir or not checkpoint_prefix:
        return None
    if not os.path.isdir(checkpoint_dir):
        return None

    pattern = re.compile(rf"^{re.escape(checkpoint_prefix)}_iter(\d+)\.tmp$")
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
):
    if not checkpoint_dir or not checkpoint_prefix:
        return None

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, checkpoint_prefix, next_index)
    payload = {
        "next_index": int(next_index),
        "X": X,
        "y": y,
        "debugs": debugs,
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    pattern = re.compile(rf"^{re.escape(checkpoint_prefix)}_iter(\d+)\.tmp$")
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


def _load_iteration_checkpoint(checkpoint_dir, checkpoint_prefix):
    latest_path = _find_latest_checkpoint(checkpoint_dir, checkpoint_prefix)
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


def _cleanup_iteration_checkpoints(checkpoint_dir, checkpoint_prefix):
    if not checkpoint_dir or not checkpoint_prefix:
        return
    if not os.path.isdir(checkpoint_dir):
        return

    pattern = re.compile(rf"^{re.escape(checkpoint_prefix)}_iter(\d+)\.tmp$")
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
):
    if resume_from_checkpoint and checkpoint_dir and checkpoint_prefix:
        start_index, X, y, debugs = _load_iteration_checkpoint(checkpoint_dir, checkpoint_prefix)
    else:
        start_index, X, y, debugs = 0, [], [], []

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
            if checkpoint_dir and checkpoint_prefix and ((row_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
                _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, row_idx + 1, X, y, debugs)
            continue

        arousal_seq = list(arousal_seq)
        valence_seq = list(valence_seq)
        sequence_length = min(len(arousal_seq), len(valence_seq))
        if sequence_length <= 0:
            if checkpoint_dir and checkpoint_prefix and ((row_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
                _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, row_idx + 1, X, y, debugs)
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if checkpoint_dir and checkpoint_prefix and ((row_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
                _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, row_idx + 1, X, y, debugs)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        candidate_indices = np.arange(total_frames, dtype=int)
        scored_candidates = []
        prev_gray = None

        for frame_idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            has_face = _has_face_detected(frame, detector)
            if not has_face:
                continue

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

            face, crop_box, landmarks = detect_and_crop_face(frame, *detector_pack)
            if face is None:
                continue

            face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
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

            X.append(combined_img)
            y.append(label)
            debugs.append(sample_debug)

        cap.release()

        if checkpoint_dir and checkpoint_prefix and ((row_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
            checkpoint_path = _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, row_idx + 1, X, y, debugs)
            if checkpoint_path is not None:
                print(f"Checkpoint saved: {checkpoint_path}")

    if checkpoint_dir and checkpoint_prefix:
        _cleanup_iteration_checkpoints(checkpoint_dir, checkpoint_prefix)

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
                face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
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
            checkpoint_path = _save_iteration_checkpoint(
                checkpoint_dir,
                checkpoint_prefix,
                sample_idx + 1,
                X,
                y,
                debugs,
            )
            if checkpoint_path is not None:
                print(f"Checkpoint saved: {checkpoint_path}")

    if checkpoint_dir and checkpoint_prefix:
        _cleanup_iteration_checkpoints(checkpoint_dir, checkpoint_prefix)

    return np.array(X), np.array(y), debugs


def process_video_sequences(
    df,
    video_dir,
    filename_col,
    label_map,
    sequence_length=8,
    max_candidates=90,
    checkpoint_dir=None,
    checkpoint_prefix=None,
    save_checkpoint_every=1,
    resume_from_checkpoint=True,
):
    if resume_from_checkpoint and checkpoint_dir and checkpoint_prefix:
        start_index, X, y, debugs = _load_iteration_checkpoint(checkpoint_dir, checkpoint_prefix)
    else:
        start_index, X, y, debugs = 0, [], [], []

    detector_pack = get_dlib_detector_predictor()

    first_run = True

    for row_idx, (_, row) in enumerate(tqdm(df.iterrows(), desc="Processing videos", total=len(df), unit="video")):
        if row_idx < start_index:
            continue

        video_name = row[filename_col]
        video_path = os.path.join(video_dir, video_name)
        label = row["label"] if label_map is None else label_map.get(row["label"], 0)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if checkpoint_dir and checkpoint_prefix and ((row_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
                _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, row_idx + 1, X, y, debugs)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            if checkpoint_dir and checkpoint_prefix and ((row_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
                _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, row_idx + 1, X, y, debugs)
            continue

        candidate_indices = _build_candidate_indices(total_frames, max_candidates=max_candidates)
        center_idx = (total_frames - 1) / 2.0
        sigma = max(3.0, total_frames * 0.22)

        detector, _ = detector_pack
        quality_values = []
        temporal_values = []
        face_detection_values = []
        prev_gray = None

        for frame_idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            quality, prev_gray = _frame_quality_score(frame, prev_gray)
            temporal_weight = float(np.exp(-((float(frame_idx) - center_idx) ** 2) / (2.0 * (sigma**2))))
            has_face = _has_face_detected(frame, detector)

            quality_values.append((int(frame_idx), float(quality)))
            temporal_values.append((int(frame_idx), temporal_weight))
            face_detection_values.append((int(frame_idx), 1.0 if has_face else 0.0))

        if not quality_values:
            cap.release()
            if checkpoint_dir and checkpoint_prefix and ((row_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
                _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, row_idx + 1, X, y, debugs)
            continue

        q_scores = np.array([q for _, q in quality_values], dtype=np.float32)
        q_min = float(np.min(q_scores))
        q_max = float(np.max(q_scores))
        q_range = max(1e-6, q_max - q_min)

        temporal_lookup = {idx: w for idx, w in temporal_values}
        face_detection_lookup = {idx: w for idx, w in face_detection_values}
        scored_candidates = []
        for frame_idx, quality in quality_values:
            quality_norm = float((quality - q_min) / q_range)
            center_w = float(temporal_lookup.get(frame_idx, 0.0))
            face_w = float(face_detection_lookup.get(frame_idx, 0.0))
            # Scoring: 35% quality, 40% temporal position, 25% face detection
            fused_score = (0.35 * quality_norm) + (0.40 * center_w) + (0.25 * face_w)
            scored_candidates.append((int(frame_idx), fused_score))

        frame_indices = _select_diverse_top_indices(scored_candidates, sequence_length, total_frames)
        if len(frame_indices) < sequence_length:
            frame_indices = _center_window_indices(total_frames, sequence_length)

        sequence_frames = []
        sequence_debug = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            face, crop_box, landmarks = detect_and_crop_face(frame, *detector_pack)
            if face is None:
                continue

            face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
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

            sequence_frames.append(combined_img)
            sequence_debug.append(
                {
                    "frame_idx": int(frame_idx),
                    "crop_box": crop_box,
                    "landmarks": landmarks,
                }
            )

        cap.release()

        if not sequence_frames:
            if checkpoint_dir and checkpoint_prefix and ((row_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
                _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, row_idx + 1, X, y, debugs)
            continue

        while len(sequence_frames) < sequence_length:
            sequence_frames.append(sequence_frames[-1].copy())
            sequence_debug.append(sequence_debug[-1].copy())

        if len(sequence_frames) > sequence_length:
            sequence_frames = sequence_frames[:sequence_length]
            sequence_debug = sequence_debug[:sequence_length]

        X.append(np.stack(sequence_frames, axis=0))
        y.append(label)
        debugs.append(
            {
                "video": video_name,
                "frames": [item["frame_idx"] for item in sequence_debug],
                "center_frame": int(sequence_debug[len(sequence_debug) // 2]["frame_idx"]),
                "crop_box": sequence_debug[len(sequence_debug) // 2].get("crop_box"),
                "landmarks": sequence_debug[len(sequence_debug) // 2].get("landmarks"),
            }
        )

        if checkpoint_dir and checkpoint_prefix and ((row_idx + 1) % max(1, int(save_checkpoint_every)) == 0):
            checkpoint_path = _save_iteration_checkpoint(checkpoint_dir, checkpoint_prefix, row_idx + 1, X, y, debugs)
            if checkpoint_path is not None:
                print(f"Checkpoint saved: {checkpoint_path}")

    if checkpoint_dir and checkpoint_prefix:
        _cleanup_iteration_checkpoints(checkpoint_dir, checkpoint_prefix)

    return np.array(X), np.array(y), debugs

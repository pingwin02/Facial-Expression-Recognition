import cv2
import numpy as np
import os
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


def process_image_directory(directory, label_map):
    X, y, debugs = [], [], []
    detector_pack = get_dlib_detector_predictor()

    first_run = True

    for label in tqdm(os.listdir(directory), desc="Processing labels", unit="label"):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for image_file in tqdm(os.listdir(label_dir), desc=f"Processing images for label {label}", unit="image"):
                image_path = os.path.join(label_dir, image_file)

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                if image is not None:
                    face, crop_box, landmarks = detect_and_crop_face(image, *detector_pack)

                    if face is not None and crop_box is not None and landmarks is not None:
                        face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
                        face_channel = np.squeeze(face)
                        if face_channel.ndim != 2:
                            continue

                        landmark_mask = normalize_and_create_mask(landmarks, crop_box)
                        combined_img = np.stack((face_channel, landmark_mask), axis=-1).astype(np.float32)

                        if first_run:
                            print(f"\nCombined shape: {combined_img.shape}")
                            first_run = False

                        X.append(combined_img)
                        y.append(label_map[label])
                        debugs.append({"crop_box": crop_box, "landmarks": landmarks})

    return np.array(X), np.array(y), debugs


def process_video_data(df, video_dir, filename_col, label_map, frames_per_video=12, max_candidates=60):
    X, y, debugs = [], [], []
    detector_pack = get_dlib_detector_predictor()

    first_run = True

    for _, row in tqdm(df.iterrows(), desc="Processing videos", total=len(df), unit="video"):
        video_path = os.path.join(video_dir, row[filename_col])
        label = row["label"] if label_map is None else label_map.get(row["label"], 0)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        candidate_indices = _build_candidate_indices(total_frames, max_candidates=max_candidates)
        detector, _ = detector_pack
        scored_candidates = []
        prev_gray = None

        for frame_idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            score, prev_gray = _frame_quality_score(frame, prev_gray)
            has_face = _has_face_detected(frame, detector)
            # Boost score by 50% if face detected to prioritize frames with faces
            if has_face:
                score = score * 1.5
            scored_candidates.append((int(frame_idx), score))

        frame_indices = _select_diverse_top_indices(scored_candidates, frames_per_video, total_frames)
        if not frame_indices:
            frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int).tolist()

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            face, crop_box, landmarks = detect_and_crop_face(frame, *detector_pack)

            if face is not None and crop_box is not None and landmarks is not None:
                face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
                face_channel = np.squeeze(face)
                if face_channel.ndim != 2:
                    continue

                landmark_mask = normalize_and_create_mask(landmarks, crop_box)
                combined_img = np.stack((face_channel, landmark_mask), axis=-1).astype(np.float32)

                if first_run:
                    print(f"\nCombined shape: {combined_img.shape}")
                    first_run = False

                X.append(combined_img)
                y.append(label)

                debugs.append(
                    {
                        "video": row[filename_col],
                        "frame_idx": int(frame_idx),
                        "crop_box": crop_box,
                        "landmarks": landmarks,
                    }
                )

        cap.release()

    return np.array(X), np.array(y), debugs


def process_video_sequences(
    df,
    video_dir,
    filename_col,
    label_map,
    sequence_length=8,
    max_candidates=90,
):
    X, y, debugs = [], [], []
    detector_pack = get_dlib_detector_predictor()

    first_run = True

    for _, row in tqdm(df.iterrows(), desc="Processing videos", total=len(df), unit="video"):
        video_name = row[filename_col]
        video_path = os.path.join(video_dir, video_name)
        label = row["label"] if label_map is None else label_map.get(row["label"], 0)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
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
            face_channel = np.squeeze(face)
            if face_channel.ndim != 2:
                continue

            if crop_box is not None and landmarks is not None:
                landmark_mask = normalize_and_create_mask(landmarks, crop_box)
            else:
                landmark_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

            combined_img = np.stack((face_channel, landmark_mask), axis=-1).astype(np.float32)

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

    return np.array(X), np.array(y), debugs

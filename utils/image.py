import bz2
import os
import urllib.request

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 1

ALIGN_EYE_X_LEFT = 0.32
ALIGN_EYE_X_RIGHT = 0.68
ALIGN_EYE_Y = 0.35
ALIGN_MOUTH_X = 0.50
ALIGN_MOUTH_Y = 0.75

MEAN_FACE_5PT_BASE = np.float32(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ]
)
MEAN_FACE_5PT_SIZE = 112.0

DETECTOR_UPSAMPLE_PRIMARY = 1
DETECTOR_UPSAMPLE_FALLBACK = 2

HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE_RATIO = 0.2
HAAR_MIN_SIZE_PX = 24

CENTER_CROP_RATIO = 0.8

_detector = None
_predictor = None
_haar_detector = None


def get_dlib_detector_predictor():
    global _detector, _predictor
    if _detector is None or _predictor is None:
        DLIB_LANDMARK_MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"
        DLIB_DOWNLOAD_DIR = "downloads"
        DLIB_FULL_PATH = os.path.join(DLIB_DOWNLOAD_DIR, DLIB_LANDMARK_MODEL_FILENAME)
        if not os.path.exists(DLIB_FULL_PATH):
            print(f"Dlib weights not found: {DLIB_FULL_PATH}")
            print("Downloading shape_predictor_68_face_landmarks.dat...")
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            bz2_path = DLIB_FULL_PATH + ".bz2"
            urllib.request.urlretrieve(url, bz2_path)
            print("Download complete. Extracting...")
            with bz2.open(bz2_path, "rb") as f_in, open(DLIB_FULL_PATH, "wb") as f_out:
                f_out.write(f_in.read())
            os.remove(bz2_path)
            print("Extraction complete.")
        _detector = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(DLIB_FULL_PATH)
    return _detector, _predictor


def _get_haar_detector():
    global _haar_detector
    if _haar_detector is None:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        _haar_detector = cv2.CascadeClassifier(cascade_path)
    return _haar_detector


def _detect_faces(gray, detector):
    faces = detector(gray, DETECTOR_UPSAMPLE_PRIMARY)
    if len(faces) == 0 and DETECTOR_UPSAMPLE_FALLBACK > DETECTOR_UPSAMPLE_PRIMARY:
        faces = detector(gray, DETECTOR_UPSAMPLE_FALLBACK)
    if len(faces) == 0:
        faces = _detect_faces_haar(gray)
    return faces


def _detect_faces_haar(gray):
    haar = _get_haar_detector()
    if haar is None or haar.empty():
        return []

    min_side = int(min(gray.shape[:2]) * HAAR_MIN_SIZE_RATIO)
    min_side = max(HAAR_MIN_SIZE_PX, min_side)
    boxes = haar.detectMultiScale(
        gray,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=(min_side, min_side),
    )
    return [dlib.rectangle(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in boxes]


def _center_crop(frame, ratio=CENTER_CROP_RATIO):
    h, w = frame.shape[:2]
    side = int(min(h, w) * float(ratio))
    if side < 2:
        return frame, None
    x1 = max(0, (w - side) // 2)
    y1 = max(0, (h - side) // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    if x2 <= x1 or y2 <= y1:
        return frame, None
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def _landmark_points_3(landmarks):
    if landmarks is None:
        return None
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.ndim != 2 or lm.shape[1] != 2 or lm.shape[0] < 60:
        return None
    left_eye = lm[36:42].mean(axis=0)
    right_eye = lm[42:48].mean(axis=0)
    mouth = lm[48:60].mean(axis=0)
    return np.stack([left_eye, right_eye, mouth], axis=0)


def _landmark_points_5(landmarks):
    if landmarks is None:
        return None
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.ndim != 2 or lm.shape[1] != 2 or lm.shape[0] < 55:
        return None
    left_eye = lm[36:42].mean(axis=0)
    right_eye = lm[42:48].mean(axis=0)
    nose = lm[30]
    mouth_left = lm[48]
    mouth_right = lm[54]
    return np.stack([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0)


def _alignment_targets_3():
    return np.float32(
        [
            [ALIGN_EYE_X_LEFT * IMG_WIDTH, ALIGN_EYE_Y * IMG_HEIGHT],
            [ALIGN_EYE_X_RIGHT * IMG_WIDTH, ALIGN_EYE_Y * IMG_HEIGHT],
            [ALIGN_MOUTH_X * IMG_WIDTH, ALIGN_MOUTH_Y * IMG_HEIGHT],
        ]
    )


def _alignment_targets_5():
    scale = np.array([IMG_WIDTH / MEAN_FACE_5PT_SIZE, IMG_HEIGHT / MEAN_FACE_5PT_SIZE], dtype=np.float32)
    return MEAN_FACE_5PT_BASE * scale


def _compute_alignment_transform(landmarks):
    src = _landmark_points_5(landmarks)
    if src is not None and np.isfinite(src).all():
        if src[0][0] < src[1][0] and src[3][0] < src[4][0]:
            eye_dist = float(np.linalg.norm(src[1] - src[0]))
            if eye_dist >= 5.0:
                eye_y = float((src[0][1] + src[1][1]) * 0.5)
                nose_y = float(src[2][1])
                mouth_y = float((src[3][1] + src[4][1]) * 0.5)
                if eye_y < nose_y < mouth_y:
                    dst = _alignment_targets_5()
                    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
                    if matrix is not None and np.isfinite(matrix).all():
                        return matrix.astype(np.float32)

    src3 = _landmark_points_3(landmarks)
    if src3 is None or not np.isfinite(src3).all():
        return None
    if src3[0][0] >= src3[1][0]:
        return None
    eye_dist = float(np.linalg.norm(src3[1] - src3[0]))
    if eye_dist < 5.0:
        return None
    dst3 = _alignment_targets_3()
    matrix = cv2.getAffineTransform(src3, dst3)
    if matrix is None or not np.isfinite(matrix).all():
        return None
    return matrix.astype(np.float32)


def _apply_affine(points, matrix):
    pts = np.asarray(points, dtype=np.float32)
    return (pts @ matrix[:, :2].T) + matrix[:, 2]


def _crop_box_from_landmarks(landmarks, frame_shape, margin_x=0.25, margin_y=0.35):
    if landmarks is None:
        return None
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.ndim != 2 or lm.shape[0] == 0:
        return None

    h, w = frame_shape[:2]
    x1 = float(lm[:, 0].min())
    y1 = float(lm[:, 1].min())
    x2 = float(lm[:, 0].max())
    y2 = float(lm[:, 1].max())
    if x2 <= x1 or y2 <= y1:
        return None

    pad_x = (x2 - x1) * float(margin_x)
    pad_y = (y2 - y1) * float(margin_y)

    x1 = max(0, int(np.floor(x1 - pad_x)))
    y1 = max(0, int(np.floor(y1 - pad_y)))
    x2 = min(w, int(np.ceil(x2 + pad_x)))
    y2 = min(h, int(np.ceil(y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def detect_and_crop_face(frame, detector, predictor, faces=None, align=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if faces is None or len(faces) == 0:
        faces = _detect_faces(gray, detector)

    target_img = frame
    crop_box = None
    landmarks = None

    if len(faces) >= 1:
        face_rect = max(faces, key=lambda rect: max(0, rect.width()) * max(0, rect.height()))
        x1, y1 = face_rect.left(), face_rect.top()
        x2, y2 = face_rect.right(), face_rect.bottom()

        margin = int(0.2 * (x2 - x1))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)

        try:
            shape = predictor(gray, face_rect)
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]
        except Exception:
            landmarks = None

        if align and landmarks is not None:
            matrix = _compute_alignment_transform(landmarks)
            if matrix is not None:
                aligned = cv2.warpAffine(
                    frame,
                    matrix,
                    (IMG_WIDTH, IMG_HEIGHT),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                aligned_landmarks = _apply_affine(landmarks, matrix)
                resized = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                normalized = resized.astype(np.float32) / 255.0
                return normalized, (0, 0, IMG_WIDTH, IMG_HEIGHT), aligned_landmarks.tolist()

        crop_box = _crop_box_from_landmarks(landmarks, frame.shape)
        if crop_box is None:
            target_img = frame[y1:y2, x1:x2]
            crop_box = (x1, y1, x2, y2)
        else:
            x1, y1, x2, y2 = crop_box
            target_img = frame[y1:y2, x1:x2]
    else:
        target_img, crop_box = _center_crop(frame)

    resized = cv2.resize(target_img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = resized.astype(np.float32) / 255.0
    processed_frame = normalized

    return processed_frame, crop_box, landmarks


def save_sample_frames(
        frames,
        preds,
        labels,
        debugs,
        output_dir,
        model_name,
        dataset_name,
        accuracy,
        filename,
        highlight_correctness_bg=False,
        cols=5,
        group_size=None,
        show_group_separator=False,
):
    def _pretty_class_name(value):
        return str(value).replace("_", " ")

    def _format_short_field(label, value, max_len=20):
        if value is None:
            return None
        text = str(value)
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        return f"{label}: {text}"

    class_map = {}
    for debug in debugs:
        if debug and "class_map" in debug and isinstance(debug["class_map"], dict):
            class_map.update(debug["class_map"])
    if not class_map:
        for lbl in list(labels) + list(preds):
            if isinstance(lbl, str):
                class_map[lbl] = lbl
        if not class_map:
            unique_vals = list(sorted(set(list(labels) + list(preds))))
            for v in unique_vals:
                class_map[v] = str(v)

    n = len(frames)
    if n == 0:
        print("No frames to save.")
        return
    cols = max(1, int(cols))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))

    title_parts = []
    if model_name:
        title_parts.append(str(model_name))
    if dataset_name:
        title_parts.append(str(dataset_name))
    title = " - ".join(title_parts) if title_parts else None
    if title:
        fig.suptitle(title, fontsize=13, y=0.995)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[a] for a in axes])

    for idx, (frame, pred, label, debug) in enumerate(zip(frames, preds, labels, debugs)):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        img = np.squeeze(frame)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        elif img.ndim == 3 and img.shape[-1] == 1:
            ax.imshow(img[:, :, 0], cmap="gray")
        elif img.ndim == 3 and img.shape[-1] == 2:
            face_channel = img[:, :, 0]
            landmark_channel = img[:, :, 1]
            preview = np.stack([face_channel, face_channel, face_channel], axis=-1)
            preview[:, :, 1] = np.maximum(preview[:, :, 1], landmark_channel)
            ax.imshow(preview)
        elif img.ndim == 3 and img.shape[-1] >= 3:
            ax.imshow(img[:, :, :3])
        else:
            ax.imshow(img, cmap="gray")

        if debug:
            lm = debug.get("landmarks")
            crop = debug.get("crop_box") or debug.get("crop")
            if lm is not None and crop is not None:
                try:
                    lm_arr = np.array(lm)
                    x1, y1, x2, y2 = crop
                    crop_w = max(1.0, float(x2 - x1))
                    crop_h = max(1.0, float(y2 - y1))

                    lm_scaled = np.zeros_like(lm_arr, dtype=np.float32)
                    lm_scaled[:, 0] = (lm_arr[:, 0] - x1) * (IMG_WIDTH / crop_w)
                    lm_scaled[:, 1] = (lm_arr[:, 1] - y1) * (IMG_HEIGHT / crop_h)

                    ax.scatter(lm_scaled[:, 0], lm_scaled[:, 1], c="lime", s=8, alpha=0.8)
                except Exception:
                    pass

        pred_name = _pretty_class_name(class_map.get(pred, str(pred)))
        label_name = _pretty_class_name(class_map.get(label, str(label)))

        top_caption = f"actual: {label_name}"
        bottom_caption = f"predicted: {pred_name}"
        top_caption_color = "#111111"

        if highlight_correctness_bg:
            is_correct = str(pred_name) == str(label_name)
            top_caption = f"✓ {top_caption}" if is_correct else f"✗ {top_caption}"
            top_caption_color = "#2e7d32" if is_correct else "#c62828"

        ax.text(
            0.5,
            1.03,
            top_caption,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=top_caption_color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9),
        )
        if debug and isinstance(debug, dict):
            frame_idx = debug.get("frame_idx")
            frame_total = debug.get("frame_total")
            frame_part = None
            if frame_idx is not None and frame_total is not None:
                frame_part = f"frame: {int(frame_idx)}/{max(0, int(frame_total) - 1)}"
            elif frame_idx is not None:
                frame_part = f"frame: {int(frame_idx)}"

            text_lines = []
            participant_part = _format_short_field("participant", debug.get("participant"))
            if participant_part:
                text_lines.append(participant_part)

            video_part = _format_short_field("video", debug.get("video"))
            if video_part:
                text_lines.append(video_part)

            text_lines.append(bottom_caption)
            if frame_part:
                text_lines.append(frame_part)

            ax.text(
                0.5,
                -0.10,
                "\n".join(text_lines),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
            )
        else:
            ax.text(
                0.5,
                -0.10,
                bottom_caption,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
            )
        ax.axis("off")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("off")

    if show_group_separator and group_size and group_size > 0 and rows > 1 and cols == int(group_size):
        left = axes[0, 0].get_position().x0
        right = axes[0, cols - 1].get_position().x1
        for row_idx in range(1, rows):
            y = axes[row_idx, 0].get_position().y1
            fig.add_artist(
                Line2D(
                    [left, right],
                    [y, y],
                    transform=fig.transFigure,
                    color="#7f8c8d",
                    linewidth=1.2,
                    alpha=0.55,
                )
            )

    top_rect = 0.988 if title else 1.0
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    print(f"Saving sample grid PNG to {os.path.join(output_dir, filename)}")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)

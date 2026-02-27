import bz2
import cv2
import dlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import urllib.request

IMG_HEIGHT = 48
IMG_WIDTH = 48
CHANNELS = 1

_detector = None
_predictor = None


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


def detect_and_crop_face(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    target_img = frame
    crop_box = None
    landmarks = None

    if len(faces) == 1:
        face_rect = faces[0]
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
            pass

        target_img = frame[y1:y2, x1:x2]
        crop_box = (x1, y1, x2, y2)

    resized = cv2.resize(target_img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
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
    if accuracy is not None:
        title_parts.append(f"Accuracy: {accuracy * 100:.2f}%")
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
        if debug and isinstance(debug, dict) and debug.get("video"):
            frame_idx = debug.get("frame_idx")
            frame_total = debug.get("frame_total")
            frame_part = None
            if frame_idx is not None and frame_total is not None:
                frame_part = f"frame: {int(frame_idx)}/{max(0, int(frame_total) - 1)}"
            elif frame_idx is not None:
                frame_part = f"frame: {int(frame_idx)}"

            text_lines = [f"video: {debug['video']}", bottom_caption]
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
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

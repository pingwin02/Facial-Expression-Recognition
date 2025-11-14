import os
import sys
import random
import dlib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

IMG_HEIGHT = 48
IMG_WIDTH = 48
CHANNELS = 1

DLIB_LANDMARK_MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"
DLIB_DOWNLOAD_DIR = "downloads"
DLIB_FULL_PATH = os.path.join(DLIB_DOWNLOAD_DIR, DLIB_LANDMARK_MODEL_FILENAME)


def get_dlib_detector_predictor():
    if not os.path.exists(DLIB_FULL_PATH):
        print(f"Dlib weights not found: {DLIB_FULL_PATH}")
        print("Downloading shape_predictor_68_face_landmarks.dat...")
        import urllib.request

        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        bz2_path = DLIB_FULL_PATH + ".bz2"
        urllib.request.urlretrieve(url, bz2_path)
        print("Download complete. Extracting...")
        import bz2

        with bz2.open(bz2_path, "rb") as f_in, open(DLIB_FULL_PATH, "wb") as f_out:
            f_out.write(f_in.read())
        os.remove(bz2_path)
        print("Extraction complete.")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_FULL_PATH)
    return detector, predictor


def detect_and_crop_face(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 1:
        face_rect = faces[0]
        x1, y1 = face_rect.left(), face_rect.top()
        x2, y2 = face_rect.right(), face_rect.bottom()
        margin = int(0.2 * (x2 - x1))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        cropped_face = gray[y1:y2, x1:x2]
        resized_face = cv2.resize(cropped_face, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
        normalized_face = resized_face.astype(np.float32) / 255.0
        normalized_face = np.expand_dims(normalized_face, axis=-1)
        crop_box = (x1, y1, x2, y2)
        return normalized_face, crop_box
    return None, None


def load_data(input_dir):
    csv_path = os.path.join(input_dir, "devemo", "_clips_info.csv")
    video_dir = os.path.join(input_dir, "devemo")
    df = pd.read_csv(csv_path, sep=";")
    unique_ids = df["id_examined"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_ids)
    split_idx = int(0.8 * len(unique_ids))
    train_ids = unique_ids[:split_idx]
    val_ids = unique_ids[split_idx:]
    train_df = df[df["id_examined"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["id_examined"].isin(val_ids)].reset_index(drop=True)
    train_dataset = FERDataset(train_df, video_dir)
    val_dataset = FERDataset(val_df, video_dir)
    return DataLoaderTF(train_dataset), DataLoaderTF(val_dataset)


class FERDataset:
    def __init__(self, df, video_dir):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.detector, self.predictor = get_dlib_detector_predictor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.video_dir, row["file"])
        label_map = {"neutral": 0, "happiness": 1, "confusion": 2, "surprise": 3, "anger": 4}
        label = label_map.get(row["label"], 0)
        cap = cv2.VideoCapture(video_path)
        debug_info = {"video_path": video_path}
        if not cap.isOpened():
            arr = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)
            debug_info["error"] = "Video not opened"
            debug_info["frame_num"] = None
            debug_info["ret"] = False
            cap.release()
            return arr, label, debug_info
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            arr = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)
            debug_info["error"] = "No frames in video"
            debug_info["frame_num"] = None
            debug_info["ret"] = False
            cap.release()
            return arr, label, debug_info
        max_attempts = 5
        arr = None
        for attempt in range(max_attempts):
            frame_num = random.randint(0, frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            debug_info["frame_num"] = frame_num
            debug_info["ret"] = ret
            if not ret:
                continue
            face_rects = self.detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)
            if len(face_rects) == 1:
                shape = self.predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), face_rects[0])
                landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
                face, crop_box = detect_and_crop_face(frame, self.detector, self.predictor)
                if face is not None:
                    arr = face
                    debug_info["landmarks"] = landmarks
                    debug_info["crop_box"] = crop_box
                    break
        if arr is None:
            arr = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)
            debug_info["error"] = "Frame not loaded or no face detected"
            debug_info["landmarks"] = None
        cap.release()
        return arr, label, debug_info


class DataLoaderTF:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[start : start + self.batch_size]
            images, labels, debugs = [], [], []
            for idx in batch_idx:
                img, label, debug = self.dataset[idx]
                if "error" in debug and debug["error"]:
                    continue
                images.append(img)
                labels.append(label)
                debugs.append(debug)
            if images:
                yield np.stack(images), np.array(labels), debugs


def plot_metrics(train_losses, val_losses, train_acc, val_acc, output_dir):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()
    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()


def save_sample_frames(frames, preds, labels, debugs, output_dir, model_name=None):
    class_map = {0: "neutral", 1: "happiness", 2: "confusion", 3: "surprise", 4: "anger"}
    n = len(frames)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    for idx, (frame, pred, label, debug) in enumerate(zip(frames, preds, labels, debugs)):
        r, c = divmod(idx, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        img = frame.squeeze()
        ax.imshow(img, cmap="gray")
        if (
            "landmarks" in debug
            and debug["landmarks"] is not None
            and "crop_box" in debug
            and debug["crop_box"] is not None
        ):
            lm = np.array(debug["landmarks"])
            x1, y1, x2, y2 = debug["crop_box"]
            lm_scaled = np.zeros_like(lm, dtype=np.float32)
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w > 0 and crop_h > 0:
                lm_scaled[:, 0] = (lm[:, 0] - x1) * (48.0 / crop_w)
                lm_scaled[:, 1] = (lm[:, 1] - y1) * (48.0 / crop_h)
                ax.scatter(lm_scaled[:, 0], lm_scaled[:, 1], c="lime", s=5)
        pred_name = class_map.get(pred, str(pred))
        label_name = class_map.get(label, str(label))
        ax.set_title(f"{label_name}\npred: {pred_name}", fontsize=8)
        ax.axis("off")
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.axis("off")
    plt.tight_layout()
    fname = f"{model_name}_samples.png" if model_name else "samples.png"
    print(f"Saving sample grid PNG: {fname} with {n} frames")
    plt.savefig(os.path.join(output_dir, fname))
    plt.close(fig)

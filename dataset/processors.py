import os

import cv2
import numpy as np
from tqdm import tqdm

from utils.image import get_dlib_detector_predictor, detect_and_crop_face

IMG_HEIGHT = 48
IMG_WIDTH = 48


def process_image_directory(directory, label_map):
    """Iterates through directory structure to load images."""
    X, y, debugs = [], [], []
    detector_pack = get_dlib_detector_predictor()

    for label in tqdm(os.listdir(directory), desc="Processing labels", unit="label"):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for image_file in tqdm(os.listdir(label_dir), desc=f"Processing images for label {label}", unit="image"):
                image_path = os.path.join(label_dir, image_file)
                image = cv2.imread(image_path)

                if image is not None:
                    face, crop_box, landmarks = detect_and_crop_face(image, *detector_pack)
                    if face is not None:
                        face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
                        X.append(face)
                        y.append(label_map[label])
                        debugs.append({"crop_box": crop_box, "landmarks": landmarks})

    return np.array(X), np.array(y), debugs


def process_video_data(df, video_dir, filename_col, label_map, num_frames=10):
    """
    Iterates through video files in DataFrame to extract multiple equidistant frames per video.
    Each extracted frame becomes a separate training sample with the same label.
    """
    X, y, debugs = [], [], []
    detector_pack = get_dlib_detector_predictor()

    for _, row in tqdm(df.iterrows(), desc="Processing videos", total=len(df), unit=" videos"):
        video_path = os.path.join(video_dir, row[filename_col])
        label = label_map.get(row["label"], 0)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        count = min(total_frames, num_frames)
        frame_indices = np.linspace(0, total_frames - 1, count, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            face, crop_box, landmarks = detect_and_crop_face(frame, *detector_pack)

            if face is not None:
                face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))

                if len(face.shape) == 2:
                    face = np.expand_dims(face, axis=-1)

                X.append(face)
                y.append(label)

                debugs.append(
                    {"video": row[filename_col], "landmarks": landmarks, "frame_idx": idx, "crop_box": crop_box}
                )

        cap.release()

    return np.array(X), np.array(y), debugs

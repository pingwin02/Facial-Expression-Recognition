import os

import cv2
import numpy as np
from tqdm import tqdm

from dataset.utils import get_safe_frame
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
            for image_file in tqdm(
                    os.listdir(label_dir), desc=f"Processing images for label {label}", unit="image"
            ):
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


def process_video_data(df, video_dir, filename_col, label_map):
    """Iterates through video files in DataFrame to extract frames and faces."""
    X, y, debugs = [], [], []
    detector_pack = get_dlib_detector_predictor()

    for _, row in tqdm(df.iterrows(), desc="Processing videos", total=len(df), unit=" videos"):
        video_path = os.path.join(video_dir, row[filename_col])
        label = label_map.get(row["label"], 0)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            cap.release()
            continue

        target_idx = frame_count // 2
        ret, frame = get_safe_frame(cap, target_idx)

        if not ret:
            cap.release()
            continue

        face, crop_box, landmarks = detect_and_crop_face(frame, *detector_pack)

        if face is not None:
            face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
            X.append(face)
            y.append(label)
            debugs.append({"crop_box": crop_box, "landmarks": landmarks})

        cap.release()

    return np.array(X), np.array(y), debugs

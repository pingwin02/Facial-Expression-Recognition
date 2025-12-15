import os
from collections import deque

import cv2
import numpy as np
from tqdm import tqdm

from utils.image import get_dlib_detector_predictor, detect_and_crop_face

IMG_HEIGHT = 48
IMG_WIDTH = 48


def process_image_directory(directory, label_map):
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


def process_video_data(df, video_dir, filename_col, label_map, max_attempts=10):
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

        queue = deque([(0, total_frames)])
        visited = set()
        attempts = 0

        while queue and attempts < max_attempts:
            start, end = queue.popleft()
            mid = (start + end) // 2

            if mid not in visited:
                visited.add(mid)
                attempts += 1

                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret, frame = cap.read()

                if ret:
                    face, crop_box, landmarks = detect_and_crop_face(frame, *detector_pack)

                    if face is not None:
                        face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
                        if len(face.shape) == 2:
                            face = np.expand_dims(face, axis=-1)

                        X.append(face)
                        y.append(label)
                        debugs.append(
                            {"video": row[filename_col], "landmarks": landmarks, "frame_idx": mid, "crop_box": crop_box}
                        )
                        break

            if end - start > 1:
                queue.append((start, mid))
                queue.append((mid, end))

        cap.release()

    return np.array(X), np.array(y), debugs

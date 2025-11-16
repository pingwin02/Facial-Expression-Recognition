import json
import os
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.image import get_dlib_detector_predictor, detect_and_crop_face

IMG_HEIGHT = 48
IMG_WIDTH = 48
CHANNELS = 1


def load_data(input_dir, input_flag="devemo"):
    """Load dataset from disk and return train/validation splits and label map.

    Args:
        input_dir (str): Root directory containing dataset folders.
        input_flag (str): Either "devemo" or "devemo+" specifying subfolder format.

    Returns:
        tuple: ((X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map)
            X arrays are numpy arrays of preprocessed face images,
            y arrays are numpy arrays of integer labels,
            debug lists contain dicts with keys 'crop_box' and 'landmarks'.
    """
    print(f"Loading {input_flag} dataset...")
    if input_flag == "devemo+":
        json_path = os.path.join(input_dir, "devemo+", "devemo+.json")
        video_dir = os.path.join(input_dir, "devemo+")
        with open(json_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["label"] = df["label"].str.lower()
        id_col = "participant"
        filename_col = "filename"
    else:
        csv_path = os.path.join(input_dir, "devemo", "_clips_info.csv")
        video_dir = os.path.join(input_dir, "devemo")
        df = pd.read_csv(csv_path, sep=";")
        df["label"] = df["label"].str.lower()
        id_col = "id_examined"
        filename_col = "file"

    unique_labels = sorted(df["label"].unique())
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    unique_ids = df[id_col].unique()
    np.random.seed(42)
    np.random.shuffle(unique_ids)
    split_idx = int(0.8 * len(unique_ids))
    train_ids = unique_ids[:split_idx]
    val_ids = unique_ids[split_idx:]
    train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[id_col].isin(val_ids)].reset_index(drop=True)

    def process_data(df):
        """Process a DataFrame of video rows into arrays and debug info.

        Returns:
            (X, y, debugs): X numpy array of faces, y numpy array of labels, debugs list of dicts
        """
        X, y, debugs = [], [], []
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                continue
            face, crop_box, landmarks = detect_and_crop_face(frame, *get_dlib_detector_predictor())
            if face is not None:
                X.append(face)
                y.append(label)
                debugs.append({"crop_box": crop_box, "landmarks": landmarks})
            cap.release()
        if not X or not y:
            print("Warning: No valid data processed in this split. Check input files and preprocessing.")
        return np.array(X), np.array(y), debugs

    X_train, y_train, train_debugs = process_data(train_df)
    X_val, y_val, val_debugs = process_data(val_df)

    try:
        y_train = np.array(y_train, dtype=np.int64)
    except Exception:
        y_train = np.array([], dtype=np.int64)
    try:
        y_val = np.array(y_val, dtype=np.int64)
    except Exception:
        y_val = np.array([], dtype=np.int64)

    if y_train.size > 0:
        min_label = int(y_train.min())
        if min_label > 0:
            y_train = y_train - min_label
            if y_val.size > 0:
                y_val = y_val - min_label
            label_map = {k: (v - min_label) for k, v in label_map.items()}

    def print_stats(name, y_arr):
        total = int(len(y_arr))
        print(f"\nDataset stats ({name}):")
        print(f"  Total samples: {total}")
        if total == 0:
            print("  No samples.")
            return
        counts = Counter(y_arr.tolist())
        for lbl_idx, cnt in sorted(counts.items()):
            label_name = None
            for k, v in label_map.items():
                if v == lbl_idx:
                    label_name = k
                    break
            print(f"  Class {lbl_idx} ({label_name}): {cnt}")

    print_stats("train", y_train)
    print_stats("val", y_val)

    return (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map

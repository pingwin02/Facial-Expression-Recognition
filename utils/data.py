import json
import os
import urllib.request
import zipfile
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.image import get_dlib_detector_predictor, detect_and_crop_face

IMG_HEIGHT = 48
IMG_WIDTH = 48
CHANNELS = 1


def load_data(input_dir, input_flag="devemo", seed=42, mode="train"):
    """Load dataset from disk and return train/validation splits and label map.

    Args:
        input_dir (str): Root directory containing dataset folders.
        input_flag (str): Dataset type (e.g., "devemo", "fer2013").
        seed (int): Random seed for reproducibility.
        mode (str): Mode for loading data ("train" or "eval").

    Returns:
        tuple: ((X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map)
    """
    print(f"Loading {input_flag} dataset...")
    np.random.seed(seed)

    def split_data(df, id_col):
        unique_ids = df[id_col].unique()
        np.random.shuffle(unique_ids)
        split_idx = int(0.8 * len(unique_ids))
        train_ids = unique_ids[:split_idx]
        val_ids = unique_ids[split_idx:]
        train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
        val_df = df[df[id_col].isin(val_ids)].reset_index(drop=True)
        return train_df, val_df

    def process_directory(directory, label_map):
        X, y, debugs = [], [], []
        for label in tqdm(os.listdir(directory), desc="Processing labels", unit="label"):
            label_dir = os.path.join(directory, label)
            if os.path.isdir(label_dir):
                for image_file in tqdm(
                        os.listdir(label_dir), desc=f"Processing images for label {label}", unit="image"
                ):
                    image_path = os.path.join(label_dir, image_file)
                    image = cv2.imread(image_path)
                    if image is not None:
                        face, crop_box, landmarks = detect_and_crop_face(image, *get_dlib_detector_predictor())
                        if face is not None:
                            face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
                            X.append(face)
                            y.append(label_map[label])
                            debugs.append({"crop_box": crop_box, "landmarks": landmarks})
        return np.array(X), np.array(y), debugs

    def print_stats(name, y_arr, label_map):
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

    if input_flag == "fer2013":
        if mode == "eval":
            print("Evaluation mode: Loading only validation dataset...")
            np.random.seed(seed + 1)
            test_dir = os.path.join(input_dir, "fer2013", "test")
            label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(test_dir)))}
            X_val, y_val, val_debugs = process_directory(test_dir, label_map)
            return (None, None, None), (X_val, y_val, val_debugs), label_map
        else:
            train_dir = os.path.join(input_dir, "fer2013", "train")
            test_dir = os.path.join(input_dir, "fer2013", "test")
            label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(train_dir)))}
            X_train, y_train, train_debugs = process_directory(train_dir, label_map)
            X_val, y_val, val_debugs = process_directory(test_dir, label_map)
            return (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map

    else:
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

        train_df, val_df = split_data(df, id_col)
        label_map = {lbl: idx for idx, lbl in enumerate(sorted(df["label"].unique()))}

        def process_data(df):
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
            return np.array(X), np.array(y), debugs

        X_train, y_train, train_debugs = process_data(train_df)
        X_val, y_val, val_debugs = process_data(val_df)

    print_stats("train", y_train, label_map)
    print_stats("val", y_val, label_map)

    return (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map


def ensure_dataset(input_dir, dataset_name):
    """Ensure the specified dataset is downloaded and available in the input directory.

    Args:
        input_dir (str): Root directory containing dataset folders.
        dataset_name (str): Name of the dataset folder.
    """
    dataset_path = os.path.join(input_dir, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"{dataset_name} dataset not found. Preparing dataset...")
        os.makedirs(dataset_path, exist_ok=True)

        if dataset_name == "fer2013":
            print("Downloading FER2013 dataset...")
            downloads_dir = "downloads"
            os.makedirs(downloads_dir, exist_ok=True)
            dataset_zip = os.path.join(downloads_dir, "fer2013.zip")
            download_url = "https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013"

            if not os.path.exists(dataset_zip):
                print(f"Downloading {dataset_zip}...")
                urllib.request.urlretrieve(download_url, dataset_zip)
                print("Download complete.")

            print("Extracting FER2013 dataset...")
            with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
                zip_ref.extractall(dataset_path)
            print(f"FER2013 dataset extracted to {dataset_path}.")
        else:
            raise ValueError(f"No preparation logic defined for dataset: {dataset_name}")

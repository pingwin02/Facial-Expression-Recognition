import json
import os
import pickle

import numpy as np
import pandas as pd

from dataset.processors import process_image_directory, process_video_data
from dataset.utils import split_data, print_stats


def load_data(input_dir, input_flag="devemo", seed=42, cache_dir="input/.cache", no_cache=False):
    """
    Load dataset from disk and return train/validation splits and label map.
    Utilizes caching to speed up subsequent loads.

    Args:
        input_dir (str): Root directory containing dataset folders.
        input_flag (str): Dataset type (e.g., "devemo", "fer2013").
        seed (int): Random seed for reproducibility.
        cache_dir (str, optional): Directory to save/load cache files.
        no_cache (bool, optional): If True, bypass cache and reload data from disk.

    Returns:
        tuple: ((X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map)
    """

    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = f"{input_flag}_seed{seed}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path) and not no_cache:
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
            print(f"{input_flag} dataset loaded from cache.")
            print_stats("train", result[0][1], result[2])
            print_stats("val", result[1][1], result[2])
            return result

    else:
        print(f"No cache found at {cache_path}, loading data from disk...")

    print(f"Loading {input_flag} dataset...")
    np.random.seed(seed)

    if input_flag == "fer2013":
        train_dir = os.path.join(input_dir, "fer2013", "train")
        test_dir = os.path.join(input_dir, "fer2013", "test")
        label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(train_dir)))}

        X_train, y_train, train_debugs = process_image_directory(train_dir, label_map)
        X_val, y_val, val_debugs = process_image_directory(test_dir, label_map)

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

        X_train, y_train, train_debugs = process_video_data(train_df, video_dir, filename_col, label_map)
        X_val, y_val, val_debugs = process_video_data(val_df, video_dir, filename_col, label_map)

    print(f"{input_flag} dataset loaded from disk.")
    print_stats("train", y_train, label_map)
    print_stats("val", y_val, label_map)

    result = ((X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map)

    print(f"Saving cache to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result

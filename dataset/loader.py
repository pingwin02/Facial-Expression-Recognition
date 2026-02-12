import json
import numpy as np
import os
import pandas as pd
import pickle

from dataset.processors import process_image_directory, process_video_data
from dataset.utils import split_data, print_stats

CACHE_VERSION = "video_v4"


def load_data(input_dir, input_flag="devemo", seed=42, cache_dir="input/.cache", no_cache=False):
    os.makedirs(cache_dir, exist_ok=True)
    cache_input_name = input_flag.replace("+", "plus")
    cache_filename = f"{cache_input_name}_seed{seed}_{CACHE_VERSION}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path) and not no_cache:
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
            print(f"{input_flag} dataset loaded from cache.")
            print_stats("X_train", result[0][0])
            print_stats("X_val", result[1][0])
            print_stats("y_train", result[0][1], result[2])
            print_stats("y_val", result[1][1], result[2])
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

        train_df, val_df = split_data(df, id_col, seed=seed)
        label_map = {lbl: idx for idx, lbl in enumerate(sorted(df["label"].unique()))}

        X_train, y_train, train_debugs = process_video_data(
            train_df, video_dir, filename_col, label_map, frames_per_video=12, max_candidates=60
        )
        X_val, y_val, val_debugs = process_video_data(
            val_df, video_dir, filename_col, label_map, frames_per_video=12, max_candidates=60
        )

    print(f"{input_flag} dataset loaded from disk.")
    print_stats("X_train", y_train)
    print_stats("X_val", y_val)
    print_stats("y_train", y_train, label_map)
    print_stats("y_val", y_val, label_map)

    result = ((X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map)

    print(f"Saving cache to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result

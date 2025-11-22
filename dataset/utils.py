from collections import Counter

import cv2
import numpy as np


def split_data(df, id_col):
    """Splits DataFrame into train and validation sets based on unique IDs."""
    unique_ids = df[id_col].unique()
    np.random.shuffle(unique_ids)
    split_idx = int(0.8 * len(unique_ids))
    train_ids = unique_ids[:split_idx]
    val_ids = unique_ids[split_idx:]

    train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[id_col].isin(val_ids)].reset_index(drop=True)

    return train_df, val_df


def join_data(data_tuples):
    """
    Joins multiple data tuples (X, y, debugs) into a single dataset.
    Handles None values gracefully and concatenates arrays/lists correctly.

    Args:
        data_tuples (list): List of tuples, e.g., [(X_train, y_train, debugs_train), ...]

    Returns:
        tuple: (X_merged, y_merged, debugs_merged)
    """
    valid_tuples = [t for t in data_tuples if t[0] is not None]

    if not valid_tuples:
        return np.array([]), np.array([]), []

    Xs, ys, debug_lists = zip(*valid_tuples)

    X_merged = np.concatenate(Xs, axis=0)
    y_merged = np.concatenate(ys, axis=0)

    debugs_merged = []
    for dbg in debug_lists:
        debugs_merged.extend(dbg)

    return X_merged, y_merged, debugs_merged


def get_safe_frame(cap, target_frame_idx):
    """Attempts to retrieve a frame at the specific index with a fallback to the first frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    return ret, frame


def print_stats(name, y_arr, label_map):
    """Prints distribution statistics for the dataset."""
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

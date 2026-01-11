from collections import Counter

import numpy as np


def split_data(df, id_col):
    unique_ids = df[id_col].unique()
    np.random.shuffle(unique_ids)
    split_idx = int(0.8 * len(unique_ids))
    train_ids = unique_ids[:split_idx]
    val_ids = unique_ids[split_idx:]

    train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[id_col].isin(val_ids)].reset_index(drop=True)

    return train_df, val_df


def join_data(data_tuples):
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


def print_stats(name, arr, label_map=None):
    print(f"\nDataset stats ({name}):")

    if len(arr) == 0:
        print("  No samples.")
        return

    if label_map is None:
        if hasattr(arr, "shape"):
            print(f"  Shape: {arr.shape}")
        else:
            print(f"  Length: {len(arr)}")
    else:
        total = int(len(arr))
        print(f"  Total samples: {total}")

        counts = Counter(arr.tolist())
        for lbl_idx, cnt in sorted(counts.items()):
            label_name = None
            for k, v in label_map.items():
                if v == lbl_idx:
                    label_name = k
                    break
            print(f"  Class {lbl_idx} ({label_name}): {cnt}")

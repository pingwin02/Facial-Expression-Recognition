import numpy as np
from collections import Counter


def split_data(df, id_col, seed=42, val_ratio=0.2, label_col="label"):
    unique_ids = np.array(df[id_col].dropna().unique())
    if len(unique_ids) < 2:
        return df.copy().reset_index(drop=True), df.iloc[0:0].copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)

    id_to_label = df.groupby(id_col)[label_col].agg(lambda series: series.value_counts().index[0]).to_dict()

    label_to_ids = {}
    for identity in unique_ids:
        label = id_to_label.get(identity)
        if label is None:
            continue
        label_to_ids.setdefault(label, []).append(identity)

    val_ids = set()
    target_val_ids = max(1, int(round(len(unique_ids) * val_ratio)))

    for _, ids in label_to_ids.items():
        ids = np.array(ids)
        rng.shuffle(ids)

        if len(ids) == 1:
            continue

        take = max(1, int(round(len(ids) * val_ratio)))
        take = min(take, len(ids) - 1)
        val_ids.update(ids[:take].tolist())

    remaining_ids = [identity for identity in unique_ids if identity not in val_ids]
    rng.shuffle(remaining_ids)

    while len(val_ids) < target_val_ids and remaining_ids:
        candidate = remaining_ids.pop()
        val_ids.add(candidate)

    if len(val_ids) >= len(unique_ids):
        val_ids.remove(next(iter(val_ids)))

    train_ids = [identity for identity in unique_ids if identity not in val_ids]
    val_ids = list(val_ids)

    train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[id_col].isin(val_ids)].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        rng.shuffle(unique_ids)
        split_idx = int((1.0 - val_ratio) * len(unique_ids))
        split_idx = min(max(1, split_idx), len(unique_ids) - 1)
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

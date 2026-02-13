import numpy as np
from collections import Counter


def split_data(df, id_col, seed=42, val_ratio=0.2, label_col="label"):
    unique_ids = np.array(df[id_col].dropna().unique())
    if len(unique_ids) < 2:
        return df.copy().reset_index(drop=True), df.iloc[0:0].copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)

    id_label_counts_df = (
        df.groupby([id_col, label_col]).size().unstack(fill_value=0).sort_index(axis=1)
    )
    labels = list(id_label_counts_df.columns)

    id_to_counts = {
        identity: id_label_counts_df.loc[identity].to_numpy(dtype=np.float64)
        for identity in id_label_counts_df.index
    }

    total_counts = id_label_counts_df.sum(axis=0).to_numpy(dtype=np.float64)
    target_val_counts = total_counts * float(val_ratio)
    denom = np.maximum(1.0, target_val_counts)

    target_val_ids = max(1, int(round(len(unique_ids) * val_ratio)))
    max_val_ids = max(1, len(unique_ids) - 1)
    target_val_ids = min(target_val_ids, max_val_ids)

    ordered_ids = list(id_to_counts.keys())
    rng.shuffle(ordered_ids)

    val_ids = set()
    current_val_counts = np.zeros(len(labels), dtype=np.float64)

    def balance_error(counts):
        diff = (counts - target_val_counts) / denom
        return float(np.sum(diff * diff))

    for identity in ordered_ids:
        if len(val_ids) >= target_val_ids:
            break

        before = balance_error(current_val_counts)
        candidate_counts = current_val_counts + id_to_counts[identity]
        after = balance_error(candidate_counts)

        if after <= before or len(val_ids) < max(1, target_val_ids // 2):
            val_ids.add(identity)
            current_val_counts = candidate_counts

    if len(val_ids) < target_val_ids:
        remaining_ids = [identity for identity in ordered_ids if identity not in val_ids]
        remaining_ids.sort(key=lambda ident: balance_error(current_val_counts + id_to_counts[ident]))
        for identity in remaining_ids:
            if len(val_ids) >= target_val_ids:
                break
            val_ids.add(identity)
            current_val_counts = current_val_counts + id_to_counts[identity]

    train_ids = [identity for identity in ordered_ids if identity not in val_ids]
    if len(train_ids) == 0:
        moved = next(iter(val_ids))
        val_ids.remove(moved)
        train_ids = [moved]

    train_label_counts = total_counts - current_val_counts

    for label_idx in range(len(labels)):
        total_for_label = total_counts[label_idx]
        if total_for_label < 2:
            continue

        if current_val_counts[label_idx] <= 0:
            candidates = [
                ident
                for ident in train_ids
                if id_to_counts[ident][label_idx] > 0
                   and (train_label_counts[label_idx] - id_to_counts[ident][label_idx]) > 0
            ]
            if candidates and len(val_ids) < max_val_ids:
                chosen = min(candidates, key=lambda ident: balance_error(current_val_counts + id_to_counts[ident]))
                train_ids.remove(chosen)
                val_ids.add(chosen)
                current_val_counts = current_val_counts + id_to_counts[chosen]
                train_label_counts = train_label_counts - id_to_counts[chosen]

        if train_label_counts[label_idx] <= 0:
            candidates = [ident for ident in val_ids if id_to_counts[ident][label_idx] > 0]
            if candidates and len(val_ids) > 1:
                chosen = max(candidates, key=lambda ident: id_to_counts[ident][label_idx])
                val_ids.remove(chosen)
                train_ids.append(chosen)
                current_val_counts = current_val_counts - id_to_counts[chosen]
                train_label_counts = train_label_counts + id_to_counts[chosen]

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


def print_stats(name, arr, label_map=None, split_arrays=None):
    print(f"\nDataset stats ({name}):")

    if split_arrays is not None:
        if len(split_arrays) != 2:
            print("  Invalid split_arrays. Expected (train_array, val_array).")
            return

        train_arr, val_arr = split_arrays
        train_arr = np.asarray(train_arr)
        val_arr = np.asarray(val_arr)

        if train_arr.size == 0 and val_arr.size == 0:
            print("  No samples.")
            return

        if label_map is None:
            print("  split summary requires label_map.")
            return

        total = np.concatenate([train_arr, val_arr], axis=0)
        total_counts = Counter(total.tolist())
        train_counts = Counter(train_arr.tolist())
        val_counts = Counter(val_arr.tolist())

        total_n = int(len(total))
        train_n = int(len(train_arr))
        val_n = int(len(val_arr))
        print(f"  all={total_n} train={train_n} val={val_n}")

        inv_label_map = {v: k for k, v in label_map.items()}
        all_label_indices = sorted(total_counts.keys())

        for lbl_idx in all_label_indices:
            label_name = inv_label_map.get(lbl_idx, str(lbl_idx))
            all_cnt = int(total_counts.get(lbl_idx, 0))
            train_cnt = int(train_counts.get(lbl_idx, 0))
            val_cnt = int(val_counts.get(lbl_idx, 0))
            val_ratio = (100.0 * val_cnt / all_cnt) if all_cnt > 0 else 0.0
            print(
                f"  {label_name}: all={all_cnt} train={train_cnt} "
                f"val={val_cnt} val%={val_ratio:.1f}"
            )
        return

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

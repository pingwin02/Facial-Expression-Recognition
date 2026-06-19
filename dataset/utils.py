import csv
import html as html_lib
import json
import numpy as np
import os
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from http.cookiejar import CookieJar


def split_data(df, id_col, seed=42, val_ratio=0.1, test_ratio=0.1, label_col="label"):
    unique_ids = np.array(df[id_col].dropna().unique())
    if len(unique_ids) < 3:
        empty = df.iloc[0:0].copy().reset_index(drop=True)
        return df.copy().reset_index(drop=True), empty, empty

    rng = np.random.default_rng(seed)

    holdout_ratio = val_ratio + test_ratio

    first_label = df[label_col].iloc[0]
    if isinstance(first_label, np.ndarray):
        target_holdout_ids = max(
            2 if float(val_ratio) > 0 and float(test_ratio) > 0 else 1,
            int(round(len(unique_ids) * holdout_ratio)),
        )
        rng.shuffle(unique_ids)
        holdout_ids = unique_ids[:target_holdout_ids]
        train_ids = set(unique_ids[target_holdout_ids:])

        val_split = max(1, int(round(len(holdout_ids) * (val_ratio / holdout_ratio))))
        val_ids = set(holdout_ids[:val_split])
        test_ids = set(holdout_ids[val_split:])

        train_df = df[df[id_col].isin(train_ids)].copy().reset_index(drop=True)
        val_df = df[df[id_col].isin(val_ids)].copy().reset_index(drop=True)
        test_df = df[df[id_col].isin(test_ids)].copy().reset_index(drop=True)
        return train_df, val_df, test_df

    id_label_counts_df = df.groupby([id_col, label_col]).size().unstack(fill_value=0).sort_index(axis=1)
    labels = list(id_label_counts_df.columns)

    id_to_counts = {
        identity: id_label_counts_df.loc[identity].to_numpy(dtype=np.float64) for identity in id_label_counts_df.index
    }

    total_counts = id_label_counts_df.sum(axis=0).to_numpy(dtype=np.float64)
    target_holdout_counts = total_counts * float(holdout_ratio)
    denom = np.maximum(1.0, target_holdout_counts)

    target_holdout_ids = max(
        2 if float(val_ratio) > 0 and float(test_ratio) > 0 else 1,
        int(round(len(unique_ids) * holdout_ratio)),
    )
    max_holdout_ids = max(1, len(unique_ids) - 1)
    target_holdout_ids = min(target_holdout_ids, max_holdout_ids)

    ordered_ids = list(id_to_counts.keys())
    rng.shuffle(ordered_ids)

    holdout_ids = set()
    current_holdout_counts = np.zeros(len(labels), dtype=np.float64)

    def balance_error(counts):
        diff = (counts - target_holdout_counts) / denom
        return float(np.sum(diff * diff))

    for identity in ordered_ids:
        if len(holdout_ids) >= target_holdout_ids:
            break

        before = balance_error(current_holdout_counts)
        candidate_counts = current_holdout_counts + id_to_counts[identity]
        after = balance_error(candidate_counts)

        if after <= before or len(holdout_ids) < max(1, target_holdout_ids // 2):
            holdout_ids.add(identity)
            current_holdout_counts = candidate_counts

    if len(holdout_ids) < target_holdout_ids:
        remaining_ids = [identity for identity in ordered_ids if identity not in holdout_ids]
        remaining_ids.sort(key=lambda ident: balance_error(current_holdout_counts + id_to_counts[ident]))
        for identity in remaining_ids:
            if len(holdout_ids) >= target_holdout_ids:
                break
            holdout_ids.add(identity)
            current_holdout_counts = current_holdout_counts + id_to_counts[identity]

    train_ids = [identity for identity in ordered_ids if identity not in holdout_ids]
    if len(train_ids) == 0:
        moved = next(iter(holdout_ids))
        holdout_ids.remove(moved)
        train_ids = [moved]

    train_label_counts = total_counts - current_holdout_counts

    for label_idx in range(len(labels)):
        total_for_label = total_counts[label_idx]
        if total_for_label < 2:
            continue

        if current_holdout_counts[label_idx] <= 0:
            candidates = [
                ident
                for ident in train_ids
                if id_to_counts[ident][label_idx] > 0
                   and (train_label_counts[label_idx] - id_to_counts[ident][label_idx]) > 0
            ]
            if candidates and len(holdout_ids) < max_holdout_ids:
                chosen = min(candidates, key=lambda ident: balance_error(current_holdout_counts + id_to_counts[ident]))
                train_ids.remove(chosen)
                holdout_ids.add(chosen)
                current_holdout_counts = current_holdout_counts + id_to_counts[chosen]
                train_label_counts = train_label_counts - id_to_counts[chosen]

        if train_label_counts[label_idx] <= 0:
            candidates = [ident for ident in holdout_ids if id_to_counts[ident][label_idx] > 0]
            if candidates and len(holdout_ids) > 1:
                chosen = max(candidates, key=lambda ident: id_to_counts[ident][label_idx])
                holdout_ids.remove(chosen)
                train_ids.append(chosen)
                current_holdout_counts = current_holdout_counts - id_to_counts[chosen]
                train_label_counts = train_label_counts + id_to_counts[chosen]

    holdout_ids = list(holdout_ids)

    train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    holdout_df = df[df[id_col].isin(holdout_ids)].reset_index(drop=True)

    if len(train_df) == 0 or len(holdout_df) == 0:
        rng.shuffle(unique_ids)
        split_idx = int((1.0 - holdout_ratio) * len(unique_ids))
        split_idx = min(max(1, split_idx), len(unique_ids) - 1)
        train_ids = unique_ids[:split_idx]
        holdout_ids = unique_ids[split_idx:]
        train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
        holdout_df = df[df[id_col].isin(holdout_ids)].reset_index(drop=True)

    holdout_unique_ids = np.array(holdout_df[id_col].dropna().unique())
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(holdout_unique_ids)
    val_split = max(1, int(round(len(holdout_unique_ids) * (val_ratio / holdout_ratio))))
    val_split = min(val_split, len(holdout_unique_ids) - 1)
    val_ids = set(holdout_unique_ids[:val_split])
    test_ids = set(holdout_unique_ids[val_split:])

    val_df = holdout_df[holdout_df[id_col].isin(val_ids)].reset_index(drop=True)
    test_df = holdout_df[holdout_df[id_col].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df


def print_stats(name, arr, label_map=None, split_arrays=None):
    print(f"\nDataset stats ({name}):")

    if split_arrays is not None:
        if len(split_arrays) not in (2, 3):
            print("  Invalid split_arrays. Expected (train, val, test) or (train, val).")
            return

        arrays = [np.asarray(a) for a in split_arrays]
        if all(a.size == 0 for a in arrays):
            print("  No samples.")
            return

        if label_map is None:
            print("  split summary requires label_map.")
            return

        total = np.concatenate(arrays, axis=0)
        total_counts = Counter(total.tolist())
        counts_per_split = [Counter(a.tolist()) for a in arrays]

        total_n = int(len(total))
        split_names = ["train", "val", "test"] if len(arrays) == 3 else ["train", "val"]
        header = f"  all={total_n}"
        for sname, a in zip(split_names, arrays):
            header += f" {sname}={int(len(a))}"
        print(header)

        inv_label_map = {v: k for k, v in label_map.items()}
        all_label_indices = sorted(total_counts.keys())

        for lbl_idx in all_label_indices:
            label_name = inv_label_map.get(lbl_idx, str(lbl_idx))
            all_cnt = int(total_counts.get(lbl_idx, 0))
            parts = f"  {label_name}: all={all_cnt}"
            for sname, sc in zip(split_names, counts_per_split):
                cnt = int(sc.get(lbl_idx, 0))
                ratio = (100.0 * cnt / all_cnt) if all_cnt > 0 else 0.0
                parts += f" {sname}={cnt} {sname}%={ratio:.1f}"
            print(parts)
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


def normalize_extracted_layout(dataset_path, required_marker):
    if not os.path.exists(dataset_path):
        return

    if not required_marker:
        while True:
            entries = [entry for entry in os.listdir(dataset_path) if not entry.startswith("!")]
            if len(entries) != 1:
                break

            only_entry = entries[0]
            only_path = os.path.join(dataset_path, only_entry)
            if not os.path.isdir(only_path):
                break

            for nested in os.listdir(only_path):
                src = os.path.join(only_path, nested)
                dst = os.path.join(dataset_path, nested)
                if os.path.exists(dst):
                    continue
                shutil.move(src, dst)
            if os.path.isdir(only_path) and not os.listdir(only_path):
                os.rmdir(only_path)

        for root, dirs, _ in os.walk(dataset_path, topdown=False):
            for directory in dirs:
                directory_path = os.path.join(root, directory)
                if os.path.isdir(directory_path) and not os.listdir(directory_path):
                    os.rmdir(directory_path)
        return

    marker_path = os.path.join(dataset_path, required_marker)

    if not os.path.exists(marker_path):
        marker_source_dir = None
        for root, _, files in os.walk(dataset_path):
            if required_marker in files:
                marker_source_dir = root
                break

        if marker_source_dir and marker_source_dir != dataset_path:
            for entry in os.listdir(marker_source_dir):
                src = os.path.join(marker_source_dir, entry)
                dst = os.path.join(dataset_path, entry)
                if os.path.exists(dst):
                    continue
                shutil.move(src, dst)

    for root, dirs, _ in os.walk(dataset_path, topdown=False):
        for directory in dirs:
            directory_path = os.path.join(root, directory)
            if os.path.isdir(directory_path) and not os.listdir(directory_path):
                os.rmdir(directory_path)


def summarize_dataset_folder(dataset_path):
    file_count = 0
    dir_count = 0
    total_size_bytes = 0

    for root, dirs, files in os.walk(dataset_path):
        dir_count += len(dirs)

        for filename in files:
            full_path = os.path.join(root, filename)

            file_count += 1
            try:
                total_size_bytes += int(os.path.getsize(full_path))
            except OSError:
                pass

    return {
        "file_count": file_count,
        "dir_count": dir_count,
        "total_size_bytes": total_size_bytes,
    }


def build_distribution_result(counts_by_label):
    counts = {str(label).strip().lower(): int(count) for label, count in counts_by_label.items() if int(count) > 0}
    return {
        "num_labels": len(counts),
        "total_items": int(sum(counts.values())),
        "counts_by_label": dict(sorted(counts.items())),
    }


def label_distribution_from_rows(rows, label_key, item_key):
    label_to_items = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        label = str(row.get(label_key, "unknown")).strip().lower()
        item_id = str(row.get(item_key, "")).strip()
        if not item_id:
            continue

        label_to_items.setdefault(label, set()).add(item_id)

    counts = {label: len(items) for label, items in label_to_items.items()}
    return build_distribution_result(counts)


def label_distribution_from_json(json_path, label_key, item_key):
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    return label_distribution_from_rows(rows, label_key=label_key, item_key=item_key)


def label_distribution_from_csv(csv_path, delimiter, label_key, item_key):
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    return label_distribution_from_rows(rows, label_key=label_key, item_key=item_key)


def label_distribution_from_image_folders(root_dir):
    if not os.path.isdir(root_dir):
        return None

    label_counts = Counter()
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            continue
        label_counts[label] = sum(
            1 for entry in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, entry))
        )

    return build_distribution_result(label_counts)


def write_dataset_details_json(
        dataset_name,
        dataset_path,
        download_url,
        archive_name,
        dataset_zip,
        required_marker,
        labels_distribution,
        participants=None,
):
    details_path = os.path.join(dataset_path, "!dataset_details.json")
    details = {
        "dataset_name": dataset_name,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "download_url": download_url,
            "archive_name": archive_name,
            "archive_path": dataset_zip,
        },
        "required_marker": required_marker,
        "dataset_path": dataset_path,
        "summary": summarize_dataset_folder(dataset_path),
        "labels": labels_distribution,
    }

    if participants is not None:
        details["participants"] = participants

    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    print(f"Saved dataset details to {details_path}")


def _extract_drive_file_id(url):
    patterns = [r"/d/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)"]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _print_progress(prefix, downloaded, total_size):
    if total_size and total_size > 0:
        pct = min(100.0, (downloaded * 100.0) / total_size)
        print(f"\r{prefix}: {pct:6.2f}% ({downloaded / (1024 ** 2):.1f}/{total_size / (1024 ** 2):.1f} MB)", end="")
    else:
        print(f"\r{prefix}: {downloaded / (1024 ** 2):.1f} MB", end="")


def _stream_response_to_file(response, destination_path, prefix):
    total_size = response.headers.get("Content-Length")
    total_size = int(total_size) if total_size and str(total_size).isdigit() else None

    downloaded = 0
    chunk_size = 1024 * 1024

    with open(destination_path, "wb") as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            _print_progress(prefix, downloaded, total_size)
    print()


def _extract_drive_confirm_request(html_text):
    form_match = re.search(r'<form[^>]*action="([^"]+)"[^>]*>', html_text)
    if not form_match:
        return None, None

    action = html_lib.unescape(form_match.group(1))
    inputs = re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"[^>]*>', html_text)

    params = {}
    for name, value in inputs:
        params[name] = html_lib.unescape(value)

    return action, params


def _download_google_drive_file(drive_url, destination_path):
    file_id = _extract_drive_file_id(drive_url)
    if not file_id:
        raise RuntimeError(f"Could not extract Google Drive file id from URL: {drive_url}")

    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    headers = {"User-Agent": "Mozilla/5.0"}

    initial_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(initial_url, headers=headers)

    with opener.open(req) as resp:
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type:
            _stream_response_to_file(resp, destination_path, prefix="Downloading")
            return

        html_text = resp.read().decode("utf-8", errors="ignore")

    action, params = _extract_drive_confirm_request(html_text)
    if not action or not params:
        snippet = html_text[:400].replace("\n", " ")
        raise RuntimeError(f"Failed to parse Google Drive confirm page. Response starts with: {snippet}")

    if action.startswith("/"):
        action = urllib.parse.urljoin("https://drive.google.com", action)

    confirm_url = f"{action}?{urllib.parse.urlencode(params)}"
    confirm_req = urllib.request.Request(confirm_url, headers=headers)

    with opener.open(confirm_req) as resp:
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            html_text = resp.read().decode("utf-8", errors="ignore")
            snippet = html_text[:400].replace("\n", " ")
            raise RuntimeError(f"Google Drive returned HTML instead of archive. Response starts with: {snippet}")
        _stream_response_to_file(resp, destination_path, prefix="Downloading")


def download_archive(download_url, dataset_zip):
    if "drive.google.com" not in str(download_url):
        with urllib.request.urlopen(download_url) as resp:
            _stream_response_to_file(resp, dataset_zip, prefix="Downloading")
        return

    _download_google_drive_file(download_url, dataset_zip)


def download_and_extract(dataset_name, download_url, archive_name, dataset_path, required_marker):
    downloads_dir = "downloads"
    os.makedirs(downloads_dir, exist_ok=True)
    dataset_zip = os.path.join(downloads_dir, archive_name)

    if os.path.exists(dataset_zip) and not zipfile.is_zipfile(dataset_zip):
        print(f"Existing archive is not a valid zip: {dataset_zip}. Re-downloading...")
        os.remove(dataset_zip)

    if not os.path.exists(dataset_zip):
        if not download_url:
            raise FileNotFoundError(
                f"Archive for {dataset_name} not found at {dataset_zip}. " f"Please add it manually before running."
            )
        print(f"Downloading {dataset_name} dataset...")
        print(f"Downloading {dataset_zip}...")
        download_archive(download_url, dataset_zip)
        print("Download complete.")

    for attempt in range(2):
        try:
            print(f"Extracting {dataset_name} dataset...")
            with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
                zip_ref.extractall(dataset_path)
            break
        except zipfile.BadZipFile:
            if attempt == 1 or not download_url:
                raise
            print(f"Archive is corrupted or not a zip: {dataset_zip}. Re-downloading once...")
            if os.path.exists(dataset_zip):
                os.remove(dataset_zip)
            download_archive(download_url, dataset_zip)
            print("Download complete.")

    normalize_extracted_layout(dataset_path, required_marker)
    return dataset_zip

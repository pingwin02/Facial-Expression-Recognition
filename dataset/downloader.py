import csv
import json
import os
import shutil
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone


def _dataset_is_ready(dataset_path, dataset_name):
    required_paths = {
        "fer2013": ["train", "test"],
        "devemo": ["_clips_info.csv"],
        "devemo+": ["devemo+.json"],
    }
    if dataset_name not in required_paths:
        return False
    return all(os.path.exists(os.path.join(dataset_path, rel_path)) for rel_path in required_paths[dataset_name])


def _normalize_extracted_layout(dataset_path, required_marker):
    if not os.path.exists(dataset_path):
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


def _summarize_dataset_folder(dataset_path):
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


def _build_distribution_result(counts_by_label):
    counts = {str(label).strip().lower(): int(count) for label, count in counts_by_label.items() if int(count) > 0}
    return {
        "num_labels": len(counts),
        "total_items": int(sum(counts.values())),
        "counts_by_label": dict(sorted(counts.items())),
    }


def _label_distribution_from_rows(rows, label_key, item_key):
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
    return _build_distribution_result(counts)


def _label_distribution_from_json(json_path, label_key, item_key):
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    return _label_distribution_from_rows(rows, label_key=label_key, item_key=item_key)


def _label_distribution_from_csv(csv_path, delimiter, label_key, item_key):
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    return _label_distribution_from_rows(rows, label_key=label_key, item_key=item_key)


def _label_distribution_from_image_folders(root_dir):
    if not os.path.isdir(root_dir):
        return None

    label_counts = Counter()
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            continue
        label_counts[label] = sum(
            1 for entry in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, entry)))

    return _build_distribution_result(label_counts)


def _load_label_distribution(dataset_name, dataset_path):
    row_based_sources = {
        "devemo+": {
            "kind": "json",
            "path": "devemo+.json",
            "label_key": "label",
            "item_key": "filename",
        },
        "devemo": {
            "kind": "csv",
            "path": "_clips_info.csv",
            "delimiter": ";",
            "label_key": "label",
            "item_key": "file",
        },
    }

    source = row_based_sources.get(dataset_name)
    if source is not None:
        source_path = os.path.join(dataset_path, source["path"])
        if source["kind"] == "json":
            return _label_distribution_from_json(source_path, label_key=source["label_key"],
                                                 item_key=source["item_key"])
        return _label_distribution_from_csv(
            source_path,
            delimiter=source.get("delimiter", ","),
            label_key=source["label_key"],
            item_key=source["item_key"],
        )

    if dataset_name == "fer2013":
        return _label_distribution_from_image_folders(os.path.join(dataset_path, "train"))

    return None


def _write_dataset_details_json(dataset_name, dataset_path, download_url, archive_name, dataset_zip, required_marker):
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
        "summary": _summarize_dataset_folder(dataset_path),
        "labels": _load_label_distribution(dataset_name, dataset_path),
    }

    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    print(f"Saved dataset details to {details_path}")


def _download_and_extract(dataset_name, download_url, archive_name, dataset_path, required_marker):
    downloads_dir = "downloads"
    os.makedirs(downloads_dir, exist_ok=True)
    dataset_zip = os.path.join(downloads_dir, archive_name)

    if not os.path.exists(dataset_zip):
        if not download_url:
            raise FileNotFoundError(
                f"Archive for {dataset_name} not found at {dataset_zip}. " f"Please add it manually before running."
            )
        print(f"Downloading {dataset_name} dataset...")
        print(f"Downloading {dataset_zip}...")
        urllib.request.urlretrieve(download_url, dataset_zip)
        print("Download complete.")

    print(f"Extracting {dataset_name} dataset...")
    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        zip_ref.extractall(dataset_path)

    _normalize_extracted_layout(dataset_path, required_marker)
    _write_dataset_details_json(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        download_url=download_url,
        archive_name=archive_name,
        dataset_zip=dataset_zip,
        required_marker=required_marker,
    )


def ensure_dataset(input_dir, dataset_name):
    dataset_path = os.path.join(input_dir, dataset_name)

    dataset_configs = {
        "fer2013": {
            "url": "https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013",
            "archive": "fer2013.zip",
            "marker": "train",
        },
        "devemo": {
            "url": None,
            "archive": "devemo.zip",
            "marker": "_clips_info.csv",
        },
        "devemo+": {
            "url": "https://zenodo.org/records/17214691/files/devemo+.zip?download=1",
            "archive": "devemo_plus.zip",
            "marker": "devemo+.json",
        },
    }

    if dataset_name not in dataset_configs:
        raise ValueError(f"No preparation logic defined for dataset: {dataset_name}")

    _normalize_extracted_layout(dataset_path, dataset_configs[dataset_name]["marker"])

    config = dataset_configs[dataset_name]
    downloads_dir = "downloads"
    dataset_zip = os.path.join(downloads_dir, config["archive"])

    if _dataset_is_ready(dataset_path, dataset_name):
        _write_dataset_details_json(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            download_url=config["url"],
            archive_name=config["archive"],
            dataset_zip=dataset_zip,
            required_marker=config["marker"],
        )
        print(f"{dataset_name} dataset already prepared at {dataset_path}.")
        return

    print(f"{dataset_name} dataset not found or incomplete. Preparing dataset...")
    os.makedirs(dataset_path, exist_ok=True)

    _download_and_extract(
        dataset_name=dataset_name,
        download_url=config["url"],
        archive_name=config["archive"],
        dataset_path=dataset_path,
        required_marker=config["marker"],
    )

    if not _dataset_is_ready(dataset_path, dataset_name):
        raise RuntimeError(f"Failed to prepare {dataset_name}. Expected files were not found in {dataset_path}.")

    print(f"{dataset_name} dataset extracted to {dataset_path}.")

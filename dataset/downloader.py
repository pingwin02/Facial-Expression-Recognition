import os
import shutil
import urllib.request
import zipfile


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

    if _dataset_is_ready(dataset_path, dataset_name):
        print(f"{dataset_name} dataset already prepared at {dataset_path}.")
        return

    print(f"{dataset_name} dataset not found or incomplete. Preparing dataset...")
    os.makedirs(dataset_path, exist_ok=True)

    config = dataset_configs[dataset_name]
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

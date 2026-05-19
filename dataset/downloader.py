import os

from dataset.sources.registry import get_dataset_source
from dataset.utils import download_and_extract, normalize_extracted_layout, write_dataset_details_json


def ensure_dataset(input_dir, dataset_name):
    if dataset_name == "devemo_combined":
        ensure_dataset(input_dir, "devemo")
        ensure_dataset(input_dir, "devemo+")
        return

    source = get_dataset_source(input_flag=dataset_name, input_dir=input_dir)
    dataset_path = source.dataset_path

    normalize_extracted_layout(dataset_path, source.required_marker)

    downloads_dir = "downloads"
    dataset_zip = os.path.join(downloads_dir, source.archive_name)

    if source.is_ready():
        write_dataset_details_json(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            download_url=source.download_url,
            archive_name=source.archive_name,
            dataset_zip=dataset_zip,
            required_marker=source.required_marker,
            labels_distribution=source.label_distribution(),
            participants=source.participants_info(),
        )
        print(f"{dataset_name} dataset already prepared at {dataset_path}.")
        return

    print(f"{dataset_name} dataset not found or incomplete. Preparing dataset...")
    os.makedirs(dataset_path, exist_ok=True)

    dataset_zip = download_and_extract(
        dataset_name=dataset_name,
        download_url=source.download_url,
        archive_name=source.archive_name,
        dataset_path=dataset_path,
        required_marker=source.required_marker,
    )

    write_dataset_details_json(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        download_url=source.download_url,
        archive_name=source.archive_name,
        dataset_zip=dataset_zip,
        required_marker=source.required_marker,
        labels_distribution=source.label_distribution(),
        participants=source.participants_info(),
    )

    if not source.is_ready():
        raise RuntimeError(f"Failed to prepare {dataset_name}. Expected files were not found in {dataset_path}.")

    print(f"{dataset_name} dataset extracted to {dataset_path}.")

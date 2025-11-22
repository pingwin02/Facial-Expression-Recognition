import os
import urllib.request
import zipfile


def ensure_dataset(input_dir, dataset_name):
    """Ensure the specified dataset is downloaded and available in the input directory."""
    dataset_path = os.path.join(input_dir, dataset_name)

    if not os.path.exists(dataset_path):
        print(f"{dataset_name} dataset not found. Preparing dataset...")
        os.makedirs(dataset_path, exist_ok=True)

        if dataset_name == "fer2013":
            print("Downloading FER2013 dataset...")
            downloads_dir = "downloads"
            os.makedirs(downloads_dir, exist_ok=True)
            dataset_zip = os.path.join(downloads_dir, "fer2013.zip")
            download_url = "https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013"

            if not os.path.exists(dataset_zip):
                print(f"Downloading {dataset_zip}...")
                urllib.request.urlretrieve(download_url, dataset_zip)
                print("Download complete.")

            print("Extracting FER2013 dataset...")
            with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
                zip_ref.extractall(dataset_path)
            print(f"FER2013 dataset extracted to {dataset_path}.")
        else:
            raise ValueError(f"No preparation logic defined for dataset: {dataset_name}")

import os

from dataset.processors import process_image_directory
from dataset.sources.base_source import DatasetSource
from dataset.utils import label_distribution_from_image_folders


class FER2013Source(DatasetSource):
    @property
    def dataset_name(self):
        return "fer2013"

    @property
    def archive_name(self):
        return "fer2013.zip"

    @property
    def download_url(self):
        return "https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013"

    @property
    def required_marker(self):
        return "train"

    @property
    def required_paths(self):
        return ["train", "test"]

    def label_distribution(self):
        return label_distribution_from_image_folders(os.path.join(self.dataset_path, "train"))

    def load(self, seed=42):
        train_dir = os.path.join(self.dataset_path, "train")
        test_dir = os.path.join(self.dataset_path, "test")

        label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(train_dir)))}

        X_train, y_train, train_debugs = process_image_directory(train_dir, label_map)
        X_val, y_val, val_debugs = process_image_directory(test_dir, label_map)

        return (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map

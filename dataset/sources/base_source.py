import os
from abc import ABC, abstractmethod


class DatasetSource(ABC):
    def __init__(self, input_dir):
        self.input_dir = input_dir

    @property
    @abstractmethod
    def dataset_name(self):
        pass

    @property
    @abstractmethod
    def archive_name(self):
        pass

    @property
    @abstractmethod
    def download_url(self):
        pass

    @property
    @abstractmethod
    def required_marker(self):
        pass

    @property
    @abstractmethod
    def required_paths(self):
        pass

    @property
    def dataset_path(self):
        return os.path.join(self.input_dir, self.dataset_name)

    def is_ready(self):
        return all(os.path.exists(os.path.join(self.dataset_path, rel_path)) for rel_path in self.required_paths)

    def label_distribution(self):
        return None

    @abstractmethod
    def load(self, seed=42):
        pass

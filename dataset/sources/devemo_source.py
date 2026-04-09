import json
import os
import pandas as pd

from dataset.processors import process_video_temporal_encoding
from dataset.processors import cleanup_iteration_checkpoints
from dataset.sources.base_source import DatasetSource
from dataset.utils import label_distribution_from_csv, label_distribution_from_json, split_data


class DevemoSource(DatasetSource):
    def __init__(self, input_dir, plus_variant=False):
        super().__init__(input_dir)
        self.plus_variant = plus_variant

    @property
    def dataset_name(self):
        return "devemo+" if self.plus_variant else "devemo"

    @property
    def archive_name(self):
        return "devemo_plus.zip" if self.plus_variant else "devemo.zip"

    @property
    def download_url(self):
        if self.plus_variant:
            return "https://zenodo.org/records/17214691/files/devemo+.zip?download=1"
        return None

    @property
    def required_marker(self):
        return "devemo+.json" if self.plus_variant else "_clips_info.csv"

    @property
    def required_paths(self):
        return ["devemo+.json"] if self.plus_variant else ["_clips_info.csv"]

    def label_distribution(self):
        if self.plus_variant:
            json_path = os.path.join(self.dataset_path, "devemo+.json")
            return label_distribution_from_json(json_path, label_key="label", item_key="filename")

        csv_path = os.path.join(self.dataset_path, "_clips_info.csv")
        return label_distribution_from_csv(csv_path, delimiter=";", label_key="label", item_key="file")

    @staticmethod
    def _to_binary_label(label):
        return "positive" if label in ("happiness", "neutral") else "negative"

    def _build_dataframe(self):
        if self.plus_variant:
            json_path = os.path.join(self.dataset_path, "devemo+.json")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df["label"] = df["label"].str.lower().map(self._to_binary_label)
            return df, self.dataset_path, "participant", "filename"

        csv_path = os.path.join(self.dataset_path, "_clips_info.csv")
        df = pd.read_csv(csv_path, sep=";")
        df["label"] = df["label"].str.lower().map(self._to_binary_label)
        return df, self.dataset_path, "id_examined", "file"

    def load(self, seed=42):
        df, video_dir, id_col, filename_col = self._build_dataframe()

        train_df, val_df = split_data(df, id_col, seed=seed)
        label_map = {lbl: idx for idx, lbl in enumerate(sorted(df["label"].unique()))}
        checkpoint_dir = os.path.join(self.input_dir, ".tmp")
        os.makedirs(checkpoint_dir, exist_ok=True)

        X_train, y_train, train_debugs = process_video_temporal_encoding(
            train_df,
            video_dir,
            filename_col,
            label_map,
            num_sample_frames=5,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=f"{self.dataset_name}_train_seed{seed}",
            save_checkpoint_every=10,
            resume_from_checkpoint=True,
        )
        X_val, y_val, val_debugs = process_video_temporal_encoding(
            val_df,
            video_dir,
            filename_col,
            label_map,
            num_sample_frames=5,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=f"{self.dataset_name}_val_seed{seed}",
            save_checkpoint_every=10,
            resume_from_checkpoint=True,
        )

        cleanup_iteration_checkpoints(checkpoint_dir, f"{self.dataset_name}_train_seed{seed}")
        cleanup_iteration_checkpoints(checkpoint_dir, f"{self.dataset_name}_val_seed{seed}")

        return (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map

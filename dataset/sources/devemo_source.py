import json
import os
import re
import pandas as pd

from dataset.processors import process_video_temporal_encoding
from dataset.processors import cleanup_iteration_checkpoints
from dataset.sources.base_source import DatasetSource
from dataset.utils import build_distribution_result, split_data


class DevemoSource(DatasetSource):
    NEGATIVE_LABELS = {"anger", "confusion", "surprise", "disgust"}

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
            if not os.path.exists(json_path):
                return None

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            if "label" not in df:
                return None

            normalized = df["label"].apply(lambda lbl: self._normalize_label(lbl)).dropna()
            return build_distribution_result(normalized.value_counts().to_dict())

        csv_path = os.path.join(self.dataset_path, "_clips_info.csv")
        if not os.path.exists(csv_path):
            return None

        df = pd.read_csv(csv_path, sep=";")
        if "label" not in df:
            return None

        normalized = df["label"].apply(lambda lbl: self._normalize_label(lbl)).dropna()
        return build_distribution_result(normalized.value_counts().to_dict())

    def participants_info(self):
        if self.plus_variant:
            json_path = os.path.join(self.dataset_path, "devemo+.json")
            if not os.path.exists(json_path):
                return None
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            if "participant" not in df:
                return None
            ids = sorted(df["participant"].dropna().unique().tolist())
        else:
            csv_path = os.path.join(self.dataset_path, "_clips_info.csv")
            if not os.path.exists(csv_path):
                return None
            df = pd.read_csv(csv_path, sep=";")
            if "id_examined" not in df:
                return None
            ids = sorted(df["id_examined"].dropna().unique().tolist())

        return {"num_participants": len(ids), "participant_ids": ids}

    @staticmethod
    def _normalize_label(label, class_split="binary"):
        if isinstance(label, str):
            label = label.strip().lower()
            if not label:
                return None
            if class_split == "all":
                return label
            if label in DevemoSource.NEGATIVE_LABELS:
                return "negative"
            return "others"
        return None

    def _build_dataframe(self, class_split="binary"):
        if self.plus_variant:
            json_path = os.path.join(self.dataset_path, "devemo+.json")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df["label"] = df["label"].apply(lambda lbl: self._normalize_label(lbl, class_split))
            df = df[df["label"].notna()].reset_index(drop=True)
            return df, self.dataset_path, "participant", "filename"

        csv_path = os.path.join(self.dataset_path, "_clips_info.csv")
        df = pd.read_csv(csv_path, sep=";")
        df["label"] = df["label"].apply(lambda lbl: self._normalize_label(lbl, class_split))
        df = df[df["label"].notna()].reset_index(drop=True)
        return df, self.dataset_path, "id_examined", "file"

    @staticmethod
    def _checkpoint_tag(train_frame_selection, test_frame_selection, num_frames, class_split):
        raw = (
            f"tr_{train_frame_selection}"
            f"__te_{test_frame_selection}"
            f"__nf_{int(num_frames)}"
            f"__cls_{class_split}"
        )
        return re.sub(r"[^a-zA-Z0-9_+-]", "_", raw)

    def load(
        self, seed=42, train_frame_selection="uniform", test_frame_selection=None, num_frames=5, class_split="binary"
    ):
        if test_frame_selection is None:
            test_frame_selection = train_frame_selection

        import dataset.processors as proc

        proc.NUM_SAMPLE_FRAMES = num_frames
        proc.CENTER_IDX = num_frames // 2

        df, video_dir, id_col, filename_col = self._build_dataframe(class_split=class_split)

        train_df, val_df, test_df = split_data(df, id_col, seed=seed)
        label_map = {lbl: idx for idx, lbl in enumerate(sorted(df["label"].unique()))}
        checkpoint_dir = os.path.join(self.input_dir, ".tmp")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_tag = self._checkpoint_tag(
            train_frame_selection=train_frame_selection,
            test_frame_selection=test_frame_selection,
            num_frames=num_frames,
            class_split=class_split,
        )
        train_checkpoint_prefix = f"{self.dataset_name}_train_seed{seed}_{checkpoint_tag}"
        val_checkpoint_prefix = f"{self.dataset_name}_val_seed{seed}_{checkpoint_tag}"
        test_checkpoint_prefix = f"{self.dataset_name}_test_seed{seed}_{checkpoint_tag}"

        def _parse_selection(method):
            manual = method.startswith("manual_")
            base = method.replace("manual_", "")
            use_transformer = base == "transformer"
            use_random = base == "random"
            return use_transformer, use_random, manual

        train_transformer, train_random, train_manual = _parse_selection(train_frame_selection)
        test_transformer, test_random, test_manual = _parse_selection(test_frame_selection)

        X_train, y_train, train_debugs = process_video_temporal_encoding(
            train_df,
            video_dir,
            filename_col,
            label_map,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=train_checkpoint_prefix,
            save_checkpoint_every=10,
            resume_from_checkpoint=True,
            use_transformer_selection=train_transformer,
            use_random_selection=train_random,
            use_manual_selection=train_manual,
        )
        X_val, y_val, val_debugs = process_video_temporal_encoding(
            val_df,
            video_dir,
            filename_col,
            label_map,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=val_checkpoint_prefix,
            save_checkpoint_every=10,
            resume_from_checkpoint=True,
            use_transformer_selection=test_transformer,
            use_random_selection=test_random,
            use_manual_selection=test_manual,
        )
        X_test, y_test, test_debugs = process_video_temporal_encoding(
            test_df,
            video_dir,
            filename_col,
            label_map,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=test_checkpoint_prefix,
            save_checkpoint_every=10,
            resume_from_checkpoint=True,
            use_transformer_selection=test_transformer,
            use_random_selection=test_random,
            use_manual_selection=test_manual,
        )

        cleanup_iteration_checkpoints(checkpoint_dir, train_checkpoint_prefix)
        cleanup_iteration_checkpoints(checkpoint_dir, val_checkpoint_prefix)
        cleanup_iteration_checkpoints(checkpoint_dir, test_checkpoint_prefix)

        return (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), (X_test, y_test, test_debugs), label_map

import numpy as np
import os
import re

from dataset.processors import process_video_temporal_encoding, cleanup_iteration_checkpoints
from dataset.sources.devemo_source import DevemoSource
from dataset.sources.base_source import DatasetSource
from dataset.utils import split_data


class DevemoCombinedSource(DatasetSource):
    def __init__(self, input_dir):
        super().__init__(input_dir)
        self._devemo = DevemoSource(input_dir, plus_variant=False)
        self._devemo_plus = DevemoSource(input_dir, plus_variant=True)

    @property
    def dataset_name(self):
        return "devemo_combined"

    @property
    def archive_name(self):
        return "devemo.zip"

    @property
    def download_url(self):
        return None

    @property
    def required_marker(self):
        return "_clips_info.csv"

    @property
    def required_paths(self):
        return []

    @property
    def dataset_path(self):
        return os.path.join(self.input_dir, "devemo")

    def label_distribution(self):
        return None

    def is_ready(self):
        return self._devemo.is_ready() and self._devemo_plus.is_ready()

    def _build_combined_dataframe(self, class_split="binary"):
        df1, video_dir1, _, filename_col1 = self._devemo._build_dataframe(class_split=class_split)
        df2, video_dir2, _, filename_col2 = self._devemo_plus._build_dataframe(class_split=class_split)

        df1 = df1.copy()
        df1["_video_path"] = df1[filename_col1].apply(lambda f: os.path.join(video_dir1, f))
        df1["_source"] = "devemo"
        df1["_combined_id"] = "d_" + df1.index.astype(str)

        df2 = df2.copy()
        df2["_video_path"] = df2[filename_col2].apply(lambda f: os.path.join(video_dir2, f))
        df2["_source"] = "devemo+"
        df2["_combined_id"] = "dp_" + df2.index.astype(str)

        combined = []
        for df_src in [df1, df2]:
            combined.append(df_src[["label", "_video_path", "_source", "_combined_id"]].copy())

        result = __import__("pandas").concat(combined, ignore_index=True)
        return result

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

        df = self._build_combined_dataframe(class_split=class_split)

        train_df, val_df, test_df = split_data(df, "_combined_id", seed=seed)
        label_map = {lbl: idx for idx, lbl in enumerate(sorted(df["label"].unique()))}

        checkpoint_dir = os.path.join(self.input_dir, ".tmp")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_tag = self._checkpoint_tag(
            train_frame_selection=train_frame_selection,
            test_frame_selection=test_frame_selection,
            num_frames=num_frames,
            class_split=class_split,
        )
        train_cp = f"devemo_combined_train_seed{seed}_{checkpoint_tag}"
        val_cp = f"devemo_combined_val_seed{seed}_{checkpoint_tag}"
        test_cp = f"devemo_combined_test_seed{seed}_{checkpoint_tag}"

        def _parse_selection(method):
            manual = method.startswith("manual_")
            base = method.replace("manual_", "")
            use_transformer = base == "transformer"
            use_random = base == "random"
            return use_transformer, use_random, manual

        train_transformer, train_random, train_manual = _parse_selection(train_frame_selection)
        test_transformer, test_random, test_manual = _parse_selection(test_frame_selection)

        # Use empty string as video_dir since _video_path already contains full path
        X_train, y_train, train_debugs = process_video_temporal_encoding(
            train_df,
            "",
            "_video_path",
            label_map,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=train_cp,
            save_checkpoint_every=10,
            resume_from_checkpoint=True,
            use_transformer_selection=train_transformer,
            use_random_selection=train_random,
            use_manual_selection=train_manual,
        )
        X_val, y_val, val_debugs = process_video_temporal_encoding(
            val_df,
            "",
            "_video_path",
            label_map,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=val_cp,
            save_checkpoint_every=10,
            resume_from_checkpoint=True,
            use_transformer_selection=test_transformer,
            use_random_selection=test_random,
            use_manual_selection=test_manual,
        )
        X_test, y_test, test_debugs = process_video_temporal_encoding(
            test_df,
            "",
            "_video_path",
            label_map,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=test_cp,
            save_checkpoint_every=10,
            resume_from_checkpoint=True,
            use_transformer_selection=test_transformer,
            use_random_selection=test_random,
            use_manual_selection=test_manual,
        )

        cleanup_iteration_checkpoints(checkpoint_dir, train_cp)
        cleanup_iteration_checkpoints(checkpoint_dir, val_cp)
        cleanup_iteration_checkpoints(checkpoint_dir, test_cp)

        return (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), (X_test, y_test, test_debugs), label_map

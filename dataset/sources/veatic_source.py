import csv
import numpy as np
import os
import pandas as pd
from collections import Counter

from dataset.processors import cleanup_iteration_checkpoints, process_video_frames_with_frame_labels
from dataset.sources.base_source import DatasetSource
from dataset.utils import build_distribution_result


def read_veatic_rating_values(csv_path):
    values = []
    if not os.path.exists(csv_path):
        return values

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                values.append(float(row[1]))
            except (TypeError, ValueError):
                continue
    return values


def veatic_quadrant_label(arousal_value, valence_value, threshold=0.0):
    arousal_state = "high_arousal" if float(arousal_value) >= threshold else "low_arousal"
    valence_state = "high_valence" if float(valence_value) >= threshold else "low_valence"
    return f"{arousal_state}_{valence_state}"


def label_distribution_from_veatic(dataset_path):
    rating_dir = os.path.join(dataset_path, "rating_averaged")
    if not os.path.isdir(rating_dir):
        return None

    counts = Counter()
    for filename in os.listdir(rating_dir):
        if not filename.endswith("_arousal.csv"):
            continue

        video_id = filename[: -len("_arousal.csv")]
        arousal_path = os.path.join(rating_dir, f"{video_id}_arousal.csv")
        valence_path = os.path.join(rating_dir, f"{video_id}_valence.csv")

        if not os.path.exists(valence_path):
            continue

        arousal_values = read_veatic_rating_values(arousal_path)
        valence_values = read_veatic_rating_values(valence_path)
        if not arousal_values or not valence_values:
            continue

        n = min(len(arousal_values), len(valence_values))
        for frame_idx in range(n):
            state = veatic_quadrant_label(arousal_values[frame_idx], valence_values[frame_idx], threshold=0.0)
            counts[state] += 1

    return build_distribution_result(counts)


class VEATICSource(DatasetSource):
    @property
    def dataset_name(self):
        return "veatic"

    @property
    def archive_name(self):
        return "veatic.zip"

    @property
    def download_url(self):
        return "https://drive.google.com/file/d/1HZIw8RGsRwwENhJlhNJRL88YyfiE442N/view"

    @property
    def required_marker(self):
        return None

    @property
    def required_paths(self):
        return ["videos", "rating_averaged"]

    def label_distribution(self):
        return label_distribution_from_veatic(self.dataset_path)

    def _build_dataframe(self):
        rating_dir = os.path.join(self.dataset_path, "rating_averaged")
        video_dir = os.path.join(self.dataset_path, "videos")

        rows = []
        for filename in os.listdir(rating_dir):
            if not filename.endswith("_arousal.csv"):
                continue

            video_id = filename[: -len("_arousal.csv")]
            arousal_path = os.path.join(rating_dir, f"{video_id}_arousal.csv")
            valence_path = os.path.join(rating_dir, f"{video_id}_valence.csv")
            video_filename = f"{video_id}.mp4"
            video_path = os.path.join(video_dir, video_filename)

            if not (os.path.exists(arousal_path) and os.path.exists(valence_path) and os.path.exists(video_path)):
                continue

            arousal_values = read_veatic_rating_values(arousal_path)
            valence_values = read_veatic_rating_values(valence_path)
            if not arousal_values or not valence_values:
                continue

            sequence_len = min(len(arousal_values), len(valence_values))
            if sequence_len <= 0:
                continue

            frame_labels = []
            for frame_idx in range(sequence_len):
                frame_labels.append(
                    veatic_quadrant_label(
                        arousal_values[frame_idx],
                        valence_values[frame_idx],
                        threshold=0.0,
                    )
                )

            rows.append(
                {
                    "video_id": video_id,
                    "filename": video_filename,
                    "full_arousal_sequence": arousal_values,
                    "full_valence_sequence": valence_values,
                    "frame_labels": frame_labels,
                    "arousal_path": arousal_path,
                    "valence_path": valence_path,
                }
            )

        df = pd.DataFrame(rows)

        return df, video_dir

    def load(self, seed=42):
        df, video_dir = self._build_dataframe()
        if df.empty:
            raise RuntimeError("VEATIC dataset appears empty or missing paired rating/video files.")

        unique_ids = np.array(df["video_id"].dropna().unique())
        if len(unique_ids) < 2:
            train_df = df.copy().reset_index(drop=True)
            val_df = df.iloc[0:0].copy().reset_index(drop=True)
        else:
            rng = np.random.default_rng(seed)
            rng.shuffle(unique_ids)
            split_idx = int(round(len(unique_ids) * 0.8))
            split_idx = min(max(1, split_idx), len(unique_ids) - 1)

            train_ids = set(unique_ids[:split_idx])
            val_ids = set(unique_ids[split_idx:])

            train_df = df[df["video_id"].isin(train_ids)].copy().reset_index(drop=True)
            val_df = df[df["video_id"].isin(val_ids)].copy().reset_index(drop=True)

        all_frame_labels = sorted({label for labels in df["frame_labels"] for label in labels})
        label_map = {lbl: idx for idx, lbl in enumerate(all_frame_labels)}

        checkpoint_dir = os.path.join(self.input_dir, ".tmp")
        os.makedirs(checkpoint_dir, exist_ok=True)

        X_train, y_train, train_debugs = process_video_frames_with_frame_labels(
            train_df,
            video_dir,
            "filename",
            label_map,
            frames_per_video=300,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=f"veatic_train_seed{seed}",
            save_checkpoint_every=1,
            resume_from_checkpoint=True,
        )
        X_val, y_val, val_debugs = process_video_frames_with_frame_labels(
            val_df,
            video_dir,
            "filename",
            label_map,
            frames_per_video=300,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=f"veatic_val_seed{seed}",
            save_checkpoint_every=1,
            resume_from_checkpoint=True,
        )

        cleanup_iteration_checkpoints(checkpoint_dir, f"veatic_train_seed{seed}")
        cleanup_iteration_checkpoints(checkpoint_dir, f"veatic_val_seed{seed}")

        return (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map

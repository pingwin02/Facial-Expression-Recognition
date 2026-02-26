import csv
import numpy as np
import os
import pandas as pd
from collections import Counter

from dataset.processors import process_video_frames_with_frame_labels
from dataset.sources.base_source import DatasetSource
from dataset.utils import build_distribution_result, split_data


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


def center_window_mean(values, window_fraction=0.4):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return None

    win = max(1, int(round(arr.size * float(window_fraction))))
    center = arr.size // 2
    start = max(0, center - win // 2)
    end = min(arr.size, start + win)

    if end <= start:
        return float(np.mean(arr))
    return float(np.mean(arr[start:end]))


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

            center_arousal = center_window_mean(arousal_values, window_fraction=0.4)
            center_valence = center_window_mean(valence_values, window_fraction=0.4)
            video_label = veatic_quadrant_label(center_arousal, center_valence, threshold=0.0)

            rows.append(
                {
                    "video_id": video_id,
                    "filename": video_filename,
                    "label": video_label,
                    "center_mean_arousal": center_arousal,
                    "center_mean_valence": center_valence,
                    "full_arousal_sequence": arousal_values,
                    "full_valence_sequence": valence_values,
                    "frame_labels": frame_labels,
                    "arousal_path": arousal_path,
                    "valence_path": valence_path,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            state_counts = Counter(df["label"].tolist())
            print("VEATIC quadrant classification (threshold=0.0):")
            for state_name, count in sorted(state_counts.items()):
                print(f"  {state_name}: {count}")
            arousal_vals = np.array([row["center_mean_arousal"] for _, row in df.iterrows()])
            valence_vals = np.array([row["center_mean_valence"] for _, row in df.iterrows()])
            print(f"  Arousal range: [{np.min(arousal_vals):.3f}, {np.max(arousal_vals):.3f}]")
            print(f"  Valence range: [{np.min(valence_vals):.3f}, {np.max(valence_vals):.3f}]")

        return df, video_dir

    def load(self, seed=42):
        df, video_dir = self._build_dataframe()
        if df.empty:
            raise RuntimeError("VEATIC dataset appears empty or missing paired rating/video files.")

        train_df, val_df = split_data(df, "video_id", seed=seed)
        all_frame_labels = sorted({label for labels in df["frame_labels"] for label in labels})
        label_map = {lbl: idx for idx, lbl in enumerate(all_frame_labels)}

        checkpoint_dir = os.path.join(self.input_dir, ".tmp")
        os.makedirs(checkpoint_dir, exist_ok=True)

        X_train, y_train, train_debugs = process_video_frames_with_frame_labels(
            train_df,
            video_dir,
            "filename",
            label_map,
            frames_per_video=100,
            max_candidates=None,
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
            frames_per_video=100,
            max_candidates=None,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=f"veatic_val_seed{seed}",
            save_checkpoint_every=1,
            resume_from_checkpoint=True,
        )

        return (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.sources.devemo_source import DevemoSource

INPUT_DIR = "input"
OUTPUT_BASE = "output/dataset_statistics"

DATASETS = {
    "devemo": {"plus_variant": False},
    "devemo+": {"plus_variant": True},
}


def _build_dataframe(dataset_key, class_split):
    cfg = DATASETS[dataset_key]
    source = DevemoSource(INPUT_DIR, plus_variant=cfg["plus_variant"])
    df, _, id_col, _ = source._build_dataframe(class_split=class_split)
    return df, id_col


def plot_class_distribution(ax, df, dataset_name, class_split):
    counts = df["label"].value_counts().sort_index()
    colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    labels = [f"{lbl}\n({cnt})" for lbl, cnt in zip(counts.index, counts.values)]

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "black", "linewidth": 0.5},
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")
    ax.set_title(f"{dataset_name} — Class Distribution ({class_split})", fontsize=12, fontweight="bold")


def plot_participant_distribution(ax, df, id_col, dataset_name, class_split):
    participant_counts = df[id_col].value_counts().sort_values(ascending=False)
    n = len(participant_counts)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n, 20)))
    if n > 20:
        colors = np.tile(colors, (n // 20 + 1, 1))[:n]

    ax.bar(range(n), participant_counts.values, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_title(
        f"{dataset_name} — Videos per Participant ({class_split})\n({n} participants)", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Participant (sorted by count)")
    ax.set_ylabel("Number of videos")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.3)

    mean_val = participant_counts.mean()
    ax.axhline(mean_val, color="red", linestyle="--", linewidth=1, label=f"Mean: {mean_val:.1f}")
    ax.legend(fontsize=9)


def plot_participant_class_breakdown(ax, df, id_col, dataset_name, class_split):
    pivot = df.groupby([id_col, "label"]).size().unstack(fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    labels_sorted = sorted(pivot.columns)
    n = len(pivot)
    x = np.arange(n)
    width = 0.8
    bottom = np.zeros(n)

    colors = plt.cm.Set2(np.linspace(0, 1, len(labels_sorted)))
    for i, label in enumerate(labels_sorted):
        vals = pivot[label].values
        ax.bar(x, vals, width, bottom=bottom, label=label, color=colors[i], edgecolor="black", linewidth=0.2)
        bottom += vals

    ax.set_title(
        f"{dataset_name} — Class per Participant ({class_split})\n({n} participants)", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Participant (sorted by total count)")
    ax.set_ylabel("Number of videos")
    ax.set_xticks([])
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)


def generate_plots(class_split):
    output_dir = os.path.join(OUTPUT_BASE, class_split)
    os.makedirs(output_dir, exist_ok=True)

    all_data = {}
    for dataset_key in DATASETS:
        try:
            df, id_col = _build_dataframe(dataset_key, class_split)
            all_data[dataset_key] = (df, id_col)
        except Exception as e:
            print(f"Warning: Could not load {dataset_key}: {e}")

    if not all_data:
        print("No datasets available.")
        return

    n_datasets = len(all_data)

    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]
    for ax, (name, (df, _)) in zip(axes, all_data.items()):
        plot_class_distribution(ax, df, name, class_split)
    plt.tight_layout()
    path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]
    for ax, (name, (df, id_col)) in zip(axes, all_data.items()):
        plot_participant_distribution(ax, df, id_col, name, class_split)
    plt.tight_layout()
    path = os.path.join(output_dir, "participant_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    fig, axes = plt.subplots(1, n_datasets, figsize=(8 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]
    for ax, (name, (df, id_col)) in zip(axes, all_data.items()):
        plot_participant_class_breakdown(ax, df, id_col, name, class_split)
    plt.tight_layout()
    path = os.path.join(output_dir, "participant_class_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    summary = {}
    for name, (df, id_col) in all_data.items():
        participants = sorted(df[id_col].dropna().unique().tolist())
        class_counts = df["label"].value_counts().to_dict()
        summary[name] = {
            "class_split": class_split,
            "total_samples": int(len(df)),
            "num_classes": int(df["label"].nunique()),
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "num_participants": len(participants),
            "participant_ids": [str(p) for p in participants],
            "videos_per_participant": {str(p): int(c) for p, c in df[id_col].value_counts().to_dict().items()},
        }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset statistics plots")
    parser.add_argument(
        "--class-split", choices=["binary", "all", "both"], default="both", help="Class split configuration"
    )
    args = parser.parse_args()

    splits = ["binary", "all"] if args.class_split == "both" else [args.class_split]
    for split in splits:
        print(f"\n{'='*60}")
        print(f"  Generating statistics for class_split = '{split}'")
        print(f"{'='*60}\n")
        generate_plots(split)

    print("\nDone.")


if __name__ == "__main__":
    main()

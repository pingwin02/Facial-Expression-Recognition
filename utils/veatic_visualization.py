import csv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D


def read_veatic_sequence(csv_path):
    values = []
    if not os.path.exists(csv_path):
        return np.array([])

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                values.append(float(row[1]))
            except (TypeError, ValueError):
                continue
    return np.array(values)


def get_quadrant_from_av(arousal, valence, threshold=0.0):
    arousal_state = "high_arousal" if arousal >= threshold else "low_arousal"
    valence_state = "high_valence" if valence >= threshold else "low_valence"
    return f"{arousal_state}_{valence_state}"


def get_quadrant_color(quadrant):
    colors = {
        "high_arousal_high_valence": "#00b894",
        "high_arousal_low_valence": "#ff3b30",
        "low_arousal_high_valence": "#007aff",
        "low_arousal_low_valence": "#6c5ce7",
    }
    return colors.get(quadrant, "#000000")


def plot_veatic_frame_label_timeline(arousal_seq, valence_seq, video_name, output_path=None, threshold=0.0):
    n = min(len(arousal_seq), len(valence_seq))
    if n == 0:
        return

    labels = [get_quadrant_from_av(float(arousal_seq[i]), float(valence_seq[i]), threshold=threshold) for i in range(n)]
    frames = np.arange(n)

    y_map = {
        "low_arousal_low_valence": 0,
        "low_arousal_high_valence": 1,
        "high_arousal_low_valence": 2,
        "high_arousal_high_valence": 3,
    }
    y_values = np.array([y_map.get(lbl, -1) for lbl in labels], dtype=np.int32)
    colors = [get_quadrant_color(lbl) for lbl in labels]

    fig, ax_values = plt.subplots(figsize=(18, 6))

    run_start = 0
    for idx in range(1, n + 1):
        run_end = idx == n
        label_changed = (not run_end) and (labels[idx] != labels[idx - 1])
        if run_end or label_changed:
            run_label = labels[idx - 1]
            left = run_start - 0.5
            right = (idx - 1) + 0.5
            ax_values.axvspan(left, right, color=get_quadrant_color(run_label), alpha=0.24, zorder=0)
            run_start = idx

    line_arousal = ax_values.plot(frames, arousal_seq[:n], "b-", linewidth=1.5, alpha=0.7, label="Arousal (CSV)")
    line_valence = ax_values.plot(frames, valence_seq[:n], "g-", linewidth=1.5, alpha=0.7, label="Valence (CSV)")
    ax_values.axhline(
        0.0,
        color="gray",
        linestyle=":",
        linewidth=2.0,
        alpha=0.95,
        zorder=3,
        label="Threshold: 0.0",
    )

    ax_values.set_xlabel("Frame Index", fontsize=12)
    ax_values.set_ylabel("Value", fontsize=11, fontweight="bold")
    ax_values.set_title(f"VEATIC frame labels timeline: {video_name}", fontsize=14, fontweight="bold")
    ax_values.grid(True, alpha=0.3)
    ax_values.set_ylim(-1.1, 1.1)

    legend_handles = [
        mpatches.Patch(color=get_quadrant_color("high_arousal_high_valence"), label="Q1 high_arousal_high_valence"),
        mpatches.Patch(color=get_quadrant_color("high_arousal_low_valence"), label="Q2 high_arousal_low_valence"),
        mpatches.Patch(color=get_quadrant_color("low_arousal_high_valence"), label="Q3 low_arousal_high_valence"),
        mpatches.Patch(color=get_quadrant_color("low_arousal_low_valence"), label="Q4 low_arousal_low_valence"),
    ]
    value_handles = line_arousal + line_valence + [ax_values.lines[-1]]
    value_labels = [h.get_label() for h in value_handles]
    legend_values = ax_values.legend(value_handles, value_labels, loc="upper left", fontsize=9, framealpha=0.9)
    ax_values.add_artist(legend_values)
    ax_values.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_veatic_timeline(
        arousal_seq,
        valence_seq,
        video_name,
        selected_frames=None,
        selected_frame_correct=None,
        showcase_frames=None,
        output_path=None,
        threshold=0.0,
):
    fig, ax = plt.subplots(figsize=(18, 6))

    n = min(len(arousal_seq), len(valence_seq))
    if n == 0:
        return

    frames = np.arange(n)
    arousal_seq = np.asarray(arousal_seq[:n], dtype=np.float32)
    valence_seq = np.asarray(valence_seq[:n], dtype=np.float32)

    ax.plot(frames, arousal_seq, color="#1f77b4", linewidth=1.6, alpha=0.85, label="Arousal")
    ax.plot(frames, valence_seq, color="#8e44ad", linewidth=1.6, alpha=0.85, label="Valence")
    ax.axhline(y=threshold, color="gray", linestyle=":", linewidth=2, label=f"Threshold: {threshold}")

    selected_accuracy_text = "n/a"
    showcase_frames = set(showcase_frames or [])

    if selected_frames is not None and len(selected_frames) > 0:
        selected_frames = sorted({int(frame) for frame in selected_frames if 0 <= int(frame) < n})
        if selected_frames:
            correct_flags = []
            showcase_flags = []
            for frame_idx in selected_frames:
                if selected_frame_correct and frame_idx in selected_frame_correct:
                    correct_flags.append(bool(selected_frame_correct[frame_idx]))
                else:
                    correct_flags.append(False)
                showcase_flags.append(frame_idx in showcase_frames)

            for frame_idx, is_correct, is_showcase in zip(selected_frames, correct_flags, showcase_flags):
                span_color = "#2ecc71" if is_correct else "#ff6b6b"
                alpha = 0.14 if is_showcase else 0.07
                ax.axvspan(frame_idx - 0.35, frame_idx + 0.35, color=span_color, alpha=alpha, zorder=1)

            selected_arousal = arousal_seq[selected_frames]
            selected_valence = valence_seq[selected_frames]
            edge_colors = ["#1e8449" if is_correct else "#c0392b" for is_correct in correct_flags]
            marker_sizes = [78 if is_showcase else 52 for is_showcase in showcase_flags]

            ax.scatter(
                selected_frames,
                selected_arousal,
                color="#1f77b4",
                s=marker_sizes,
                zorder=5,
                marker="o",
                edgecolors=edge_colors,
                linewidths=1.2,
                label="Frames used in evaluation (Arousal)",
            )
            ax.scatter(
                selected_frames,
                selected_valence,
                color="#8e44ad",
                s=marker_sizes,
                zorder=5,
                marker="o",
                edgecolors=edge_colors,
                linewidths=1.2,
                label="Frames used in evaluation (Valence)",
            )

            if any(showcase_flags):
                showcase_x = [frame for frame, show in zip(selected_frames, showcase_flags) if show]
                showcase_arousal = [val for val, show in zip(selected_arousal.tolist(), showcase_flags) if show]
                showcase_valence = [val for val, show in zip(selected_valence.tolist(), showcase_flags) if show]

                ax.scatter(
                    showcase_x,
                    showcase_arousal,
                    facecolors="none",
                    edgecolors="#f1c40f",
                    s=120,
                    linewidths=1.4,
                    zorder=6,
                    label="Example frames",
                )
                ax.scatter(
                    showcase_x,
                    showcase_valence,
                    facecolors="none",
                    edgecolors="#f1c40f",
                    s=120,
                    linewidths=1.4,
                    zorder=6,
                )

            selected_accuracy = 100.0 * (float(np.sum(correct_flags)) / float(len(correct_flags)))
            selected_accuracy_text = f"{selected_accuracy:.1f}%"

    ax.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frame Index", fontsize=12)
    video_id = os.path.splitext(os.path.basename(str(video_name)))[0] if video_name else "unknown"
    ax.set_title(
        f"Video ID: {video_id} | Accuracy: {selected_accuracy_text} | Arousal & Valence Timeline",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    correct_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor="#1e8449",
        markeredgewidth=1.6,
        markersize=8,
        label="Correct prediction",
    )
    incorrect_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor="#c0392b",
        markeredgewidth=1.6,
        markersize=8,
        label="Incorrect prediction",
    )
    existing_labels = set(legend_labels)
    if correct_handle.get_label() not in existing_labels:
        legend_handles.append(correct_handle)
        legend_labels.append(correct_handle.get_label())
    if incorrect_handle.get_label() not in existing_labels:
        legend_handles.append(incorrect_handle)
        legend_labels.append(incorrect_handle.get_label())
    ax.legend(legend_handles, legend_labels, loc="upper left", fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_arousal_valence_quadrants(arousal_vals, valence_vals, labels, predictions, output_path=None, threshold=0.0):
    fig, ax = plt.subplots(figsize=(14, 12))

    colors = []
    for true_label, pred_label in zip(labels, predictions):
        if true_label == pred_label:
            colors.append("green")
        else:
            colors.append("red")

    scatter = ax.scatter(
        valence_vals, arousal_vals, c=colors, s=90, alpha=0.55, edgecolors="black", linewidth=1.2, zorder=8
    )

    ax.axhline(y=threshold, color="black", linestyle="-", linewidth=3, label="Classification Threshold")
    ax.axvline(x=threshold, color="black", linestyle="-", linewidth=3)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    ax.set_xlabel("Valence (Unpleasant ← → Pleasant)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Arousal (Low ← → High)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Arousal-Valence Space: Model Predictions",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.text(
        0.55,
        0.55,
        "Q1\nHigh Arousal\n+ High Valence\n\n(Excited, Happy)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        zorder=30,
        bbox=dict(boxstyle="round", facecolor="#2ecc71", alpha=0.4, edgecolor="black", linewidth=2),
    )

    ax.text(
        -0.55,
        0.55,
        "Q2\nHigh Arousal\n+ Low Valence\n\n(Angry, Stressed)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        zorder=30,
        bbox=dict(boxstyle="round", facecolor="#e74c3c", alpha=0.4, edgecolor="black", linewidth=2),
    )

    ax.text(
        -0.55,
        -0.55,
        "Q4\nLow Arousal\n+ Low Valence\n\n(Sad, Bored)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        zorder=30,
        bbox=dict(boxstyle="round", facecolor="#95a5a6", alpha=0.4, edgecolor="black", linewidth=2),
    )

    ax.text(
        0.55,
        -0.55,
        "Q3\nLow Arousal\n+ High Valence\n\n(Calm, Content)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        zorder=30,
        bbox=dict(boxstyle="round", facecolor="#3498db", alpha=0.4, edgecolor="black", linewidth=2),
    )

    correct_count = sum(color == "green" for color in colors)
    incorrect_count = sum(color == "red" for color in colors)
    correct_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="green",
        markeredgecolor="black",
        markersize=10,
        label=f"Correct prediction ({correct_count})",
    )
    incorrect_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=10,
        label=f"Incorrect prediction ({incorrect_count})",
    )
    ax.legend(handles=[correct_handle, incorrect_handle], loc="upper right", fontsize=12, framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle="--")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_veatic_visualizations(
        val_data,
        predictions,
        true_labels,
        label_map,
        output_dir,
        dataset_path,
        max_samples=10,
        model=None,
        selected_videos=None,
        selected_sample_indices=None,
):
    os.makedirs(output_dir, exist_ok=True)

    X_val, y_val, val_debugs = val_data
    selected_sample_indices = set(selected_sample_indices or [])

    frame_arousal_vals = []
    frame_valence_vals = []
    frame_true_labels = []
    frame_pred_labels = []

    video_groups = {}
    for idx, debug in enumerate(val_debugs):
        if not isinstance(debug, dict):
            continue
        video_name = debug.get("video")
        if not video_name:
            continue
        video_groups.setdefault(video_name, []).append(idx)

    sample_count = 0

    for video_name, indices in video_groups.items():
        should_generate_timeline = False
        if selected_videos is not None:
            should_generate_timeline = video_name in selected_videos
        else:
            should_generate_timeline = sample_count < max_samples

        video_id = video_name.replace(".mp4", "")
        arousal_path = os.path.join(dataset_path, "rating_averaged", f"{video_id}_arousal.csv")
        valence_path = os.path.join(dataset_path, "rating_averaged", f"{video_id}_valence.csv")

        if not (os.path.exists(arousal_path) and os.path.exists(valence_path)):
            continue

        arousal_seq = read_veatic_sequence(arousal_path)
        valence_seq = read_veatic_sequence(valence_path)

        if len(arousal_seq) == 0 or len(valence_seq) == 0:
            continue

        selected_frames = []
        selected_frame_correct = {}
        showcase_frames = set()
        for sample_idx in indices:
            if sample_idx >= len(predictions) or sample_idx >= len(true_labels):
                continue
            debug = val_debugs[sample_idx]

            frame_idx = debug.get("frame_idx") if isinstance(debug, dict) else None
            if frame_idx is None and isinstance(debug, dict):
                frames_list = debug.get("frames", [])
                if isinstance(frames_list, list) and len(frames_list) > 0:
                    frame_idx = int(frames_list[len(frames_list) // 2])

            if frame_idx is None:
                continue

            frame_idx = int(frame_idx)
            if frame_idx < 0 or frame_idx >= len(arousal_seq) or frame_idx >= len(valence_seq):
                continue

            selected_frames.append(frame_idx)
            if sample_idx in selected_sample_indices:
                showcase_frames.add(frame_idx)

            true_idx = int(true_labels[sample_idx])
            pred_idx = int(predictions[sample_idx])
            selected_frame_correct[frame_idx] = bool(true_idx == pred_idx)

            frame_arousal_vals.append(float(arousal_seq[frame_idx]))
            frame_valence_vals.append(float(valence_seq[frame_idx]))
            frame_true_labels.append(true_idx)
            frame_pred_labels.append(pred_idx)

        if should_generate_timeline and selected_frames:
            output_path = os.path.join(output_dir, f"timeline_{video_id}.png")
            plot_veatic_timeline(
                arousal_seq,
                valence_seq,
                video_name,
                selected_frames=selected_frames,
                selected_frame_correct=selected_frame_correct,
                showcase_frames=showcase_frames,
                output_path=output_path,
            )

            sample_count += 1

    quadrant_path = os.path.join(output_dir, "arousal_valence_quadrants.png")
    if len(frame_arousal_vals) > 0:
        plot_arousal_valence_quadrants(
            np.array(frame_arousal_vals, dtype=np.float32),
            np.array(frame_valence_vals, dtype=np.float32),
            np.array(frame_true_labels, dtype=np.int32),
            np.array(frame_pred_labels, dtype=np.int32),
            output_path=quadrant_path,
        )
        print(f"Created quadrant plot: {quadrant_path}")

    print(f"\nCreated {sample_count} timeline visualizations in {output_dir}")

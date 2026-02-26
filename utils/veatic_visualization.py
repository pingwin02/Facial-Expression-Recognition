import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
import os


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


def get_quadrant_label(quadrant):
    labels = {
        "high_arousal_high_valence": "High Arousal + High Valence\n(Excited, Happy)",
        "high_arousal_low_valence": "High Arousal + Low Valence\n(Angry, Stressed)",
        "low_arousal_high_valence": "Low Arousal + High Valence\n(Calm, Content)",
        "low_arousal_low_valence": "Low Arousal + Low Valence\n(Sad, Bored)",
    }
    return labels.get(quadrant, quadrant)


def get_quadrant_code(quadrant):
    codes = {
        "high_arousal_high_valence": "Q1",
        "high_arousal_low_valence": "Q2",
        "low_arousal_high_valence": "Q3",
        "low_arousal_low_valence": "Q4",
    }
    return codes.get(quadrant, quadrant)


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
    true_label,
    pred_label,
    video_name,
    selected_frames=None,
    selected_frame_preds=None,
    selected_frame_true=None,
    output_path=None,
    threshold=0.0,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    frames = np.arange(len(arousal_seq))

    true_arousal_mean = np.mean(arousal_seq)
    true_valence_mean = np.mean(valence_seq)
    pred_code = get_quadrant_code(pred_label)
    pred_color = get_quadrant_color(pred_label)

    # Arousal plot
    ax1.plot(frames, arousal_seq, "b-", linewidth=1.5, alpha=0.7, label="Arousal per frame")
    ax1.axhline(
        y=true_arousal_mean, color="b", linestyle="--", linewidth=2.5, label=f"Mean Arousal: {true_arousal_mean:.3f}"
    )
    ax1.axhline(y=threshold, color="gray", linestyle=":", linewidth=2, label=f"Threshold: {threshold}")

    marker_edges = None
    if selected_frames is not None and len(selected_frames) > 0:
        selected_arousal = [arousal_seq[f] if f < len(arousal_seq) else 0 for f in selected_frames]
        marker_edges = []
        for frame_idx in selected_frames:
            pred_name = selected_frame_preds.get(frame_idx) if selected_frame_preds else None
            true_name = selected_frame_true.get(frame_idx) if selected_frame_true else None
            marker_edges.append("green" if pred_name and true_name and pred_name == true_name else "red")

        ax1.scatter(
            selected_frames,
            selected_arousal,
            color="#f39c12",
            s=120,
            zorder=5,
            marker="o",
            edgecolors=marker_edges,
            linewidths=2,
            label="Frames fed to model",
        )

        # Per-frame labels removed by request

    ax1.set_ylabel("Arousal", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Frame Index", fontsize=12)
    ax1.set_title(f"Video: {video_name} - Arousal Over Time", fontsize=15, fontweight="bold")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)

    # Valence plot
    ax2.plot(frames, valence_seq, "g-", linewidth=1.5, alpha=0.7, label="Valence per frame")
    ax2.axhline(
        y=true_valence_mean, color="g", linestyle="--", linewidth=2.5, label=f"Mean Valence: {true_valence_mean:.3f}"
    )
    ax2.axhline(y=threshold, color="gray", linestyle=":", linewidth=2, label=f"Threshold: {threshold}")

    if selected_frames is not None and len(selected_frames) > 0:
        selected_valence = [valence_seq[f] if f < len(valence_seq) else 0 for f in selected_frames]
        ax2.scatter(
            selected_frames,
            selected_valence,
            color="#f39c12",
            s=120,
            zorder=5,
            marker="o",
            edgecolors=marker_edges if marker_edges is not None else "#d35400",
            linewidths=2,
            label="Frames fed to model",
        )

    ax2.set_ylabel("Valence", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Frame Index", fontsize=12)
    ax2.set_title(f"Video: {video_name} - Valence Over Time", fontsize=15, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)

    true_code = get_quadrant_code(true_label)
    true_color = get_quadrant_color(true_label)
    true_desc = get_quadrant_label(true_label)
    pred_desc = get_quadrant_label(pred_label)

    fig.text(
        0.98,
        0.08,
        f"True label: {true_code}\n{true_desc}",
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor=true_color, alpha=0.7, edgecolor="black", linewidth=1),
    )

    fig.text(
        0.98,
        0.02,
        f"Model prediction: {pred_code}\n{pred_desc}",
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor=pred_color, alpha=0.7, edgecolor="black", linewidth=1),
    )

    plt.tight_layout(rect=(0, 0.12, 1, 1))

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_arousal_valence_quadrants(arousal_vals, valence_vals, labels, predictions, output_path=None, threshold=0.0):
    fig, ax = plt.subplots(figsize=(14, 12))

    # Determine colors based on correctness
    colors = []
    for true_label, pred_label in zip(labels, predictions):
        if true_label == pred_label:
            colors.append("green")
        else:
            colors.append("red")

    # Plot points
    scatter = ax.scatter(
        valence_vals, arousal_vals, c=colors, s=180, alpha=0.7, edgecolors="black", linewidth=2, zorder=10
    )

    # Draw threshold lines
    ax.axhline(y=threshold, color="black", linestyle="-", linewidth=3, label="Classification Threshold")
    ax.axvline(x=threshold, color="black", linestyle="-", linewidth=3)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    ax.set_xlabel("Valence (Unpleasant ← → Pleasant)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Arousal (Low ← → High)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Arousal-Valence Space: Model Predictions\n(Each dot = one video, color = correct/incorrect)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Quadrant labels with better descriptions
    ax.text(
        0.55,
        0.55,
        "Q1\nHigh Arousal\n+ High Valence\n\n(Excited, Happy)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
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
        bbox=dict(boxstyle="round", facecolor="#3498db", alpha=0.4, edgecolor="black", linewidth=2),
    )

    # Legend (dots only)
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
):
    os.makedirs(output_dir, exist_ok=True)

    X_val, y_val, val_debugs = val_data

    inv_label_map = {v: k for k, v in label_map.items()}

    arousal_vals = []
    valence_vals = []

    sample_count = 0
    frame_label_timeline_created = False
    processed_videos = set()

    for idx, debug in enumerate(val_debugs):
        if not isinstance(debug, dict):
            continue

        video_name = debug.get("video", "")
        if not video_name or video_name in processed_videos:
            continue

        video_id = video_name.replace(".mp4", "")
        arousal_path = os.path.join(dataset_path, "rating_averaged", f"{video_id}_arousal.csv")
        valence_path = os.path.join(dataset_path, "rating_averaged", f"{video_id}_valence.csv")

        if not (os.path.exists(arousal_path) and os.path.exists(valence_path)):
            continue

        arousal_seq = read_veatic_sequence(arousal_path)
        valence_seq = read_veatic_sequence(valence_path)

        if len(arousal_seq) == 0 or len(valence_seq) == 0:
            continue

        true_label_idx = true_labels[idx]
        pred_label_idx = predictions[idx]

        true_label = inv_label_map.get(true_label_idx, "unknown")
        pred_label = inv_label_map.get(pred_label_idx, "unknown")

        arousal_vals.append(np.mean(arousal_seq))
        valence_vals.append(np.mean(valence_seq))

        # Determine if timeline should be generated
        should_generate_timeline = False
        if selected_videos is not None:
            should_generate_timeline = video_name in selected_videos
        else:
            should_generate_timeline = sample_count < max_samples

        if should_generate_timeline:
            selected_frames = debug.get("frames", [])
            selected_frame_preds = {}
            selected_frame_true = {}

            if model is not None and selected_frames and idx < len(X_val):
                sequence = X_val[idx]
                if isinstance(sequence, np.ndarray) and sequence.ndim == 4:
                    seq_len = sequence.shape[0]
                    for pos_idx, frame_idx in enumerate(selected_frames):
                        if pos_idx >= seq_len:
                            continue
                        frame = sequence[pos_idx].astype("float32")
                        if np.max(frame) > 1.0:
                            frame = frame / 255.0
                        tiled = np.repeat(frame[np.newaxis, ...], seq_len, axis=0)
                        batch = np.expand_dims(tiled, axis=0)
                        preds = model.predict(batch, verbose=0)
                        pred_idx = int(np.argmax(preds, axis=1)[0])
                        selected_frame_preds[frame_idx] = inv_label_map.get(pred_idx, str(pred_idx))

            if selected_frames:
                for frame_idx in selected_frames:
                    if frame_idx >= len(arousal_seq) or frame_idx >= len(valence_seq):
                        continue
                    true_frame_label = get_quadrant_from_av(
                        float(arousal_seq[frame_idx]),
                        float(valence_seq[frame_idx]),
                        threshold=0.0,
                    )
                    selected_frame_true[frame_idx] = true_frame_label

            output_path = os.path.join(output_dir, f"timeline_{video_id}.png")
            plot_veatic_timeline(
                arousal_seq,
                valence_seq,
                true_label,
                pred_label,
                video_name,
                selected_frames=selected_frames,
                selected_frame_preds=selected_frame_preds,
                selected_frame_true=selected_frame_true,
                output_path=output_path,
            )

            if not frame_label_timeline_created:
                labels_timeline_path = os.path.join(output_dir, f"timeline_labels_{video_id}.png")
                plot_veatic_frame_label_timeline(
                    arousal_seq,
                    valence_seq,
                    video_name,
                    output_path=labels_timeline_path,
                    threshold=0.0,
                )
                frame_label_timeline_created = True

            sample_count += 1
            processed_videos.add(video_name)

    quadrant_path = os.path.join(output_dir, "arousal_valence_quadrants.png")
    plot_arousal_valence_quadrants(
        np.array(arousal_vals),
        np.array(valence_vals),
        true_labels[: len(arousal_vals)],
        predictions[: len(arousal_vals)],
        output_path=quadrant_path,
    )

    print(f"\nCreated {sample_count} timeline visualizations in {output_dir}")
    if frame_label_timeline_created:
        print("Created 1 frame-label timeline visualization for VEATIC test.")
    print(f"Created quadrant plot: {quadrant_path}")

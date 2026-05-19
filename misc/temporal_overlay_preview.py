import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.processors import temporal_encoding_preview_from_video, NUM_SAMPLE_FRAMES, CENTER_IDX


def _draw_frame(ax, image, title, fontsize=9):
    if image is None:
        ax.text(0.5, 0.5, "no face", ha="center", va="center", fontsize=fontsize)
        ax.set_facecolor("#f2f2f2")
    else:
        ax.imshow(np.clip(image, 0.0, 1.0))
    ax.set_title(title, fontsize=fontsize, pad=4)
    ax.axis("off")


def save_preview_figure(overlay_pack, output_path):
    sample_indices = overlay_pack["sample_indices"]
    sampled_faces = overlay_pack["sampled_faces"]
    diff_layers = overlay_pack["diff_layers"]
    base_gray = overlay_pack["base_gray"]
    encoded = overlay_pack["encoded_frame"]

    n = NUM_SAMPLE_FRAMES
    fig = plt.figure(figsize=(2.8 * n, 8.5))
    gs = fig.add_gridspec(3, n, height_ratios=[1.0, 1.0, 1.5], hspace=0.22, wspace=0.06)

    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        label = f"frame {sample_indices[i]}"
        if i == CENTER_IDX:
            label += " (base)"
        _draw_frame(ax, sampled_faces[i], label)

    for i in range(n):
        ax = fig.add_subplot(gs[1, i])
        if i == CENTER_IDX:
            _draw_frame(ax, base_gray, "grayscale base")
        else:
            _draw_frame(ax, diff_layers[i], f"diff shadow {i}")

    ax_result = fig.add_subplot(gs[2, 1 : n - 1])
    ax_result.imshow(np.clip(encoded, 0.0, 1.0))
    ax_result.set_title("composed result", fontsize=11, fontweight="bold", pad=6)
    ax_result.axis("off")

    fig.suptitle("Temporal Chromatic Encoding", fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved temporal encoding preview to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preview temporal chromatic encoding for a single video file.")
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output PNG preview.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    overlay_pack = temporal_encoding_preview_from_video(args.video)
    if args.output is None:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join("output", "temporal_previews", f"{video_name}.png")
    save_preview_figure(overlay_pack, args.output)


if __name__ == "__main__":
    main()

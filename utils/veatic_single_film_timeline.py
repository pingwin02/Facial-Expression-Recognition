import argparse
import os

from veatic_visualization import (
    plot_veatic_frame_label_timeline,
    read_veatic_sequence,
)


def _find_first_video_id(rating_dir):
    candidates = []
    for name in os.listdir(rating_dir):
        if name.endswith("_arousal.csv"):
            candidates.append(name[: -len("_arousal.csv")])
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (len(x), x))[0]


def main():
    parser = argparse.ArgumentParser(description="Generate VEATIC frame-label timeline for a single film")
    parser.add_argument("--dataset-path", type=str, default="input/veatic", help="Path to VEATIC dataset directory")
    parser.add_argument("--video-id", type=str, default=None, help="Video id (e.g. 0, 1, 100)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/veatic_single_test",
        help="Directory where timeline image will be saved",
    )
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for arousal/valence quadrant labeling")

    args = parser.parse_args()

    rating_dir = os.path.join(args.dataset_path, "rating_averaged")
    if not os.path.isdir(rating_dir):
        raise FileNotFoundError(f"Missing rating directory: {rating_dir}")

    video_id = args.video_id or _find_first_video_id(rating_dir)
    if video_id is None:
        raise RuntimeError("No VEATIC rating CSV files found.")

    arousal_path = os.path.join(rating_dir, f"{video_id}_arousal.csv")
    valence_path = os.path.join(rating_dir, f"{video_id}_valence.csv")

    arousal_seq = read_veatic_sequence(arousal_path)
    valence_seq = read_veatic_sequence(valence_path)

    if len(arousal_seq) == 0 or len(valence_seq) == 0:
        raise RuntimeError(f"No values in CSV files for video_id={video_id}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"timeline_{video_id}.png")

    plot_veatic_frame_label_timeline(
        arousal_seq=arousal_seq,
        valence_seq=valence_seq,
        video_name=f"{video_id}.mp4",
        output_path=output_path,
        threshold=args.threshold,
    )

    print(f"Saved timeline for video_id={video_id}: {output_path}")


if __name__ == "__main__":
    main()

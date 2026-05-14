import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dataset.downloader import ensure_dataset
from dataset.loader import build_cache_version, load_data

import sys
import argparse

from utils.model_io import load_model_class, prepare_output_directory, cleanup_empty_dirs


def main():
    parser = argparse.ArgumentParser(description="Facial Expression Recognition Training/Evaluation")
    parser.add_argument("--model", type=str, help="Model class name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (train mode only)")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        choices=["devemo", "devemo+", "fer2013", "veatic"],
        help="Input format/folder",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument(
        "--train-frame-selection",
        type=str,
        default="uniform",
        choices=["uniform", "transformer", "random", "manual_uniform", "manual_random", "manual_transformer"],
        help="Frame selection method for training (devemo/devemo+ only)",
    )
    parser.add_argument(
        "--test-frame-selection",
        type=str,
        default=None,
        choices=["uniform", "transformer", "random", "manual_uniform", "manual_random", "manual_transformer"],
        help="Frame selection method for testing (devemo/devemo+ only, defaults to train method)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of frames to select per video (default: 5)",
    )
    parser.add_argument(
        "--class-split",
        type=str,
        default="binary",
        choices=["binary", "all"],
        help="Class split: 'binary' (negative/others) or 'all' (all original classes)",
    )
    args = parser.parse_args()

    if args.test_frame_selection is None:
        args.test_frame_selection = args.train_frame_selection

    cache_version = build_cache_version(
        input_flag=args.input,
        train_frame_selection=args.train_frame_selection,
        test_frame_selection=args.test_frame_selection,
        num_frames=args.num_frames,
        class_split=args.class_split,
    )

    INPUT_DIR = "input"

    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print(f"  cache_version: {cache_version}")

    ensure_dataset(INPUT_DIR, dataset_name=args.input)

    (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), (X_test, y_test, test_debugs), label_map = load_data(
        INPUT_DIR,
        args.input,
        no_cache=args.no_cache,
        train_frame_selection=args.train_frame_selection,
        test_frame_selection=args.test_frame_selection,
        num_frames=args.num_frames,
        class_split=args.class_split,
        cache_version=cache_version,
    )

    try:
        model_class = load_model_class(args.model)
        model = model_class()
    except (AttributeError, ImportError):
        raise ValueError(f"Unknown model: {args.model}")
    except TypeError as e:
        raise ValueError(str(e))

    try:
        if args.mode == "train":
            if args.epochs is None:
                print("Error: --epochs must be specified for training.")
                sys.exit(1)
            OUTPUT_DIR, MODEL_PATH = prepare_output_directory(
                model, args.mode, dataset=args.input, cache_version=cache_version
            )
            model.train(
                X_train,
                y_train,
                X_val,
                y_val,
                OUTPUT_DIR,
                MODEL_PATH,
                args.epochs,
                label_map=label_map,
                train_debugs=train_debugs,
                val_debugs=val_debugs,
                dataset_name=args.input,
                cache_label=cache_version,
            )

        if args.mode == "eval":
            if args.epochs is not None:
                print("Warning: --epochs argument is ignored in eval mode.")
            dataset_path = os.path.join(INPUT_DIR, args.input)
            model.eval(
                (X_test, y_test, test_debugs),
                label_map=label_map,
                dataset_name=args.input,
                dataset_path=dataset_path,
                train_tuple=(X_train, y_train, train_debugs),
                cache_label=cache_version,
            )
    finally:
        cleanup_empty_dirs()


if __name__ == "__main__":
    main()

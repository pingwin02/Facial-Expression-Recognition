import os

from dataset.utils import join_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dataset.downloader import ensure_dataset
from dataset.loader import load_data

import sys
import argparse

from utils.model_io import load_model_class, prepare_output_directory


def main():
    parser = argparse.ArgumentParser(description="Facial Expression Recognition Training/Evaluation")
    parser.add_argument("--model", type=str, help="Model class name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (train mode only)")
    parser.add_argument(
        "--input", type=str, required=True, choices=["devemo", "devemo+", "fer2013"], help="Input format/folder"
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    args = parser.parse_args()

    INPUT_DIR = "input"

    try:
        model_class = load_model_class(args.model)
        model = model_class()
    except (AttributeError, ImportError):
        raise ValueError(f"Unknown model: {args.model}")
    except TypeError as e:
        raise ValueError(str(e))

    ensure_dataset(INPUT_DIR, dataset_name=args.input)

    (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map = load_data(
        INPUT_DIR, args.input, no_cache=args.no_cache
    )

    OUTPUT_DIR, MODEL_PATH = prepare_output_directory(model, args.mode, dataset=args.input)

    if args.mode == "train":
        if args.epochs is None:
            print("Error: --epochs must be specified for training.")
            sys.exit(1)
        model.train(X_train, y_train, X_val, y_val, OUTPUT_DIR, MODEL_PATH, args.epochs, label_map=label_map)

    if args.mode == "eval":
        if args.epochs is not None:
            print("Warning: --epochs argument is ignored in eval mode.")
        merged = join_data([(X_train, y_train, train_debugs), (X_val, y_val, val_debugs)])
        model.eval(merged, OUTPUT_DIR, label_map=label_map, dataset_name=args.input)


if __name__ == "__main__":
    main()

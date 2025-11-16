import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import argparse
from utils.data import load_data
from utils.model_io import load_model_class, prepare_output_directory


def main():
    """Entry point for training or evaluation.

    Parses CLI arguments, loads data and model, and dispatches to training or
    evaluation routines depending on the --mode flag.
    """
    parser = argparse.ArgumentParser(description="Facial Expression Recognition Training/Evaluation")
    parser.add_argument("--model", type=str, help="Model class name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (train mode only)")
    parser.add_argument(
        "--input", type=str, default="devemo", choices=["devemo", "devemo+"], help="Input format/folder"
    )
    args = parser.parse_args()

    INPUT_DIR = "input"

    try:
        model_class = load_model_class(args.model)
        model = model_class()
    except (AttributeError, ImportError):
        raise ValueError(f"Unknown model: {args.model}")
    except TypeError as e:
        raise ValueError(str(e))

    (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map = load_data(INPUT_DIR, args.input)

    OUTPUT_DIR, MODEL_PATH = prepare_output_directory(model, args.mode, dataset=args.input)

    if args.mode == "train":
        if args.epochs is None:
            print("Error: --epochs must be specified for training.")
            sys.exit(1)
        model.train(X_train, y_train, X_val, y_val, OUTPUT_DIR, args.epochs)

    if args.mode == "eval":
        dataset_name = args.input
        model.eval(
            (X_val, y_val, val_debugs), y_val, OUTPUT_DIR, MODEL_PATH, label_map=label_map, dataset_name=dataset_name
        )


if __name__ == "__main__":
    main()

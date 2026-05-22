import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dataset.downloader import ensure_dataset
from dataset.loader import build_cache_version, load_data

from utils.cli import parse_args
from utils.model_io import load_model_class, prepare_output_directory, cleanup_empty_dirs


def run_once(args, mode):
    cache_version = build_cache_version(
        input_flag=args.input,
        train_frame_selection=args.train_frame_selection,
        test_frame_selection=args.test_frame_selection,
        num_frames=args.num_frames,
        class_split=args.class_split,
    )

    INPUT_DIR = "input"

    print(f"\n{'=' * 50}")
    print(f"  mode: {mode}")
    for arg in (
            "model",
            "input",
            "epochs",
            "train_frame_selection",
            "test_frame_selection",
            "num_frames",
            "class_split",
    ):
        print(f"  {arg}: {getattr(args, arg)}")
    print(f"  cache_version: {cache_version}")
    print(f"{'=' * 50}\n")

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
        if mode == "train":
            OUTPUT_DIR, MODEL_PATH = prepare_output_directory(
                model, mode, dataset=args.input, cache_version=cache_version
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

        if mode == "eval":
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


def main():
    args = parse_args()

    if args.mode == "both":
        modes = ["train", "eval"]
    else:
        modes = [args.mode]

    for loop_idx in range(1, args.loop + 1):
        if args.loop > 1:
            print(f"\n>>> Loop {loop_idx}/{args.loop}")
        for mode in modes:
            run_once(args, mode)


if __name__ == "__main__":
    main()

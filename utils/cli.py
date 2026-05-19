import argparse
import os
import sys

FRAME_SELECTION_METHODS = ["uniform", "transformer", "random", "manual_uniform", "manual_random", "manual_transformer"]
CLASS_SPLITS = ["binary", "all"]
MODES = ["train", "eval", "both"]

DEVEMO_INPUTS = {"devemo", "devemo+", "devemo_combined"}


def _discover_models():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    models = []
    for f in sorted(os.listdir(models_dir)):
        if f.endswith(".py") and not f.startswith("_"):
            models.append(f[:-3])
    return models


def _discover_inputs():
    input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")
    inputs = []
    for d in sorted(os.listdir(input_dir)):
        full = os.path.join(input_dir, d)
        if os.path.isdir(full) and d not in (".cache", "__pycache__", ".tmp"):
            inputs.append(d)
    inputs.append("devemo_combined")
    return inputs


def _prompt_choice(label, options, default_index=0):
    print(f"\n{label}:")
    for i, opt in enumerate(options):
        marker = " (default)" if i == default_index else ""
        print(f"  {i}) {opt}{marker}")
    while True:
        raw = input(f"Select [{default_index}]: ").strip()
        if raw == "":
            return options[default_index]
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(options):
                return options[idx]
        if raw in options:
            return raw
        print(f"Invalid selection. Enter 0-{len(options)-1} or a valid name.")


def _prompt_int(label, default):
    while True:
        raw = input(f"\n{label} [{default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit() and int(raw) >= 1:
            return int(raw)
        print("Invalid input. Enter an integer >= 1.")


def _resolve_arg(value, options, arg_name):
    """Resolve a CLI value that can be numeric index or string name."""
    if value is None:
        return None
    if value.isdigit():
        idx = int(value)
        if 0 <= idx < len(options):
            return options[idx]
        raise argparse.ArgumentTypeError(f"--{arg_name}: index {idx} out of range (0-{len(options)-1})")
    if value in options:
        return value
    raise argparse.ArgumentTypeError(f"--{arg_name}: '{value}' not in {options}")


def parse_args():
    available_models = _discover_models()
    available_inputs = _discover_inputs()

    parser = argparse.ArgumentParser(
        description="Facial Expression Recognition — Train & Evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Available models (use name or index):
  {', '.join(f'{i}={m}' for i, m in enumerate(available_models))}

Available inputs (use name or index):
  {', '.join(f'{i}={inp}' for i, inp in enumerate(available_inputs))}

Frame selection methods (use name or index):
  {', '.join(f'{i}={m}' for i, m in enumerate(FRAME_SELECTION_METHODS))}

Class splits (use name or index):
  {', '.join(f'{i}={c}' for i, c in enumerate(CLASS_SPLITS))}

Modes (use name or index):
  {', '.join(f'{i}={m}' for i, m in enumerate(MODES))}

Examples:
  python main.py --model 0 --input 4 --mode both --epochs 100
  python main.py --model ResNetModel --input devemo_combined --mode train --epochs 50
  python main.py  (interactive mode — prompts for all parameters)
""",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name or index")
    parser.add_argument("--input", type=str, default=None, help="Dataset name or index")
    parser.add_argument("--mode", type=str, default=None, help="Mode: train, eval, both (or index 0-2)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: 100)")
    parser.add_argument(
        "--train-frame-selection", type=str, default=None, help="Training frame selection method (name or index)"
    )
    parser.add_argument(
        "--test-frame-selection", type=str, default=None, help="Test frame selection method (name or index)"
    )
    parser.add_argument("--num-frames", type=int, default=None, help="Frames per video (default: 5)")
    parser.add_argument("--class-split", type=str, default=None, help="Class split: binary, all (or index 0-1)")
    parser.add_argument("--loop", type=int, default=None, help="Number of full run loops (default: 1)")
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset caching")

    args = parser.parse_args()

    interactive = sys.stdin.isatty() if hasattr(sys.stdin, "isatty") else False

    if args.model is not None:
        args.model = _resolve_arg(args.model, available_models, "model")
    elif interactive:
        args.model = _prompt_choice("Model", available_models, default_index=0)
    else:
        parser.error("--model is required in non-interactive mode")

    if args.input is not None:
        args.input = _resolve_arg(args.input, available_inputs, "input")
    elif interactive:
        args.input = _prompt_choice(
            "Input dataset",
            available_inputs,
            default_index=available_inputs.index("devemo_combined") if "devemo_combined" in available_inputs else 0,
        )
    else:
        parser.error("--input is required in non-interactive mode")

    if args.mode is not None:
        args.mode = _resolve_arg(args.mode, MODES, "mode")
    elif interactive:
        args.mode = _prompt_choice("Mode", MODES, default_index=2)
    else:
        args.mode = "both"

    if args.epochs is None:
        if interactive:
            args.epochs = _prompt_int("Number of epochs", 100)
        else:
            args.epochs = 100

    is_devemo = args.input in DEVEMO_INPUTS
    if is_devemo:
        if args.train_frame_selection is not None:
            args.train_frame_selection = _resolve_arg(
                args.train_frame_selection, FRAME_SELECTION_METHODS, "train-frame-selection"
            )
        elif interactive:
            args.train_frame_selection = _prompt_choice(
                "Training frame selection", FRAME_SELECTION_METHODS, default_index=0
            )
        else:
            args.train_frame_selection = "uniform"

        if args.test_frame_selection is not None:
            args.test_frame_selection = _resolve_arg(
                args.test_frame_selection, FRAME_SELECTION_METHODS, "test-frame-selection"
            )
        elif interactive:
            test_options = ["same_as_train"] + FRAME_SELECTION_METHODS
            choice = _prompt_choice("Test frame selection", test_options, default_index=0)
            args.test_frame_selection = args.train_frame_selection if choice == "same_as_train" else choice
        else:
            args.test_frame_selection = args.train_frame_selection

        if args.num_frames is None:
            if interactive:
                args.num_frames = _prompt_int("Number of frames per video", 5)
            else:
                args.num_frames = 5

        if args.class_split is not None:
            args.class_split = _resolve_arg(args.class_split, CLASS_SPLITS, "class-split")
        elif interactive:
            args.class_split = _prompt_choice("Class split", CLASS_SPLITS, default_index=0)
        else:
            args.class_split = "binary"
    else:
        args.train_frame_selection = args.train_frame_selection or "uniform"
        args.test_frame_selection = args.test_frame_selection or args.train_frame_selection
        args.num_frames = args.num_frames or 5
        if args.class_split is not None:
            args.class_split = _resolve_arg(args.class_split, CLASS_SPLITS, "class-split")
        else:
            args.class_split = "binary"

    if args.loop is None:
        if interactive:
            args.loop = _prompt_int("Number of loops", 1)
        else:
            args.loop = 1

    return args

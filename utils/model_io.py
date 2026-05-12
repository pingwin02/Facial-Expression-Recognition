import datetime
import glob
import os
import select
import shutil
import sys

import tensorflow as tf


def _input_with_timeout(prompt, timeout_seconds=5):
    print(prompt, end="", flush=True)

    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    except (OSError, ValueError):
        print()
        return None

    if not ready:
        print()
        return None

    try:
        return sys.stdin.readline().strip()
    except OSError:
        print()
        return None


def _is_interactive_stdin():
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def _path_has_dataset_segment(path, dataset_name):
    if not dataset_name:
        return True

    normalized = os.path.normpath(path).replace("\\", "/").lower()
    parts = [part for part in normalized.split("/") if part]
    return str(dataset_name).strip().lower() in parts


def _path_has_cache_segment(path, cache_version):
    if not cache_version:
        return True

    normalized = os.path.normpath(path).replace("\\", "/").lower()
    parts = [part for part in normalized.split("/") if part]
    return str(cache_version).strip().lower() in parts


def _path_timestamp_folder(path):
    return os.path.basename(os.path.dirname(path))


def find_and_load_model(model_prefix="SimpleModel", dataset_name=None, cache_version=None):
    if dataset_name and cache_version:
        print(
            f"Searching for models matching '{model_prefix}', dataset '{dataset_name}', and cache '{cache_version}'..."
        )
    elif dataset_name:
        print(f"Searching for models matching '{model_prefix}' and dataset '{dataset_name}'...")
    elif cache_version:
        print(f"Searching for models matching '{model_prefix}' and cache '{cache_version}'...")
    else:
        print(f"Searching for models matching '{model_prefix}'...")

    models = glob.glob(os.path.join(".", "**", "*.keras"), recursive=True)
    all_models = [path for path in models if "backup" not in path.lower()]

    candidates = []
    for path in all_models:
        normalized_path = path.lower()
        if (
            model_prefix.lower() in normalized_path
            and _path_has_dataset_segment(path, dataset_name)
            and _path_has_cache_segment(path, cache_version)
        ):
            candidates.append(path)

    candidates.sort(key=_path_timestamp_folder, reverse=True)

    if not candidates:
        if dataset_name and cache_version:
            print(
                "Error: No trained models found matching prefix "
                f"'{model_prefix}' for dataset '{dataset_name}' and cache '{cache_version}'."
            )
        elif dataset_name:
            print(f"Error: No trained models found matching prefix '{model_prefix}' for dataset '{dataset_name}'.")
        elif cache_version:
            print(f"Error: No trained models found matching prefix '{model_prefix}' for cache '{cache_version}'.")
        else:
            print(f"Error: No trained models found matching prefix '{model_prefix}'.")
        return None, None

    selected_path = candidates[0]

    if len(candidates) > 1:
        print(f"\nMultiple matching models found:")
        for i, path in enumerate(candidates):
            print(f"[{i}] {path}")

        if not _is_interactive_stdin():
            print("Non-interactive stdin detected; selecting default model index 0.")
            selected_path = candidates[0]
        else:
            while True:
                user_input = _input_with_timeout(
                    f"\nSelect model index to load (0-{len(candidates) - 1}) [default: 0, timeout: 5s]: ",
                    timeout_seconds=5,
                )
                if user_input is None or user_input == "":
                    index = 0
                    selected_path = candidates[index]
                    break

                if user_input.isdigit():
                    index = int(user_input)
                    if 0 <= index < len(candidates):
                        selected_path = candidates[index]
                        break

                print("Invalid selection. Please try again.")

    print(f"Loading trained model from: {selected_path}")
    model_dir = os.path.dirname(selected_path)
    try:
        loaded_model = tf.keras.models.load_model(selected_path, compile=False)
        return loaded_model, model_dir
    except Exception as e:
        print(f"Standard load failed: {e}")
        print("Retrying model load with safe_mode=False for trusted local artifact...")
        try:
            loaded_model = tf.keras.models.load_model(selected_path, safe_mode=False, compile=False)
            return loaded_model, model_dir
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return None, None


def load_model_class(model_name):
    try:
        if "." in model_name:
            module_name, class_name = model_name.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
        else:
            module = __import__("models." + model_name, fromlist=[model_name])
            class_name = model_name

        model_class = getattr(module, class_name, None)
        if model_class is None or not callable(model_class):
            raise TypeError(f"The specified model '{model_name}' is not a class or cannot be instantiated.")
        print(f"Resolved model class: {model_class}")
        return model_class
    except (AttributeError, ImportError):
        raise ValueError(f"Unknown model: {model_name}")
    except TypeError as e:
        raise ValueError(str(e))


def cleanup_empty_dirs(output_root="output"):
    if os.path.exists(output_root):
        for root, dirs, files in os.walk(output_root, topdown=False):
            for d in dirs:
                dir_path = os.path.join(root, d)
                if os.path.isdir(dir_path) and not os.listdir(dir_path):
                    shutil.rmtree(dir_path)


def prepare_output_directory(model, mode, output_root="output", dataset=None, cache_version=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = model.__class__.__name__
    data_part = dataset if dataset is not None else "unknown"
    cache_part = cache_version if cache_version else "default"

    run_dir = timestamp
    output_dir = os.path.join(output_root, cache_part, model_name, data_part, run_dir)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.keras")

    return output_dir, model_path

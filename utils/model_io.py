import datetime
import glob
import os
import shutil

import tensorflow as tf


def find_and_load_model(model_prefix="SimpleModel"):
    """
    Locate and load a Keras model from disk based on prefix.

    Searches recursively for *.keras files. If multiple files match the
    folder/filename criteria, prompts the user to select one.

    Args:
        model_prefix (str): Substring expected in the file path (e.g., model architecture).

    Returns:
        A loaded Keras model instance or None if not found.
    """
    print(f"Searching for models matching '{model_prefix}'...")

    all_models = glob.glob(os.path.join(".", "**", "*.keras"), recursive=True)

    candidates = []
    for path in all_models:
        normalized_path = path.lower()
        if model_prefix.lower() in normalized_path in normalized_path:
            candidates.append(path)

    candidates.sort(key=os.path.getmtime, reverse=True)

    if not candidates:
        print(f"Error: No trained models found matching prefix '{model_prefix}'.")
        return None

    selected_path = candidates[0]

    if len(candidates) > 1:
        print(f"\nMultiple matching models found:")
        for i, path in enumerate(candidates):
            print(f"[{i}] {path}")

        while True:
            user_input = input(f"\nSelect model index to load (0-{len(candidates) - 1}) [default: 0]: ").strip()
            if user_input == "":
                index = 0
                break

            if user_input.isdigit():
                index = int(user_input)
                if 0 <= index < len(candidates):
                    selected_path = candidates[index]
                    break

            print("Invalid selection. Please try again.")

    print(f"Loading trained model from: {selected_path}")
    try:
        loaded_model = tf.keras.models.load_model(selected_path)
        return loaded_model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


def load_model_class(model_name):
    """Resolve and return a model class by name.

    The function supports either 'Module.ClassName' or just 'ClassName' where the
    latter will be imported from the 'models' package.

    Args:
        model_name (str): Name of the model class to import.

    Returns:
        The model class object.

    Raises:
        ValueError: if the model cannot be found or is not instantiable.
    """
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


def prepare_output_directory(model, mode, output_root="output", dataset=None):
    """Prepare output directory for a run and return its path and expected model path.

    Args:
        model: Instantiated model object used to name the output directory.
        mode (str): Mode of operation (e.g. 'train' or 'eval').
        output_root (str): Root folder in which to create subdirectories.

    Returns:
        tuple: (output_dir, model_path) where model_path is the expected filename for saving the model.
    """
    if os.path.exists(output_root):
        for root, dirs, files in os.walk(output_root, topdown=False):
            for d in dirs:
                dir_path = os.path.join(root, d)
                if os.path.isdir(dir_path) and not os.listdir(dir_path):
                    shutil.rmtree(dir_path)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    data_part = dataset if dataset is not None else "unknown"
    output_subdir = f"{model.__class__.__name__}_{data_part}_{timestamp}_{mode}"
    output_dir = os.path.join(output_root, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model.__class__.__name__}_model.keras")

    return output_dir, model_path

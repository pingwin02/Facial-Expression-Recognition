import datetime
import glob
import os
import shutil

import tensorflow as tf


def find_and_load_model(model_path, model_prefix="SimpleModel"):
    """Locate and load a Keras model from disk.

    If `model_path` does not exist, search upward for a matching '*model.keras' file
    whose filename contains `model_prefix`.

    Args:
        model_path (str): Expected path to the saved model.
        model_prefix (str): Substring to match model filenames when searching.

    Returns:
        A loaded Keras model instance or None if not found.
    """
    if not os.path.exists(model_path):
        model_dir = os.path.dirname(model_path)
        base_dir = os.path.dirname(model_dir)
        candidates = sorted(glob.glob(os.path.join(base_dir, "**", "*model.keras"), recursive=True), reverse=True)
        found_model_path = None
        for candidate in candidates:
            if model_prefix.lower() in os.path.basename(candidate).lower():
                found_model_path = candidate
                break
        if found_model_path:
            model_path = found_model_path
            print(f"Loading {model_path} for evaluation...")
        else:
            print(f"Error: trained model not found")
            return None
    print("Loading trained model for evaluation...")
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model


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

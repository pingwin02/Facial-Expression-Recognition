import numpy as np
import os
import pickle

from dataset.sources.registry import get_dataset_source
from dataset.utils import print_stats

CACHE_VERSION = "video_v13_veatic_landmarks_rgb"


def load_data(
    input_dir,
    input_flag="devemo",
    seed=42,
    cache_dir="input/.cache",
    no_cache=False,
):
    os.makedirs(cache_dir, exist_ok=True)
    cache_input_name = input_flag.replace("+", "plus")
    cache_filename = f"{cache_input_name}_seed{seed}_{CACHE_VERSION}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path) and not no_cache:
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
            print(f"{input_flag} dataset loaded from cache.")
            print_stats("X_train", result[0][0])
            print_stats("X_val", result[1][0])
            print_stats("y_train", result[0][1], result[2])
            print_stats("y_val", result[1][1], result[2])
            print_stats("split", None, result[2], split_arrays=(result[0][1], result[1][1]))
            return result

    else:
        print(f"No cache found at {cache_path}, loading data from disk...")

    print(f"Loading {input_flag} dataset...")
    np.random.seed(seed)

    source = get_dataset_source(input_flag=input_flag, input_dir=input_dir)
    (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map = source.load(seed=seed)

    print(f"{input_flag} dataset loaded from disk.")
    print_stats("X_train", X_train)
    print_stats("X_val", X_val)
    print_stats("y_train", y_train, label_map)
    print_stats("y_val", y_val, label_map)
    print_stats("split", None, label_map, split_arrays=(y_train, y_val))

    result = ((X_train, y_train, train_debugs), (X_val, y_val, val_debugs), label_map)

    print(f"Saving cache to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result

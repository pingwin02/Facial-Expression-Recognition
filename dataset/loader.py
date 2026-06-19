import numpy as np
import os
import pickle

from dataset.sources.registry import get_dataset_source
from dataset.utils import print_stats

CACHE_VERSION = "v25"


def _normalize_cache_token(value):
    return str(value).strip().replace("+", "plus").replace("/", "-").replace(" ", "_")


def build_cache_version(
        input_flag="devemo",
        train_frame_selection="uniform",
        test_frame_selection=None,
        num_frames=5,
        class_split="binary",
):
    if test_frame_selection is None:
        test_frame_selection = train_frame_selection

    resolved_num_frames = max(1, int(num_frames))

    if input_flag == "veatic":
        return f"{CACHE_VERSION}" f"__nf-{resolved_num_frames}"

    if input_flag not in ("devemo", "devemo+", "devemo_combined"):
        return CACHE_VERSION

    return (
        f"{CACHE_VERSION}"
        f"__tr-{_normalize_cache_token(train_frame_selection)}"
        f"__te-{_normalize_cache_token(test_frame_selection)}"
        f"__nf-{resolved_num_frames}"
        f"__cls-{_normalize_cache_token(class_split)}"
    )


def _supports_directory_cache(input_flag):
    return input_flag == "veatic"


def _directory_cache_path(cache_dir, cache_input_name, seed, cache_version):
    return os.path.join(cache_dir, f"{cache_input_name}_seed{seed}_{cache_version}")


def _directory_cache_split_paths(cache_path, split_name):
    base_path = os.path.join(cache_path, f"{split_name}_224x224")
    return {
        "X": f"{base_path}_X.npy",
        "y": f"{base_path}_y.npy",
        "meta": f"{base_path}_meta.pkl",
        "debugs": f"{base_path}_debugs.pkl",
    }


def _load_directory_cache_split(cache_path, split_name):
    split_paths = _directory_cache_split_paths(cache_path, split_name)

    with open(split_paths["meta"], "rb") as f:
        meta = pickle.load(f)

    sample_count = int(meta.get("sample_count", 0))
    X = np.load(split_paths["X"], mmap_mode="r")[:sample_count]
    y = np.load(split_paths["y"], mmap_mode="r")[:sample_count]

    if os.path.exists(split_paths["debugs"]):
        with open(split_paths["debugs"], "rb") as f:
            debugs = pickle.load(f)
    else:
        debugs = []

    return X, y, debugs


def _load_directory_cache(cache_path):
    with open(os.path.join(cache_path, "label_map.pkl"), "rb") as f:
        label_map = pickle.load(f)

    train = _load_directory_cache_split(cache_path, "train")
    val = _load_directory_cache_split(cache_path, "val")
    test = _load_directory_cache_split(cache_path, "test")
    return train, val, test, label_map


def _is_complete_directory_cache(cache_path):
    if not cache_path or not os.path.isdir(cache_path):
        return False

    required_paths = [os.path.join(cache_path, "label_map.pkl")]
    for split_name in ("train", "val", "test"):
        split_paths = _directory_cache_split_paths(cache_path, split_name)
        required_paths.extend([split_paths["X"], split_paths["y"], split_paths["meta"]])

    return all(os.path.exists(path) for path in required_paths)


def load_data(
        input_dir,
        input_flag="devemo",
        seed=42,
        cache_dir="input/.cache",
        no_cache=False,
        train_frame_selection="uniform",
        test_frame_selection=None,
        num_frames=5,
        class_split="binary",
        cache_version=None,
):
    if test_frame_selection is None:
        test_frame_selection = train_frame_selection

    if cache_version is None:
        cache_version = build_cache_version(
            input_flag=input_flag,
            train_frame_selection=train_frame_selection,
            test_frame_selection=test_frame_selection,
            num_frames=num_frames,
            class_split=class_split,
        )

    os.makedirs(cache_dir, exist_ok=True)
    cache_input_name = input_flag.replace("+", "plus")

    if _supports_directory_cache(input_flag):
        directory_cache_path = _directory_cache_path(cache_dir, cache_input_name, seed, cache_version)

        if _is_complete_directory_cache(directory_cache_path) and not no_cache:
            print(f"Loading cached data from {directory_cache_path}...")
            result = _load_directory_cache(directory_cache_path)
            print(f"{input_flag} dataset loaded from cache.")
            print_stats("X_train", result[0][0])
            print_stats("X_val", result[1][0])
            print_stats("X_test", result[2][0])
            print_stats("y_train", result[0][1], result[3])
            print_stats("y_val", result[1][1], result[3])
            print_stats("y_test", result[2][1], result[3])
            print_stats("split", None, result[3], split_arrays=(result[0][1], result[1][1], result[2][1]))
            return result

        if no_cache and os.path.isdir(directory_cache_path):
            print(f"Ignoring cached data at {directory_cache_path} and rebuilding from disk/checkpoints...")
        elif os.path.isdir(directory_cache_path):
            print(f"Incomplete cache found at {directory_cache_path}, resuming build from disk artifacts.")
        else:
            print(f"No cache found at {directory_cache_path}, loading data from disk...")

        print(f"Loading {input_flag} dataset...")
        np.random.seed(seed)

        source = get_dataset_source(input_flag=input_flag, input_dir=input_dir)
        (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), (X_test, y_test, test_debugs), label_map = (
            source.load(seed=seed, num_frames=num_frames, cache_artifact_dir=directory_cache_path)
        )

        print(f"{input_flag} dataset loaded from disk.")
        print_stats("X_train", X_train)
        print_stats("X_val", X_val)
        print_stats("X_test", X_test)
        print_stats("y_train", y_train, label_map)
        print_stats("y_val", y_val, label_map)
        print_stats("y_test", y_test, label_map)
        print_stats("split", None, label_map, split_arrays=(y_train, y_val, y_test))

        result = (
            (X_train, y_train, train_debugs),
            (X_val, y_val, val_debugs),
            (X_test, y_test, test_debugs),
            label_map,
        )

        print(f"Saving cache to {directory_cache_path}...")
        return result

    cache_filename = f"{cache_input_name}_seed{seed}_{cache_version}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path) and not no_cache:
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
            print(f"{input_flag} dataset loaded from cache.")
            print_stats("X_train", result[0][0])
            print_stats("X_val", result[1][0])
            print_stats("X_test", result[2][0])
            print_stats("y_train", result[0][1], result[3])
            print_stats("y_val", result[1][1], result[3])
            print_stats("y_test", result[2][1], result[3])
            print_stats("split", None, result[3], split_arrays=(result[0][1], result[1][1], result[2][1]))
            return result

    else:
        print(f"No cache found at {cache_path}, loading data from disk...")

    print(f"Loading {input_flag} dataset...")
    np.random.seed(seed)

    source = get_dataset_source(input_flag=input_flag, input_dir=input_dir)
    if input_flag in ("devemo", "devemo+", "devemo_combined"):
        (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), (X_test, y_test, test_debugs), label_map = (
            source.load(
                seed=seed,
                train_frame_selection=train_frame_selection,
                test_frame_selection=test_frame_selection,
                num_frames=num_frames,
                class_split=class_split,
            )
        )
    else:
        (X_train, y_train, train_debugs), (X_val, y_val, val_debugs), (X_test, y_test, test_debugs), label_map = (
            source.load(seed=seed)
        )

    print(f"{input_flag} dataset loaded from disk.")
    print_stats("X_train", X_train)
    print_stats("X_val", X_val)
    print_stats("X_test", X_test)
    print_stats("y_train", y_train, label_map)
    print_stats("y_val", y_val, label_map)
    print_stats("y_test", y_test, label_map)
    print_stats("split", None, label_map, split_arrays=(y_train, y_val, y_test))

    result = (
        (X_train, y_train, train_debugs),
        (X_val, y_val, val_debugs),
        (X_test, y_test, test_debugs),
        label_map,
    )

    print(f"Saving cache to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result

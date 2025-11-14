import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_processor import (
    download_dlib_weights,
    initialize_dlib_components,
    load_and_prepare_data,
    create_combined_strip_visualization,
    TIME_STEPS,
    NUM_CLIPS_FOR_VISUALIZATION,
    FRAMES_TO_VISUALIZE,
    VIDEO_DIR,
    CSV_FILE,
    OUTPUT_VISUALIZATION_FILE,
    IMG_HEIGHT,
    IMG_WIDTH,
    CHANNELS,
)
from model import (
    build_hybrid_vfer_model,
    save_vfer_model,
    load_vfer_model,
    MODEL_PATH,
)

EPOCHS = 80
BATCH_SIZE = 4
TEST_SIZE = 0.2


def check_gpu():
    """
    Check for available GPU devices and configure TensorFlow to use the first GPU if present.

    This function will attempt to enable memory growth for the selected GPU to avoid allocation issues.
    """
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[{'=' * 10} HARDWARE CHECK {'=' * 10}]")
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"INFO: GPU found and configured: {gpus[0].name}. Proceeding with GPU acceleration.")
        except RuntimeError as e:
            print(f"CRITICAL ERROR: GPU configuration failed: {e}")
            print("INFO: Proceeding using CPU.")
    else:
        print("INFO: No GPU detected. Proceeding using CPU.")
    print(f"[{'=' * 10} HARDWARE CHECK {'=' * 10}]\n")


def main():
    """
    Main entry point for the training/evaluation/visualization pipeline.

    The function handles downloading dlib weights, initializing detectors, loading or building the model,
    optionally training, performing evaluation if a model is available and there are test samples,
    and finally creating the visualization image.
    """
    check_gpu()
    if not download_dlib_weights():
        print("CRITICAL ERROR: Dlib model weights download failed. Exiting.")
        sys.exit(1)

    detector, predictor = initialize_dlib_components()

    model = load_vfer_model(MODEL_PATH)
    should_train = False

    if model is None:
        print("\nINFO: Model file not found. Training is required.")
        should_train = True
    else:
        user_input = input("INFO: Model found. Do you want to retrain it? (yes/no): ").lower().strip()
        if user_input.startswith("y"):
            should_train = True
        elif user_input.startswith("n"):
            print("INFO: Skipping training.")
        else:
            print("WARNING: Invalid input. Skipping training.")

    max_clips_to_load = None
    if not should_train and model is not None:
        max_clips_to_load = NUM_CLIPS_FOR_VISUALIZATION
        print(f"\nINFO: Loading only {max_clips_to_load} clips for visualization (training/full evaluation skipped).")
    else:
        print("\nINFO: Loading all video clips for training/full evaluation.")

    X, y, all_visualization_data, EMOTION_CLASSES, NUM_CLASSES = load_and_prepare_data(
        CSV_FILE,
        VIDEO_DIR,
        detector,
        predictor,
        TIME_STEPS,
        max_clips=max_clips_to_load,
    )

    if X is None:
        print("\nCRITICAL ERROR: Data loading failed. Exiting.")
        sys.exit(1)

    if model is not None and y is not None:
        try:
            model_output_units = model.output_shape[-1]
        except Exception:
            model_output_units = None

        if model_output_units is not None and getattr(y, "ndim", 1) > 1:
            current_units = y.shape[1]
            if model_output_units != current_units:
                print(
                    f"WARNING: Model expects {model_output_units} classes but data has {current_units} one-hot columns."
                )
                if model_output_units > current_units:
                    pad_width = model_output_units - current_units
                    pad = np.zeros((y.shape[0], pad_width), dtype=y.dtype)
                    y = np.concatenate([y, pad], axis=1)
                    print(f"INFO: Padded label one-hot vectors with {pad_width} zero columns to match model outputs.")
                else:
                    y = y[:, :model_output_units]
                    print(
                        f"INFO: Truncated label one-hot vectors to first {model_output_units} columns to match model outputs."
                    )
        elif model_output_units is not None and getattr(y, "ndim", 1) == 1:
            print(
                "WARNING: Labels are not one-hot encoded. Evaluation requires one-hot vectors matching model outputs."
            )

    X_train, X_test, y_train, y_test = (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )

    if max_clips_to_load is not None:
        X_test, y_test = X, y
        if should_train:
            print("\nWARNING: Only a small data subset was loaded. Training is disabled.")
            should_train = False
    else:
        print(f"INFO: Total valid clips loaded: {X.shape[0]}")
        stratify_labels = np.argmax(y, axis=1) if getattr(y, "ndim", 1) > 1 else y
        from collections import Counter

        label_counts = Counter(stratify_labels)
        min_count = min(label_counts.values()) if label_counts else 0
        if min_count < 2:
            print(
                "WARNING: Some classes have fewer than 2 samples. Stratified split is not possible. Proceeding without stratify."
            )
            stratify_arg = None
        else:
            stratify_arg = stratify_labels

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=42, stratify=stratify_arg
            )
        except ValueError as e:
            print(f"WARNING: train_test_split failed with stratify={stratify_arg}: {e}. Retrying without stratify.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=42, stratify=None
            )

        print(f"INFO: Training clips: {X_train.shape[0]}, Test clips: {X_test.shape[0]}")
        print(f"INFO: Class distribution in Training set (Top 5):")

        train_labels = np.argmax(y_train, axis=1) if getattr(y_train, "ndim", 1) > 1 else y_train
        label_counts = Counter(train_labels)
        sorted_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
        top_5_counts = [f"{EMOTION_CLASSES[k]}: {v}" for k, v in sorted_counts[:5]]
        print(f"INFO: {', '.join(top_5_counts)}...")

    if should_train:
        if model is None:
            print(f"\nBuilding model architecture with {NUM_CLASSES} output classes...")
            model = build_hybrid_vfer_model(TIME_STEPS, IMG_HEIGHT, IMG_WIDTH, CHANNELS, NUM_CLASSES)

        model.summary()
        print(f"\nTraining data: {X_train.shape[0]} clips (Shape: {X_train.shape}).")

        print(f"\n--- Starting Model Training (EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}) ---")
        model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=1,
        )

        save_vfer_model(model, MODEL_PATH)

    loaded_model = model

    if loaded_model and len(X_test) > 0:
        if max_clips_to_load is not None:
            print("\nINFO: Only a small subset of clips was loaded; evaluation will run on the available subset.")
        print(f"\nTest data: {X_test.shape[0]} clips (Shape: {X_test.shape}).")
        print("\n--- Starting Model Performance Evaluation on Test Set ---")
        loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=1)

        print("\n--- TEST RESULTS ---")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("--------------------")
    elif loaded_model and len(X_test) == 0:
        print("\nINFO: No test clips available for evaluation.")

    if all_visualization_data:
        create_combined_strip_visualization(
            all_visualization_data,
            OUTPUT_VISUALIZATION_FILE,
            FRAMES_TO_VISUALIZE,
        )
    else:
        print("WARNING: Could not generate visualization image (no valid clips loaded for visualization).")


if __name__ == "__main__":
    main()

import os
import sys
import tensorflow as tf
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
)
from model import (
    build_hybrid_vfer_model,
    save_vfer_model,
    load_vfer_model,
    MODEL_PATH,
    NUM_CLASSES,
    IMG_HEIGHT,
    IMG_WIDTH,
    CHANNELS,
)
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

EPOCHS = 5
BATCH_SIZE = 4
TEST_SIZE = 0.2


def check_gpu():
    """Checks for GPU availability and configures TensorFlow."""
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[{'='*10} HARDWARE CHECK {'='*10}]")
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"INFO: GPU found and configured: {gpus[0].name}. Proceeding with GPU acceleration.")
        except RuntimeError as e:
            print(f"CRITICAL ERROR: GPU configuration failed: {e}")
            print("INFO: Proceeding using CPU.")
    else:
        print(
            "WARNING: No GPU found. Training and evaluation will proceed using CPU, which may be significantly slower."
        )
    print("-" * 50)


def main():
    check_gpu()

    if not download_dlib_weights():
        sys.exit(1)

    detector, predictor = initialize_dlib_components()

    print(f"\nLoading and processing full dataset from '{CSV_FILE}'...")
    X_data, y_data, all_visualization_data = load_and_prepare_data(CSV_FILE, VIDEO_DIR, detector, predictor, TIME_STEPS)

    if X_data is None or X_data.shape[0] == 0:
        print("\n--- EXECUTION CANCELED ---")
        print("No data loaded or video processing error.")
        sys.exit(1)
    else:
        y_indices = np.argmax(y_data, axis=1)
        print(f"\nClass distribution in loaded data ({X_data.shape[0]} total samples): {Counter(y_indices)}")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, test_size=TEST_SIZE, random_state=42, stratify=y_indices
            )
        except ValueError as e:
            print(f"\n[WARNING] Stratified split failed: {e}. Falling back to non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=TEST_SIZE, random_state=42)

        model_exists = os.path.exists(MODEL_PATH)
        should_train = True
        model = None

        if model_exists:
            print(f"\n[Model Check] Found existing model at {MODEL_PATH}.")
            print("--- Training/Loading Decision ---")

            try:
                user_input = input(f"Model found. Do you want to overwrite and train a new model? (yes/no): ").lower()
            except EOFError:
                print("\n[WARNING] Non-interactive environment detected. Defaulting to 'no' (load).")
                user_input = "no"

            if user_input == "yes":
                should_train = True
                print("Decision: 'yes' received. Proceeding with training and overwrite.")
            else:
                should_train = False
                print("Decision: 'no' received. Attempting to load existing model.")
                model = load_vfer_model(MODEL_PATH)

                if model is None:
                    print("WARNING: Loading failed, even though 'no' was selected. Forced to train new model.")
                    should_train = True

        if should_train:
            if model is None:
                print(f"\nBuilding model architecture with TIME_STEPS={TIME_STEPS}...")
                model = build_hybrid_vfer_model(TIME_STEPS, IMG_HEIGHT, IMG_WIDTH, CHANNELS, NUM_CLASSES)

            print(f"\nTraining data: {X_train.shape[0]} clips (Shape: {X_train.shape}).")

            print(f"\n--- Starting Model Training (EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}) ---")
            model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

            save_vfer_model(model, MODEL_PATH)

        loaded_model = model

        if loaded_model:
            print(f"\nTest data: {X_test.shape[0]} clips (Shape: {X_test.shape}).")

            print("\n--- Starting Model Performance Evaluation on Test Set ---")
            loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=1)

            print("\n--- TEST RESULTS ---")
            print(f"Loss: {loss:.4f}")
            print(f"Accuracy: {accuracy*100:.2f}%")
            print("--------------------")

        if all_visualization_data:
            create_combined_strip_visualization(all_visualization_data, OUTPUT_VISUALIZATION_FILE, FRAMES_TO_VISUALIZE)


if __name__ == "__main__":
    main()

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import argparse
import numpy as np
import tensorflow as tf
from utils import load_data, plot_metrics, save_sample_frames


def get_model(model_name):
    if model_name == "simple":
        from models.simple_model import SimpleModel

        return SimpleModel()
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Facial Expression Recognition Training/Evaluation")
    parser.add_argument("--model", type=str, default="simple", help="Model name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (train mode only)")
    args = parser.parse_args()

    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    MODEL_PATH = os.path.join(OUTPUT_DIR, f"{args.model}_model.keras")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader = load_data(INPUT_DIR)

    model = get_model(args.model)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    train_losses, val_losses = [], []
    train_acc, val_acc = [], []

    def generator(loader):
        for images, labels, debugs in loader:
            images = np.array(images)
            images = np.reshape(images, (images.shape[0], 48, 48, 1))
            labels = np.array(labels)
            yield images, labels

    train_data = (
        tf.data.Dataset.from_generator(
            lambda: generator(train_loader),
            output_signature=(
                tf.TensorSpec(shape=(None, 48, 48, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int64),
            ),
        )
        .unbatch()
        .batch(64)
        .repeat()
    )
    val_data = (
        tf.data.Dataset.from_generator(
            lambda: generator(val_loader),
            output_signature=(
                tf.TensorSpec(shape=(None, 48, 48, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int64),
            ),
        )
        .unbatch()
        .batch(64)
    )

    if args.mode == "train":
        if args.epochs is None:
            print("Error: --epochs must be specified for training.")
            sys.exit(1)
        print("Training model...")
        history = model.fit(train_data, validation_data=val_data, epochs=args.epochs, steps_per_epoch=len(train_loader))
        model.save(MODEL_PATH)
        train_losses = history.history["loss"]
        val_losses = history.history["val_loss"]
        train_acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        plot_metrics(train_losses, val_losses, train_acc, val_acc, OUTPUT_DIR)

    if args.mode == "eval":
        if not os.path.exists(MODEL_PATH):
            print(f"Error: trained model not found: {MODEL_PATH}")
            sys.exit(1)
        print("Loading trained model for evaluation...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Selecting 1 frame from each of up to 10 different videos in the validation set...")
        video_frames = {}
        all_preds = []
        all_labels = []
        batch_num = 0
        for images, labels, debugs in val_loader:
            batch_num += 1
            print(f"Processing batch {batch_num} with {len(images)} frames...")
            images_np = np.array(images)
            images_np = np.reshape(images_np, (images_np.shape[0], 48, 48, 1))
            if images_np.shape[0] == 0:
                print(f"Batch {batch_num} is empty, skipping.")
                continue
            preds = np.argmax(model.predict(images_np), axis=1)
            for i in range(images_np.shape[0]):
                debug = debugs[i]
                video_path = debug.get("video_path")
                if video_path not in video_frames and len(video_frames) < 10:
                    video_frames[video_path] = {
                        "frame": images_np[i],
                        "pred": int(preds[i]),
                        "label": int(labels[i]),
                        "debug": debug,
                    }
                    print(f"Selected frame {i} from video {video_path} (total selected: {len(video_frames)})")
                all_preds.append(int(preds[i]))
                all_labels.append(int(labels[i]))
            if len(video_frames) == 10:
                print("Collected 10 frames from 10 unique videos. Stopping batch processing.")
                break
        print(f"Total unique videos selected: {len(video_frames)}")
        frames = [v["frame"] for v in video_frames.values()]
        preds = [v["pred"] for v in video_frames.values()]
        labels = [v["label"] for v in video_frames.values()]
        debugs = [v["debug"] for v in video_frames.values()]
        print(f"Generating sample PNG for {len(frames)} videos...")
        save_sample_frames(frames, preds, labels, debugs, OUTPUT_DIR, model_name=f"{args.model}_sample_grid")
        print("Sample PNG generation complete.")
        correct = sum([p == l for p, l in zip(all_preds, all_labels)])
        total = len(all_labels)
        accuracy = correct / total if total > 0 else 0.0
        print(f"Validation accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()

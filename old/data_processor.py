import bz2
import os
import urllib.request
from typing import List, Tuple, Dict

import cv2
import dlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

TIME_STEPS = 8
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 1

NUM_CLIPS_FOR_VISUALIZATION = 10
FRAMES_TO_VISUALIZE = 8

VIDEO_DIR = "./input/devemo"
CSV_FILE = os.path.join(VIDEO_DIR, "_clips_info.csv")
OUTPUT_VISUALIZATION_FILE = "OUTPUT.png"

DLIB_LANDMARK_MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"
DLIB_DOWNLOAD_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
DLIB_DOWNLOAD_DIR = "downloads"
DLIB_FULL_PATH = os.path.join(DLIB_DOWNLOAD_DIR, DLIB_LANDMARK_MODEL_FILENAME)


def download_dlib_weights():
    """
    Download and extract the dlib facial landmark model if it does not already exist.

    Returns:
        bool: True if the model file is present or successfully downloaded and extracted, False otherwise.
    """
    if os.path.exists(DLIB_FULL_PATH):
        print(f"Dlib weights file already exists at {DLIB_FULL_PATH}.")
        return True

    os.makedirs(os.path.dirname(DLIB_FULL_PATH), exist_ok=True)

    print(f"Downloading Dlib weights from: {DLIB_DOWNLOAD_URL}")
    try:
        bz2_file_path = DLIB_FULL_PATH + ".bz2"
        urllib.request.urlretrieve(DLIB_DOWNLOAD_URL, bz2_file_path)
        print("Download complete. Extracting...")

        with bz2.BZ2File(bz2_file_path) as compressed_file:
            data = compressed_file.read()
        with open(DLIB_FULL_PATH, "wb") as decompressed_file:
            decompressed_file.write(data)

        os.remove(bz2_file_path)
        print(f"File '{DLIB_LANDMARK_MODEL_FILENAME}' is ready.")
        return True
    except Exception as e:
        print(f"ERROR: Could not download/extract Dlib weights file: {e}")
        return False


def initialize_dlib_components():
    """
    Initialize and return dlib's face detector and shape predictor.

    Returns:
        tuple: (detector, predictor) where detector is the frontal face detector and predictor is the shape predictor.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_FULL_PATH)
    return detector, predictor


def detect_and_crop_face(frame: np.ndarray, detector, predictor) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """
    Detect a single face in the provided frame, crop, resize and normalize it, and compute adjusted landmarks.

    Args:
        frame (np.ndarray): BGR image frame.
        detector: dlib face detector.
        predictor: dlib shape predictor.

    Returns:
        Tuple[np.ndarray | None, np.ndarray | None]: Normalized grayscale face array with shape (IMG_HEIGHT, IMG_WIDTH, 1)
        and adjusted landmarks array, or (None, None) if detection fails or multiple/no faces exist.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 1:
        face_rect = faces[0]
        shape = predictor(gray, face_rect)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        x1, y1 = face_rect.left(), face_rect.top()
        x2, y2 = face_rect.right(), face_rect.bottom()
        margin = int(0.2 * (x2 - x1))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)

        cropped_face = gray[y1:y2, x1:x2]
        resized_face = cv2.resize(cropped_face, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)

        scale_x = IMG_WIDTH / (x2 - x1)
        scale_y = IMG_HEIGHT / (y2 - y1)
        adjusted_landmarks = np.copy(landmarks)
        adjusted_landmarks[:, 0] = ((landmarks[:, 0] - x1) * scale_x).astype(int)
        adjusted_landmarks[:, 1] = ((landmarks[:, 1] - y1) * scale_y).astype(int)

        normalized_face_2d = resized_face.astype(np.float32) / 255.0
        normalized_face = np.expand_dims(normalized_face_2d, axis=-1)

        return normalized_face, adjusted_landmarks

    return None, None


def preprocess_video(
    video_path: str, detector, predictor, time_steps: int
) -> Tuple[np.ndarray | None, List[np.ndarray]]:
    """
    Sample frames from a video, detect and preprocess faces per frame, and prepare visualization frames.

    Args:
        video_path (str): Path to the video file.
        detector: dlib face detector.
        predictor: dlib shape predictor.
        time_steps (int): Number of frames to sample from the video.

    Returns:
        Tuple[np.ndarray | None, List[np.ndarray]]: Array of processed frames with shape (time_steps, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
        or None if loading failed, and a list of visualization BGR frames for each sampled frame.
    """
    cap = cv2.VideoCapture(video_path)
    video_filename = os.path.basename(video_path)

    if not cap.isOpened():
        tqdm.write(f"[ERROR] Could not load video file: {video_filename}")
        return None, []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < 1:
        tqdm.write(f"[ERROR] Video has no frames: {video_filename}")
        cap.release()
        return None, []

    indices = np.linspace(0, frame_count - 1, time_steps, dtype=int)

    processed_frames = []
    visualization_frames_full = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if ret:
            processed_face, landmarks = detect_and_crop_face(frame, detector, predictor)

            if processed_face is not None:
                processed_frames.append(processed_face)

                vis_frame = np.squeeze(processed_face) * 255.0
                vis_frame = vis_frame.astype(np.uint8)
                vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)

                for x, y in landmarks:
                    cv2.circle(vis_frame_bgr, (x, y), 1, (0, 255, 0), -1)
                visualization_frames_full.append(vis_frame_bgr)
            else:
                processed_frames.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32))
                vis_frame_fail = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
                vis_frame_fail = vis_frame_fail.astype(np.uint8)
                cv2.rectangle(
                    vis_frame_fail,
                    (0, 0),
                    (IMG_HEIGHT - 1, IMG_WIDTH - 1),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    vis_frame_fail,
                    "NO FACE",
                    (5, IMG_HEIGHT - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                visualization_frames_full.append(vis_frame_fail)

        if len(processed_frames) == time_steps:
            break

    cap.release()

    padding_needed = time_steps - len(processed_frames)
    if padding_needed > 0:
        pass

    while len(processed_frames) < time_steps:
        processed_frames.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32))

        padding_vis_frame = np.full((IMG_HEIGHT, IMG_WIDTH, 3), 100, dtype=np.uint8)

        visualization_frames_full.append(padding_vis_frame)

    return np.array(processed_frames), visualization_frames_full


def load_and_prepare_data(
    csv_path: str,
    video_dir: str,
    detector,
    predictor,
    time_steps: int,
    max_clips: int | None = None,
) -> Tuple[np.ndarray | None, np.ndarray | None, List[Dict], List[str], int]:
    """
    Load clip metadata from CSV, optionally sample a subset, preprocess videos and prepare labels and visualization data.

    Args:
        csv_path (str): Path to the CSV file with 'file' and 'label' columns.
        video_dir (str): Directory where video files are located.
        detector: dlib face detector.
        predictor: dlib shape predictor.
        time_steps (int): Number of frames per clip to process.
        max_clips (int | None): If provided, limit processing to this many clips (random sampling from CSV).

    Returns:
        Tuple[np.ndarray | None, np.ndarray | None, List[Dict], List[str], int]: X array, y array (one-hot),
        visualization metadata list, list of emotion classes, and number of classes.
    """
    try:
        df = pd.read_csv(csv_path, sep=";")
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {csv_path}")
        return None, None, [], [], 0

    df.dropna(subset=["file", "label"], inplace=True)
    df["label"] = df["label"].str.lower()

    if max_clips is not None and max_clips > 0 and len(df) > max_clips:
        df = df.sample(n=max_clips, random_state=None).reset_index(drop=True)
        print(f"INFO: Randomly sampled {max_clips} clips from CSV for visualization.")

    EMOTION_CLASSES = sorted(df["label"].unique())
    NUM_CLASSES = len(EMOTION_CLASSES)

    if df.empty:
        print("ERROR: No valid data found in CSV after cleaning.")
        return None, None, [], [], 0

    print(f"\nDynamically detected {NUM_CLASSES} classes: {EMOTION_CLASSES}")

    video_paths = [os.path.join(video_dir, f) for f in df["file"]]
    labels = df["label"].values

    X_list = []
    y_labels = []
    all_visualization_data = []
    vis_count = 0

    for idx, (path, label) in enumerate(
        tqdm(
            zip(video_paths, labels),
            total=len(video_paths),
            desc="Processing Clips (Total)",
        )
    ):
        if max_clips is not None and len(X_list) >= max_clips:
            tqdm.write(f"[INFO] Hit max_clips limit ({max_clips}). Stopping video processing.")
            break

        if not os.path.exists(path):
            tqdm.write(f"[SKIP] Video file not found: {path}")
            continue

        clip, visualization_frames_full = preprocess_video(path, detector, predictor, time_steps)

        if clip is not None and clip.shape == (
            TIME_STEPS,
            IMG_HEIGHT,
            IMG_WIDTH,
            CHANNELS,
        ):
            X_list.append(clip)
            y_labels.append(label)

            if vis_count < NUM_CLIPS_FOR_VISUALIZATION:
                all_visualization_data.append(
                    {
                        "frames": visualization_frames_full,
                        "label": label,
                        "clip_index": len(X_list),
                    }
                )
                vis_count += 1
        elif clip is None:
            tqdm.write(f"[SKIP] Video processing failed for: {path}")
        else:
            tqdm.write(f"[SKIP] Invalid clip shape {clip.shape} for: {path}")

    if not X_list:
        print("ERROR: No valid clips were loaded after preprocessing.")
        return None, None, [], EMOTION_CLASSES, NUM_CLASSES

    X = np.array(X_list)

    le = LabelEncoder()
    le.fit(EMOTION_CLASSES)
    integer_encoded = le.transform(y_labels)
    y = to_categorical(integer_encoded, num_classes=NUM_CLASSES)

    return X, y, all_visualization_data, EMOTION_CLASSES, NUM_CLASSES


def create_combined_strip_visualization(visualization_data: List[Dict], output_path: str, frames_to_visualize: int):
    """
    Create and save a combined strip visualization image for multiple clips.

    Args:
        visualization_data (List[Dict]): List of dicts with keys 'frames', 'label' and 'clip_index'.
        output_path (str): Path to save the combined visualization image.
        frames_to_visualize (int): Number of frames to show per clip in the final image.

    Returns:
        None
    """
    if not visualization_data or len(visualization_data) == 0:
        print("[Warning] No clips loaded for visualization.")
        return

    total_available = len(visualization_data)
    max_to_take = NUM_CLIPS_FOR_VISUALIZATION
    if total_available > max_to_take:
        selected_indices = np.random.choice(range(total_available), max_to_take, replace=False)
        visualization_data = [visualization_data[i] for i in selected_indices]
        print(f"INFO: Randomly selected {max_to_take} clips for visualization out of {total_available} available.")
    else:
        print(f"INFO: Using all {total_available} available clips for visualization.")

    combined_strips = []
    LABEL_BAR_HEIGHT = 20
    SEPARATION_HEIGHT = 5

    full_frame_count = TIME_STEPS
    frames_to_show = min(full_frame_count, frames_to_visualize)
    vis_indices = np.linspace(0, full_frame_count - 1, frames_to_show, dtype=int)

    STRIP_WIDTH = IMG_WIDTH * frames_to_show

    for clip_data in visualization_data:
        full_frame_sequence = clip_data["frames"]
        video_label = clip_data["label"]
        clip_index = clip_data["clip_index"]

        if len(full_frame_sequence) != full_frame_count:
            print(
                f"[Visualization Warning] Clip {clip_index} has {len(full_frame_sequence)} frames, expected {full_frame_count}. Skipping."
            )
            continue

        frame_sequence = [full_frame_sequence[i] for i in vis_indices]

        final_frames = [frame.astype(np.uint8) for frame in frame_sequence]
        strip = np.concatenate(final_frames, axis=1)

        label_text = f"CLIP {clip_index} - Emotion: {video_label.upper()} (Frames: {full_frame_count})"
        label_bar = np.zeros((LABEL_BAR_HEIGHT, STRIP_WIDTH, 3), dtype=np.uint8)

        cv2.putText(
            label_bar,
            label_text,
            (5, LABEL_BAR_HEIGHT - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        clip_visual_block = np.concatenate([strip, label_bar], axis=0)

        separator = np.zeros((SEPARATION_HEIGHT, STRIP_WIDTH, 3), dtype=np.uint8)

        combined_strips.append(clip_visual_block)
        combined_strips.append(separator)

    if combined_strips:
        combined_strips.pop()
    else:
        print("ERROR: No visualization strips were generated.")
        return

    final_image = np.concatenate(combined_strips, axis=0)

    cv2.imwrite(output_path, final_image)
    print(f"\n--- PROCESSING VISUALIZATION ---")
    print(
        f"A combined visualization image showing {len(visualization_data)} clips, each with {frames_to_show} frames, has been saved as '{output_path}'."
    )
    print(f"Visualization: {len(visualization_data)} rows x {frames_to_show} columns in '{output_path}'.")

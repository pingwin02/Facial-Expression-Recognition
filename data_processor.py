import pandas as pd
import numpy as np
import os
import cv2
import dlib
import urllib.request
import bz2
from typing import List, Tuple, Dict
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# --- CONFIGURATION ---
EMOTION_CLASSES = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
CLASS_TO_INDEX = {emotion: i for i, emotion in enumerate(EMOTION_CLASSES)}

TIME_STEPS = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 3
NUM_CLASSES = len(EMOTION_CLASSES)

NUM_CLIPS_FOR_VISUALIZATION = 10
FRAMES_TO_VISUALIZE = 8

VIDEO_DIR = "./input/devemo"
CSV_FILE = os.path.join(VIDEO_DIR, "_clips_info.csv")
OUTPUT_VISUALIZATION_FILE = "OUTPUT.png"

# Dlib Configuration
DLIB_LANDMARK_MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"
DLIB_DOWNLOAD_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
DLIB_DOWNLOAD_DIR = "downloads"
DLIB_FULL_PATH = os.path.join(DLIB_DOWNLOAD_DIR, DLIB_LANDMARK_MODEL_FILENAME)

# --- DLIB UTILITIES ---


def download_dlib_weights():
    """Downloads and extracts the Dlib shape predictor weights file."""
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
    """Initializes and returns the Dlib face detector and shape predictor."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_FULL_PATH)
    return detector, predictor


# --- PREPROCESSING FUNCTIONS ---


def detect_and_crop_face(frame: np.ndarray, detector, predictor) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Detects face, finds landmarks, and returns the normalized, resized face and landmarks."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 1:
        face_rect = faces[0]
        shape = predictor(gray, face_rect)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        # Cropping logic
        x1, y1 = face_rect.left(), face_rect.top()
        x2, y2 = face_rect.right(), face_rect.bottom()
        margin = int(0.2 * (x2 - x1))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)

        cropped_face = frame[y1:y2, x1:x2]
        resized_face = cv2.resize(cropped_face, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)

        # Adjust landmarks coordinates
        scale_x = IMG_WIDTH / (x2 - x1)
        scale_y = IMG_HEIGHT / (y2 - y1)
        adjusted_landmarks = np.copy(landmarks)
        adjusted_landmarks[:, 0] = ((landmarks[:, 0] - x1) * scale_x).astype(int)
        adjusted_landmarks[:, 1] = ((landmarks[:, 1] - y1) * scale_y).astype(int)

        # Normalize pixels
        normalized_face = resized_face.astype(np.float32) / 255.0

        return normalized_face, adjusted_landmarks

    return None, None


def preprocess_video(
    video_path: str, detector, predictor, time_steps: int
) -> Tuple[np.ndarray | None, List[np.ndarray]]:
    """Loads video, processes frames, and returns a sequence of 'time_steps' frames."""
    cap = cv2.VideoCapture(video_path)
    video_filename = os.path.basename(video_path)

    if not cap.isOpened():
        tqdm.write(f"[ERROR] Could not load video file: {video_filename}")
        return None, []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

                # Visualization frame (0-255 scale with landmarks)
                vis_frame = np.copy(processed_face) * 255
                vis_frame = vis_frame.astype(np.uint8)
                for x, y in landmarks:
                    cv2.circle(vis_frame, (x, y), 2, (0, 255, 0), -1)
                visualization_frames_full.append(vis_frame)

            else:
                # Face Detection Failure Case
                processed_frames.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32))
                vis_frame_fail = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
                vis_frame_fail = vis_frame_fail.astype(np.uint8)
                cv2.rectangle(vis_frame_fail, (0, 0), (IMG_HEIGHT - 1, IMG_WIDTH - 1), (0, 0, 255), 3)
                cv2.putText(
                    vis_frame_fail,
                    "NO FACE",
                    (5, IMG_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                visualization_frames_full.append(vis_frame_fail)

        if len(processed_frames) == time_steps:
            break

    cap.release()

    # Padding with zeros if not enough frames were collected
    padding_needed = time_steps - len(processed_frames)
    if padding_needed > 0:
        tqdm.write(
            f"[WARN] Padding clip '{video_filename}' ({frame_count} total frames): {padding_needed} frames missing."
        )

    while len(processed_frames) < time_steps:
        processed_frames.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32))
        padding_vis_frame = np.full((IMG_HEIGHT, IMG_WIDTH, CHANNELS), 100, dtype=np.uint8)
        cv2.putText(
            padding_vis_frame,
            "PADDING",
            (5, IMG_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        visualization_frames_full.append(padding_vis_frame)

    return np.array(processed_frames), visualization_frames_full


def load_and_prepare_data(
    csv_path: str, video_dir: str, detector, predictor, time_steps: int
) -> Tuple[np.ndarray | None, np.ndarray | None, List[Dict]]:
    """Loads all metadata, processes videos, and encodes labels. Collects visualization data for the first 10 clips."""
    try:
        df = pd.read_csv(csv_path, sep=";")
    except FileNotFoundError:
        return None, None, []

    df.dropna(subset=["file", "label"], inplace=True)
    df["label"] = df["label"].str.lower()
    df = df[df["label"].isin(CLASS_TO_INDEX.keys())]

    video_paths = [os.path.join(video_dir, f) for f in df["file"]]
    labels = df["label"].values

    X_list = []
    y_labels = []
    all_visualization_data = []
    vis_count = 0

    for idx, (path, label) in enumerate(
        tqdm(zip(video_paths, labels), total=len(video_paths), desc="Processing Clips (Total)")
    ):

        if not os.path.exists(path):
            tqdm.write(f"[SKIP] Video file not found: {path}")
            continue

        clip, visualization_frames_full = preprocess_video(path, detector, predictor, time_steps)

        if clip is not None:
            X_list.append(clip)
            y_labels.append(label)

            if vis_count < NUM_CLIPS_FOR_VISUALIZATION:
                all_visualization_data.append(
                    {"frames": visualization_frames_full, "label": label, "clip_index": len(X_list)}
                )
                vis_count += 1

    if not X_list:
        return None, None, []

    X = np.array(X_list)

    # Encode labels
    le = LabelEncoder()
    le.fit(EMOTION_CLASSES)
    integer_encoded = le.transform(y_labels)
    y = to_categorical(integer_encoded, num_classes=NUM_CLASSES)

    return X, y, all_visualization_data


# --- VISUALIZATION FUNCTION ---


def create_combined_strip_visualization(visualization_data: List[Dict], output_path: str, frames_to_visualize: int):
    """
    Combines 10 clips' frame sequences into one large vertical image (10 rows x 8 columns), adds labels, and saves it as OUTPUT.png.
    """
    if not visualization_data or len(visualization_data) == 0:
        print("[Warning] No clips loaded for visualization.")
        return

    combined_strips = []

    # Calculate indices for even sampling of the 32 frames
    full_frame_count = TIME_STEPS
    vis_indices = np.linspace(0, full_frame_count - 1, frames_to_visualize, dtype=int)

    for clip_data in visualization_data:
        full_frame_sequence = clip_data["frames"]
        video_label = clip_data["label"]
        clip_index = clip_data["clip_index"]

        # Select the sampled frames for visualization
        frame_sequence = [full_frame_sequence[i] for i in vis_indices]

        # Create the horizontal strip (row)
        final_frames = [frame.astype(np.uint8) for frame in frame_sequence]
        strip = np.concatenate(final_frames, axis=1)

        # Add label text
        label_text = f"CLIP {clip_index} - Emotion: {video_label.upper()} (Frames: {full_frame_count})"
        cv2.putText(
            strip, label_text, (10, IMG_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Add a separation line/margin
        separator_height = 5
        separator = np.zeros((separator_height, strip.shape[1], 3), dtype=np.uint8)

        combined_strips.append(strip)
        combined_strips.append(separator)

    if combined_strips:
        combined_strips.pop()

    # Concatenate all strips vertically
    final_image = np.concatenate(combined_strips, axis=0)

    # Save the combined image
    cv2.imwrite(output_path, final_image)
    print(f"\n--- PROCESSING VISUALIZATION ---")
    print(
        f"A combined visualization image showing {len(visualization_data)} clips (10 rows), each with {frames_to_visualize} frames, has been saved as '{output_path}'."
    )
    print(f"Visualization: {len(visualization_data)} rows x {frames_to_visualize} columns in '{output_path}'.")

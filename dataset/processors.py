import os
import cv2
import numpy as np
from tqdm import tqdm
from utils.image import get_dlib_detector_predictor, detect_and_crop_face

IMG_HEIGHT = 48
IMG_WIDTH = 48


def normalize_and_create_mask(landmarks, crop_box, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    x1, y1, x2, y2 = crop_box
    crop_w = x2 - x1
    crop_h = y2 - y1

    scale_x = target_size[0] / crop_w
    scale_y = target_size[1] / crop_h

    mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    for lx, ly in landmarks:
        new_x = int((lx - x1) * scale_x)
        new_y = int((ly - y1) * scale_y)

        if 0 <= new_x < target_size[0] and 0 <= new_y < target_size[1]:
            cv2.circle(mask, (new_x, new_y), 1, 255, -1)

    return mask


def process_image_directory(directory, label_map):
    X, y, debugs = [], [], []
    detector_pack = get_dlib_detector_predictor()

    first_run = True

    for label in tqdm(os.listdir(directory), desc="Processing labels", unit="label"):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for image_file in tqdm(os.listdir(label_dir), desc=f"Processing images for label {label}", unit="image"):
                image_path = os.path.join(label_dir, image_file)

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                if image is not None:
                    face, crop_box, landmarks = detect_and_crop_face(image, *detector_pack)

                    if face is not None and crop_box is not None and landmarks is not None:
                        face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))

                        if len(face.shape) == 2:
                            face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)

                        landmark_mask = normalize_and_create_mask(landmarks, crop_box)
                        combined_img = np.dstack((face, landmark_mask))

                        if first_run:
                            print(f"\nCombined shape: {combined_img.shape}")
                            first_run = False

                        X.append(combined_img)
                        y.append(label_map[label])
                        debugs.append({"crop_box": crop_box})

    return np.array(X), np.array(y), debugs


def process_video_data(df, video_dir, filename_col, label_map, frames_per_video=5):
    X, y, debugs = [], [], []
    detector_pack = get_dlib_detector_predictor()

    first_run = True

    for _, row in tqdm(df.iterrows(), desc="Processing videos", total=len(df), unit="video"):
        video_path = os.path.join(video_dir, row[filename_col])
        label = label_map.get(row["label"], 0)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            face, crop_box, landmarks = detect_and_crop_face(frame, *detector_pack)

            if face is not None and crop_box is not None and landmarks is not None:
                face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))

                if len(face.shape) == 2:
                    face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)

                landmark_mask = normalize_and_create_mask(landmarks, crop_box)
                combined_img = np.dstack((face, landmark_mask))

                if first_run:
                    print(f"\nCombined shape: {combined_img.shape}")
                    first_run = False

                X.append(combined_img)
                y.append(label)

                debugs.append({"video": row[filename_col], "frame_idx": frame_idx, "crop_box": crop_box})

        cap.release()

    return np.array(X), np.array(y), debugs

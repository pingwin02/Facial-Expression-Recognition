import cv2
import json
import numpy as np
import os
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from torchvision import models, transforms

DEVEMO_DIR = "devemo"
DEVEMO_PLUS_DIR = "devemo+"
DEVEMO_PLUS_JSON = os.path.join(DEVEMO_PLUS_DIR, "devemo+.json")

OUTPUT_DIR = "output"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def map_to_binary_class(label):
    label = label.lower()
    negative_keywords = ["confusion", "anger", "disgust", "surprise", "dezorientacja", "złość", "wstręt", "zaskoczenie"]
    other_keywords = ["happiness", "neutral", "radość", "neutralna"]

    if any(neg in label for neg in negative_keywords):
        return "negative"
    if any(oth in label for oth in other_keywords):
        return "other"
    return None


def extract_label_from_name(filename):
    name = filename.lower()
    return map_to_binary_class(name)


def load_devemo_plus_labels():
    if not os.path.exists(DEVEMO_PLUS_JSON):
        return {}
    with open(DEVEMO_PLUS_JSON, "r", encoding="utf8") as f:
        data = json.load(f)
    labels = {}
    for item in data:
        labels[item["filename"]] = map_to_binary_class(item["label"])
    return labels


def read_video_frames(path, max_frames=200):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25.0

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
        if count > max_frames:
            break
    cap.release()
    return frames, fps


DEEP_FEATURE_EXTRACTOR = None
DEEP_TRANSFORM = None


def get_deep_features(frames):
    global DEEP_FEATURE_EXTRACTOR, DEEP_TRANSFORM
    if DEEP_FEATURE_EXTRACTOR is None:
        print("[INFO] Inicjalizacja modelu ResNet do ekstrakcji cech...")
        DEEP_FEATURE_EXTRACTOR = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        DEEP_FEATURE_EXTRACTOR.fc = torch.nn.Identity()
        DEEP_FEATURE_EXTRACTOR.eval()
        DEEP_TRANSFORM = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    features = []
    with torch.no_grad():
        for f in frames:
            t = DEEP_TRANSFORM(f).unsqueeze(0)
            feat = DEEP_FEATURE_EXTRACTOR(t).squeeze().numpy()
            features.append(feat)
    return np.array(features)


def select_frames(frames, n_frames, strategy="uniform"):
    total = len(frames)
    if total == 0:
        return [], []

    if total <= n_frames:
        res = list(frames)
        res_idxs = list(range(total))
        while len(res) < n_frames:
            res.append(frames[-1])
            res_idxs.append(total - 1)
        return res, res_idxs

    idxs = []

    if strategy == "uniform":
        idxs = np.linspace(0, total - 1, n_frames, dtype=int)
    elif strategy == "random":
        idxs = np.sort(np.random.choice(total, n_frames, replace=False))
    elif strategy == "edges":
        if n_frames == 3:
            idxs = [0, total // 2, total - 1]
        else:
            half = n_frames // 2
            idxs = list(range(half)) + list(range(total - (n_frames - half), total))
    elif strategy == "center_dense":
        mu, sigma = total / 2, total / 6
        raw_idxs = np.random.normal(mu, sigma, n_frames * 3)
        raw_idxs = np.clip(np.round(raw_idxs), 0, total - 1).astype(int)
        raw_idxs = np.unique(raw_idxs)
        if len(raw_idxs) >= n_frames:
            idxs = np.sort(np.random.choice(raw_idxs, n_frames, replace=False))
        else:
            idxs = np.linspace(0, total - 1, n_frames, dtype=int)
    elif strategy == "motion_max":
        diffs = [0.0] + [np.sum(np.abs(frames[i].astype(float) - frames[i - 1].astype(float))) for i in range(1, total)]
        top_idxs = np.argsort(diffs)[-n_frames:]
        idxs = np.sort(top_idxs)
    elif strategy == "sharpest":
        variances = [cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames]
        top_idxs = np.argsort(variances)[-n_frames:]
        idxs = np.sort(top_idxs)
    elif strategy == "hog_diff":
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        hog_feats = [hog.compute(cv2.resize(f, (64, 64))) for f in frames]
        diffs = [0.0] + [np.linalg.norm(hog_feats[i] - hog_feats[i - 1]) for i in range(1, total)]
        top_idxs = np.argsort(diffs)[-n_frames:]
        idxs = np.sort(top_idxs)
    elif strategy == "kmeans_rgb":
        small_frames = np.array([cv2.resize(f, (32, 32)).flatten() for f in frames])
        kmeans = KMeans(n_clusters=n_frames, n_init=10, random_state=42).fit(small_frames)
        closest_idxs = pairwise_distances_argmin(kmeans.cluster_centers_, small_frames)
        idxs = np.sort(closest_idxs)
    elif strategy == "resnet_kmeans":
        features = get_deep_features(frames)
        kmeans = KMeans(n_clusters=n_frames, n_init=10, random_state=42).fit(features)
        closest_idxs = pairwise_distances_argmin(kmeans.cluster_centers_, features)
        idxs = np.sort(closest_idxs)
    else:
        idxs = np.linspace(0, total - 1, n_frames, dtype=int)

    idxs = np.sort(np.unique(idxs))

    while len(idxs) < n_frames:
        idxs = np.append(idxs, idxs[-1] if len(idxs) > 0 else 0)

    idxs = idxs[:n_frames]
    return [frames[i] for i in idxs], idxs


def _get_temporal_tints_alphas(n_frames):
    if n_frames == 5:
        return [
            np.array([1.0, 0.1, 0.2], dtype=np.float32),
            np.array([0.2, 1.0, 0.1], dtype=np.float32),
            None,
            np.array([0.1, 0.6, 1.0], dtype=np.float32),
            np.array([0.3, 0.1, 1.0], dtype=np.float32),
        ], [1.2, 0.9, 0.0, 0.9, 1.2]

    center = n_frames // 2
    tints = []
    alphas = []
    base_colors = [
        np.array([1.0, 0.1, 0.2], dtype=np.float32),
        np.array([0.2, 1.0, 0.1], dtype=np.float32),
        np.array([0.1, 0.6, 1.0], dtype=np.float32),
        np.array([0.3, 0.1, 1.0], dtype=np.float32),
        np.array([0.8, 0.8, 0.1], dtype=np.float32),
        np.array([0.1, 0.9, 0.9], dtype=np.float32),
    ]

    for i in range(n_frames):
        if i == center:
            tints.append(None)
            alphas.append(0.0)
        else:
            color_idx = i % len(base_colors)
            if i > center:
                color_idx = (i - 1) % len(base_colors)
            tints.append(base_colors[color_idx])
            distance = abs(i - center) / max(1, center)
            alphas.append(0.9 + 0.3 * distance)

    return tints, alphas


def method_temporal_chromatic(sel_frames):
    n_frames = len(sel_frames)
    if n_frames == 0:
        return None

    frames_norm = [f.astype(np.float32) / 255.0 for f in sel_frames]
    center_idx = n_frames // 2
    tints, alphas = _get_temporal_tints_alphas(n_frames)
    center_frame = frames_norm[center_idx]

    gray = np.dot(center_frame[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    gray = np.clip(gray, 0.0, 1.0).astype(np.float32)
    base_gray = np.repeat(gray[..., np.newaxis], 3, axis=-1)

    result = base_gray.copy()

    for i in range(n_frames):
        if tints[i] is None:
            continue
        frame = frames_norm[i]
        color = tints[i]
        alpha = alphas[i]
        diff = np.abs(frame - center_frame)
        diff_magnitude = np.mean(diff, axis=-1, keepdims=True)
        colored_diff = diff_magnitude * color.reshape(1, 1, 3)
        result = result + alpha * colored_diff

    result = np.clip(result, 0.0, 1.0)
    return (result * 255.0).astype(np.uint8)


def add_timeline(img_rgb, selected_idxs, total_frames, fps):
    h, w, c = img_rgb.shape
    timeline_h = 60

    new_img = np.zeros((h + timeline_h, w, c), dtype=np.uint8)
    new_img[:h, :] = img_rgb
    new_img[h:, :] = (30, 30, 30)

    margin = max(40, w // 10)
    line_y = h + 20

    cv2.line(new_img, (margin, line_y), (w - margin, line_y), (150, 150, 150), 2)

    tints, _ = _get_temporal_tints_alphas(len(selected_idxs))

    for i, frame_idx in enumerate(selected_idxs):
        ratio = frame_idx / max(1, total_frames - 1)
        x = margin + int(ratio * (w - 2 * margin))
        if tints[i] is None:
            color = (255, 255, 255)
        else:
            color = (tints[i] * 255).astype(int).tolist()

        cv2.line(new_img, (x, line_y - 8), (x, line_y + 8), color, 3)

        t_sec = frame_idx / fps if fps > 0 else 0.0
        text = f"{t_sec:.1f}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = line_y + 25

        cv2.putText(new_img, text, (text_x, text_y), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)

    return new_img


def process_video(path, label, output_root):
    frames, fps = read_video_frames(path)
    total_frames = len(frames)

    if not frames:
        return

    strategies = [
        "uniform",
        "random",
        "edges",
        "center_dense",
        "motion_max",
        "sharpest",
        "hog_diff",
        "kmeans_rgb",
        "resnet_kmeans",
    ]

    for n in [3, 4, 5]:
        for strategy in strategies:
            out_dir = os.path.join(output_root, strategy, f"{n}frames", label)
            ensure_dir(out_dir)

            try:
                selected_frames, selected_idxs = select_frames(frames, n, strategy)
                img = method_temporal_chromatic(selected_frames)

                if img is not None:
                    img_with_timeline = add_timeline(img, selected_idxs, total_frames, fps)

                    Image.fromarray(img_with_timeline).save(os.path.join(out_dir, os.path.basename(path) + ".png"))
            except Exception as e:
                print(f"[BŁĄD] Nie udało się przetworzyć strategii {strategy} dla wideo {path}. Błąd: {e}")


def main():
    ensure_dir(OUTPUT_DIR)
    print("[INFO] Start przetwarzania...")

    if os.path.exists(DEVEMO_DIR):
        print(f"[INFO] Przetwarzanie folderu {DEVEMO_DIR}")
        for file in os.listdir(DEVEMO_DIR):
            if not file.endswith(".mp4"):
                continue
            label = extract_label_from_name(file)
            if label is None:
                continue
            process_video(os.path.join(DEVEMO_DIR, file), label, OUTPUT_DIR)

    if os.path.exists(DEVEMO_PLUS_DIR):
        print(f"[INFO] Przetwarzanie folderu {DEVEMO_PLUS_DIR}")
        plus_labels = load_devemo_plus_labels()
        for file in os.listdir(DEVEMO_PLUS_DIR):
            if not file.endswith(".mp4"):
                continue
            label = plus_labels.get(file, None)
            if label is None:
                continue
            process_video(os.path.join(DEVEMO_PLUS_DIR, file), label, OUTPUT_DIR)

    print("[INFO] Przetwarzanie zakończone sukcesem!")


if __name__ == "__main__":
    main()

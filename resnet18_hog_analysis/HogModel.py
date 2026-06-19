import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # DODANO: do rysowania macierzy konfuzji

# TensorFlow / Keras (do uczenia klasyfikatora z HOG)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# Skimage (do ekstrakcji HOG)
from skimage.feature import hog

# PyTorch (zostawiamy WYŁĄCZNIE do działania MTCNN)
import torch
from facenet_pytorch import MTCNN

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # DODANO: confusion_matrix

# ----------------------------
# KONFIGURACJA
# ----------------------------
DATASET_DIR = "devemo"
SEQ_LEN = 8
IMG_SIZE = 64  # Zmniejszono do 64x64. Przy 224x224 wektory HOG z 8 klatek byłyby gigantyczne!
device = "cuda" if torch.cuda.is_available() else "cpu"

# Inicjalizacja MTCNN
mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, post_process=False, device=device)


# ----------------------------
# KLASA HOG MODEL
# ----------------------------
class HogModel:
    def __init__(self, input_shape, num_classes=2):  # Zmiana domyślnego na 2 klasy wg parsowania
        inputs = layers.Input(shape=input_shape, name="hog_features")

        x = layers.Dense(512, activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)

    @staticmethod
    def compute_hog_batch(frames_batch, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
        hog_features = []
        for video in frames_batch:
            frame_hogs = []
            for frame in video:
                # Konwersja do skali szarości lub wzięcie pierwszego kanału
                if len(frame.shape) == 3 and frame.shape[-1] == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame[..., 0]

                hog_vec = hog(
                    gray,
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    block_norm="L2-Hys",
                    visualize=False,
                )
                frame_hogs.append(hog_vec)
            hog_features.append(np.concatenate(frame_hogs))
        return np.array(hog_features, dtype=np.float32)

    @staticmethod
    def _oversample_minority_classes(X, y, min_target_ratio=0.35, max_multiplier=8, seed=42):
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) <= 1:
            return X, y

        max_count = int(np.max(counts))
        rng = np.random.default_rng(seed)

        extra_indices = []
        for class_id, class_count in zip(classes, counts):
            target_from_ratio = int(max_count * min_target_ratio)
            target_from_multiplier = int(class_count * max_multiplier)
            target_count = max(class_count, min(target_from_ratio, target_from_multiplier))

            if target_count <= class_count:
                continue

            class_indices = np.where(y == class_id)[0]
            add_count = int(target_count - class_count)
            sampled = rng.choice(class_indices, size=add_count, replace=True)
            extra_indices.append(sampled)

        if not extra_indices:
            return X, y

        extra_indices = np.concatenate(extra_indices)
        X_balanced = np.concatenate([X, X[extra_indices]], axis=0)
        y_balanced = np.concatenate([y, y[extra_indices]], axis=0)

        shuffle_idx = rng.permutation(len(y_balanced))
        return X_balanced[shuffle_idx], y_balanced[shuffle_idx]

    @staticmethod
    def _build_class_weight_map(y, min_weight=0.6, max_weight=8.0):
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        total = float(np.sum(counts))
        n_classes = float(len(classes))
        class_weights = {}
        for class_id, class_count in zip(classes, counts):
            raw = total / (n_classes * float(class_count))
            class_weights[int(class_id)] = float(np.clip(raw, min_weight, max_weight))
        return class_weights

    @staticmethod
    def _build_sample_weights(y, class_weights):
        return np.array([class_weights.get(int(lbl), 1.0) for lbl in y], dtype=np.float32)

    def compile(self, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=None, learning_rate=2e-4):
        if metrics is None:
            metrics = ["accuracy"]
        if optimizer == "adam":
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        if loss == "sparse_categorical_crossentropy":
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    @classmethod
    def train(cls, X_train_raw, y_train, X_val_raw, y_val, output_dir, model_filename, epochs):
        print("\nExtracting HOG features for Training...")
        X_train = cls.compute_hog_batch(X_train_raw)
        X_val = cls.compute_hog_batch(X_val_raw)

        input_shape = X_train.shape[1:]
        num_classes = len(np.unique(np.concatenate([y_train, y_val])))

        print(f"HOG Vector shape per sequence: {input_shape}")
        model = cls(input_shape=input_shape, num_classes=num_classes)

        class_weights = cls._build_class_weight_map(y_train)
        X_train, y_train = cls._oversample_minority_classes(X_train, y_train)
        train_weights = cls._build_sample_weights(y_train, class_weights)
        val_weights = cls._build_sample_weights(y_val, class_weights)

        model.compile(learning_rate=1e-3)

        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, model_filename)
        checkpoint = callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, verbose=1
        )

        history = model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val, val_weights),
            epochs=epochs,
            batch_size=32,
            sample_weight=train_weights,
            callbacks=[checkpoint],
        )

        model.model.load_weights(model_path)
        return history.history, model

    def predict(self, images_np):
        hog_vecs = self.compute_hog_batch(images_np)
        return np.argmax(self.model.predict(hog_vecs), axis=1)


# ----------------------------
# POMOCNICZE FUNKCJE WIDEO
# ----------------------------
def parse_label(path):
    p = path.lower()
    if any(x in p for x in ["confusion", "surprise", "angry"]):
        return 0
    if any(x in p for x in ["neutral", "happiness"]):
        return 1
    return None


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []

    idxs = np.linspace(0, total - 1, SEQ_LEN).astype(int)
    frames = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Konwersja dla MTCNN (oczekuje PIL lub numpy array)
        from PIL import Image
        frame_pil = Image.fromarray(frame)
        face = mtcnn(frame_pil)

        if face is None:
            continue

        # Zamiast trzymać w Pytorch, konwertujemy z powrotem na numpy array (H, W, 3) dla skimage/Keras
        face_np = face.permute(1, 2, 0).byte().cpu().numpy()
        frames.append(face_np)

    cap.release()

    # Padding na wypadek braku wykrycia twarzy (zastępczy czarny obraz)
    if len(frames) == 0:
        return [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)] * SEQ_LEN

    while len(frames) < SEQ_LEN:
        frames.append(frames[-1])

    return frames


def load_dataset():
    print("Skanowanie plików wideo...")
    paths, labels = [], []
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if not f.endswith((".mp4", ".avi", ".mkv", ".mov")):
                continue
            path = os.path.join(root, f)
            label = parse_label(path)
            if label is None:
                continue
            paths.append(path)
            labels.append(label)

    print(f"Znaleziono {len(paths)} plików z przypisanymi etykietami. Ekstrakcja klatek (to potrwa chwilę)...")

    X, Y = [], []
    for path, label in zip(paths, labels):
        frames = extract_frames(path)
        if len(frames) == SEQ_LEN:
            X.append(np.array(frames))  # Shape: (SEQ_LEN, H, W, 3)
            Y.append(label)

    return np.array(X), np.array(Y)


# ----------------------------
# MAIN
# ----------------------------
def main():
    # 1. Wczytanie i przygotowanie danych do pamięci RAM jako numpy arrays
    X_data, y_data = load_dataset()

    if len(X_data) == 0:
        print("Nie znaleziono żadnych plików lub nie udało się wyekstrahować twarzy. Sprawdź DATASET_DIR.")
        return

    print(f"\nZebrano dane. Kształt X: {X_data.shape}, Kształt y: {y_data.shape}")

    # 2. Podział na zbiory
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, stratify=y_data, random_state=42
    )
    print(f"Zbiór treningowy: {len(X_train)} próbek. Zbiór testowy: {len(X_test)} próbek.")

    # 3. Trening HogModel
    print("\n--- Rozpoczynanie treningu HogModel ---")
    output_directory = "hog_output"

    history, trained_model = HogModel.train(
        X_train_raw=X_train,
        y_train=y_train,
        X_val_raw=X_test,
        y_val=y_test,
        output_dir=output_directory,
        model_filename="best_hog_model.keras",
        epochs=100
    )

    # 4. Ewaluacja i Testowanie
    print("\n--- Rozpoczęto testowanie na zbiorze walidacyjnym ---")
    predictions = trained_model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"\n=> DOKŁADNOŚĆ TESTOWA (Test Accuracy): {acc * 100:.2f}%")

    print("\nRaport Klasyfikacji:")
    print(classification_report(y_test, predictions))

    # 5. Generowanie Wykresów i Macierzy Konfuzji
    print("\nGenerowanie wizualizacji...")

    # 5a. Wykresy Historii Treningu (Accuracy i Loss)
    plt.figure(figsize=(12, 5))

    # Wykres Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Wykres Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "training_history.png"))
    plt.close()

    # 5b. Macierz Konfuzji
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negatywne (0)', 'Pozytywne/Neutralne (1)'],
                yticklabels=['Negatywne (0)', 'Pozytywne/Neutralne (1)'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "confusion_matrix.png"))
    plt.close()

    print(f"Zapisano wizualizacje do folderu: {output_directory}")


if __name__ == "__main__":
    main()
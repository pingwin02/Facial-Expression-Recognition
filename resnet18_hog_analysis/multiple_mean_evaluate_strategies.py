import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "binary_output/output"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSES = ["negative", "other"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def load_dataset(data_path):
    dataset = datasets.ImageFolder(data_path, transform=transform)

    print(f"[DEBUG] Znalezione klasy w folderze {data_path}: {dataset.classes}")

    global CLASSES
    CLASSES = dataset.classes

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

def get_cnn():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(DEVICE)


def train_model(model, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item()

        val_losses.append(val_loss)

        print(f"Epoch {epoch}: train_loss={total_loss:.3f} val_loss={val_loss:.3f}")

    return train_losses, val_losses


def evaluate(model, test_loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            out = model(X)
            p = torch.argmax(out, dim=1).cpu().numpy()

            preds.extend(p)
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds)

    cm = confusion_matrix(labels, preds, labels=np.arange(len(CLASSES)))

    return acc, cm, preds, labels


def plot_confusion(cm, strategy_name, model_name):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"{strategy_name} - {model_name}")
    plt.savefig(f"{RESULTS_DIR}/cm_{strategy_name}_{model_name}.png")
    plt.close()


def plot_combined_mean_confusion(strategy_name, cms_dict):
    sorted_keys = sorted(cms_dict.keys())
    num_plots = len(sorted_keys)

    if num_plots == 0:
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=True)

    if num_plots == 1:
        axes = [axes]

    fig.suptitle(f'Mean Confusion Matrix - Strategia: {strategy_name}', fontsize=18, y=1.05)
    annot_font_size = 10 if len(CLASSES) > 4 else 12

    for ax, key in zip(axes, sorted_keys):
        cm_sum = np.array(cms_dict[key], dtype=float)

        cm_norm = cm_sum / (cm_sum.sum(axis=1, keepdims=True) + 1e-9)

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                    vmin=0.0, vmax=1.0, cbar=False, ax=ax,
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    square=True, annot_kws={"size": annot_font_size})

        ax.set_title(f"Konfiguracja: {key}", fontsize=14, pad=15)
        ax.set_xlabel('Przewidywana klasa', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        if ax == axes[0]:
            ax.set_ylabel('Rzeczywista klasa', fontsize=12)
            ax.tick_params(axis='y', rotation=0)

    plt.subplots_adjust(right=0.9, bottom=0.2)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=11)

    plt.savefig(f"{RESULTS_DIR}/mean_cm_{strategy_name}_combined.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_training(train_losses, val_losses, strategy, model):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.title(f"{strategy} - {model}")
    plt.savefig(f"{RESULTS_DIR}/loss_{strategy}_{model}.png")
    plt.close()


def save_gradcam_examples(dataset, model, strategy, model_name):
    print("[INFO] Generuję GradCAM...")

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    plt.figure(figsize=(12, 6))

    for i, (X, y) in enumerate(loader):
        if i >= 4:
            break

        X = X.to(DEVICE)
        grayscale_cam = cam(input_tensor=X)[0]

        rgb_img = X.cpu().squeeze().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        plt.subplot(1, 4, i + 1)
        plt.imshow(cam_image)
        plt.title(f"GT:{CLASSES[y.item()]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/gradcam_{strategy}_{model_name}.png")
    plt.close()


def save_examples(dataset, model, strategy, model_name):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    plt.figure(figsize=(10, 10))

    for i, (X, y) in enumerate(loader):
        if i >= 5:
            break

        X = X.to(DEVICE)
        out = model(X)
        pred = torch.argmax(out, dim=1).item()

        img = X.cpu().squeeze().permute(1, 2, 0).numpy()

        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(f"P:{CLASSES[pred]}\nT:{CLASSES[y.item()]}")
        plt.axis("off")

    plt.savefig(f"{RESULTS_DIR}/examples_{strategy}_{model_name}.png")
    plt.close()


def run_for_strategy(strategy_name, run_idx=None):
    results = {}
    base_path = os.path.join(DATA_ROOT, strategy_name)

    if not os.path.exists(base_path):
        print(f"Brak folderu: {base_path}")
        return results

    for frame_dir in os.listdir(base_path):
        path = os.path.join(base_path, frame_dir)

        if not os.path.isdir(path):
            continue

        print(f"\n--- STRATEGIA: {strategy_name} | KLATKI: {frame_dir} ---")

        try:
            train_ds, val_ds, test_ds = load_dataset(path)
        except Exception as e:
            print(f"[BŁĄD W ZBIORZE DANYCH] {path}: {e}")
            continue

        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            print(f"[SKIP] Za mało danych: {path}")
            continue

        print(f"[INFO] train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16)

        model_name = "cnn"
        model = get_cnn()
        print(f"\n=== Trenuję: {model_name} ===")

        train_losses, val_losses = train_model(model, train_loader, val_loader)
        acc, cm, preds, labels = evaluate(model, test_loader)

        print(f"[WYNIK] Accuracy: {acc:.4f}")

        safe_name = f"{strategy_name}_{frame_dir}"
        if run_idx is not None:
            safe_name += f"_run{run_idx}"

        plot_confusion(cm, safe_name, model_name)
        plot_training(train_losses, val_losses, safe_name, model_name)
        save_examples(test_ds, model, safe_name, model_name)
        save_gradcam_examples(test_ds, model, safe_name, model_name)

        results[frame_dir] = {
            "acc": acc,
            "cm": cm
        }

    return results


def run_multiple_times(strategy_name="edges", num_runs=10):
    all_runs_acc = {}
    all_runs_cms = {}
    strategy_summary = {}

    if not os.path.exists(DATA_ROOT):
        print(f"Brak folderu głównego: {DATA_ROOT}")
        return {}

    print(f"\n######## ROZPOCZYNAM TRENING {num_runs} RAZY DLA STRATEGII: {strategy_name.upper()} ########")

    for i in range(1, num_runs + 1):
        print(f"\n======================================")
        print(f"       STRATEGIA: {strategy_name} | PRÓBA {i}/{num_runs} ")
        print(f"======================================")

        res = run_for_strategy(strategy_name, run_idx=i)

        for frame_dir, data in res.items():
            config_key = f"{frame_dir}_cnn"

            if config_key not in all_runs_acc:
                all_runs_acc[config_key] = []
            all_runs_acc[config_key].append(data["acc"])

            if frame_dir not in all_runs_cms:
                all_runs_cms[frame_dir] = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)
            all_runs_cms[frame_dir] += data["cm"]

    print(f"\n============= PODSUMOWANIE {num_runs} PRÓB ({strategy_name}) =============")
    for config, acc_list in all_runs_acc.items():
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)

        strategy_summary[config] = mean_acc

        print(f"Konfiguracja: {config}")
        print(f"  - Średnie Accuracy: {mean_acc:.4f}")
        print(f"  - Odchylenie Std:   {std_acc:.4f}")
        print(f"  - Wszystkie wyniki: {[round(a, 4) for a in acc_list]}")
        print("-" * 50)

    if all_runs_cms:
        plot_combined_mean_confusion(strategy_name, all_runs_cms)

    return strategy_summary


if __name__ == "__main__":
    strategies = [
        "center_dense", "edges", "hog_diff", "kmeans_rgb",
        "motion_max", "random", "resnet_kmeans", "sharpest", "uniform"
    ]
    NUM_RUNS = 10

    final_results = {}

    for strat in strategies:
        strat_summary = run_multiple_times(strategy_name=strat, num_runs=NUM_RUNS)
        if strat_summary:
            final_results[strat] = strat_summary

    if final_results:
        print("\n\n" + "*" * 80)
        print(f"   MACIERZ WYNIKÓW: ŚREDNIE ACCURACY (Z {NUM_RUNS} PRÓB DLA KAŻDEJ METODY)   ")
        print("*" * 80)

        df_results = pd.DataFrame(final_results).T
        print(df_results.to_string(na_rep="Brak danych"))

        csv_path = os.path.join(RESULTS_DIR, "final_summary_matrix_multiple_classes.csv")
        df_results.to_csv(csv_path)
        print(f"\nMacierz zapisano również w: {csv_path}")
        print("*" * 80 + "\n")
    else:
        print("\nNie udało się zebrać żadnych wyników. Sprawdź strukturę folderów w 'output'.")
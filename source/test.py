import os
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from dataset import Traubenwelke


def load_packed_samples(data_dir: str, seed: int, overwrite: bool = False):
    samples_path = Path("all_samples.pkl")
    labels_path = Path("all_labels.pkl")

    if samples_path.exists() and labels_path.exists() and not overwrite:
        with samples_path.open("rb") as f:
            samples = pickle.load(f)
        with labels_path.open("rb") as f:
            labels = pickle.load(f)
    else:
        healthy_path = Path(data_dir) / "healthy_images"
        diseased_path = Path(data_dir) / "BS_images"

        samples = []
        labels = []

        for image_path in sorted(healthy_path.iterdir()):
            samples.append(Image.open(image_path))
            labels.append(0)

        for image_path in sorted(diseased_path.iterdir()):
            samples.append(Image.open(image_path))
            labels.append(1)

        with samples_path.open("wb") as f:
            pickle.dump(samples, f)
        with labels_path.open("wb") as f:
            pickle.dump(labels, f)

    rng = random.Random(seed)
    combined = list(zip(samples, labels))
    rng.shuffle(combined)
    samples, labels = zip(*combined)
    return list(samples), list(labels)


def build_model(device: torch.device) -> nn.Module:
    classifier = resnet18(ResNet18_Weights.DEFAULT)
    in_features = classifier.fc.in_features
    classifier.fc = nn.Linear(in_features, 2)
    return classifier.to(device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            predictions = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["healthy", "BS"])
    ax.set_yticklabels(["healthy", "BS"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path)
    #fig.close()


def main():
    seed = 42
    data_dir = "../data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 224
    batch_size = 32

    samples, labels = load_packed_samples(data_dir, seed, overwrite=False)

    _, test_samples, _, test_labels = train_test_split(
        samples,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )

    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = Traubenwelke(test_samples, test_labels, transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(device)
    ckpt_path = Path("best_ResNet18_traubenwelke.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
    else:
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found; run train.py first.")

    y_true, y_pred = evaluate(model, test_loader, device)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)

    with (results_dir / "metrics.txt").open("w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

    plot_confusion_matrix(y_true, y_pred, results_dir / "confusion_matrix.png")

    print("Test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print(f"Confusion matrix saved to {results_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()

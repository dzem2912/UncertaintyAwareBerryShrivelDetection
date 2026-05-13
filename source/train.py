import os
import torch
import torch.nn as nn
from pathlib import Path
import random
import pickle
import numpy as np
from dataset import Traubenwelke
from torchvision import transforms
from collections import Counter

from tqdm import tqdm
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Runs the model over a data loader and returns a metrics dict containing
    loss, accuracy, balanced accuracy, macro precision, macro recall,
    macro F1, and Brier score, plus the raw predictions and labels.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    y_true, y_pred, y_prob_pos = [], [], []

    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        correct    += (preds == y).sum().item()
        total      += x.size(0)

        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob_pos.extend(probs[:, 1].cpu().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_prob_np = np.array(y_prob_pos)

    metrics = {
        "loss":              total_loss / total,
        "accuracy":          correct / total,
        "balanced_accuracy": balanced_accuracy_score(y_true_np, y_pred_np),
        "precision":         precision_score(y_true_np, y_pred_np, average="macro", zero_division=0),
        "recall":            recall_score(y_true_np, y_pred_np, average="macro", zero_division=0),
        "f1":                f1_score(y_true_np, y_pred_np, average="macro", zero_division=0),
        "brier_loss":        brier_score_loss(y_true_np, y_prob_np),
    }

    return metrics, y_true, y_pred


def print_metrics(split: str, metrics: dict):
    print(
        f"  [{split}] loss {metrics['loss']:.4f} | "
        f"acc {metrics['accuracy']:.4f} | "
        f"bal-acc {metrics['balanced_accuracy']:.4f} | "
        f"precision {metrics['precision']:.4f} | "
        f"recall {metrics['recall']:.4f} | "
        f"F1 {metrics['f1']:.4f} | "
        f"brier {metrics['brier_loss']:.4f}"
    )


def main():
    seed: int = 42
    seed_everything(seed)
    data_dir: str = '../data'
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size: int = 224
    batch_size: int = 32
    overwrite: bool = False

    healthy_images_path: str = os.path.join(data_dir, 'healthy_images')
    traubenwelke_images_path: str = os.path.join(data_dir, 'BS_images')

    healthy_samples: list[str] = [os.path.join(healthy_images_path, file) for file in os.listdir(healthy_images_path)]
    traubenwelke_samples: list[str] = [os.path.join(traubenwelke_images_path, file) for file in os.listdir(traubenwelke_images_path)]

    if os.path.exists('all_samples.pkl') and os.path.exists('all_labels.pkl') and not overwrite:
        with open("all_samples.pkl", "rb") as f:
            all_samples = pickle.load(f)
        with open("all_labels.pkl", "rb") as f:
            all_labels = pickle.load(f)
    else:
        all_samples = []
        all_labels = []

        for healthy_sample in tqdm(healthy_samples):
            with Image.open(healthy_sample) as img:
                img = img.convert("RGB")
                healthy_image = img.copy()
            all_samples.append(healthy_image)
            all_labels.append(0)

        for traubenwelke_sample in tqdm(traubenwelke_samples):
            with Image.open(traubenwelke_sample) as img:
                img = img.convert("RGB")
                traubenwelke_image = img.copy()
            all_samples.append(traubenwelke_image)
            all_labels.append(1)

        with open("all_samples.pkl", "wb") as f:
            pickle.dump(all_samples, f)
        with open("all_labels.pkl", "wb") as f:
            pickle.dump(all_labels, f)

    rng = random.Random(seed)
    combined = list(zip(all_samples, all_labels))
    rng.shuffle(combined)
    all_samples, all_labels = zip(*combined)

    # First split: 85% train+val, 15% test
    trainval_samples, test_samples, trainval_labels, test_labels = train_test_split(
        all_samples,
        all_labels,
        test_size=0.15,
        stratify=all_labels,
        random_state=seed,
    )

    # Second split: 70% train, 15% val (i.e. 15/85 ≈ 0.176 of the trainval set)
    train_samples, val_samples, train_labels, val_labels = train_test_split(
        trainval_samples,
        trainval_labels,
        test_size=0.15 / 0.85,
        stratify=trainval_labels,
        random_state=seed,
    )

    counts = Counter(train_labels)
    print(f"There are {counts[0]} healthy and {counts[1]} Traubenwelke samples for training!")
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Traubenwelke(train_samples, train_labels, transform=train_tf)
    val_dataset = Traubenwelke(val_samples,   val_labels,   transform=eval_tf)
    test_dataset = Traubenwelke(test_samples,  test_labels,  transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    classifier_name: str = 'ResNet18'
    ckpt_path = f'best_{classifier_name}_traubenwelke_v2.pt'

    classifier = resnet18(ResNet18_Weights.DEFAULT)
    in_features = classifier.fc.in_features
    classifier.fc = nn.Linear(in_features, 2)
    classifier = classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)

    best_val_accuracy: float = -1.0
    epochs: int = 20
    tolerance: int = 7
    tolerance_counter: int = 0

    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0

        for img, label in tqdm(train_loader, desc=f'Epoch: {epoch}/{epochs}', leave=False):
            imgs   = img.to(device)
            labels = label.to(device)

            optimizer.zero_grad()
            logits: torch.Tensor = classifier(imgs)
            loss: torch.Tensor   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            predictions = logits.argmax(dim=1)
            correct    += (predictions == labels).sum().item()
            total      += imgs.size(0)

        train_loss     = total_loss / total
        train_accuracy = correct / total

        # Early stopping is driven by the validation set
        val_metrics, _, _ = evaluate(classifier, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}: train loss {train_loss:.4f} | train acc {train_accuracy:.4f}")
        print_metrics("val", val_metrics)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state": classifier.state_dict(),
                    "img_size":    image_size,
                    "classes":     ["healthy", "BS"],
                },
                ckpt_path,
            )
            print(f"  Saved best -> {ckpt_path} (val acc {best_val_accuracy:.4f})")
            tolerance_counter = 0
        else:
            tolerance_counter += 1
            if tolerance_counter >= tolerance:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Final evaluation on the held-out test set
    print("\nLoading best checkpoint for final test evaluation...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state"])

    test_metrics, _, _ = evaluate(classifier, test_loader, criterion, device)
    print("\nFinal test results:")
    print_metrics("test", test_metrics)


if __name__ == "__main__":
    main()

"""
import os
import torch
import torch.nn as nn
from pathlib import Path
import random
import pickle
import numpy as np
from dataset import Traubenwelke
from torchvision import transforms
from collections import Counter

from tqdm import tqdm
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    y_true, y_pred = [], []

    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += x.size(0)

        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return total_loss / total, correct / total, y_true, y_pred


def main():
    seed: int = 42
    seed_everything(seed)
    data_dir: str = '../data'
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size: int = 224
    batch_size: int = 32
    overwrite: bool = False

    healthy_images_path: str = os.path.join(data_dir, 'healthy_images')
    traubenwelke_images_path: str = os.path.join(data_dir, 'BS_images')

    healthy_samples: list[str] = [os.path.join(healthy_images_path, file) for file in os.listdir(healthy_images_path)]
    traubenwelke_samples: list[str] = [os.path.join(traubenwelke_images_path, file) for file in os.listdir(traubenwelke_images_path)]

    if os.path.exists('all_samples.pkl') and os.path.exists('all_labels.pkl') and not overwrite:
        with open("all_samples.pkl", "rb") as f:
            all_samples = pickle.load(f)
        with open("all_labels.pkl", "rb") as f:
            all_labels = pickle.load(f)
    else:
        all_samples = []
        all_labels = []

        for healthy_sample in tqdm(healthy_samples):
            with Image.open(healthy_sample) as img:
                img = img.convert("RGB")
                healthy_image = img.copy()
            all_samples.append(healthy_image)
            all_labels.append(0)

        for traubenwelke_sample in tqdm(traubenwelke_samples):
            with Image.open(traubenwelke_sample) as img:
                img = img.convert("RGB")
                traubenwelke_image = img.copy()
            all_samples.append(traubenwelke_image)
            all_labels.append(1)

        with open("all_samples.pkl", "wb") as f:
            pickle.dump(all_samples, f)
        with open("all_labels.pkl", "wb") as f:
            pickle.dump(all_labels, f)

    rng = random.Random(seed)
    combined = list(zip(all_samples, all_labels))
    rng.shuffle(combined)
    all_samples, all_labels = zip(*combined)

    # First split: 85% train+val, 15% test
    trainval_samples, test_samples, trainval_labels, test_labels = train_test_split(
        all_samples,
        all_labels,
        test_size=0.15,
        stratify=all_labels,
        random_state=seed,
    )

    # Second split: 70% train, 15% val (i.e. 15/85 ≈ 0.176 of the trainval set)
    train_samples, val_samples, train_labels, val_labels = train_test_split(
        trainval_samples,
        trainval_labels,
        test_size=0.15 / 0.85,
        stratify=trainval_labels,
        random_state=seed,
    )

    counts = Counter(train_labels)
    print(f"There are {counts[0]} healthy and {counts[1]} Traubenwelke samples for training!")
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Traubenwelke(train_samples, train_labels, transform=train_tf)
    val_dataset = Traubenwelke(val_samples, val_labels, transform=eval_tf)
    test_dataset = Traubenwelke(test_samples, test_labels, transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    classifier_name: str = 'ResNet18'
    ckpt_path = f'best_{classifier_name}_traubenwelke_v2.pt'

    classifier = resnet18(ResNet18_Weights.DEFAULT)
    in_features = classifier.fc.in_features
    classifier.fc = nn.Linear(in_features, 2)
    classifier = classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)

    best_val_accuracy: float = -1.0
    epochs: int = 20
    tolerance: int = 7
    tolerance_counter: int = 0

    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0

        for img, label in tqdm(train_loader, desc=f'Epoch: {epoch}/{epochs}', leave=False):
            imgs = img.to(device)
            labels = label.to(device)

            optimizer.zero_grad()
            logits: torch.Tensor = classifier(imgs)
            loss: torch.Tensor = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += imgs.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total

        # Early stopping is driven by the validation set
        val_loss, val_accuracy, _, _ = evaluate(classifier, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}: train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
              f"val loss {val_loss:.4f} acc {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            
            torch.save(
                {
                    "model_state": classifier.state_dict(),
                    "img_size": image_size,
                    "classes": ["healthy", "BS"],
                },
                ckpt_path,
            )

            print(f"  Saved best -> {ckpt_path} (val acc {best_val_accuracy:.4f})")
            tolerance_counter = 0
        else:
            tolerance_counter += 1
            if tolerance_counter >= tolerance:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Final evaluation on the held-out test set
    print("\nLoading best checkpoint for final test evaluation...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state"])

    test_loss, test_accuracy, _, _ = evaluate(classifier, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
"""
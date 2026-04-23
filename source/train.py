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
from torch.utils.data import DataLoader, random_split
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
            #healthy_image = Image.open(healthy_sample) # Height x Width x Channel
            all_samples.append(healthy_image)
            all_labels.append(0)

        for traubenwelke_sample in tqdm(traubenwelke_samples):
            #print(f'Traubenwelke sample: {traubenwelke_sample}')
            with Image.open(traubenwelke_sample) as img:
                img = img.convert("RGB")
                traubenwelke_image = img.copy()
            #traubenwelke_image = Image.open(traubenwelke_sample)
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

    train_samples, test_samples, train_labels, test_labels = train_test_split(
        all_samples,
        all_labels,
        test_size=0.2,
        stratify=all_labels,
        random_state=seed
    )

    counts = Counter(train_labels)
    zeros = counts[0]
    ones  = counts[1]

    print(f'There are {zeros} healthy and {ones} Traubenwelke samples for training!')

    print(f"Train: {len(train_samples)} | Test: {len(test_samples)}")

    
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Traubenwelke(train_samples, train_labels, transform=train_tf)
    test_dataset = Traubenwelke(test_samples, test_labels, transform=test_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    classifier_name: str = 'ResNet18'
    ckpt_path = (f'best_{classifier_name}_traubenwelke.pt')

    classifier = resnet18(ResNet18_Weights.DEFAULT)
    in_features = classifier.fc.in_features
    classifier.fc = nn.Linear(in_features, 2)
    classifier = classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)

    best_accuracy: float = -1.0
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

            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += batch_size

        train_loss = total_loss / total
        train_accuracy = correct / total

        test_loss, test_accuracy, _, _ = evaluate(classifier, test_loader, criterion, device)

        print(f"Epoch {epoch:02d}: train loss {train_loss:.4f} acc {train_accuracy:.4f} |"
              f"test loss {test_loss:.4f} acc {test_accuracy:.4f}")


        if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save({
                            "model_state": classifier.state_dict(),
                            "img_size": image_size,
                            "classes": ["healthy", "BS"],
                            },
                            ckpt_path)
                
                print(f"  Saved best -> {ckpt_path} (test acc {best_accuracy:.4f})")
                tolerance_counter = 0
        else:
            tolerance_counter += 1

if __name__ == "__main__":
    main()
import os
import random
import pickle
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from dataset import Traubenwelke


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class EvidentialResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = resnet18(ResNet18_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)          # [B,K] real
        evidence = F.softplus(logits)   # [B,K] >= 0
        alpha = evidence + 1.0          # [B,K] > 0
        return alpha

@torch.no_grad()
def edl_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    """
    Subjective logic uncertainty:
      u = K / sum(alpha)
    """
    K = alpha.size(1)
    S = alpha.sum(dim=1)
    return K / S   

@torch.no_grad()
def collect_edl_confidence(model, loader, device):
    model.eval()
    conf_all = []
    u_all = []
    y_all = []
    for x, y in tqdm(loader, desc="collect conf", leave=False):
        x = x.to(device)
        alpha = model(x)                                  # [B,K]
        probs = alpha / alpha.sum(dim=1, keepdim=True)    # [B,K] mean
        conf = probs.max(dim=1).values                    # [B]
        u = edl_uncertainty(alpha)                        # [B] (optional)
        conf_all.append(conf.cpu())
        u_all.append(u.cpu())
        y_all.append(y.cpu())
    return torch.cat(conf_all), torch.cat(u_all), torch.cat(y_all)


if __name__ == "__main__":
    seed: int = 42
    seed_everything(seed)

    data_dir: str = "../data"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size: int = 224
    batch_size: int = 8
    overwrite: bool = True

    num_classes: int = 2
    epochs: int = 20

    # 2025 dataset
    healthy_2025_images_path: str = os.path.join(data_dir, "healthy_images")
    traubenwelke_2025_images_path: str = os.path.join(data_dir, "BS_images")

    healthy_2025_samples: list[str] = [os.path.join(healthy_2025_images_path, file) for file in os.listdir(healthy_2025_images_path)]
    traubenwelke_2025_samples: list[str] = [os.path.join(traubenwelke_2025_images_path, file) for file in os.listdir(traubenwelke_2025_images_path)]

    all_2025_samples = []
    all_2025_labels = []

    for healthy_sample in tqdm(healthy_2025_samples, desc="load healthy"):
        with Image.open(healthy_sample) as img:
                all_2025_samples.append(img.convert("RGB").copy())
        #healthy_image = Image.open(healthy_sample)
        #all_2025_samples.append(healthy_image)
        all_2025_labels.append(0)

    for traubenwelke_sample in tqdm(traubenwelke_2025_samples, desc="load BS"):
        with Image.open(traubenwelke_sample) as img:
            all_2025_samples.append(img.convert('RGB').copy())
        #traubenwelke_image = Image.open(traubenwelke_sample)    
        #all_2025_samples.append(traubenwelke_image)
        all_2025_labels.append(1)

    rng = random.Random(seed)
    combined = list(zip(all_2025_samples, all_2025_labels))
    rng.shuffle(combined)
    all_2025_samples, all_2025_labels = zip(*combined)

    train_2025_samples, test_2025_samples, train_2025_labels, test_2025_labels = train_test_split(
        all_2025_samples,
        all_2025_labels,
        test_size=0.2,
        stratify=all_2025_labels,
        random_state=seed,
    )

    counts = Counter(train_2025_labels)
    zeros = counts[0]
    ones = counts[1]
    print(f"There are {zeros} healthy and {ones} traubenwelke samples for training!")
    print(f"Train: {len(train_2025_samples)} | Test: {len(test_2025_samples)}")

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

    #train_2025_dataset = Traubenwelke(train_2025_samples, train_2025_labels, transform=train_tf)
    test_2025_dataset = Traubenwelke(test_2025_samples, test_2025_labels, transform=test_tf)
    
    test_2025_loader = DataLoader(test_2025_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


    # 2023 dataset
    data_dir = '../data/2023'
    healthy_ood_path = os.path.join(data_dir, "healthy_images") 
    traubenwelke_ood_path = os.path.join(data_dir, "BS_images")

    healthy_ood_samples: list[str] = [os.path.join(healthy_ood_path, file) for file in os.listdir(healthy_ood_path)]
    traubenwelke_ood_samples: list[str] = [os.path.join(traubenwelke_ood_path, file) for file in os.listdir(traubenwelke_ood_path)]
    
    all_ood_samples = []
    all_ood_labels = []
    for healthy_ood_sample in tqdm(healthy_ood_samples):
        with Image.open(healthy_ood_sample) as img:
            all_ood_samples.append(img.convert("RGB").copy())
        #healthy_ood_image = Image.open(healthy_ood_sample) # Height x Width x Channel
        #all_ood_samples.append(healthy_ood_image)
        all_ood_labels.append(0)

    for traubenwelke_ood_sample in tqdm(traubenwelke_ood_samples):
        with Image.open(traubenwelke_ood_sample) as img:
                all_ood_samples.append(img.convert("RGB").copy())
        #traubenwelke_ood_image = Image.open(traubenwelke_ood_sample)
        #all_ood_samples.append(traubenwelke_ood_image)
        all_ood_labels.append(1)
    
    counts = Counter(all_ood_labels)
    zeros = counts[0]
    ones  = counts[1]

    rng = random.Random(seed)
    combined = list(zip(all_ood_samples, all_ood_labels))
    rng.shuffle(combined)
    all_ood_samples, all_ood_labels = zip(*combined)

    print(f'There are {zeros} healthy and {ones} traubenwellke samples for training!')

    ood_dataset = Traubenwelke(all_ood_samples, all_ood_labels, transform=test_tf)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    classifier_name: str = "ResNet_EDL"
    ckpt_path = f"best_{classifier_name}_traubenwelke.pt"

    classifier = EvidentialResNet(num_classes=num_classes)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        classifier.load_state_dict(checkpoint["model_state"])

    
    classifier = classifier.to(device)
    classifier.eval()

    conf_2025, u_2025, y_2025 = collect_edl_confidence(classifier, test_2025_loader, device)
    conf_2023, u_2023, y_2023 = collect_edl_confidence(classifier, ood_loader, device)

    print("Standard deviation confidence 2025:", conf_2025.std().item())
    print("Standard deviation confidence 2023:", conf_2023.std().item())

    conf_2025_np = conf_2025.numpy()
    conf_2023_np = conf_2023.numpy()

    plt.figure(figsize=(12, 5))
    bins = np.linspace(0, 1, 21)

    # Left plot
    plt.subplot(1, 2, 1)
    plt.hist(conf_2025_np, bins=bins, density=False)
    plt.title("2025 (ID)")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)

    # Right plot
    plt.subplot(1, 2, 2)
    plt.hist(conf_2023_np, bins=bins, density=False)
    plt.title("2023 (OoD)")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)

    plt.suptitle("EDL Confidence Scores for 2025 and 2023")
    plt.tight_layout()
    plt.show()
    plt.close()

    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve
    
    id_scores = u_2025.numpy()
    ood_scores = u_2023.numpy()

    y_true  = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])

    auroc_u = roc_auc_score(y_true, y_score)
    print("EDL AUROC (using uncertainty u):", auroc_u)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fpr95 = fpr[np.searchsorted(tpr, 0.95)]
    print("EDL FPR@95TPR (u):", fpr95)
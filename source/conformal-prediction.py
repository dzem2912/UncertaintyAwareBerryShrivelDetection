import csv
import os
import pickle
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from dataset import Traubenwelke


CLASS_NAMES = ["healthy", "BS"]
NUM_CLASSES = 2


class UnlabeledImageDataset(Dataset):
    """Lightweight wrapper for unlabeled OoD pilot images."""

    def __init__(self, samples, filenames, transform=None):
        self.samples = samples
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = self.samples[index].convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.filenames[index]


class LabeledImageDatasetWithPaths(Dataset):
    """Evaluation wrapper that keeps filenames alongside labels."""

    def __init__(self, samples, labels, filenames, transform=None):
        self.samples = samples
        self.labels = labels
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.samples[index].convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label, self.filenames[index]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
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

    return total_loss / max(total, 1), correct / max(total, 1), y_true, y_pred


@torch.no_grad()
def compute_calibration_alphas_by_class(model, loader, device, num_classes=NUM_CLASSES):
    model.eval()
    cal_by_class = {c: [] for c in range(num_classes)}

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        true_probs = probs[torch.arange(len(y), device=y.device), y]
        alpha = 1.0 - true_probs

        for c in range(num_classes):
            mask = y == c
            if mask.any():
                cal_by_class[c].append(alpha[mask].detach().cpu())

    for c in range(num_classes):
        if len(cal_by_class[c]) == 0:
            raise ValueError(f"No calibration samples for class {c}.")
        cal_by_class[c] = torch.cat(cal_by_class[c], dim=0)

    return cal_by_class


@torch.no_grad()
def compute_pvalues_class_conditional(model, loader, cal_by_class, device, num_classes=NUM_CLASSES):
    model.eval()
    cal_by_class_dev = {c: cal_by_class[c].to(device) for c in range(num_classes)}
    n_by_class = {c: len(cal_by_class[c]) for c in range(num_classes)}

    all_pvals = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        alpha_xy = 1.0 - probs

        pvals = torch.empty_like(alpha_xy)
        for c in range(num_classes):
            cal_c = cal_by_class_dev[c]
            n_c = n_by_class[c]
            counts = (cal_c[:, None] >= alpha_xy[:, c][None, :]).sum(dim=0)
            pvals[:, c] = (counts + 1.0) / (n_c + 1.0)

        all_pvals.append(pvals.detach().cpu())
        all_labels.append(y.detach().cpu())

    return torch.cat(all_pvals, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def collect_labeled_results(model, loader, cal_by_class, device):
    model.eval()
    cal_by_class_dev = {c: cal_by_class[c].to(device) for c in range(NUM_CLASSES)}
    n_by_class = {c: len(cal_by_class[c]) for c in range(NUM_CLASSES)}

    rows = []
    for x, y, filenames in tqdm(loader, desc="collect labeled", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        alpha_xy = 1.0 - probs
        pred_idx = probs.argmax(dim=1)
        pred_conf = probs.max(dim=1).values

        pvals = torch.empty_like(alpha_xy)
        for c in range(NUM_CLASSES):
            cal_c = cal_by_class_dev[c]
            counts = (cal_c[:, None] >= alpha_xy[:, c][None, :]).sum(dim=0)
            pvals[:, c] = (counts + 1.0) / (n_by_class[c] + 1.0)

        max_pvals = pvals.max(dim=1).values
        ood_scores = 1.0 - max_pvals
        true_label_pvals = pvals[torch.arange(len(y), device=y.device), y]

        for idx, filename in enumerate(filenames):
            rows.append({
                "filename": filename,
                "true_label": CLASS_NAMES[int(y[idx].item())],
                "true_label_index": int(y[idx].item()),
                "predicted_class": CLASS_NAMES[int(pred_idx[idx].item())],
                "predicted_class_index": int(pred_idx[idx].item()),
                "predicted_softmax_confidence": float(pred_conf[idx].item()),
                "pvalue_healthy": float(pvals[idx, 0].item()),
                "pvalue_BS": float(pvals[idx, 1].item()),
                "max_pvalue": float(max_pvals[idx].item()),
                "ood_score": float(ood_scores[idx].item()),
                "true_label_pvalue": float(true_label_pvals[idx].item()),
            })

    return rows


@torch.no_grad()
def collect_unlabeled_results(model, loader, cal_by_class, device):
    model.eval()
    cal_by_class_dev = {c: cal_by_class[c].to(device) for c in range(NUM_CLASSES)}
    n_by_class = {c: len(cal_by_class[c]) for c in range(NUM_CLASSES)}

    rows = []
    for x, filenames in tqdm(loader, desc="collect unlabeled", leave=False):
        x = x.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        alpha_xy = 1.0 - probs
        pred_idx = probs.argmax(dim=1)
        pred_conf = probs.max(dim=1).values

        pvals = torch.empty_like(alpha_xy)
        for c in range(NUM_CLASSES):
            cal_c = cal_by_class_dev[c]
            counts = (cal_c[:, None] >= alpha_xy[:, c][None, :]).sum(dim=0)
            pvals[:, c] = (counts + 1.0) / (n_by_class[c] + 1.0)

        max_pvals = pvals.max(dim=1).values
        ood_scores = 1.0 - max_pvals

        for idx, filename in enumerate(filenames):
            rows.append({
                "filename": filename,
                "predicted_class": CLASS_NAMES[int(pred_idx[idx].item())],
                "predicted_class_index": int(pred_idx[idx].item()),
                "predicted_softmax_confidence": float(pred_conf[idx].item()),
                "pvalue_healthy": float(pvals[idx, 0].item()),
                "pvalue_BS": float(pvals[idx, 1].item()),
                "max_pvalue": float(max_pvals[idx].item()),
                "ood_score": float(ood_scores[idx].item()),
            })

    return rows


def load_labeled_samples(healthy_dir: Path, bs_dir: Path, cache_prefix: str, overwrite: bool = False):
    samples_cache = Path(f"{cache_prefix}_samples.pkl")
    labels_cache = Path(f"{cache_prefix}_labels.pkl")
    filenames_cache = Path(f"{cache_prefix}_filenames.pkl")

    healthy_paths = sorted([path for path in healthy_dir.iterdir() if path.is_file()])
    bs_paths = sorted([path for path in bs_dir.iterdir() if path.is_file()])

    if samples_cache.exists() and labels_cache.exists() and filenames_cache.exists() and not overwrite:
        with samples_cache.open("rb") as f:
            samples = pickle.load(f)
        with labels_cache.open("rb") as f:
            labels = pickle.load(f)
        with filenames_cache.open("rb") as f:
            filenames = pickle.load(f)
        return samples, labels, filenames

    samples = []
    labels = []
    filenames = []

    for image_path in tqdm(healthy_paths, desc=f"load {cache_prefix} healthy", leave=False):
        with Image.open(image_path) as img:
            samples.append(img.convert("RGB").copy())
        labels.append(0)
        filenames.append(image_path.name)

    for image_path in tqdm(bs_paths, desc=f"load {cache_prefix} BS", leave=False):
        with Image.open(image_path) as img:
            samples.append(img.convert("RGB").copy())
        labels.append(1)
        filenames.append(image_path.name)

    with samples_cache.open("wb") as f:
        pickle.dump(samples, f)
    with labels_cache.open("wb") as f:
        pickle.dump(labels, f)
    with filenames_cache.open("wb") as f:
        pickle.dump(filenames, f)

    return samples, labels, filenames


def load_unlabeled_samples(folder: Path):
    samples = []
    filenames = []
    image_paths = sorted([path for path in folder.iterdir() if path.is_file()])

    for image_path in tqdm(image_paths, desc="load internet OoD", leave=False):
        with Image.open(image_path) as img:
            samples.append(img.convert("RGB").copy())
        filenames.append(image_path.name)

    return samples, filenames


def save_csv(rows, out_path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_histogram(groups, value_key: str, title: str, xlabel: str, out_path: Path):
    plt.figure(figsize=(9, 6))
    bins = np.linspace(0, 1, 31)
    for label, rows in groups:
        values = [row[value_key] for row in rows]
        if len(values) == 0:
            continue
        plt.hist(values, bins=bins, density=True, alpha=0.5, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def compute_ood_detection_metrics(id_rows, ood_rows):
    id_scores = np.array([row["ood_score"] for row in id_rows], dtype=np.float64)
    ood_scores = np.array([row["ood_score"] for row in ood_rows], dtype=np.float64)

    y_true = np.concatenate([
        np.zeros(len(id_scores), dtype=np.int64),
        np.ones(len(ood_scores), dtype=np.int64),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    roc_auc = float("nan")
    fpr95 = float("nan")

    if len(np.unique(y_true)) == 2 and len(scores) > 0:
        roc_auc = roc_auc_score(y_true, scores)
        fpr, tpr, _ = roc_curve(y_true, scores)
        idx = int(np.argmin(np.abs(tpr - 0.95)))
        fpr95 = float(fpr[idx])

    return roc_auc, fpr95


def summarize_group(rows):
    max_p = np.array([row["max_pvalue"] for row in rows], dtype=np.float64)
    ood_scores = np.array([row["ood_score"] for row in rows], dtype=np.float64)
    summary = {
        "count": len(rows),
        "max_pvalue_mean": float(np.mean(max_p)) if len(max_p) else float("nan"),
        "max_pvalue_median": float(np.median(max_p)) if len(max_p) else float("nan"),
        "ood_score_mean": float(np.mean(ood_scores)) if len(ood_scores) else float("nan"),
        "ood_score_median": float(np.median(ood_scores)) if len(ood_scores) else float("nan"),
    }
    return summary


def write_summary_report(out_path: Path, id_rows, ood_2023_rows, internet_rows, metrics_2023, metrics_internet):
    id_summary = summarize_group(id_rows)
    ood_2023_summary = summarize_group(ood_2023_rows)
    internet_summary = summarize_group(internet_rows)

    internet_farther = internet_summary["ood_score_mean"] > ood_2023_summary["ood_score_mean"]
    internet_lower_max_p = internet_summary["max_pvalue_mean"] < ood_2023_summary["max_pvalue_mean"]
    internet_auroc_higher = metrics_internet[0] > metrics_2023[0]

    if internet_farther and internet_lower_max_p:
        conclusion = (
            "Pilot result: the random internet images appear farther from the 2025 in-distribution set "
            "than the 2023 dataset, based on higher OoD scores and lower max conformal p-values."
        )
    elif not internet_farther and not internet_lower_max_p:
        conclusion = (
            "Pilot result: the random internet images do not appear farther from the 2025 in-distribution set "
            "than the 2023 dataset under these conformal metrics."
        )
    else:
        conclusion = (
            "Pilot result: the comparison is mixed. Some conformal metrics suggest the internet images are farther "
            "from the 2025 data, while others do not."
        )

    with out_path.open("w") as f:
        f.write("Conformal OoD comparison summary\n")
        f.write("================================\n\n")
        f.write("Caution: the internet-image set is a small pilot sample (currently about 20 images), so results should be interpreted carefully.\n\n")

        f.write("Sample counts\n")
        f.write(f"2025 ID samples: {id_summary['count']}\n")
        f.write(f"2023 OoD samples: {ood_2023_summary['count']}\n")
        f.write(f"Internet OoD samples: {internet_summary['count']}\n\n")

        f.write("Group summaries\n")
        for name, summary in [
            ("2025 ID", id_summary),
            ("2023 OoD", ood_2023_summary),
            ("Internet OoD", internet_summary),
        ]:
            f.write(f"{name}:\n")
            f.write(f"  mean max_pvalue: {summary['max_pvalue_mean']:.6f}\n")
            f.write(f"  median max_pvalue: {summary['max_pvalue_median']:.6f}\n")
            f.write(f"  mean ood_score: {summary['ood_score_mean']:.6f}\n")
            f.write(f"  median ood_score: {summary['ood_score_median']:.6f}\n")
        f.write("\n")

        f.write("OoD detection metrics\n")
        f.write(f"ID vs 2023 AUROC: {metrics_2023[0]:.6f}\n")
        f.write(f"ID vs 2023 FPR@95TPR: {metrics_2023[1]:.6f}\n")
        f.write(f"ID vs internet AUROC: {metrics_internet[0]:.6f}\n")
        f.write(f"ID vs internet FPR@95TPR: {metrics_internet[1]:.6f}\n\n")

        f.write("Interpretation helper\n")
        f.write("Higher OoD score = farther from the calibration-supported in-distribution region.\n")
        f.write("Lower max conformal p-value = farther from the in-distribution region.\n")
        f.write(f"Internet AUROC higher than 2023 AUROC: {internet_auroc_higher}\n\n")

        f.write("Conclusion template\n")
        f.write(conclusion + "\n")


def build_model(device):
    classifier = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_feats = classifier.fc.in_features
    classifier.fc = nn.Linear(in_feats, NUM_CLASSES)
    return classifier.to(device)


def main():
    data_dir = Path("../data")
    overwrite = False
    seed = 42
    image_size = 512
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("analysis_outputs_conformal_ood_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)

    healthy_images_path = data_dir / "healthy_images"
    traubenwelke_images_path = data_dir / "BS_images"
    healthy_ood_path = data_dir / "2023" / "healthy_images"
    traubenwelke_ood_path = data_dir / "2023" / "BS_images"
    internet_ood_path = data_dir / "ood_test_data"

    all_samples, all_labels, all_filenames = load_labeled_samples(
        healthy_images_path,
        traubenwelke_images_path,
        cache_prefix="all_2025",
        overwrite=overwrite,
    )

    rng = random.Random(seed)
    combined = list(zip(all_samples, all_labels, all_filenames))
    rng.shuffle(combined)
    all_samples, all_labels, all_filenames = zip(*combined)

    train_samples, test_samples, train_labels, test_labels, train_filenames, test_filenames = train_test_split(
        all_samples,
        all_labels,
        all_filenames,
        test_size=0.4,
        stratify=all_labels,
        random_state=seed,
    )

    calibration_samples, id_samples, calibration_labels, id_labels, calibration_filenames, id_filenames = train_test_split(
        test_samples,
        test_labels,
        test_filenames,
        test_size=0.5,
        stratify=test_labels,
        random_state=seed,
    )

    dataset_names = ["train", "calibration", "id"]
    for dataset_name, dataset_labels in zip(dataset_names, [train_labels, calibration_labels, id_labels]):
        counts = Counter(dataset_labels)
        print(
            f"There are {counts[0]} healthy and {counts[1]} traubenwelke samples in {dataset_name} dataset!"
        )

    print(f"Train: {len(train_samples)} | Calibration: {len(calibration_samples)} | ID: {len(id_labels)}")

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Traubenwelke(train_samples, train_labels, transform=train_tf)
    calibration_dataset = Traubenwelke(calibration_samples, calibration_labels, transform=test_tf)
    id_dataset = Traubenwelke(id_samples, id_labels, transform=test_tf)
    id_eval_dataset = LabeledImageDatasetWithPaths(id_samples, id_labels, id_filenames, transform=test_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    id_eval_loader = DataLoader(id_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    all_ood_samples, all_ood_labels, all_ood_filenames = load_labeled_samples(
        healthy_ood_path,
        traubenwelke_ood_path,
        cache_prefix="all_2023_ood",
        overwrite=overwrite,
    )

    rng = random.Random(seed)
    combined = list(zip(all_ood_samples, all_ood_labels, all_ood_filenames))
    rng.shuffle(combined)
    all_ood_samples, all_ood_labels, all_ood_filenames = zip(*combined)

    counts = Counter(all_ood_labels)
    print(f"There are {counts[0]} healthy and {counts[1]} traubenwelke samples in 2023 OoD dataset!")

    ood_dataset = Traubenwelke(all_ood_samples, all_ood_labels, transform=test_tf)
    ood_eval_dataset = LabeledImageDatasetWithPaths(all_ood_samples, all_ood_labels, all_ood_filenames, transform=test_tf)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ood_eval_loader = DataLoader(ood_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    internet_samples, internet_filenames = load_unlabeled_samples(internet_ood_path)
    print(
        f"Loaded {len(internet_samples)} internet OoD images from {internet_ood_path}. "
        "This is a small pilot sample, so interpret the results cautiously."
    )
    internet_dataset = UnlabeledImageDataset(internet_samples, internet_filenames, transform=test_tf)
    internet_loader = DataLoader(internet_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    classifier = build_model(device)
    ckpt_path = Path("BEST_RESNET18_CONFORMAL_PREDICTION.pt")

    if not ckpt_path.exists():
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
        best_accuracy = -1.0
        epochs = 20

        for epoch in range(1, epochs + 1):
            classifier.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for imgs, labels in tqdm(train_loader, total=len(train_loader), desc=f"Epoch: {epoch}/{epochs}"):
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = classifier(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += imgs.size(0)

            train_loss = total_loss / max(total, 1)
            train_accuracy = correct / max(total, 1)
            test_loss, test_accuracy, _, _ = evaluate(classifier, id_loader, criterion, device)

            print(
                f"Epoch {epoch:02d}: train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
                f"test loss {test_loss:.4f} acc {test_accuracy:.4f}"
            )

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(
                    {
                        "model_state": classifier.state_dict(),
                        "img_size": image_size,
                        "classes": CLASS_NAMES,
                    },
                    ckpt_path,
                )
                print(f"  Saved best -> {ckpt_path} (test acc {best_accuracy:.4f})")

    checkpoint = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state"])
    classifier.eval()

    cal_by_class = compute_calibration_alphas_by_class(classifier, calibration_loader, device, num_classes=NUM_CLASSES)
    print({c: len(v) for c, v in cal_by_class.items()})

    id_pvals_cd, id_labels_tensor = compute_pvalues_class_conditional(
        classifier, id_loader, cal_by_class, device, num_classes=NUM_CLASSES
    )
    ood_pvals_cd, ood_labels_tensor = compute_pvalues_class_conditional(
        classifier, ood_loader, cal_by_class, device, num_classes=NUM_CLASSES
    )

    id_true_p = id_pvals_cd[torch.arange(len(id_labels_tensor)), id_labels_tensor]
    ood_true_p = ood_pvals_cd[torch.arange(len(ood_labels_tensor)), ood_labels_tensor]

    print(f"Mean ID true-label p-value: {id_true_p.mean().item():.6f}")
    print(f"Mean 2023 true-label p-value: {ood_true_p.mean().item():.6f}")

    id_rows = collect_labeled_results(classifier, id_eval_loader, cal_by_class, device)
    ood_2023_rows = collect_labeled_results(classifier, ood_eval_loader, cal_by_class, device)
    internet_rows = collect_unlabeled_results(classifier, internet_loader, cal_by_class, device)

    save_csv(id_rows, output_dir / "id_2025_results.csv")
    save_csv(ood_2023_rows, output_dir / "ood_2023_results.csv")
    save_csv(internet_rows, output_dir / "internet_ood_results.csv")

    plot_histogram(
        [("2025 ID", id_rows), ("2023 OoD", ood_2023_rows), ("Internet OoD", internet_rows)],
        value_key="ood_score",
        title="Distribution of 1 - Max Conformal p-value",
        xlabel="1 - max_y p_y(x)",
        out_path=output_dir / "ood_score_histogram_all_groups.png",
    )
    plot_histogram(
        [("2025 ID", id_rows), ("2023 OoD", ood_2023_rows), ("Internet OoD", internet_rows)],
        value_key="max_pvalue",
        title="Distribution of Max Conformal p-value",
        xlabel="max_y p_y(x)",
        out_path=output_dir / "max_pvalue_histogram_all_groups.png",
    )

    metrics_2023 = compute_ood_detection_metrics(id_rows, ood_2023_rows)
    metrics_internet = compute_ood_detection_metrics(id_rows, internet_rows)

    summary_rows = [
        {"comparison": "ID_vs_2023", "auroc": metrics_2023[0], "fpr_at_95_tpr": metrics_2023[1]},
        {"comparison": "ID_vs_internet", "auroc": metrics_internet[0], "fpr_at_95_tpr": metrics_internet[1]},
    ]
    save_csv(summary_rows, output_dir / "ood_detection_metrics.csv")

    write_summary_report(
        output_dir / "summary_report.txt",
        id_rows=id_rows,
        ood_2023_rows=ood_2023_rows,
        internet_rows=internet_rows,
        metrics_2023=metrics_2023,
        metrics_internet=metrics_internet,
    )

    print(f"Saved conformal OoD analysis to: {output_dir.resolve()}")
    print(f"ID vs 2023 AUROC: {metrics_2023[0]:.6f} | FPR@95TPR: {metrics_2023[1]:.6f}")
    print(f"ID vs internet AUROC: {metrics_internet[0]:.6f} | FPR@95TPR: {metrics_internet[1]:.6f}")


if __name__ == "__main__":
    main()

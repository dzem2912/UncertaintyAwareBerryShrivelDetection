import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from dataset import Traubenwelke


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_samples(data_dir: Path, overwrite: bool = False):
    samples_path = Path("all_samples.pkl")
    labels_path = Path("all_labels.pkl")

    healthy_dir = data_dir / "healthy_images"
    bs_dir = data_dir / "BS_images"

    healthy_samples = sorted(healthy_dir.iterdir())
    bs_samples = sorted(bs_dir.iterdir())

    if len(healthy_samples) == 0 or len(bs_samples) == 0:
        raise RuntimeError(
            f"Found {len(healthy_samples)} healthy and {len(bs_samples)} BS images. "
            f"Check folder names and that images exist."
        )

    if samples_path.exists() and labels_path.exists() and not overwrite:
        with samples_path.open("rb") as f:
            all_samples = pickle.load(f)
        with labels_path.open("rb") as f:
            all_labels = pickle.load(f)
    else:
        all_samples = []
        all_labels = []

        for healthy_sample in tqdm(healthy_samples, desc="load healthy", leave=False):
            with Image.open(healthy_sample) as img:
                all_samples.append(img.convert("RGB").copy())
            all_labels.append(0)

        for bs_sample in tqdm(bs_samples, desc="load BS", leave=False):
            with Image.open(bs_sample) as img:
                all_samples.append(img.convert("RGB").copy())
            all_labels.append(1)

        with samples_path.open("wb") as f:
            pickle.dump(all_samples, f)
        with labels_path.open("wb") as f:
            pickle.dump(all_labels, f)

    return all_samples, all_labels


def train_one_epoch_amp(model, loader, optimizer, criterion, device, scaler, use_amp: bool):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(enabled=use_amp, device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(x)
            loss = criterion(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_with_probs(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    y_true, y_pred, y_prob_pos = [], [], []

    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        p1 = probs[:, 1]
        threshold = 0.85
        preds = (p1 >= threshold).long()

        total_loss += loss.item() * x.size(0)
        correct += (preds == y).sum().item()
        total += x.size(0)

        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_prob_pos.extend(p1.detach().cpu().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    tp = np.sum((y_true_np == 1) & (y_pred_np == 1))
    fp = np.sum((y_true_np == 0) & (y_pred_np == 1))
    fn = np.sum((y_true_np == 1) & (y_pred_np == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return total_loss / max(total, 1), correct / max(total, 1), y_true, y_pred, y_prob_pos, precision, recall


def build_model(device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model.to(device)


def save_roc_pr_images(y_true, y_prob_pos, out_prefix: Path, roc_auc=None, pr_auc=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title = "ROC curve"
    if roc_auc is not None and not np.isnan(roc_auc):
        title += f" (AUC={roc_auc:.4f})"
    plt.title(title)
    plt.grid(True)
    fig.savefig(str(out_prefix) + "_roc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_prob_pos)
    fig = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    title = "Precision-Recall curve"
    if pr_auc is not None and not np.isnan(pr_auc):
        title += f" (AP={pr_auc:.4f})"
    plt.title(title)
    plt.grid(True)
    fig.savefig(str(out_prefix) + "_pr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    seed = 42
    n_splits = 5
    n_repeats = 2

    epochs = 10
    batch_size = 32
    lr = 1e-3
    img_size = 256
    num_workers = 0
    pin_memory = True
    overwrite = False

    save_curves = True
    curves_out_dir = Path("cv_curves")
    if save_curves:
        curves_out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("../data")
    all_samples, all_labels = load_samples(data_dir=data_dir, overwrite=overwrite)

    all_samples = np.array(all_samples, dtype=object)
    all_labels = np.array(all_labels)

    print(
        f"Found healthy: {(all_labels == 0).sum()} | BS: {(all_labels == 1).sum()} | total: {len(all_samples)}"
    )

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    use_amp = device.type == "cuda"
    fold_metrics = []

    for repeat in range(1, n_repeats + 1):
        repeat_seed = seed + (repeat - 1)
        print("\n" + "#" * 70)
        print(f"Repeat {repeat}/{n_repeats} (seed={repeat_seed})")

        seed_everything(repeat_seed)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)

        for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, all_labels), start=1):
            print("\n" + "=" * 70)
            print(f"Repeat {repeat}/{n_repeats} | Fold {fold}/{n_splits}")

            train_samples = all_samples[train_idx].tolist()
            train_labels = all_labels[train_idx].tolist()
            val_samples = all_samples[val_idx].tolist()
            val_labels = all_labels[val_idx].tolist()

            print(f"  Train: {len(train_samples)} | Val: {len(val_samples)}")

            train_ds = Traubenwelke(train_samples, train_labels, transform=train_tf)
            val_ds = Traubenwelke(val_samples, val_labels, transform=val_tf)

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory
            )

            model = build_model(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scaler = torch.amp.GradScaler(enabled=use_amp)

            best_val_acc = -1.0
            best_path = f"best_resnet18_grapes_repeat{repeat}_fold{fold}.pt"

            if not os.path.exists(best_path):
                for epoch in range(1, epochs + 1):
                    train_loss, train_acc = train_one_epoch_amp(
                        model, train_loader, optimizer, criterion, device, scaler, use_amp
                    )
                    val_loss, val_acc, _, _, _, precision, recall = evaluate_with_probs(
                        model, val_loader, criterion, device
                    )

                    print(
                        f"  Epoch {epoch:02d}/{epochs}: "
                        f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                        f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                        f"precision {precision:.4f} | recall {recall:.4f}"
                    )

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(
                            {
                                "model_state": model.state_dict(),
                                "img_size": img_size,
                                "classes": ["healthy", "BS"],
                                "repeat": repeat,
                                "fold": fold,
                                "seed": repeat_seed,
                            },
                            best_path,
                        )
                        print(f"    Saved best -> {best_path} (val acc {best_val_acc:.4f})")

            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])

            val_loss, val_acc, y_true, y_pred, y_prob_pos, val_precision, val_recall = evaluate_with_probs(
                model, val_loader, criterion, device
            )

            roc_auc = float("nan")
            pr_auc = float("nan")
            try:
                roc_auc = roc_auc_score(y_true, y_prob_pos)
            except ValueError:
                pass
            try:
                pr_auc = average_precision_score(y_true, y_prob_pos)
            except ValueError:
                pass

            if save_curves:
                out_prefix = curves_out_dir / f"repeat{repeat}_fold{fold}"
                try:
                    save_roc_pr_images(y_true, y_prob_pos, out_prefix=out_prefix, roc_auc=roc_auc, pr_auc=pr_auc)
                except ValueError:
                    pass

            print("\n  Confusion matrix (rows=true, cols=pred):")
            print(confusion_matrix(y_true, y_pred))

            print("\n  Classification report:")
            print(classification_report(y_true, y_pred, target_names=["healthy", "BS"]))

            print(f"Repeat {repeat} | Fold {fold} metrics:")
            print(f"Val accuracy: {val_acc:.4f}")
            print(f"Val precision: {val_precision:.4f}")
            print(f"Val recall: {val_recall:.4f}")

            fold_metrics.append({
                "repeat": repeat,
                "fold": fold,
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            })

    print("\n" + "=" * 70)
    print("Repeated cross-validation summary")

    def _drop_nan(values):
        values = np.array(values, dtype=np.float64)
        return values[~np.isnan(values)]

    def _median_iqr(values):
        values = _drop_nan(values)
        if len(values) == 0:
            return float("nan"), float("nan")
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        return float(np.median(values)), float(q3 - q1)

    accs = [m["val_acc"] for m in fold_metrics]
    precisions = [m["val_precision"] for m in fold_metrics]
    recalls = [m["val_recall"] for m in fold_metrics]
    rocs = [m["roc_auc"] for m in fold_metrics]
    prs = [m["pr_auc"] for m in fold_metrics]

    for m in fold_metrics:
        print(
            f"  Repeat {m['repeat']} Fold {m['fold']}: "
            f"acc={m['val_acc']:.4f}, precision={m['val_precision']:.4f}, recall={m['val_recall']:.4f}"
        )

    acc_med, acc_iqr = _median_iqr(accs)
    precision_median, precision_iqr = _median_iqr(precisions)
    recall_median, recall_iqr = _median_iqr(recalls)
    roc_med, roc_iqr = _median_iqr(rocs)
    pr_med, pr_iqr = _median_iqr(prs)

    print(f"\nOverall over {n_repeats * n_splits} folds (median +/- IQR):")
    print(f"Accuracy: {acc_med:.4f} +/- {acc_iqr:.4f}")
    print(f"Precision: {precision_median:.4f} +/- {precision_iqr:.4f}")
    print(f"Recall: {recall_median:.4f} +/- {recall_iqr:.4f}")
    print(f"ROC-AUC: {roc_med:.4f} +/- {roc_iqr:.4f}")
    print(f"PR-AUC: {pr_med:.4f} +/- {pr_iqr:.4f}")

    if save_curves:
        print(f"\nSaved per-fold ROC/PR images to: {curves_out_dir.resolve()}")


if __name__ == "__main__":
    main()

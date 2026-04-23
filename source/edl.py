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

def unnormalize_img(x: torch.Tensor) -> torch.Tensor:
    # x: [3,H,W] normalized
    return (x * IMAGENET_STD.to(x.device)) + IMAGENET_MEAN.to(x.device)

def save_tensor_image(x: torch.Tensor, out_path: str) -> None:
    # x: [3,H,W] float tensor, normalized
    #x_img = unnormalize_img(x).clamp(0, 1)

    #img_np = (x_img.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)

    img_np = x.detach().cpu().numpy()
    Image.fromarray(img_np).save(out_path)

def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()

def kl_dirichlet_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL( Dir(alpha) || Dir(1) ) per sample.
    alpha: [B, K], alpha > 0
    returns: [B]
    """
    K = alpha.size(1)
    sum_alpha = alpha.sum(dim=1, keepdim=True)  # [B,1]

    # log B(alpha) = sum lgamma(alpha_k) - lgamma(sum alpha)
    log_B_alpha = torch.lgamma(alpha).sum(dim=1) - torch.lgamma(sum_alpha.squeeze(1))  # [B]

    digamma_sum = torch.digamma(sum_alpha)     # [B,1]
    digamma_alpha = torch.digamma(alpha)       # [B,K]

    kl = -log_B_alpha + ((alpha - 1.0) * (digamma_alpha - digamma_sum)).sum(dim=1)
    return kl

@torch.no_grad()
def evaluate_edl_and_collect(model, loader, device, epoch: int, num_classes: int):
    print(f'Evaluating EDL ...')
    model.eval()

    records = []
    for (x, y, images) in tqdm(loader, desc="eval+collect", leave=False):
        x, y = x.to(device), y.to(device)

        alpha = model(x)
        probs = alpha / alpha.sum(dim=1, keepdim=True)
        preds = probs.argmax(dim=1)

        u = edl_uncertainty(alpha)
        conf = probs.max(dim=1).values

        # store per-sample data including the actual tensor for saving later
        for i in range(x.size(0)):
            records.append({
                "x": images[i].detach().cpu(),      # keep on CPU to save later
                "y_true": int(y[i].cpu().item()),
                "y_pred": int(preds[i].cpu().item()),
                "u": float(u[i].cpu().item()),
                "conf": float(conf[i].cpu().item()),
            })

    return records


def edl_mse_loss(
    alpha: torch.Tensor,
    y: torch.Tensor,
    epoch: int,
    num_classes: int,
    kl_anneal_epochs: int = 10,
    kl_coef: float = 1.0,
) -> torch.Tensor:
    """
    EDL loss from Sensoy et al.:
      MSE Bayes risk + annealed KL regularizer to uniform Dirichlet.

    alpha: [B, K] Dirichlet parameters (evidence + 1)
    y:     [B] long labels
    """
    y_onehot = one_hot(y, num_classes=num_classes)  # [B, K]

    S = alpha.sum(dim=1, keepdim=True)  # [B, 1]
    p = alpha / S                       # Dirichlet mean [B, K]

    # Bayes risk under squared error:
    # sum_k (y_k - E[p_k])^2 + Var[p_k]
    err = (y_onehot - p).pow(2)
    var = alpha * (S - alpha) / (S * S * (S + 1.0))
    mse_risk = (err + var).sum(dim=1).mean()

    # KL regularizer uses "non-misleading evidence removed":
    # alpha_tilde = y + (1-y) * alpha
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    kl = kl_dirichlet_to_uniform(alpha_tilde).mean()

    anneal = min(1.0, float(epoch) / float(kl_anneal_epochs))
    return mse_risk + kl_coef * anneal * kl


@torch.no_grad()
def edl_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    """
    Subjective logic uncertainty:
      u = K / sum(alpha)
    """
    K = alpha.size(1)
    S = alpha.sum(dim=1)
    return K / S


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
def evaluate_edl(model, loader, device, epoch: int, num_classes: int):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    y_true, y_pred = [], []
    u_all = []
    conf_all = []

    for x, y, image in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)

        alpha = model(x)
        loss = edl_mse_loss(alpha, y, epoch=epoch, num_classes=num_classes)

        probs = alpha / alpha.sum(dim=1, keepdim=True)
        preds = probs.argmax(dim=1)

        u = edl_uncertainty(alpha)                  # [B]
        conf = probs.max(dim=1).values              # [B]

        total_loss += loss.item() * x.size(0)
        correct += (preds == y).sum().item()
        total += x.size(0)

        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        u_all.extend(u.cpu().tolist())
        conf_all.extend(conf.cpu().tolist())

    return total_loss / total, correct / total, y_true, y_pred, u_all, conf_all


def plot_curves(out_dir: str, history: dict):
    os.makedirs(out_dir, exist_ok=True)

    # Loss curve
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"])
    plt.plot(history["epoch"], history["test_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "test"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train-test-loss-curve.png"))
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(history["epoch"], history["train_acc"])
    plt.plot(history["epoch"], history["test_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "test"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train-test-accuracy-curve.png"))
    plt.close()

    # Uncertainty curve (mean u)
    plt.figure()
    plt.plot(history["epoch"], history["train_u_mean"])
    plt.plot(history["epoch"], history["test_u_mean"])
    plt.xlabel("Epoch")
    plt.ylabel("Mean uncertainty (u = K / sum(alpha))")
    plt.legend(["train", "test"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train-test-uncertainty-curve.png"))
    plt.close()


def plot_uncertainty_diagnostics(out_dir: str, y_true, y_pred, u_list, conf_list):
    u_0_correct = [u for yt, yp, u in zip(y_true, y_pred, u_list) if yt == 0 and yp == 0]
    u_0_wrong   = [u for yt, yp, u in zip(y_true, y_pred, u_list) if yt == 0 and yp == 1]

    u_1_correct = [u for yt, yp, u in zip(y_true, y_pred, u_list) if yt == 1 and yp == 1]
    u_1_wrong   = [u for yt, yp, u in zip(y_true, y_pred, u_list) if yt == 1 and yp == 0]

    plt.figure()
    plt.hist(u_0_correct, bins=40, alpha=0.7)
    plt.hist(u_0_wrong, bins=40, alpha=0.7)
    plt.title("Histogram Uncertainty and Correctness: Healthy Class")
    plt.xlabel("Uncertainty u")
    plt.ylabel("Count")
    plt.legend([
        "H correct",
        "H wrong"
        ])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "uncertainty_hist_H_and_correctness.png"))
    plt.close()

    plt.figure()
    plt.hist(u_1_correct, bins=40, alpha=0.7)
    plt.hist(u_1_wrong, bins=40, alpha=0.7)

    plt.title("Histogram Uncertainty and Correctness: BS Class")
    plt.xlabel("Uncertainty u")
    plt.ylabel("Count")
    plt.legend([
        "BS correct",
        "BS wrong",
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "uncertainty_hist_BS_and_correctness.png"))
    plt.close()

    # Scatter: uncertainty vs confidence
    plt.figure()
    plt.scatter(conf_list, u_list, s=10, alpha=0.6)
    plt.xlabel("Confidence (max Dirichlet-mean prob)")
    plt.ylabel("Uncertainty u")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "uncertainty_vs_confidence_scatter.png"))
    plt.close()

def save_uncertainty_extremes(records, out_dir: str, target_class: int,
                            top_frac: float = 0.10,      # top 10% most uncertain
                            bottom_frac: float = 0.10,   # bottom 10% most certain
                            k: int = 10,                 # save at most k from each side
                            only_correct: bool = False,  # optionally only look at correct predictions
                            ):
    os.makedirs(out_dir, exist_ok=True)

    # filter by class (true label)
    subset = [r for r in records if r["y_true"] == target_class]
    if only_correct:
        subset = [r for r in subset if r["y_true"] == r["y_pred"]]

    if len(subset) == 0:
        print(f"[warn] No samples found for class={target_class} after filtering.")
        return

    # sort by uncertainty
    subset_sorted = sorted(subset, key=lambda r: r["u"])
    n = len(subset_sorted)

    hi_start = int((1.0 - top_frac) * n)
    lo_end = int(bottom_frac * n)

    low_pool = subset_sorted[:max(1, lo_end)]
    high_pool = subset_sorted[max(0, hi_start):]

    # choose up to k from each pool (take extremes, not random, for clarity)
    low_sel = low_pool[:k]
    high_sel = list(reversed(high_pool))[:k]  # highest u first

    # save
    low_dir = os.path.join(out_dir, f'class_{target_class}_LOW_u')
    high_dir = os.path.join(out_dir, f'class_{target_class}_HIGH_u')
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)

    def dump(sel, folder, tag):
        meta_lines = ["filename,y_true,y_pred,u,conf"]
        for idx, r in enumerate(sel):
            fname = f"{tag}_{idx:03d}_u{r['u']:.4f}_c{r['conf']:.4f}_yt{r['y_true']}_yp{r['y_pred']}.png"
            fpath = os.path.join(folder, fname)
            #fpath = str(folder / fname)
            save_tensor_image(r["x"], fpath)
            meta_lines.append(f"{fname},{r['y_true']},{r['y_pred']},{r['u']:.6f},{r['conf']:.6f}")

        with open(os.path.join(folder, 'metadata.csv'), 'w') as f:
            f.write("\n".join(meta_lines))

    dump(low_sel, low_dir, "certain")
    dump(high_sel, high_dir, "uncertain")

    print(f"Saved {len(low_sel)} low-u and {len(high_sel)} high-u samples for class={target_class} in {out_dir}")


def main():
    seed: int = 42
    seed_everything(seed)

    data_dir: str = "../data"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size: int = 224
    batch_size: int = 8
    overwrite: bool = True

    num_classes: int = 2
    epochs: int = 20

    out_dir = "edl_runs"
    os.makedirs(out_dir, exist_ok=True)

    healthy_images_path: str = os.path.join(data_dir, "healthy_images")
    traubenwelke_images_path: str = os.path.join(data_dir, "BS_images")

    healthy_samples: list[str] = [os.path.join(healthy_images_path, file) for file in os.listdir(healthy_images_path)]
    traubenwelke_samples: list[str] = [os.path.join(traubenwelke_images_path, file) for file in os.listdir(traubenwelke_images_path)]

    if os.path.exists("all_samples.pkl") and os.path.exists("all_labels.pkl") and not overwrite:
        with open("all_samples.pkl", "rb") as f:
            all_samples = pickle.load(f)
        with open("all_labels.pkl", "rb") as f:
            all_labels = pickle.load(f)
    else:
        all_samples = []
        all_labels = []

        for healthy_sample in tqdm(healthy_samples, desc="load healthy"):
            healthy_image = Image.open(healthy_sample)
            all_samples.append(healthy_image)
            all_labels.append(0)

        for traubenwelke_sample in tqdm(traubenwelke_samples, desc="load BS"):
            traubenwelke_image = Image.open(traubenwelke_sample)
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
        random_state=seed,
    )

    counts = Counter(train_labels)
    zeros = counts[0]
    ones = counts[1]
    print(f"There are {zeros} healthy and {ones} traubenwelke samples for training!")
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

    classifier_name: str = "ResNet_EDL"
    ckpt_path = f"best_{classifier_name}_traubenwelke.pt"

    classifier = EvidentialResNet(num_classes=num_classes)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        classifier.load_state_dict(checkpoint["model_state"])
    
    classifier = classifier.to(device)

    best_accuracy: float = -1.0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_u_mean": [],
        "test_loss": [],
        "test_acc": [],
        "test_u_mean": [],
    }

    if not os.path.exists(ckpt_path):
        for epoch in range(1, epochs + 1):
            classifier.train()

            total_loss: float = 0.0
            correct: int = 0
            total: int = 0
            u_train_all = []

            for img, label, _ in tqdm(train_loader, desc=f"Epoch: {epoch}/{epochs}", leave=False):
                imgs = img.to(device)
                labels = label.to(device)

                optimizer.zero_grad()

                alpha = classifier(imgs)
                loss = edl_mse_loss(alpha, labels, epoch=epoch, num_classes=num_classes)

                loss.backward()
                optimizer.step()

                probs = alpha / alpha.sum(dim=1, keepdim=True)
                predictions = probs.argmax(dim=1)

                u = edl_uncertainty(alpha)

                total_loss += loss.item() * imgs.size(0)
                correct += (predictions == labels).sum().item()
                total += imgs.size(0)
                u_train_all.extend(u.detach().cpu().tolist())

            train_loss = total_loss / total
            train_accuracy = correct / total
            train_u_mean = float(sum(u_train_all) / max(1, len(u_train_all)))

            test_loss, test_accuracy, y_true, y_pred, u_test_all, conf_test_all = evaluate_edl(
                classifier, test_loader, device, epoch=epoch, num_classes=num_classes
            )

            test_u_mean = float(sum(u_test_all) / max(1, len(u_test_all)))

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_accuracy)
            history["train_u_mean"].append(train_u_mean)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_accuracy)
            history["test_u_mean"].append(test_u_mean)

            print(
                f"Epoch {epoch:02d}: train loss {train_loss:.4f} acc {train_accuracy:.4f} u {train_u_mean:.4f} | "
                f"test loss {test_loss:.4f} acc {test_accuracy:.4f} u {test_u_mean:.4f}"
            )

            # Save curves each epoch (overwriteimgs)
            plot_curves(out_dir, history)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(
                    {
                        "model_state": classifier.state_dict(),
                        "img_size": image_size,
                        "classes": ["healthy", "BS"],
                        "num_classes": num_classes,
                    },
                    ckpt_path,
                )
                print(f"  Saved best -> {ckpt_path} (test acc {best_accuracy:.4f})")

                # Save "best checkpoint" diagnostics based on the current best
                plot_uncertainty_diagnostics(out_dir, y_true, y_pred, u_test_all, conf_test_all)

    records = evaluate_edl_and_collect(classifier, test_loader, device, epoch=epochs, num_classes=num_classes)

    save_uncertainty_extremes(records, os.path.join(out_dir, "saved_samples"), target_class=0, top_frac=0.10, bottom_frac=0.10, k=12)
    save_uncertainty_extremes(records, os.path.join(out_dir, "saved_samples"), target_class=1, top_frac=0.10, bottom_frac=0.10, k=12)


    print(f"Done. Best test accuracy: {best_accuracy:.4f}")
    print(f"Plots saved to: {out_dir}/")

if __name__ == "__main__":
    main()

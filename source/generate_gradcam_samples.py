from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import cm
from torchvision import transforms
from torchvision.models import resnet18


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(self, x: torch.Tensor, class_idx: int) -> tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.detach().cpu().numpy(), logits.detach()


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    )


def overlay_heatmap(
    image: Image.Image,
    cam_map: np.ndarray,
    out_path: Path,
    alpha: float = 0.45,
) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]

    cam_img = Image.fromarray((cam_map * 255.0).astype(np.uint8), mode="L")
    cam_img = cam_img.resize((width, height), resample=Image.BILINEAR)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0

    heatmap = cm.jet(cam_resized)[..., :3]
    overlay = ((1.0 - alpha) * rgb) + (alpha * heatmap)
    overlay = np.clip(overlay, 0.0, 1.0)

    Image.fromarray((overlay * 255.0).astype(np.uint8)).save(out_path)


def build_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, int, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    img_size = int(checkpoint.get("img_size", 224))
    classes = checkpoint.get("classes", ["healthy", "BS"])
    return model, img_size, classes


def main() -> None:
    source_dir = Path(__file__).resolve().parent
    project_dir = source_dir.parent

    checkpoint_path = source_dir / "best_ResNet18_traubenwelke.pt"
    samples_root = project_dir / "data" / "grad_cam_higlights"
    set_dirs = [samples_root / f"set_{idx}" for idx in range(1, 5)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size, classes = build_model(checkpoint_path, device)

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    cam_engine = GradCAM(model, target_layer=model.layer3)

    try:
        for set_dir in set_dirs:
            if not set_dir.exists():
                print(f"Skipping missing directory: {set_dir}")
                continue

            image_paths = list_images(set_dir)
            print(f"Processing {set_dir.name}: {len(image_paths)} images")

            for image_path in image_paths:
                image = Image.open(image_path).convert("RGB")
                x = preprocess(image).unsqueeze(0).to(device)

                model.zero_grad(set_to_none=True)
                logits = model(x)
                pred_idx = int(torch.argmax(logits, dim=1).item())

                cam_map, logits = cam_engine.generate(x, class_idx=pred_idx)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

                out_path = image_path.with_name(f"{image_path.stem}_gradcam.png")
                overlay_heatmap(image, cam_map, out_path)

                print(
                    f"  {image_path.name} -> {out_path.name} | "
                    f"pred={classes[pred_idx] if pred_idx < len(classes) else pred_idx} "
                    f"prob={probs[pred_idx]:.4f}"
                )
    finally:
        cam_engine.remove()


if __name__ == "__main__":
    main()

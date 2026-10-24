"""Train a pixel-wise semantic segmentation model on CVAT indexed masks."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class EchoMaskDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: int, class_count: int):
        self.image_dir = root / "images" / split
        self.mask_dir = root / "masks" / split
        self.image_size = image_size
        self.class_count = class_count
        self.images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        missing = [p.name for p in self.images if not (self.mask_dir / f"{p.stem}.png").is_file()]
        if not self.images or missing:
            raise SystemExit(f"Invalid {split} split: {len(self.images)} images, {len(missing)} masks missing under {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"
        with Image.open(image_path) as im:
            image = im.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        with Image.open(mask_path) as im:
            mask = im.convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            target = np.asarray(mask, dtype=np.uint8)
        if int(target.max()) >= self.class_count:
            raise SystemExit(f"Mask {mask_path} has class ID {target.max()}, but the dataset declares {self.class_count} classes including background")
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - MEAN) / STD
        return tensor, torch.from_numpy(target.astype(np.int64))


def make_model(class_count: int, pretrained: bool, auxiliary: bool | None = None) -> nn.Module:
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    aux_loss = pretrained if auxiliary is None else auxiliary
    model = deeplabv3_resnet50(weights=weights, weights_backbone=None, aux_loss=aux_loss)
    model.classifier[-1] = nn.Conv2d(256, class_count, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, class_count, kernel_size=1)
    return model


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, class_count: int) -> torch.Tensor:
    ce = F.cross_entropy(logits, target)
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target, num_classes=class_count).permute(0, 3, 1, 2).float()
    intersection = (probabilities[:, 1:] * one_hot[:, 1:]).sum(dim=(0, 2, 3))
    denominator = probabilities[:, 1:].sum(dim=(0, 2, 3)) + one_hot[:, 1:].sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return ce + (1.0 - dice.mean())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_count: int) -> tuple[float, list[tuple[float, float]]]:
    model.eval()
    confusion = torch.zeros((class_count, class_count), dtype=torch.int64)
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        predictions = model(images)["out"].argmax(dim=1)
        encoded = (targets * class_count + predictions).reshape(-1)
        confusion += torch.bincount(encoded.cpu(), minlength=class_count * class_count).reshape(class_count, class_count)
    scores = []
    for class_id in range(1, class_count):
        tp = confusion[class_id, class_id].item()
        fp = confusion[:, class_id].sum().item() - tp
        fn = confusion[class_id, :].sum().item() - tp
        iou = tp / max(tp + fp + fn, 1)
        dice = (2 * tp) / max(2 * tp + fp + fn, 1)
        scores.append((dice, iou))
    mean_dice = float(np.mean([score[0] for score in scores])) if scores else 0.0
    return mean_dice, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "dataset_semantic")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/use pretrained ImageNet weights")
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "semantic" / "mitral_valve")
    args = parser.parse_args()

    root = args.data.resolve()
    classes_file = root / "classes.json"
    if not classes_file.is_file():
        raise SystemExit(f"Missing {classes_file}; run prepare_cvat_semantic.py first")
    classes = json.loads(classes_file.read_text(encoding="utf-8"))["classes"]
    class_count = len(classes) + 1  # background + foreground classes
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available in this PyTorch environment")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train_set = EchoMaskDataset(root, "train", args.image_size, class_count)
    val_set = EchoMaskDataset(root, "val", args.image_size, class_count)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = make_model(class_count, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.run_dir / "metrics.csv"
    best_dice = -1.0

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "mean_val_dice", "mean_val_iou"] + [v for name in classes for v in (f"{name}_dice", f"{name}_iou")])
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(images)
                loss = segmentation_loss(output["out"], targets, class_count)
                if "aux" in output:
                    loss = loss + 0.4 * F.cross_entropy(output["aux"], targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            mean_dice, class_scores = evaluate(model, val_loader, device, class_count)
            mean_iou = float(np.mean([score[1] for score in class_scores])) if class_scores else 0.0
            train_loss = total_loss / max(len(train_loader), 1)
            row = [epoch, round(train_loss, 5), round(mean_dice, 5), round(mean_iou, 5)]
            row.extend(round(value, 5) for pair in class_scores for value in pair)
            writer.writerow(row)
            f.flush()
            print(f"Epoch {epoch:03d}/{args.epochs} | loss {train_loss:.4f} | val Dice {mean_dice:.4f} | val IoU {mean_iou:.4f}")
            for class_name, (dice, iou) in zip(classes, class_scores):
                print(f"  {class_name}: Dice {dice:.4f}, IoU {iou:.4f}")
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save({"model": model.state_dict(), "classes": classes, "image_size": args.image_size,
                            "epoch": epoch, "val_dice": mean_dice, "auxiliary": model.aux_classifier is not None},
                           args.run_dir / "best.pt")

    print(f"Best mean validation Dice: {best_dice:.4f}")
    print(f"Weights: {(args.run_dir / 'best.pt').resolve()}")
    print(f"Metrics: {csv_path.resolve()}")


if __name__ == "__main__":
    main()

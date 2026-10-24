"""Save semantic mask PNGs and colored overlays for an image or image folder."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_semantic import make_model

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
COLORS = [(255, 64, 64), (64, 200, 255), (255, 200, 40), (160, 80, 255)]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="Image file or image directory")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "semantic" / "predictions")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity in [0, 1]")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Checkpoint not found: {args.weights}")
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")
    if not 0 <= args.alpha <= 1:
        raise SystemExit("--alpha must be between 0 and 1")

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available in this PyTorch environment")
    checkpoint = torch.load(args.weights, map_location=device)
    classes = checkpoint["classes"]
    image_size = int(checkpoint["image_size"])
    model = make_model(len(classes) + 1, pretrained=False, auxiliary=bool(checkpoint.get("auxiliary", False)))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    images = [args.source] if args.source.is_file() else sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No supported images found in {args.source}")
    args.out.mkdir(parents=True, exist_ok=True)

    for path in images:
        with Image.open(path) as source:
            original = source.convert("RGB")
        resized = original.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = ((tensor - MEAN) / STD).unsqueeze(0).to(device)
        with torch.no_grad():
            indices = model(tensor)["out"].argmax(dim=1)[0].byte().cpu().numpy()
        mask = Image.fromarray(indices, mode="L").resize(original.size, Image.Resampling.NEAREST)
        mask_array = np.asarray(mask, dtype=np.uint8)
        base = np.asarray(original, dtype=np.float32)
        colored = np.zeros_like(base)
        for class_id in range(1, len(classes) + 1):
            colored[mask_array == class_id] = COLORS[(class_id - 1) % len(COLORS)]
        selected = mask_array > 0
        base[selected] = (1.0 - args.alpha) * base[selected] + args.alpha * colored[selected]
        Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(args.out / f"{path.stem}_overlay.png")
        mask.save(args.out / f"{path.stem}_mask.png")

    print(f"Processed {len(images)} image(s); classes: {', '.join(classes)}")
    print(f"Overlays and masks saved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()

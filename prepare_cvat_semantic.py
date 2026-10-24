"""Convert a CVAT Segmentation Mask ZIP into indexed semantic masks."""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
LABELMAP_RE = re.compile(r"^\s*([^:#]+):\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")


def parse_labelmap(text: str) -> list[tuple[str, tuple[int, int, int]]]:
    labels = []
    for line in text.splitlines():
        match = LABELMAP_RE.match(line)
        if match:
            name = match.group(1).strip()
            rgb = tuple(int(match.group(i)) for i in (2, 3, 4))
            labels.append((name, rgb))
    if not labels:
        raise SystemExit("Could not parse labelmap.txt in CVAT ZIP")
    return labels


def make_index_mask(mask_bytes: bytes, labelmap: list[tuple[str, tuple[int, int, int]]], classes: list[str]) -> Image.Image:
    class_ids = {name: i + 1 for i, name in enumerate(classes)}  # 0 is background
    missing = sorted(set(classes) - {name for name, _ in labelmap})
    if missing:
        raise SystemExit(f"Requested classes missing from labelmap.txt: {', '.join(missing)}")

    with Image.open(io.BytesIO(mask_bytes)) as source:
        if source.mode in {"L", "P", "I", "I;16", "1"}:
            indices = np.asarray(source).astype(np.int64)
            result = np.zeros(indices.shape, dtype=np.uint8)
            # CVAT grayscale masks use the labelmap line order as pixel IDs.
            for index, (name, _) in enumerate(labelmap):
                if name in class_ids:
                    result[indices == index] = class_ids[name]
        else:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            result = np.zeros(rgb.shape[:2], dtype=np.uint8)
            for name, color in labelmap:
                if name in class_ids:
                    result[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = class_ids[name]
    return Image.fromarray(result, mode="L")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path, help="CVAT Segmentation Mask 1.1 export ZIP")
    parser.add_argument("--images", required=True, type=Path, help="Original images directory")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "dataset_semantic")
    parser.add_argument("--classes", nargs="+", default=["mitral_valve"], help="Foreground labels, in output ID order")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.zip.is_file():
        raise SystemExit(f"CVAT ZIP not found: {args.zip}")
    if not args.images.is_dir():
        raise SystemExit(f"Original image directory not found: {args.images}")
    if not 0.05 <= args.val_ratio <= 0.5:
        raise SystemExit("--val-ratio must be between 0.05 and 0.5")
    if not args.classes or len(set(args.classes)) != len(args.classes):
        raise SystemExit("--classes must contain unique label names")
    if len(args.classes) > 254:
        raise SystemExit("This indexed PNG pipeline supports up to 254 foreground classes")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.out}; choose another --out")

    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
        labelmap_entry = next((n for n in names if Path(n).name.lower() == "labelmap.txt"), None)
        if labelmap_entry is None:
            raise SystemExit("CVAT ZIP has no labelmap.txt; export as Segmentation Mask 1.1")
        labelmap = parse_labelmap(archive.read(labelmap_entry).decode("utf-8-sig"))
        mask_entries = [n for n in names if "SegmentationClass/" in n and Path(n).suffix.lower() == ".png"]
        if not mask_entries:
            raise SystemExit("No PNG masks found under SegmentationClass/ in the CVAT ZIP")
        masks = {Path(n).stem: archive.read(n) for n in mask_entries}

    image_files = {p.stem: p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    missing_images = sorted(set(masks) - set(image_files))
    if missing_images:
        raise SystemExit(f"No source image found for {len(missing_images)} masks; first missing stem: {missing_images[0]}")
    if len(masks) < 2:
        raise SystemExit("At least two annotated images are required for train/validation split")

    stems = sorted(masks)
    random.Random(args.seed).shuffle(stems)
    n_val = max(1, round(len(stems) * args.val_ratio))
    val_stems = set(stems[:n_val])
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "masks" / split).mkdir(parents=True, exist_ok=True)
    for stem, content in masks.items():
        split = "val" if stem in val_stems else "train"
        image_path = image_files[stem]
        shutil.copy2(image_path, args.out / "images" / split / image_path.name)
        indexed = make_index_mask(content, labelmap, args.classes)
        indexed.save(args.out / "masks" / split / f"{stem}.png")

    (args.out / "classes.json").write_text(json.dumps({"classes": args.classes}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(masks)} images: {len(masks) - n_val} train, {n_val} val")
    print(f"Foreground classes: {', '.join(args.classes)} (mask IDs start at 1; background is 0)")
    print(f"Dataset: {args.out.resolve()}")
    print("If all images come from one examination, this random validation split is not independent validation.")


if __name__ == "__main__":
    main()

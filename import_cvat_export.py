"""Pair a CVAT YOLO-seg label-only export with the original JPGs and split it."""
from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

import yaml

EXPECTED_CLASSES = ["mitral_valve", "fossa_ovalis"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path, help="CVAT Ultralytics YOLO Segmentation export")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing the original images")
    parser.add_argument("--out", type=Path, default=Path("dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.zip.is_file():
        raise SystemExit(f"CVAT ZIP not found: {args.zip}")
    if not args.images.is_dir():
        raise SystemExit(f"Image directory not found: {args.images}")
    if not 0.05 <= args.val_ratio <= 0.5:
        raise SystemExit("--val-ratio must be between 0.05 and 0.5")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.out}; move it or choose another --out")

    with zipfile.ZipFile(args.zip) as archive:
        try:
            spec = yaml.safe_load(archive.read("data.yaml"))
        except (KeyError, yaml.YAMLError) as exc:
            raise SystemExit(f"Could not read data.yaml from CVAT ZIP: {exc}") from exc
        names = spec.get("names", {})
        class_names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
        if class_names != EXPECTED_CLASSES:
            raise SystemExit(f"Expected classes in this order: {EXPECTED_CLASSES}; ZIP has: {class_names}")
        labels = {}
        for entry in archive.infolist():
            path = Path(entry.filename)
            if path.parts[:1] == ("labels",) and path.suffix.lower() == ".txt" and not entry.is_dir():
                labels[path.stem] = archive.read(entry.filename)

    source_images = {p.stem: p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    missing = sorted(set(labels) - set(source_images))
    if missing:
        raise SystemExit(f"No source image found for {len(missing)} labels; first missing stem: {missing[0]}")
    if len(labels) < 2:
        raise SystemExit("At least two labeled images are required for a train/validation split")

    stems = sorted(labels)
    random.Random(args.seed).shuffle(stems)
    n_val = max(1, round(len(stems) * args.val_ratio))
    val_stems = set(stems[:n_val])
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as archive:
        for stem, label_content in labels.items():
            split = "val" if stem in val_stems else "train"
            shutil.copy2(source_images[stem], args.out / "images" / split / source_images[stem].name)
            (args.out / "labels" / split / f"{stem}.txt").write_bytes(label_content)

    print(f"Imported {len(labels)} labeled images: {len(labels) - n_val} train, {n_val} val")
    print(f"Dataset: {args.out.resolve()}")
    print("Review the README: this random split is not independent validation if images come from one exam.")


if __name__ == "__main__":
    main()

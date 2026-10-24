"""Train a two-class YOLO segmentation model on CVAT YOLO-seg exports."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

CLASSES = ["mitral_valve", "fossa_ovalis"]


def validate_dataset(root: Path) -> None:
    missing = []
    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        if not images.is_dir() or not labels.is_dir():
            missing.append(f"{split}: expected {images} and {labels}")
            continue
        image_files = [p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not image_files:
            missing.append(f"{split}: no JPG/PNG images in {images}")
            continue
        absent = [p.name for p in image_files if not (labels / f"{p.stem}.txt").exists()]
        if absent:
            missing.append(f"{split}: {len(absent)} images have no matching label (e.g. {absent[0]})")
    if missing:
        raise SystemExit("Dataset is not ready:\n- " + "\n- ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset"), help="Dataset root")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Pretrained segmentation checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="e.g. 0, cpu; defaults to Ultralytics auto-detection")
    args = parser.parse_args()

    root = args.data.resolve()
    validate_dataset(root)
    data_yaml = root / "echo.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val", "names": {i: n for i, n in enumerate(CLASSES)}},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    model = YOLO(args.model)
    run_project = Path(__file__).resolve().parent / "runs" / "segment"
    options = dict(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                   project=str(run_project), name="echo_structures", exist_ok=True)
    if args.device is not None:
        options["device"] = args.device
    model.train(**options)


if __name__ == "__main__":
    main()

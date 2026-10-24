"""Run segmentation inference on an image, image folder, or video."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True, help="Image, directory, or video path")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    run_project = Path(__file__).resolve().parent / "runs" / "predict"
    options = dict(source=args.source, imgsz=args.imgsz, conf=args.conf,
                   project=str(run_project), name="echo_structures", exist_ok=True, save=True)
    if args.device is not None:
        options["device"] = args.device
    YOLO(str(args.weights)).predict(**options)


if __name__ == "__main__":
    main()

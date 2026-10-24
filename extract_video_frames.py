"""Sample frames from an echo video for manual annotation."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--every-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")
    if args.every_seconds <= 0:
        raise SystemExit("--every-seconds must be greater than zero")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise SystemExit("Could not read video frame rate")
    step = max(1, round(fps * args.every_seconds))
    args.out.mkdir(parents=True, exist_ok=True)
    frame_index = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step == 0:
            path = args.out / f"frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(path), frame):
                cap.release()
                raise SystemExit(f"Failed to save {path}")
            saved += 1
        frame_index += 1
    cap.release()
    print(f"Saved {saved} frames from {frame_index} total frames to {args.out}")


if __name__ == "__main__":
    main()

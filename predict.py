"""predict.py —— 用训练好的分割模型 + 相对位置模型做推理。

支持三种输入：
    * 单张图像        -> 标注图 + 同名 JSON
    * 图像目录        -> 标注图（逐张）+ predictions.json
    * 视频            -> 标注视频 + 同名 JSON（逐帧摘要）

核心行为（对应"只找到一个结构时如何给出另一个的方向"）：
    1. YOLO-seg 检测二尖瓣/卵圆孔；
    2. 若两个都检测到 -> 画出连线，并用马氏距离检查这对组合是否符合位置先验；
    3. 若只检测到其中一个 -> 用 relation.json 预测另一个结构的**方向、距离、不确定度**，
       画出"实线箭头 + 扇形锥 + 虚线 ROI"；目标落在画面外时提示 OUTSIDE FOV；
    4. 若开启 --guided，则在预测 ROI 内裁剪放大再检测一次，找回来漏检的结构，
       找回的结果用洋红色标记，并在 JSON 里标注 source="guided"。

用法示例：
    python predict.py --weights runs/segment/echo_relation_seg/weights/best.pt \
        --source D:\\data\\echo --out runs/predict
    python predict.py --weights ... --source case.avi --out runs/predict --max-frames 300
    python predict.py --weights ... --source frame.jpg --no-guided --no-relation
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from common import CLASS_NAMES, image_files, imread_unicode, imwrite_unicode, save_json
from inference import EchoSegmenter, FrameResult, InferenceConfig, analyze_frame
from relation import RelationPredictor
from visualize import render_result, render_side_by_side

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg"}


# ---------------------------------------------------------------------------
# 视频路径（Windows 上 cv2.VideoCapture/VideoWriter 处理中文路径会失败）
# ---------------------------------------------------------------------------
def _needs_temp_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def open_video(path: Path) -> tuple[cv2.VideoCapture, Path | None]:
    """打开视频；路径含中文时先复制到临时 ASCII 路径。返回 (cap, 临时文件)。"""
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap, None
    if not _needs_temp_path(path):
        return cap, None
    temp_dir = Path(tempfile.mkdtemp(prefix="echo_video_"))
    temp_path = temp_dir / ("input" + path.suffix)
    shutil.copyfile(path, temp_path)
    return cv2.VideoCapture(str(temp_path)), temp_dir


def create_video_writer(
    path: Path, fps: float, size: tuple[int, int]
) -> tuple[cv2.VideoWriter, Path, Path | None]:
    """创建 VideoWriter。

    返回 `(writer, 实际写入路径, 临时目录或 None)`。路径含中文时先写进临时 ASCII 目录，
    收尾再把文件移到目标位置；`临时目录` 只用于清理，**绝不能拿写入路径的 parent 当清理目标**
    （非临时分支下它就是输出目录本身）。
    """
    if not _needs_temp_path(path):
        return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size), path, None
    temp_dir = Path(tempfile.mkdtemp(prefix="echo_out_"))
    temp_path = temp_dir / ("output" + (path.suffix or ".mp4"))
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return writer, temp_path, temp_dir


# ---------------------------------------------------------------------------
# 关系模型发现
# ---------------------------------------------------------------------------
def discover_relation(weights: Path, explicit: str | None) -> Path | None:
    if explicit is not None and explicit.lower() in {"none", "off", "no"}:
        return None
    if explicit and explicit.lower() != "auto":
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"找不到关系模型文件：{path}")
        return path
    candidates = [
        Path("dataset/relation.json"),
        weights.parent / "relation.json",
        weights.parent.parent / "relation.json",
        weights.parent.parent.parent / "relation.json",
        Path("relation.json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# 单帧处理
# ---------------------------------------------------------------------------
def _frame_brief(frame: FrameResult, source: str = "") -> str:
    mitral = len(frame.by_class(0))
    fossa = len(frame.by_class(1))
    guided = sum(1 for link in frame.links if link.guided)
    plausible = sum(1 for link in frame.links if link.consistency and link.consistency["plausible"])
    implausible = sum(1 for link in frame.links if link.consistency and not link.consistency["plausible"])
    parts = [f"mitral={mitral}", f"fossa={fossa}"]
    if guided:
        parts.append(f"guided={guided}")
    if implausible:
        parts.append(f"implausible={implausible}")
    elif plausible:
        parts.append("pair=ok")
    return f"{source}: " + " ".join(parts)


def _config_payload(cfg: InferenceConfig) -> dict:
    """把本次推理用到的关键阈值写进 JSON，便于复现。"""
    return {
        "conf": cfg.conf, "iou": cfg.iou, "imgsz": cfg.imgsz,
        "guided": cfg.guided, "guide_conf": cfg.guide_conf,
        "guide_max_angle": cfg.guide_max_angle, "guide_upscale": cfg.guide_upscale,
        "min_area_px": cfg.min_area_px,
    }


def _summarize(records: list[dict]) -> dict:
    both = sum(1 for r in records if not r["missing_classes"])
    guided = sum(1 for r in records if any(link["guided_recovery"] for link in r["relations"]))
    implausible = sum(
        1 for r in records
        for link in r["relations"]
        if link["consistency"] and not link["consistency"]["plausible"]
    )
    missing_mitral = sum(1 for r in records if "mitral_valve" in r["missing_classes"])
    missing_fossa = sum(1 for r in records if "fossa_ovalis" in r["missing_classes"])
    return {
        "frames": len(records),
        "frames_with_both": both,
        "frames_with_guided_recovery": guided,
        "implausible_pairs": implausible,
        "frames_missing_mitral": missing_mitral,
        "frames_missing_fossa": missing_fossa,
    }


def process_image(
    segmenter: EchoSegmenter,
    predictor: RelationPredictor | None,
    image: np.ndarray,
    cfg: InferenceConfig,
) -> FrameResult:
    return analyze_frame(segmenter, predictor, image, cfg)


# ---------------------------------------------------------------------------
# 三种输入模式
# ---------------------------------------------------------------------------
def run_images(
    segmenter: EchoSegmenter,
    predictor: RelationPredictor | None,
    cfg: InferenceConfig,
    files: list[Path],
    out_dir: Path,
    args: argparse.Namespace,
) -> int:
    records: list[dict] = []
    saved = 0
    for index, path in enumerate(files, start=1):
        image = imread_unicode(path)
        if image is None:
            print(f"[skip] 读不到图像：{path}")
            continue
        frame = process_image(segmenter, predictor, image, cfg)
        record = frame.to_json(str(path))
        records.append(record)
        print(f"[{index}/{len(files)}] {_frame_brief(frame, path.name)}")
        if args.save_images:
            if args.side_by_side:
                canvas = render_side_by_side(
                    image, frame,
                    show_masks=args.show_masks,
                    show_cone=args.show_cone,
                    show_roi=args.show_roi,
                )
            else:
                canvas = render_result(
                    image, frame,
                    show_masks=args.show_masks,
                    show_cone=args.show_cone,
                    show_roi=args.show_roi,
                )
            target = out_dir / f"{path.stem}.jpg"
            if imwrite_unicode(target, canvas):
                saved += 1
            else:
                print(f"[warn] 写图失败：{target}")

    payload = {
        "weights": str(args.weights),
        "relation": str(args.relation_path) if args.relation_path else None,
        "config": _config_payload(cfg),
        "summary": _summarize(records),
        "frames": records,
    }
    save_json(out_dir / "predictions.json", payload)
    print(f"\n图像 {len(records)} 张，标注图 {saved} 张 -> {out_dir}")
    print(f"汇总：{json.dumps(payload['summary'], ensure_ascii=False)}")
    return 0


def run_video(
    segmenter: EchoSegmenter,
    predictor: RelationPredictor | None,
    cfg: InferenceConfig,
    path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> int:
    cap, temp_input = open_video(path)
    if not cap.isOpened():
        if temp_input is not None:
            shutil.rmtree(temp_input, ignore_errors=True)
        print(f"[error] 打不开视频：{path}", file=sys.stderr)
        return 1
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (1.0 <= fps <= 240.0):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = None
    temp_output: Path | None = None
    temp_output_dir: Path | None = None
    records: list[dict] = []
    frame_index = 0
    processed = 0
    failed = False
    try:
        while True:
            ok, image = cap.read()
            if not ok or image is None:
                break
            frame_index += 1
            if args.frame_stride > 1 and (frame_index - 1) % args.frame_stride != 0:
                continue
            if args.max_frames and processed >= args.max_frames:
                break
            processed += 1
            if image.shape[1] != width or image.shape[0] != height:
                width, height = image.shape[1], image.shape[0]
            frame = process_image(segmenter, predictor, image, cfg)
            record = frame.to_json(f"frame_{frame_index:06d}")
            record["frame_index"] = frame_index
            records.append(record)
            if args.side_by_side:
                canvas = render_side_by_side(
                    image, frame,
                    show_masks=args.show_masks,
                    show_cone=args.show_cone,
                    show_roi=args.show_roi,
                )
            else:
                canvas = render_result(
                    image, frame,
                    show_masks=args.show_masks,
                    show_cone=args.show_cone,
                    show_roi=args.show_roi,
                )
            if args.save_video:
                if writer is None:
                    writer, temp_output, temp_output_dir = create_video_writer(
                        out_dir / f"{path.stem}_pred.mp4", fps, (canvas.shape[1], canvas.shape[0])
                    )
                    if not writer.isOpened():
                        print("[error] 无法创建视频写入器（缺少 mp4v 编码器）", file=sys.stderr)
                        failed = True
                        break
                writer.write(canvas)
            if processed % 25 == 0:
                progress = f"{processed}/{total}" if total else str(processed)
                print(f"  视频进度 {progress} 帧：{_frame_brief(frame)}")
    except Exception:
        failed = True
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if temp_input is not None:
            shutil.rmtree(temp_input, ignore_errors=True)
        if failed and temp_output_dir is not None:
            shutil.rmtree(temp_output_dir, ignore_errors=True)
    if failed:
        return 1

    video_out = None
    if args.save_video and temp_output is not None:
        final_path = out_dir / f"{path.stem}_pred.mp4"
        if temp_output != final_path:
            shutil.move(str(temp_output), str(final_path))
            if temp_output_dir is not None:
                shutil.rmtree(temp_output_dir, ignore_errors=True)
        video_out = final_path

    payload = {
        "weights": str(args.weights),
        "relation": str(args.relation_path) if args.relation_path else None,
        "video": str(path),
        "fps": fps,
        "frame_stride": args.frame_stride,
        "config": _config_payload(cfg),
        "summary": _summarize(records),
        "frames": records,
    }
    save_json(out_dir / f"{path.stem}_predictions.json", payload)
    print(f"\n视频处理 {processed} 帧 -> {video_out if video_out else '(未写视频)'}")
    print(f"汇总：{json.dumps(payload['summary'], ensure_ascii=False)}")
    return 0


def run_single_image(
    segmenter: EchoSegmenter,
    predictor: RelationPredictor | None,
    cfg: InferenceConfig,
    path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> int:
    image = imread_unicode(path)
    if image is None:
        print(f"[error] 读不到图像：{path}", file=sys.stderr)
        return 1
    frame = process_image(segmenter, predictor, image, cfg)
    print(_frame_brief(frame, path.name))
    for link in frame.links:
        prediction = link.prediction
        if prediction is not None and link.target is None:
            print(f"  只找到 {link.anchor.name}：预测 {CLASS_NAMES[prediction.target_cls]} 在 "
                  f"{prediction.angle_deg:.1f}° 方向、{prediction.distance_px:.0f}px 处"
                  f"（±{prediction.cone_half_deg:.0f}°，P(存在)={prediction.presence_rate:.2f}）"
                  + ("，预计在画面外" if prediction.outside_fov else ""))
    payload = frame.to_json(str(path))
    payload["relation"] = str(args.relation_path) if args.relation_path else None
    payload["config"] = _config_payload(cfg)
    save_json(out_dir / f"{path.stem}.json", payload)
    if args.save_images:
        if args.side_by_side:
            canvas = render_side_by_side(
                image, frame,
                show_masks=args.show_masks,
                show_cone=args.show_cone,
                show_roi=args.show_roi,
            )
        else:
            canvas = render_result(
                image, frame,
                show_masks=args.show_masks,
                show_cone=args.show_cone,
                show_roi=args.show_roi,
            )
        target = out_dir / f"{path.stem}.jpg"
        if not imwrite_unicode(target, canvas):
            print(f"[warn] 写图失败：{target}", file=sys.stderr)
        else:
            print(f"  标注图 -> {target}")
    print(f"  JSON -> {out_dir / (path.stem + '.json')}")
    return 0


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="二尖瓣/卵圆孔分割 + 相对方向预测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="YOLO-seg 权重，如 runs/segment/.../best.pt")
    parser.add_argument("--source", required=True, help="图像文件 / 图像目录 / 视频文件")
    parser.add_argument("--out", default="runs/predict", help="输出目录")
    parser.add_argument("--relation", default="auto",
                        help="relation.json 路径；auto=自动查找；none=不做关系预测")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=30)
    parser.add_argument("--device", default=None, help="如 0 / cpu；留空自动")
    parser.add_argument("--guided", dest="guided", action="store_true", default=True,
                        help="只找到锚结构时，在预测 ROI 内二次检测找回目标")
    parser.add_argument("--no-guided", dest="guided", action="store_false")
    parser.add_argument("--guide-conf", type=float, default=0.25,
                        help="二次检测的置信度阈值（不要低于 --conf，否则容易引入误检）")
    parser.add_argument("--guide-max-angle", type=float, default=45.0,
                        help="二次检测结果相对预测方向允许的最大偏差（度）")
    parser.add_argument("--guide-upscale", type=float, default=2.0, help="二次检测的最大放大倍数")
    parser.add_argument("--min-area", type=float, default=150.0, help="小于该像素面积的检测被丢弃")
    parser.add_argument("--no-relation", dest="relation_disabled", action="store_true",
                        help="完全不加载关系模型（只做分割）")
    parser.add_argument("--save-images", dest="save_images", action="store_true", default=True)
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    parser.add_argument("--save-video", dest="save_video", action="store_true", default=True)
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.add_argument("--side-by-side", action="store_true", help="标注图与原图左右拼接")
    parser.add_argument("--show-masks", dest="show_masks", action="store_true", default=True)
    parser.add_argument("--no-masks", dest="show_masks", action="store_false")
    parser.add_argument("--show-cone", dest="show_cone", action="store_true", default=True)
    parser.add_argument("--no-cone", dest="show_cone", action="store_false")
    parser.add_argument("--show-roi", dest="show_roi", action="store_true", default=True)
    parser.add_argument("--no-roi", dest="show_roi", action="store_false")
    parser.add_argument("--frame-stride", type=int, default=1, help="视频抽帧间隔")
    parser.add_argument("--max-frames", type=int, default=0, help="视频最多处理多少帧（0=全部）")
    parser.add_argument("--limit", type=int, default=0, help="目录模式只处理前 N 张（调试用）")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    weights = Path(args.weights)
    if not weights.is_file():
        print(f"[error] 找不到权重：{weights}", file=sys.stderr)
        return 1
    source = Path(args.source)
    if not source.exists():
        print(f"[error] 找不到输入：{source}", file=sys.stderr)
        return 1
    if args.frame_stride < 1:
        print("[error] --frame-stride 必须 >= 1", file=sys.stderr)
        return 1

    relation_path = None
    if not args.relation_disabled:
        relation_path = discover_relation(weights, args.relation)
    args.relation_path = relation_path
    predictor = None
    if relation_path is not None:
        predictor = RelationPredictor.load(relation_path)
        print(f"关系模型: {relation_path}")
    else:
        print("关系模型: 未使用（只做分割）")

    cfg = InferenceConfig(
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        imgsz=args.imgsz,
        guided=args.guided,
        guide_conf=args.guide_conf,
        guide_max_angle=args.guide_max_angle,
        guide_upscale=args.guide_upscale,
        min_area_px=args.min_area,
    )
    segmenter = EchoSegmenter(weights, imgsz=args.imgsz, device=args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        files = image_files(source)
        if args.limit:
            files = files[: args.limit]
        if not files:
            print(f"[error] 目录里没有图像：{source}", file=sys.stderr)
            return 1
        print(f"图像目录: {source}（{len(files)} 张）")
        return run_images(segmenter, predictor, cfg, files, out_dir, args)
    if source.suffix.lower() in VIDEO_SUFFIXES:
        print(f"视频: {source}")
        return run_video(segmenter, predictor, cfg, source, out_dir, args)
    return run_single_image(segmenter, predictor, cfg, source, out_dir, args)


if __name__ == "__main__":
    raise SystemExit(main())

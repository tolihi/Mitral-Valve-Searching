"""evaluate.py —— 两条独立的评估线，避免"自己说自己好"。

评估线 1：分割精度
    直接调用 ultralytics 的 `model.val()`，在 `--split` 指定的划分上得到
    逐类 mask/box mAP50、mAP50-95。这是与训练框架一致的官方指标。

评估线 2：关系预测 / 引导式找回的实际效果
    对划分内每一张有标注的帧跑完整推理链（分割 -> 关系 -> 引导找回），
    与标注多边形比对，得到：
      * 逐类召回/误检（仅模型 vs 模型+引导找回）
      * 引导找回率（模型漏检、引导找回并命中的比例）
      * 只找到锚结构时，预测方向与真实方向的角度误差
      * 两个结构都在时，这对组合被判为"合理"的比例（马氏距离）
    注意：关系模型是在**全部**成对样本上拟合的，因此这里的角度误差是"训练集内"的
    乐观估计；无偏估计见 relation_report.md 里的 LOO 指标。

用法：
    python evaluate.py --weights runs/segment/echo_relation_seg/weights/best.pt --data dataset
    python evaluate.py --weights ... --data dataset --split train --no-val   # 只看关系链
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from common import (
    CLASS_NAMES,
    Structure,
    box_iou,
    image_size,
    imread_unicode,
    read_seg_label,
    resolve_data_yaml,
    save_json,
    structure_from_polygon,
    wrap_angle_deg,
)
from inference import EchoSegmenter, InferenceConfig, analyze_frame
from relation import RelationPredictor
from predict import discover_relation


# ---------------------------------------------------------------------------
# 标注加载
# ---------------------------------------------------------------------------
def _image_for_label(label_path: Path, image_dir: Path) -> Path | None:
    for candidate in sorted(image_dir.glob(label_path.stem + ".*")):
        if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            return candidate
    return None


def load_gt_frames(data: Path, splits: list[str], limit: int = 0) -> list[dict]:
    frames: list[dict] = []
    for split in splits:
        image_dir = data / "images" / split
        label_dir = data / "labels" / split
        if not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = _image_for_label(label_path, image_dir)
            if image_path is None:
                continue
            width, height = image_size(image_path)
            structures: list[Structure] = []
            for cls, points in read_seg_label(label_path):
                if cls not in (0, 1):
                    continue
                structures.append(structure_from_polygon(points, width, height, cls))
            frames.append(
                {"stem": label_path.stem, "image": image_path, "width": width,
                 "height": height, "structures": structures}
            )
            if limit and len(frames) >= limit:
                return frames
    return frames


# ---------------------------------------------------------------------------
# 检测 <-> 标注 匹配
# ---------------------------------------------------------------------------
def _polygon_contains(structure: Structure, point: tuple[float, float]) -> bool:
    if structure.points_px is None or len(structure.points_px) < 3:
        return False
    contour = np.asarray(structure.points_px, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def match_detections(detections, gt_structures) -> list[tuple[int, int]]:
    """贪心匹配：先按 bbox IoU，IoU 低但检测中心落在标注多边形内也算命中。"""
    if not detections or not gt_structures:
        return []
    scores = np.zeros((len(detections), len(gt_structures)), dtype=np.float64)
    for i, detection in enumerate(detections):
        for j, gt in enumerate(gt_structures):
            iou = box_iou(detection.structure.box_px, gt.box_px)
            if iou < 0.3 and _polygon_contains(gt, detection.structure.center_px):
                iou = 0.3
            scores[i, j] = iou
    pairs: list[tuple[int, int]] = []
    used_det: set[int] = set()
    used_gt: set[int] = set()
    while True:
        flat = scores.copy()
        if used_det:
            flat[list(used_det), :] = -1.0
        if used_gt:
            flat[:, list(used_gt)] = -1.0
        index = int(np.argmax(flat))
        best = flat.reshape(-1)[index]
        if best <= 0.0:
            break
        i, j = np.unravel_index(index, flat.shape)
        pairs.append((int(i), int(j)))
        used_det.add(int(i))
        used_gt.add(int(j))
    return pairs


# ---------------------------------------------------------------------------
# 评估线 1：ultralytics val
# ---------------------------------------------------------------------------
def evaluate_segmentation(
    weights: Path, data_yaml: Path, split: str, imgsz: int, batch: int, device: str | None
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs = dict(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch,
                  conf=0.001, iou=0.6, plots=False, verbose=False)
    if device:
        kwargs["device"] = device
    metrics = model.val(**kwargs)

    names = getattr(model, "names", None) or {0: "mitral_valve", 1: "fossa_ovalis"}
    per_class: dict[str, dict] = {}
    try:
        box, seg = metrics.box, metrics.seg
        indices = list(getattr(box, "ap_class_index", []))
        for position, cls_id in enumerate(indices):
            name = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else str(cls_id)
            per_class[name] = {
                "box_map50": float(box.ap50[position]),
                "box_map50_95": float(box.ap[position]),
                "seg_map50": float(seg.ap50[position]),
                "seg_map50_95": float(seg.ap[position]),
            }
    except Exception as error:  # noqa: BLE001 - 指标结构随版本变化，降级不致命
        print(f"[warn] 逐类指标解析失败：{error}", file=sys.stderr)

    return {
        "split": split,
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "seg_map50": float(metrics.seg.map50),
        "seg_map50_95": float(metrics.seg.map),
        "per_class": per_class,
        "speed_ms": {key: float(value) for key, value in (metrics.speed or {}).items()},
    }


# ---------------------------------------------------------------------------
# 评估线 2：检测召回 + 关系 + 引导找回
# ---------------------------------------------------------------------------
def _distance(anchor: Structure, other: Structure) -> float:
    dx = (other.cx - anchor.cx) * anchor.image_w
    dy = (other.cy - anchor.cy) * anchor.image_h
    return float(math.hypot(dx, dy))


def _true_angle_deg(anchor: Structure, target: Structure) -> float:
    dx = (target.cx - anchor.cx) * anchor.image_w
    dy = (target.cy - anchor.cy) * anchor.image_h
    return math.degrees(math.atan2(dy, dx))


def evaluate_relation_chain(
    segmenter: EchoSegmenter,
    predictor: RelationPredictor | None,
    cfg: InferenceConfig,
    frames: list[dict],
) -> dict:
    counters = {
        "frames": 0,
        "frames_no_detection": 0,
        "frames_both_gt": 0,
        "gt": {0: 0, 1: 0},
        "det_model": {0: 0, 1: 0},
        "tp_model": {0: 0, 1: 0},
        "det_final": {0: 0, 1: 0},
        "tp_final": {0: 0, 1: 0},
        "guided_attempts": 0,
        "guided_recovered": 0,
        "guided_false_positive": 0,
        "pair_links": 0,
        "pair_correct": 0,
        "pair_implausible": 0,
        "miss_cases": 0,
        "miss_target_absent": 0,
        "miss_angle_errors": [],
        "miss_center_errors": [],
        "miss_outside_fov_hits": 0,
        "miss_outside_fov_total": 0,
    }
    per_frame: list[dict] = []

    for index, frame in enumerate(frames, start=1):
        image = imread_unicode(frame["image"])
        if image is None:
            print(f"[skip] 读不到图像：{frame['image']}")
            continue
        counters["frames"] += 1
        result = analyze_frame(segmenter, predictor, image, cfg)
        if not result.detections:
            counters["frames_no_detection"] += 1

        gt_by_class = {cls: [s for s in frame["structures"] if s.cls == cls] for cls in (0, 1)}
        for cls in (0, 1):
            counters["gt"][cls] += len(gt_by_class[cls])

        model_dets = [d for d in result.detections if d.source == "model"]
        final_dets = result.detections
        for cls in (0, 1):
            model_cls = [d for d in model_dets if d.cls == cls]
            final_cls = [d for d in final_dets if d.cls == cls]
            counters["det_model"][cls] += len(model_cls)
            counters["tp_model"][cls] += len(match_detections(model_cls, gt_by_class[cls]))
            counters["det_final"][cls] += len(final_cls)
            counters["tp_final"][cls] += len(match_detections(final_cls, gt_by_class[cls]))

        # 引导找回：模型漏检 -> 引导找回的检测是否命中标注
        guided_dets = [d for d in final_dets if d.source == "guided"]
        for detection in guided_dets:
            gt_list = gt_by_class[detection.cls]
            matched = match_detections([detection], gt_list)
            if matched:
                counters["guided_recovered"] += 1
            else:
                counters["guided_false_positive"] += 1
        if cfg.guided:
            counters["guided_attempts"] += sum(
                1
                for link in result.links
                if link.guided or (link.prediction is not None and link.target is None)
            )

        # 两个结构都在：检查连线是否正确、位置先验是否被判为合理
        if gt_by_class[0] and gt_by_class[1]:
            counters["frames_both_gt"] += 1
        for link in result.links:
            if link.target is not None and link.prediction is not None:
                gt_targets = gt_by_class[link.prediction.target_cls]
                matched = match_detections([link.target], gt_targets)
                counters["pair_links"] += 1
                if matched:
                    counters["pair_correct"] += 1
                if link.consistency and not link.consistency["plausible"]:
                    counters["pair_implausible"] += 1

        # 只找到锚结构：预测方向 vs 真实方向
        for link in result.links:
            if link.prediction is None or link.target is not None:
                continue
            target_cls = link.prediction.target_cls
            gt_targets = gt_by_class[target_cls]
            counters["miss_cases"] += 1
            if not gt_targets:
                counters["miss_target_absent"] += 1
                continue
            nearest = min(gt_targets, key=lambda gt: _distance(link.anchor.structure, gt))
            true_angle = _true_angle_deg(link.anchor.structure, nearest)
            error = abs(wrap_angle_deg(link.prediction.angle_deg - true_angle))
            counters["miss_angle_errors"].append(float(error))
            counters["miss_center_errors"].append(
                abs(_distance(link.anchor.structure, nearest) - link.prediction.distance_px)
            )
            if link.prediction.outside_fov:
                counters["miss_outside_fov_total"] += 1
                if error <= 30.0:
                    counters["miss_outside_fov_hits"] += 1

        per_frame.append({
            "stem": frame["stem"],
            "detections": len(final_dets),
            "model_detections": len(model_dets),
            "missing": [CLASS_NAMES[c] for c in result.missing],
            "guided": bool(guided_dets),
            "warnings": result.warnings,
        })
        if index % 20 == 0 or index == len(frames):
            print(f"  关系链评估 {index}/{len(frames)} 帧 ...")

    def _rate(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    angle_errors = np.asarray(counters.pop("miss_angle_errors"), dtype=np.float64)
    center_errors = np.asarray(counters.pop("miss_center_errors"), dtype=np.float64)
    summary = dict(counters)
    summary["guided_recovery_rate"] = _rate(counters["guided_recovered"], counters["guided_attempts"])
    summary["guided_false_positive_rate"] = _rate(
        counters["guided_false_positive"], counters["guided_recovered"] + counters["guided_false_positive"]
    )
    summary["pair_correct_rate"] = _rate(counters["pair_correct"], counters["pair_links"])
    summary["pair_implausible_rate"] = _rate(counters["pair_implausible"], counters["pair_links"])
    summary["miss_angle_mae_deg"] = float(angle_errors.mean()) if angle_errors.size else float("nan")
    summary["miss_angle_median_deg"] = float(np.median(angle_errors)) if angle_errors.size else float("nan")
    summary["miss_within_15_deg"] = float((angle_errors <= 15).mean()) if angle_errors.size else float("nan")
    summary["miss_center_mae_px"] = float(center_errors.mean()) if center_errors.size else float("nan")
    summary["miss_outside_fov_rate"] = _rate(
        counters["miss_outside_fov_hits"], counters["miss_outside_fov_total"]
    )
    summary["recall_model"] = {
        CLASS_NAMES[cls]: _rate(counters["tp_model"][cls], counters["gt"][cls]) for cls in (0, 1)
    }
    summary["recall_final"] = {
        CLASS_NAMES[cls]: _rate(counters["tp_final"][cls], counters["gt"][cls]) for cls in (0, 1)
    }
    summary["precision_model"] = {
        CLASS_NAMES[cls]: _rate(counters["tp_model"][cls], counters["det_model"][cls]) for cls in (0, 1)
    }
    summary["precision_final"] = {
        CLASS_NAMES[cls]: _rate(counters["tp_final"][cls], counters["det_final"][cls]) for cls in (0, 1)
    }
    return {"summary": summary, "frames": per_frame}


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _pct(value: float) -> str:
    return "n/a" if value != value else f"{value:.1%}"


def _num(value: float, digits: int = 2) -> str:
    return "n/a" if value != value else f"{value:.{digits}f}"


def write_report(
    path: Path,
    segmentation: dict | None,
    chain: dict,
    meta: dict,
) -> None:
    rows: list[str] = ["# 评估报告", ""]
    rows.append(f"- 权重：`{meta['weights']}`")
    rows.append(f"- 数据集：`{meta['data']}`")
    rows.append(f"- 划分：`{meta['split']}`（{chain['summary']['frames']} 张有标注帧）")
    rows.append(f"- 关系模型：`{meta.get('relation') or '未使用'}`")
    rows.append(f"- 推理阈值：conf={meta['conf']}，guided={meta['guided']}")
    rows.append("")

    if segmentation is not None:
        rows += ["## 1. 分割精度（ultralytics val）", ""]
        rows.append(
            f"- 总体：mask mAP50 **{segmentation['seg_map50']:.3f}**，mask mAP50-95 "
            f"**{segmentation['seg_map50_95']:.3f}**；box mAP50 {segmentation['box_map50']:.3f}，"
            f"box mAP50-95 {segmentation['box_map50_95']:.3f}"
        )
        rows.append("")
        if segmentation["per_class"]:
            rows += [
                "| 类别 | mask mAP50 | mask mAP50-95 | box mAP50 | box mAP50-95 |",
                "| --- | --- | --- | --- | --- |",
            ]
            for name, values in segmentation["per_class"].items():
                rows.append(
                    f"| {name} | {values['seg_map50']:.3f} | {values['seg_map50_95']:.3f} | "
                    f"{values['box_map50']:.3f} | {values['box_map50_95']:.3f} |"
                )
            rows.append("")
        if segmentation["speed_ms"]:
            rows.append("- 速度（ms/图）：" + "，".join(
                f"{key} {value:.1f}" for key, value in segmentation["speed_ms"].items()
            ))
            rows.append("")
    else:
        rows += ["## 1. 分割精度", "", "（本次用 `--no-val` 跳过）", ""]

    summary = chain["summary"]
    rows += ["## 2. 检测召回 / 误检", ""]
    rows += [
        "| 类别 | 标注数 | 召回（仅模型） | 召回（含引导找回） | 精确率（仅模型） | 精确率（含引导找回） |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cls in (0, 1):
        name = CLASS_NAMES[cls]
        rows.append(
            f"| {name} | {summary['gt'][cls]} | {_pct(summary['recall_model'][name])} | "
            f"{_pct(summary['recall_final'][name])} | {_pct(summary['precision_model'][name])} | "
            f"{_pct(summary['precision_final'][name])} |"
        )
    rows.append("")

    rows += ["## 3. 引导式找回", ""]
    rows.append(f"- 触发次数：{summary['guided_attempts']}")
    rows.append(f"- 找回并命中标注：{summary['guided_recovered']}")
    rows.append(f"- 找回但为误检：{summary['guided_false_positive']}")
    rows.append(f"- 找回率：**{_pct(summary['guided_recovery_rate'])}**")
    rows.append(f"- 找回结果里误检占比：{_pct(summary['guided_false_positive_rate'])}")
    rows.append("")

    rows += ["## 4. 关系预测", ""]
    rows.append(f"- 两结构都检测到并连线的次数：{summary['pair_links']}")
    rows.append(f"- 连线指向正确结构的比例：**{_pct(summary['pair_correct_rate'])}**")
    rows.append(f"- 真实配对被判为“不合理”的比例（误报）：{_pct(summary['pair_implausible_rate'])}")
    rows.append("")
    rows.append(f"- 只找到锚结构的次数：{summary['miss_cases']}（其中目标确实不在画面内 "
                f"{summary['miss_target_absent']} 次）")
    rows.append(f"- 预测方向角度 MAE：**{_num(summary['miss_angle_mae_deg'])}°**，"
                f"中位 {_num(summary['miss_angle_median_deg'])}°，"
                f"≤15° 比例 {_pct(summary['miss_within_15_deg'])}")
    rows.append(f"- 预测距离误差：{_num(summary['miss_center_mae_px'], 0)} px")
    rows.append(f"- 预测“目标在画面外”且方向误差≤30° 的比例：{_pct(summary['miss_outside_fov_rate'])}")
    rows.append("")

    rows += [
        "## 解读注意",
        "",
        "- 关系模型是在**全部**成对样本上拟合的，因此第 4 节的角度误差是训练集内的乐观估计；",
        "  无偏估计请看 `dataset/relation_report.md` 中的 LOO 指标。",
        "- 本数据集只有一个检查号，指标反映“同检查内换一帧”的稳定性，不代表跨患者泛化。",
        "- 引导式找回会提高召回，但也可能引入误检；第 3 节的“误检占比”就是它的代价。",
        "",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="评估分割精度 + 关系预测/引导找回效果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="dataset", help="数据集目录（含 data.yaml）")
    parser.add_argument("--split", default="val", help="val / train / all（all=train+val）")
    parser.add_argument("--relation", default="auto", help="relation.json 路径；auto/none")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=8, help="val 的 batch（关系链评估逐张）")
    parser.add_argument("--device", default=None)
    parser.add_argument("--guided", dest="guided", action="store_true", default=True)
    parser.add_argument("--no-guided", dest="guided", action="store_false")
    parser.add_argument("--guide-conf", type=float, default=0.25,
                        help="二次检测的置信度阈值（不要低于 --conf，否则容易引入误检）")
    parser.add_argument("--no-val", action="store_true", help="跳过 ultralytics val（只评估关系链）")
    parser.add_argument("--no-relation", dest="relation_disabled", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 张（调试用）")
    parser.add_argument("--out", default="runs/evaluate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    weights = Path(args.weights)
    if not weights.is_file():
        print(f"[error] 找不到权重：{weights}", file=sys.stderr)
        return 1
    data_yaml = resolve_data_yaml(Path(args.data))
    data_dir = data_yaml.parent

    relation_path = None
    predictor = None
    if not args.relation_disabled:
        relation_path = discover_relation(weights, args.relation)
        if relation_path is not None:
            predictor = RelationPredictor.load(relation_path)
            print(f"关系模型: {relation_path}")
    if predictor is None:
        print("关系模型: 未使用")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    segmentation = None
    if not args.no_val:
        split_for_val = "val" if args.split == "all" else args.split
        print(f"分割评估（ultralytics val，split={split_for_val}）...")
        segmentation = evaluate_segmentation(
            weights, data_yaml, split_for_val, args.imgsz, args.batch, args.device
        )
        print(f"  mask mAP50={segmentation['seg_map50']:.3f}  "
              f"mask mAP50-95={segmentation['seg_map50_95']:.3f}")

    splits = ["train", "val"] if args.split == "all" else [args.split]
    frames = load_gt_frames(data_dir, splits, limit=args.limit)
    if not frames:
        print(f"[error] 在 {data_dir} 的 {splits} 里没有找到有标注的帧", file=sys.stderr)
        return 1
    print(f"关系链评估：{len(frames)} 张有标注帧 ...")

    cfg = InferenceConfig(
        conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        guided=args.guided, guide_conf=args.guide_conf,
    )
    segmenter = EchoSegmenter(weights, imgsz=args.imgsz, device=args.device)
    chain = evaluate_relation_chain(segmenter, predictor, cfg, frames)

    payload = {
        "weights": str(weights),
        "data": str(data_yaml),
        "split": args.split,
        "relation": str(relation_path) if relation_path else None,
        "config": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
                   "guided": args.guided, "guide_conf": args.guide_conf},
        "segmentation": segmentation,
        "relation_chain": chain,
    }
    save_json(out_dir / "evaluate.json", payload)
    write_report(
        out_dir / "evaluate_report.md",
        segmentation,
        chain,
        {"weights": weights, "data": data_yaml, "split": args.split,
         "relation": str(relation_path) if relation_path else None,
         "conf": args.conf, "guided": args.guided},
    )

    summary = chain["summary"]
    print("\n=== 结果摘要 ===")
    for cls in (0, 1):
        name = CLASS_NAMES[cls]
        print(f"  {name}: 标注 {summary['gt'][cls]}，召回(模型) {_pct(summary['recall_model'][name])}，"
              f"召回(含引导) {_pct(summary['recall_final'][name])}")
    print(f"  引导找回率 {_pct(summary['guided_recovery_rate'])}"
          f"（误检占比 {_pct(summary['guided_false_positive_rate'])}）")
    print(f"  连线正确率 {_pct(summary['pair_correct_rate'])}，"
          f"真实配对误报率 {_pct(summary['pair_implausible_rate'])}")
    print(f"  漏检时方向误差 {_num(summary['miss_angle_mae_deg'])}°（中位 "
          f"{_num(summary['miss_angle_median_deg'])}°）")
    print(f"\n已保存：{out_dir / 'evaluate.json'}，{out_dir / 'evaluate_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

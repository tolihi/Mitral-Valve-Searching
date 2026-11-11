"""inference.py —— 单帧推理：YOLO 分割 -> 结构 -> 关系预测 -> 引导式二次检测。

predict.py / evaluate.py 共用这一层，保证"离线评估"和"实际使用"走同一条代码路径。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from common import (
    CLASS_NAMES,
    Structure,
    canonical_class,
    clip_box,
    point_in_box,
    structure_from_polygon,
    wrap_angle_deg,
)
from relation import DIRECTIONS, RelationPrediction, RelationPredictor


@dataclass
class Detection:
    """一次检测：规范类别号 + 置信度 + 结构几何。"""

    cls: int
    conf: float
    structure: Structure
    source: str = "model"          # "model" | "guided"

    @property
    def name(self) -> str:
        return CLASS_NAMES[self.cls]

    def to_json(self) -> dict:
        payload = self.structure.to_json()
        payload.update({"source": self.source, "conf": round(float(self.conf), 4)})
        return payload


@dataclass
class InferenceConfig:
    """推理参数（predict.py 的命令行一一对应）。"""

    conf: float = 0.25
    iou: float = 0.5
    max_det: int = 30
    imgsz: int = 1024
    guided: bool = True
    guide_conf: float = 0.25
    guide_max_angle: float = 45.0
    guide_upscale: float = 2.0
    min_area_px: float = 150.0
    max_links: int = 3


@dataclass
class RelationLink:
    """一条"锚结构 -> 目标结构"的关系结论。"""

    anchor: Detection
    prediction: RelationPrediction | None
    target: Detection | None = None
    consistency: dict | None = None
    guided: bool = False

    @property
    def confirmed(self) -> bool:
        return self.target is not None

    def to_json(self) -> dict:
        return {
            "anchor": {"cls": self.anchor.cls, "name": self.anchor.name,
                       "conf": round(float(self.anchor.conf), 4),
                       "center_px": [round(v, 1) for v in self.anchor.structure.center_px],
                       "source": self.anchor.source},
            "target": None if self.target is None else {
                "cls": self.target.cls, "name": self.target.name,
                "conf": round(float(self.target.conf), 4),
                "center_px": [round(v, 1) for v in self.target.structure.center_px],
                "source": self.target.source,
            },
            "prediction": None if self.prediction is None else self.prediction.to_json(),
            "consistency": self.consistency,
            "guided_recovery": bool(self.guided),
        }


@dataclass
class FrameResult:
    """一帧的完整结果。"""

    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    links: list[RelationLink] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def by_class(self, cls: int) -> list[Detection]:
        return [d for d in self.detections if d.cls == cls]

    def to_json(self, source: str = "") -> dict:
        return {
            "source": source,
            "image_size": [self.width, self.height],
            "detections": [d.to_json() for d in self.detections],
            "relations": [link.to_json() for link in self.links],
            "missing_classes": [CLASS_NAMES[c] for c in self.missing],
            "warnings": self.warnings,
        }


class EchoSegmenter:
    """封装 ultralytics YOLO-seg，把结果转成与类别顺序无关的 Detection。"""

    def __init__(self, weights: str | Path, imgsz: int = 1024, device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.imgsz = int(imgsz)
        self.device = device
        self.names = self._canonical_names()

    def _canonical_names(self) -> dict[int, str]:
        raw = getattr(self.model, "names", {}) or {}
        items = raw.items() if isinstance(raw, dict) else enumerate(raw)
        mapping: dict[int, str] = {}
        for index, name in items:
            kind = canonical_class(name)
            if kind is None:
                raise SystemExit(f"权重里的类别 {name!r} 无法映射到 mitral_valve/fossa_ovalis。")
            if kind == "combined":
                raise SystemExit(
                    f"权重里包含组合类 {name!r}，请先用 prepare_dataset.py --combined drop 重新训练。"
                )
            if kind in mapping.values():
                raise SystemExit(f"权重里出现重复类别 {name!r}，无法确定类别顺序。")
            mapping[int(index)] = kind
        missing = [name for name in CLASS_NAMES if name not in mapping.values()]
        if missing:
            raise SystemExit(f"权重缺少类别 {missing}，本脚本需要两个类都参与训练。")
        return {index: CLASS_NAMES.index(name) for index, name in mapping.items()}

    # -- 基础检测 -------------------------------------------------------
    def detect(
        self,
        image: np.ndarray,
        cfg: InferenceConfig,
        classes: Sequence[int] | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
        inv_scale: float = 1.0,
        full_size: tuple[int, int] | None = None,
    ) -> list[Detection]:
        """对（可能是裁剪/放大后的）图像推理，并把坐标映射回整帧。

        classes 是**权重里的原始类别索引**（ultralytics 的约定），不是规范类别号；
        调用方要用 self.raw_indices() 做一次换算。
        origin/inv_scale/full_size 用于引导式二次检测：把裁剪块上的结果还原到整帧。
        """
        if image.size == 0 or image.shape[0] < 8 or image.shape[1] < 8:
            return []
        full_w, full_h = full_size or (image.shape[1], image.shape[0])
        # cfg.imgsz 是权威值（JSON 里记录的也是它）；self.imgsz 只作为兜底。
        imgsz = int(cfg.imgsz) if int(cfg.imgsz) > 0 else self.imgsz
        result = self.model.predict(
            source=image,
            conf=float(cfg.conf),
            iou=float(cfg.iou),
            imgsz=imgsz,
            max_det=int(cfg.max_det),
            classes=list(classes) if classes is not None else None,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )[0]

        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections
        confs = boxes.conf.detach().cpu().numpy()
        classes_raw = boxes.cls.detach().cpu().numpy().astype(int)
        polys = result.masks.xy if result.masks is not None else None
        for index in range(len(confs)):
            canonical = self.names.get(int(classes_raw[index]))
            if canonical is None:
                continue
            if polys is not None and index < len(polys) and len(polys[index]) >= 3:
                points = np.asarray(polys[index], dtype=np.float64)
                points = points * float(inv_scale) + np.asarray(origin, dtype=np.float64)
            else:  # 没有掩膜时退回 bbox
                x1, y1, x2, y2 = boxes.xyxy[index].detach().cpu().numpy().astype(np.float64)
                points = np.asarray(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
                ) * float(inv_scale) + np.asarray(origin, dtype=np.float64)
            norm = np.clip(points / np.asarray([full_w, full_h], dtype=np.float64), 0.0, 1.0)
            structure = structure_from_polygon(norm, full_w, full_h, canonical, conf=float(confs[index]))
            if structure.area_px < float(cfg.min_area_px):
                continue
            detections.append(Detection(cls=canonical, conf=float(confs[index]), structure=structure))
        return detections

    # -- 引导式二次检测 -------------------------------------------------
    def raw_indices(self, canonical: int) -> list[int]:
        """规范类别号 -> 权重里的原始类别索引（可能多个，取决于权重类别顺序）。"""
        return [raw for raw, canon in self.names.items() if canon == int(canonical)]

    def guided_detect(
        self,
        image: np.ndarray,
        prediction: RelationPrediction,
        cfg: InferenceConfig,
    ) -> Detection | None:
        """按关系模型给出的 ROI 裁剪放大后再检测一次，用于找回漏检的目标结构。"""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = prediction.roi
        pad = 0.12 * max(x2 - x1, y2 - y1)
        box = clip_box((x1 - pad, y1 - pad, x2 + pad, y2 + pad), width, height)
        bx1, by1, bx2, by2 = box
        if bx2 - bx1 < 24 or by2 - by1 < 24:
            return None
        crop = image[by1:by2, bx1:bx2]
        upscale = min(float(cfg.guide_upscale), max(1.0, 640.0 / max(crop.shape[:2])))
        if upscale > 1.001:
            crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        guide_cfg = InferenceConfig(
            conf=float(cfg.guide_conf),
            iou=float(cfg.iou),
            max_det=8,
            imgsz=max(self.imgsz, 640),  # 小裁剪块要放大到 ≥640 才有足够分辨率
            guided=False,
            min_area_px=float(cfg.min_area_px) * 0.5,
        )
        raw_classes = self.raw_indices(prediction.target_cls)
        if not raw_classes:
            return None
        candidates = self.detect(
            crop,
            guide_cfg,
            classes=raw_classes,
            origin=(float(bx1), float(by1)),
            inv_scale=1.0 / upscale,
            full_size=(width, height),
        )
        if not candidates:
            return None

        # 只接受落在 ROI 内、且方向偏差不超过阈值的候选
        rx1, ry1, rx2, ry2 = prediction.roi
        anchor_center = (
            prediction.center_px[0] - prediction.mean_px[0],
            prediction.center_px[1] - prediction.mean_px[1],
        )
        best: tuple[float, Detection] | None = None
        for candidate in candidates:
            cx, cy = candidate.structure.center_px
            if not point_in_box((cx, cy), (rx1, ry1, rx2, ry2), pad=0.05 * max(rx2 - rx1, ry2 - ry1)):
                continue
            observed = math.degrees(math.atan2(cy - anchor_center[1], cx - anchor_center[0]))
            deviation = abs(wrap_angle_deg(observed - prediction.angle_deg))
            if deviation > float(cfg.guide_max_angle):
                continue
            score = float(candidate.conf) * (1.0 - deviation / 180.0)
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is None:
            return None
        found = best[1]
        found.source = "guided"
        return found


# ---------------------------------------------------------------------------
# 单帧完整分析
# ---------------------------------------------------------------------------
def _nearest(anchor: Detection, candidates: Sequence[Detection]) -> Detection:
    ax, ay = anchor.structure.center_px
    return min(
        candidates,
        key=lambda d: (d.structure.center_px[0] - ax) ** 2 + (d.structure.center_px[1] - ay) ** 2,
    )


def analyze_frame(
    segmenter: EchoSegmenter,
    predictor: RelationPredictor | None,
    image: np.ndarray,
    cfg: InferenceConfig,
) -> FrameResult:
    """检测 + 关系推理 + （可选）引导式找回漏检结构。"""
    height, width = image.shape[:2]
    result = FrameResult(width=width, height=height)
    result.detections = segmenter.detect(image, cfg)
    if not result.detections:
        result.missing = [0, 1]
        result.warnings.append("no detection")
        return result

    for anchor_cls, target_cls in DIRECTIONS:
        anchors = result.by_class(anchor_cls)
        if not anchors:
            continue
        anchors.sort(key=lambda d: d.structure.area_px, reverse=True)
        targets = result.by_class(target_cls)
        if predictor is None:
            if targets:
                for target in targets[: cfg.max_links]:
                    result.links.append(RelationLink(anchor=_nearest(target, anchors), prediction=None, target=target))
            continue

        if targets:
            for target in targets[: cfg.max_links]:
                anchor = _nearest(target, anchors)
                prediction = predictor.predict(anchor.structure)
                result.links.append(
                    RelationLink(
                        anchor=anchor,
                        prediction=prediction,
                        target=target,
                        consistency=predictor.consistency(anchor.structure, target.structure),
                    )
                )
            continue

        # 只找到锚结构：给出方向预测，并尝试用 ROI 找回目标
        anchor = anchors[0]
        prediction = predictor.predict(anchor.structure)
        link = RelationLink(anchor=anchor, prediction=prediction)
        if prediction is not None and prediction.outside_fov:
            # 落点在画面外时 ROI 会被裁到画面边缘，引导找回多半无从下手，提前说明
            result.warnings.append(
                f"{CLASS_NAMES[target_cls]} predicted outside FOV "
                f"(center=({prediction.center_px[0]:.0f}, {prediction.center_px[1]:.0f}))"
            )
        if prediction is not None and cfg.guided:
            found = segmenter.guided_detect(image, prediction, cfg)
            if found is not None:
                result.detections.append(found)
                link.target = found
                link.guided = True
                link.consistency = predictor.consistency(anchor.structure, found.structure)
                result.warnings.append(f"guided recovery: {CLASS_NAMES[target_cls]}")
        result.links.append(link)

    present = {d.cls for d in result.detections}
    result.missing = [cls for cls in (0, 1) if cls not in present]
    for cls in result.missing:
        result.warnings.append(f"{CLASS_NAMES[cls]} not detected")
    for link in result.links:
        if link.consistency and not link.consistency["plausible"]:
            result.warnings.append(
                f"implausible pair (mahalanobis={link.consistency['mahalanobis']})"
            )
    return result

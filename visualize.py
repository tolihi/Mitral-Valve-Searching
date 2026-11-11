"""visualize.py —— 把 FrameResult 画成可直接查看的标注图。

图中文字一律英文：cv2.putText 不支持中文字形，混用会显示成方块。
"""
from __future__ import annotations

import math

import numpy as np

from common import (
    COLORS_BGR,
    GUIDED_BGR,
    RELATION_BGR,
    TEXT_BGR,
    WARN_BGR,
    draw_arrow,
    draw_box,
    draw_cone,
    draw_contour,
    draw_polygon_mask,
    draw_text,
)
from inference import Detection, FrameResult, RelationLink


def _font_scale(height: int) -> float:
    return max(0.42, min(0.9, height / 1300.0))


def _draw_detection(image: np.ndarray, detection: Detection, scale: float, show_mask: bool) -> None:
    structure = detection.structure
    color = GUIDED_BGR if detection.source == "guided" else COLORS_BGR[detection.cls]
    if structure.points_px is not None and len(structure.points_px) >= 3:
        if show_mask:
            draw_polygon_mask(image, structure.points_px, color, alpha=0.28)
        draw_contour(image, structure.points_px, color, thickness=2)
    box = structure.box_px
    if detection.source == "guided":
        draw_box(image, box, color, thickness=2, dashed=True)
    cx, cy = structure.center_px
    label = f"{detection.name.split('_')[0]} {detection.conf:.2f}"
    if detection.source == "guided":
        label += " (guided)"
    draw_text(image, label, (box[0] + 2, box[1] - 4), color=color, scale=scale)


def _draw_link(image: np.ndarray, link: RelationLink, scale: float, show_cone: bool, show_roi: bool) -> None:
    anchor_center = link.anchor.structure.center_px
    prediction = link.prediction

    if link.target is not None:
        target_center = link.target.structure.center_px
        draw_arrow(image, anchor_center, target_center, color=RELATION_BGR, thickness=2)
        if prediction is not None:
            angle = math.degrees(
                math.atan2(target_center[1] - anchor_center[1], target_center[0] - anchor_center[0])
            )
            distance = math.hypot(target_center[0] - anchor_center[0], target_center[1] - anchor_center[1])
            text = f"{angle:.0f} deg / {distance:.0f} px"
            if link.consistency and not link.consistency["plausible"]:
                text += " !"
            mid = ((anchor_center[0] + target_center[0]) / 2.0, (anchor_center[1] + target_center[1]) / 2.0)
            draw_text(image, text, (mid[0] + 6, mid[1] - 6), color=RELATION_BGR, scale=scale * 0.92)
        return

    if prediction is None:
        return
    distance = max(float(prediction.distance_px), 1.0)
    if show_cone:
        draw_cone(
            image,
            anchor_center,
            prediction.angle_deg,
            prediction.cone_half_deg,
            radius=max(distance * 1.25, 0.18 * max(image.shape[:2])),
            color=RELATION_BGR,
            alpha=0.10,
        )
    if show_roi:
        draw_box(image, prediction.roi, color=RELATION_BGR, thickness=1, dashed=True)

    end = draw_arrow(image, anchor_center, prediction.center_px, color=RELATION_BGR, thickness=3)
    text = (
        f"{prediction.direction} {prediction.angle_deg:.0f} deg"
        f" +/-{prediction.cone_half_deg:.0f} deg"
    )
    draw_text(image, text, (end[0] + 8, end[1] - 10), color=RELATION_BGR, scale=scale)
    if prediction.outside_fov:
        draw_text(image, "target likely OUTSIDE FOV", (end[0] + 8, end[1] + 14), color=WARN_BGR, scale=scale * 0.9)
    if not math.isnan(prediction.presence_rate):
        draw_text(
            image,
            f"P(present)~{prediction.presence_rate:.2f}",
            (end[0] + 8, end[1] + 32),
            color=TEXT_BGR,
            scale=scale * 0.85,
        )


def render_result(
    image: np.ndarray,
    frame: FrameResult,
    show_masks: bool = True,
    show_cone: bool = True,
    show_roi: bool = True,
    show_panel: bool = True,
) -> np.ndarray:
    """返回画好标注的 BGR 图像（不修改入参）。"""
    canvas = image.copy()
    scale = _font_scale(canvas.shape[0])

    for detection in frame.detections:
        _draw_detection(canvas, detection, scale, show_masks)
    for link in frame.links:
        _draw_link(canvas, link, scale, show_cone, show_roi)

    if show_panel:
        y = int(24 * scale / 0.6)
        step = int(26 * scale / 0.6)
        for cls in (0, 1):
            items = frame.by_class(cls)
            color = COLORS_BGR[cls]
            draw_text(
                canvas,
                f"{'mitral' if cls == 0 else 'fossa'}: {len(items)}",
                (10, y),
                color=color,
                scale=scale,
            )
            y += step
        for warning in frame.warnings[:4]:
            draw_text(canvas, warning, (10, y), color=WARN_BGR, scale=scale * 0.92)
            y += step
        legend = "solid=detected  dashed=predicted  magenta=guided"
        draw_text(canvas, legend, (10, canvas.shape[0] - 12), color=TEXT_BGR, scale=scale * 0.85)
    return canvas


def render_side_by_side(image: np.ndarray, frame: FrameResult, **kwargs) -> np.ndarray:
    """原图 | 标注图 左右拼接，便于快速目检。"""
    annotated = render_result(image, frame, **kwargs)
    return np.hstack([image, annotated])

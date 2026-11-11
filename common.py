"""common.py —— 共享工具：类别定义、YOLO-seg 标注解析、结构几何、关系特征与绘图。

被 prepare_dataset.py / train_seg.py / relation.py / predict.py / evaluate.py 复用。
只依赖 numpy；绘图函数内部按需导入 cv2，因此数据整理脚本不需要 OpenCV。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 类别定义
# ---------------------------------------------------------------------------
# 统一使用这两个规范类名。标注导出里可能出现 "mitral valve" / "fossal" /
# "mitral and fossil" 等写法，由 canonical_class() 归一化。
CLASS_NAMES: tuple[str, ...] = ("mitral_valve", "fossa_ovalis")
CLASS_ZH: dict[str, str] = {"mitral_valve": "二尖瓣", "fossa_ovalis": "卵圆孔"}
MITRAL: int = 0
FOSSA: int = 1

# 可视化配色（BGR）
COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (0, 165, 255),   # 二尖瓣：橙色
    1: (80, 220, 80),   # 卵圆孔：绿色
}
RELATION_BGR = (0, 255, 255)   # 方向箭头 / 锥形搜索区：黄色
GUIDED_BGR = (255, 0, 255)     # 引导式二次检测结果：品红
WARN_BGR = (0, 0, 255)
TEXT_BGR = (240, 240, 240)

IMAGE_SUFFIXES: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# 帧文件名形如 "<检查号>_<17 位时间戳>.jpg"；用时间戳切分/排序。
FRAME_TS_RE = re.compile(r"^(?P<exam>.+?)_(?P<ts>\d{15,17})$")

# 结构特征的固定顺序，训练与推理必须一致。
FEATURE_NAMES: tuple[str, ...] = (
    "cx",            # 锚结构中心 x（0~1）
    "cy",            # 锚结构中心 y（0~1）
    "sqrt_area",     # sqrt(锚结构面积占比)，尺度代理
    "log_aspect",    # log(bbox 宽 / bbox 高)
    "cos2theta",     # 锚结构主轴方向（2θ 编码，消除 π 歧义）
    "sin2theta",
    "log_elong",     # 主轴伸长比对数
)


# ---------------------------------------------------------------------------
# 名称归一化
# ---------------------------------------------------------------------------
def canonical_class(name: object) -> str | None:
    """把数据集/模型里的类名映射为规范名。

    返回 "mitral_valve" / "fossa_ovalis" / "combined"（组合类，需要策略处理），
    无法识别时返回 None。
    """
    text = re.sub(r"[\s_\-]+", " ", str(name)).strip().lower()
    if not text:
        return None
    has_mitral = "mitral" in text
    has_fossa = ("foss" in text) or ("oval" in text) or ("sept" in text)
    if has_mitral and has_fossa:
        return "combined"
    if has_mitral:
        return "mitral_valve"
    if has_fossa:
        return "fossa_ovalis"
    return None


def class_index(name: str) -> int:
    """规范名 -> 类别下标（0 二尖瓣，1 卵圆孔）。"""
    if name not in CLASS_NAMES:
        raise ValueError(f"unknown class name: {name!r}")
    return CLASS_NAMES.index(name)


# ---------------------------------------------------------------------------
# 标注解析 / 序列化（YOLO segmentation：class x1 y1 x2 y2 ...，坐标归一化）
# ---------------------------------------------------------------------------
def parse_seg_line(line: str) -> tuple[int, np.ndarray] | None:
    """解析一行 YOLO-seg 标注，返回 (cls, Nx2 归一化坐标)；非法行返回 None。"""
    fields = line.split()
    if len(fields) < 7:          # 类别 + 至少 3 个点
        return None
    if (len(fields) - 1) % 2 != 0:
        return None
    try:
        cls = int(float(fields[0]))
        coords = np.asarray([float(v) for v in fields[1:]], dtype=np.float64).reshape(-1, 2)
    except ValueError:
        return None
    if not np.isfinite(coords).all():
        return None
    return cls, coords


def read_seg_label(path: Path) -> list[tuple[int, np.ndarray]]:
    """读取一个 YOLO-seg txt，返回 [(cls, Nx2), ...]（保留原始类别号）。"""
    instances: list[tuple[int, np.ndarray]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parsed = parse_seg_line(raw)
        if parsed is None:
            continue
        instances.append(parsed)
    return instances


def format_seg_line(cls: int, points: np.ndarray) -> str:
    """序列化一行标注；坐标裁剪到 [0,1] 并保留 6 位小数。"""
    pts = np.clip(np.asarray(points, dtype=np.float64), 0.0, 1.0).reshape(-1)
    body = " ".join(f"{v:.6f}" for v in pts)
    return f"{int(cls)} {body}"


# ---------------------------------------------------------------------------
# 多边形几何
# ---------------------------------------------------------------------------
def polygon_area(points: np.ndarray) -> float:
    """鞋带公式求绝对面积（输入坐标单位与输出面积单位一致）。"""
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_centroid(points: np.ndarray) -> tuple[float, float]:
    """多边形面积质心；退化时退回顶点均值。"""
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    signed = cross.sum() / 2.0
    if abs(signed) < 1e-12:
        return float(x.mean()), float(y.mean())
    cx = float(((x + np.roll(x, -1)) * cross).sum() / (6.0 * signed))
    cy = float(((y + np.roll(y, -1)) * cross).sum() / (6.0 * signed))
    return cx, cy


def polygon_bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    """返回 (x1, y1, x2, y2)。"""
    pts = np.asarray(points, dtype=np.float64)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


# ---------------------------------------------------------------------------
# 结构（一次检测 / 一条标注的几何摘要）
# ---------------------------------------------------------------------------
@dataclass
class Structure:
    """一个解剖结构的几何摘要（归一化到图像坐标系，x 右、y 下）。

    同时保存像素尺度下的 bbox，便于画图与 ROI 计算。
    """

    cls: int
    cx: float
    cy: float
    w: float            # bbox 宽（归一化）
    h: float            # bbox 高（归一化）
    area_frac: float    # 掩膜面积 / 图像面积
    theta: float        # 主轴方向（弧度，[0, π)）
    elong: float        # 主轴/次轴长度比（>=1）
    image_w: int
    image_h: int
    conf: float = 1.0
    mask: np.ndarray | None = None       # 原图分辨率的 bool 掩膜（可选，便于画图）
    points_px: np.ndarray | None = None  # 原图像素坐标多边形（可选）

    # -- 常用派生量 ---------------------------------------------------------
    @property
    def center_px(self) -> tuple[float, float]:
        return self.cx * self.image_w, self.cy * self.image_h

    @property
    def box_px(self) -> tuple[float, float, float, float]:
        half_w, half_h = self.w * self.image_w / 2.0, self.h * self.image_h / 2.0
        cx, cy = self.center_px
        return cx - half_w, cy - half_h, cx + half_w, cy + half_h

    @property
    def area_px(self) -> float:
        return self.area_frac * self.image_w * self.image_h

    def to_json(self) -> dict:
        return {
            "cls": self.cls,
            "name": CLASS_NAMES[self.cls],
            "conf": round(float(self.conf), 4),
            "center_norm": [round(self.cx, 5), round(self.cy, 5)],
            "box_norm": [
                round(self.cx - self.w / 2, 5), round(self.cy - self.h / 2, 5),
                round(self.cx + self.w / 2, 5), round(self.cy + self.h / 2, 5),
            ],
            "area_frac": round(float(self.area_frac), 6),
            "theta_deg": round(math.degrees(self.theta) % 180.0, 2),
            "elong": round(float(self.elong), 3),
        }


def mask_moments(mask: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """二值掩膜的 (cx, cy, mu20, mu02, mu11, elong)，坐标单位=像素。"""
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        h, w = mask.shape[:2]
        return w / 2.0, h / 2.0, 0.0, 0.0, 0.0, 1.0
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    cx, cy = x.mean(), y.mean()
    dx, dy = x - cx, y - cy
    mu20 = float((dx * dx).mean())
    mu02 = float((dy * dy).mean())
    mu11 = float((dx * dy).mean())
    trace = mu20 + mu02
    det = mu20 * mu02 - mu11 * mu11
    disc = math.sqrt(max(trace * trace / 4.0 - det, 0.0))
    l1 = trace / 2.0 + disc
    l2 = max(trace / 2.0 - disc, 1e-9)
    return cx, cy, mu20, mu02, mu11, math.sqrt(max(l1, 1e-9) / l2)


def theta_from_moments(mu20: float, mu02: float, mu11: float) -> float:
    """主轴方向（弧度，mod π）。"""
    return float(0.5 * math.atan2(2.0 * mu11, mu20 - mu02)) % math.pi


def structure_from_mask(mask: np.ndarray, cls: int, conf: float = 1.0) -> Structure:
    """由原图分辨率的 bool 掩膜构造 Structure。"""
    mask = np.asarray(mask, dtype=bool)
    image_h, image_w = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("empty mask")
    cx, cy, mu20, mu02, mu11, elong = mask_moments(mask)
    x1, x2 = float(xs.min()), float(xs.max()) + 1.0
    y1, y2 = float(ys.min()), float(ys.max()) + 1.0
    return Structure(
        cls=int(cls),
        cx=cx / image_w,
        cy=cy / image_h,
        w=max(x2 - x1, 1.0) / image_w,
        h=max(y2 - y1, 1.0) / image_h,
        area_frac=float(mask.mean()),
        theta=theta_from_moments(mu20, mu02, mu11),
        elong=elong,
        image_w=image_w,
        image_h=image_h,
        conf=float(conf),
        mask=mask,
    )


def rasterize_polygon(points_px: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
    """把像素坐标多边形栅格化成 bool 掩膜（内部导入 cv2）。"""
    import cv2

    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def structure_from_polygon(
    points_norm: np.ndarray,
    image_w: int,
    image_h: int,
    cls: int,
    conf: float = 1.0,
) -> Structure:
    """由归一化多边形构造 Structure（低分辨率栅格化以计算主轴/伸长比）。"""
    pts = np.clip(np.asarray(points_norm, dtype=np.float64), 0.0, 1.0)
    pts_px = np.column_stack([pts[:, 0] * image_w, pts[:, 1] * image_h])
    scale = min(1.0, 256.0 / max(image_w, image_h))
    work_w = max(8, int(round(image_w * scale)))
    work_h = max(8, int(round(image_h * scale)))
    mask = rasterize_polygon(pts_px * scale, work_h, work_w)
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        # 退化多边形：退回 bbox 几何
        x1, y1, x2, y2 = polygon_bbox(pts_px)
        w = max(x2 - x1, 1.0) / image_w
        h = max(y2 - y1, 1.0) / image_h
        cx, cy = polygon_centroid(pts_px)
        return Structure(
            cls=int(cls), cx=cx / image_w, cy=cy / image_h, w=w, h=h,
            area_frac=max(polygon_area(pts_px), 1.0) / (image_w * image_h),
            theta=0.0, elong=1.0, image_w=image_w, image_h=image_h,
            conf=float(conf), points_px=pts_px,
        )
    cx, cy, mu20, mu02, mu11, elong = mask_moments(mask)
    x1, x2 = float(xs.min()) / work_w, (float(xs.max()) + 1.0) / work_w
    y1, y2 = float(ys.min()) / work_h, (float(ys.max()) + 1.0) / work_h
    return Structure(
        cls=int(cls),
        cx=cx / work_w,
        cy=cy / work_h,
        w=max(x2 - x1, 1e-3),
        h=max(y2 - y1, 1e-3),
        area_frac=float(mask.mean()),
        theta=theta_from_moments(mu20, mu02, mu11),
        elong=elong,
        image_w=image_w,
        image_h=image_h,
        conf=float(conf),
        points_px=pts_px,
    )


# ---------------------------------------------------------------------------
# 关系特征 / 偏移量
# ---------------------------------------------------------------------------
def scale_px(structure: Structure) -> float:
    """锚结构的尺度代理（像素）：sqrt(掩膜面积)。偏移量用它归一化。"""
    return math.sqrt(max(structure.area_px, 1.0))


def structure_features(structure: Structure) -> np.ndarray:
    """7 维锚结构特征，顺序与 FEATURE_NAMES 一致。"""
    aspect = math.log(max(structure.w, 1e-6) / max(structure.h, 1e-6))
    return np.asarray(
        [
            structure.cx,
            structure.cy,
            math.sqrt(max(structure.area_frac, 1e-8)),
            aspect,
            math.cos(2.0 * structure.theta),
            math.sin(2.0 * structure.theta),
            math.log(max(structure.elong, 1.0)),
        ],
        dtype=np.float32,
    )


def offset_units(anchor: Structure, target: Structure) -> np.ndarray:
    """target 相对 anchor 的中心偏移，除以锚结构尺度（无量纲）。"""
    dx = (target.cx - anchor.cx) * anchor.image_w
    dy = (target.cy - anchor.cy) * anchor.image_h
    return np.asarray([dx, dy], dtype=np.float64) / scale_px(anchor)


AUGMENT_MODES: tuple[str, ...] = ("identity", "fliplr", "flipud", "rot180")


def transform_structure(structure: Structure, mode: str) -> Structure:
    """几何增广：只变换几何量（图像坐标 x 右、y 下）。

    fliplr/flipud/rot180 对任意宽高比都成立，因此不会引入错误的图像几何。
    """
    if mode == "identity":
        return replace(structure)
    cx, cy, theta = structure.cx, structure.cy, structure.theta
    if mode == "fliplr":
        cx, theta = 1.0 - cx, (math.pi - theta) % math.pi
    elif mode == "flipud":
        cy, theta = 1.0 - cy, (-theta) % math.pi
    elif mode == "rot180":
        cx, cy = 1.0 - cx, 1.0 - cy
        theta = theta % math.pi
    else:
        raise ValueError(f"unknown augment mode: {mode}")
    return replace(structure, cx=cx, cy=cy, theta=theta)


def transform_offset(dx: float, dy: float, mode: str) -> tuple[float, float]:
    """与 transform_structure 配套的偏移量变换。"""
    if mode == "identity":
        return dx, dy
    if mode == "fliplr":
        return -dx, dy
    if mode == "flipud":
        return dx, -dy
    if mode == "rot180":
        return -dx, -dy
    raise ValueError(f"unknown augment mode: {mode}")


# ---------------------------------------------------------------------------
# 方向 / 角度
# ---------------------------------------------------------------------------
def wrap_angle_deg(angle: float) -> float:
    """把角度折到 (-180, 180]。"""
    a = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


_DIRECTION_SECTORS_EN = ("right", "lower-right", "down", "lower-left",
                         "left", "upper-left", "up", "upper-right")
_DIRECTION_SECTORS_ZH = ("右", "右下", "下", "左下", "左", "左上", "上", "右上")


def _sector_index(angle_deg: float) -> int:
    """把角度映射到 8 个扇区（每 45° 一格，格心对齐 0/45/90…）。"""
    a = wrap_angle_deg(angle_deg) % 360.0
    return int(math.floor((a + 22.5) / 45.0)) % 8


def direction_label(angle_deg: float) -> str:
    """图像坐标系（x 右、y 下）下的八方向英文标签。"""
    return _DIRECTION_SECTORS_EN[_sector_index(angle_deg)]


def direction_label_zh(angle_deg: float) -> str:
    """图像坐标系下的八方向中文标签。"""
    return _DIRECTION_SECTORS_ZH[_sector_index(angle_deg)]


def circular_mean_std_deg(angles_deg: Sequence[float]) -> tuple[float, float]:
    """角度（度）的圆周均值与圆周标准差。"""
    if len(angles_deg) == 0:
        return float("nan"), float("nan")
    a = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    mean = math.atan2(float(np.sin(a).mean()), float(np.cos(a).mean()))
    r = float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))
    std = math.degrees(math.sqrt(max(-2.0 * math.log(max(r, 1e-9)), 0.0)))
    return math.degrees(mean), std


# ---------------------------------------------------------------------------
# 框 / 视频 / JSON 小工具
# ---------------------------------------------------------------------------
def clip_box(box: Sequence[float], image_w: int, image_h: int, pad: float = 0.0) -> tuple[int, int, int, int]:
    """裁剪到图像范围，返回整数 (x1, y1, x2, y2)，保证 x2>x1、y2>y1。"""
    x1, y1, x2, y2 = [float(v) for v in box]
    if pad:
        x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    x1 = int(max(0, min(round(x1), image_w - 2)))
    y1 = int(max(0, min(round(y1), image_h - 2)))
    x2 = int(max(x1 + 1, min(round(x2), image_w - 1)))
    y2 = int(max(y1 + 1, min(round(y2), image_h - 1)))
    return x1, y1, x2, y2


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max((ax2 - ax1) * (ay2 - ay1), 0.0) + max((bx2 - bx1) * (by2 - by1), 0.0) - inter
    return float(inter / union) if union > 0 else 0.0


def point_in_box(point: Sequence[float], box: Sequence[float], pad: float = 0.0) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def image_files(directory: Path) -> list[Path]:
    """目录内按名称排序的图像文件。"""
    return sorted(
        (p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )


def exam_key(stem: str) -> tuple[str, int]:
    """从帧文件名提取（检查/视频组, 时间戳）。用于按检查分组划分，避免同检查帧泄漏。"""
    m = FRAME_TS_RE.match(stem)
    if m:
        return m.group("exam"), int(m.group("ts"))
    return stem, 0


def load_json(path: Path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value):
    """把 numpy 标量 / NaN / inf 转成标准 JSON 能表达的值（NaN -> null）。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text, encoding="utf-8")


def resolve_data_yaml(data: Path) -> Path:
    """--data 既可以是数据集目录（内含 data.yaml），也可以直接是 yaml 文件。"""
    data = Path(data)
    if data.is_dir():
        candidate = data / "data.yaml"
        if not candidate.is_file():
            raise SystemExit(f"找不到 data.yaml: {candidate}")
        return candidate
    if data.is_file():
        return data
    raise SystemExit(f"数据集路径不存在: {data}")


# ---------------------------------------------------------------------------
# 绘图（内部导入 cv2）
# ---------------------------------------------------------------------------
def _cv2():
    import cv2

    return cv2


def imread_unicode(path: Path) -> np.ndarray | None:
    """读取图像，支持中文/空格路径（Windows 上 cv2.imread 不支持非 ASCII 路径）。"""
    cv2 = _cv2()
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """写出图像，支持中文/空格路径。"""
    cv2 = _cv2()
    path = Path(path)
    ok, buffer = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        return False
    buffer.tofile(str(path))
    return True


def image_size(path: Path) -> tuple[int, int]:
    """返回 (width, height)，不把整张图读进内存。"""
    from PIL import Image

    with Image.open(path) as handle:
        return int(handle.width), int(handle.height)


def draw_polygon_mask(
    image: np.ndarray,
    points_px: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.35,
) -> None:
    """半透明填充多边形（就地修改）。"""
    cv2 = _cv2()
    pts = np.round(np.asarray(points_px, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def draw_contour(image: np.ndarray, points_px: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    cv2 = _cv2()
    pts = np.round(np.asarray(points_px, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_box(
    image: np.ndarray,
    box: Sequence[float],
    color: tuple[int, int, int],
    thickness: int = 2,
    dashed: bool = False,
) -> None:
    cv2 = _cv2()
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    if not dashed:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        return
    dash, gap = 10, 6
    for x in range(x1, x2, dash + gap):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, dash + gap):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness, cv2.LINE_AA)


def draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = TEXT_BGR,
    scale: float = 0.6,
    thickness: int = 1,
    bg: bool = True,
) -> None:
    cv2 = _cv2()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = int(org[0]), int(org[1])
    h, w = image.shape[:2]
    x = max(0, min(x, w - tw - 4))
    y = max(th + 4, min(y, h - 2))
    if bg:
        cv2.rectangle(image, (x - 3, y - th - 4), (x + tw + 3, y + base + 2), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_arrow(
    image: np.ndarray,
    start: Sequence[float],
    end: Sequence[float],
    color: tuple[int, int, int] = RELATION_BGR,
    thickness: int = 3,
) -> tuple[int, int]:
    """画箭头，返回实际终点（若目标在画面外，会截到图像边缘）。"""
    cv2 = _cv2()
    h, w = image.shape[:2]
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    # Liang-Barsky 裁剪到画面内
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, w - 1 - x0), (-dy, y0), (dy, h - 1 - y0)):
        if abs(p) < 1e-9:
            if q < 0:
                return int(round(x0)), int(round(y0))
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t0 > t1:
        return int(round(x0)), int(round(y0))
    ex, ey = x0 + dx * t1, y0 + dy * t1
    cv2.arrowedLine(image, (int(round(x0)), int(round(y0))), (int(round(ex)), int(round(ey))),
                    color, thickness, cv2.LINE_AA, tipLength=0.12)
    return int(round(ex)), int(round(ey))


def draw_cone(
    image: np.ndarray,
    apex: Sequence[float],
    angle_deg: float,
    half_angle_deg: float,
    radius: float,
    color: tuple[int, int, int] = RELATION_BGR,
    alpha: float = 0.12,
) -> None:
    """以 apex 为顶点、沿 angle_deg 方向张角 ±half_angle_deg 的扇形搜索区。"""
    cv2 = _cv2()
    h, w = image.shape[:2]
    half = max(2.0, min(float(half_angle_deg), 85.0))
    radius = max(10.0, float(radius))
    ax, ay = float(apex[0]), float(apex[1])
    # 用足够多的采样点逼近扇形（半径按画面对角线限制）
    radius = min(radius, math.hypot(w, h))
    angles = np.linspace(math.radians(angle_deg - half), math.radians(angle_deg + half), 40)
    pts = [(ax, ay)] + [(ax + radius * math.cos(a), ay + radius * math.sin(a)) for a in angles]
    pts_arr = np.round(np.asarray(pts, dtype=np.float32)).astype(np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts_arr], color)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)
    cv2.polylines(image, [pts_arr], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

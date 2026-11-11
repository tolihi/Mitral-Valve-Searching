"""relation.py —— 二尖瓣 ↔ 卵圆孔的相对位置模型（方向 + 距离 + 不确定度）。

核心思想
--------
两个结构的相对位置由解剖/探头几何决定，变化远小于它们在画面中的绝对位置。因此把
"目标相对锚点的偏移"作为回归目标，而不是让分割网络隐式学习：

    u = (c_target - c_anchor) / sqrt(anchor_mask_area_px)        # 无量纲、近似尺度无关

预测时由锚结构的几何（中心、尺度、长宽比、主轴方向）条件化地回归 u 的分布：

    p(u | anchor) = N(u; mu(anchor), Sigma(anchor))     # 线性（岭回归）或 MDN 混合高斯

于是即使画面里只检测到二尖瓣，也能给出：
  * 期望方向（箭头）      angle = atan2(mu_y, mu_x)
  * 方向不确定度（锥角）  cone = atan(sigma_perp / |mu|)
  * 目标落点与搜索 ROI     center = c_anchor + mu * sqrt(area)，ROI 由 Sigma 张开
  * 目标存在的先验概率     P(target | anchor)（由标注统计得到）

为什么用线性/岭回归而不是大网络：成对样本通常只有几十个（本数据集 24 对），
线性 + 残差协方差已经能给出校准的不确定度；样本变多后可用 --families 切到 MDN。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from common import (
    CLASS_NAMES,
    FEATURE_NAMES,
    Structure,
    _json_safe,
    direction_label,
    direction_label_zh,
    image_files,
    image_size,
    offset_units,
    read_seg_label,
    scale_px,
    structure_features,
    structure_from_polygon,
    transform_offset,
    transform_structure,
    wrap_angle_deg,
)

# 方向键：锚类 -> 目标类
DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0))


def direction_key(anchor_cls: int, target_cls: int) -> str:
    return f"{CLASS_NAMES[anchor_cls]}->{CLASS_NAMES[target_cls]}"


# ---------------------------------------------------------------------------
# 成对几何样本
# ---------------------------------------------------------------------------
@dataclass
class PairSample:
    """一张图里"锚结构 + 目标结构"的一对样本。"""

    stem: str
    anchor: Structure
    target: Structure
    anchor_cls: int
    target_cls: int
    features: np.ndarray = field(default=None)   # type: ignore[assignment]
    offset_u: np.ndarray = field(default=None)   # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.features is None:
            self.features = structure_features(self.anchor)
        if self.offset_u is None:
            self.offset_u = offset_units(self.anchor, self.target)


@dataclass
class PresenceStats:
    """锚类出现时，目标类也出现的经验概率（仅统计有标注的帧）。"""

    anchor_cls: int
    target_cls: int
    n_anchor_images: int = 0
    n_both_images: int = 0

    @property
    def rate(self) -> float:
        return float(self.n_both_images / self.n_anchor_images) if self.n_anchor_images else float("nan")


def _labels_dir_for(images_dir: Path) -> Path:
    """images/<split> -> labels/<split>（prepare_dataset.py 的固定布局）。"""
    return images_dir.parent.parent / "labels" / images_dir.name


def _image_size(path: Path) -> tuple[int, int]:
    """保留薄封装，便于测试替换（实际实现见 common.image_size）。"""
    return image_size(path)


def load_pairs(data: Path, splits: Sequence[str] = ("train", "val")) -> tuple[list[PairSample], dict[str, PresenceStats], dict]:
    """从准备好的数据集里提取所有成对样本。

    多实例处理：一张图里可能有 2~3 个二尖瓣标注实例，对每个目标实例取"质心最近"的
    锚实例配对。预测时则取置信度最高（或面积最大）的锚实例。
    """
    from common import resolve_data_yaml

    data_yaml = resolve_data_yaml(data)
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(spec.get("path", data_yaml.parent))

    samples: list[PairSample] = []
    presence = {direction_key(a, t): PresenceStats(a, t) for a, t in DIRECTIONS}
    stats = {"images": 0, "images_with_labels": 0, "image_sizes": []}

    for split in splits:
        images_dir = root / str(spec.get(split, f"images/{split}"))
        if not images_dir.is_dir():
            continue
        labels_dir = _labels_dir_for(images_dir)
        for image_path in image_files(images_dir):
            stats["images"] += 1
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            instances = read_seg_label(label_path)
            if not instances:
                continue
            stats["images_with_labels"] += 1
            width, height = _image_size(image_path)
            stats["image_sizes"].append([width, height])
            by_class: dict[int, list[Structure]] = {0: [], 1: []}
            for cls, points in instances:
                if cls not in (0, 1):
                    continue
                by_class[cls].append(
                    structure_from_polygon(points, width, height, cls, conf=1.0)
                )
            present = {cls for cls, items in by_class.items() if items}
            for anchor_cls, target_cls in DIRECTIONS:
                stat = presence[direction_key(anchor_cls, target_cls)]
                if anchor_cls in present:
                    stat.n_anchor_images += 1
                    if target_cls in present:
                        stat.n_both_images += 1
                for target in by_class[target_cls]:
                    anchors = by_class[anchor_cls]
                    if not anchors:
                        continue
                    anchor = min(
                        anchors,
                        key=lambda s: (s.cx - target.cx) ** 2 + (s.cy - target.cy) ** 2,
                    )
                    samples.append(
                        PairSample(
                            stem=image_path.stem,
                            anchor=anchor,
                            target=target,
                            anchor_cls=anchor_cls,
                            target_cls=target_cls,
                        )
                    )
    return samples, presence, stats


def build_arrays(
    samples: Sequence[PairSample],
    modes: Sequence[str] = ("identity",),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把成对样本展开成 (X, Y, groups)。

    默认只用 identity：本数据集是同一台机器的固定切面，"二尖瓣在右、卵圆孔在左"是探头
    方向决定的解剖事实。左右/上下翻转出来的样本在解剖上并不存在，且会让常数先验的均值
    互相抵消（左右镜像后 x 偏移反号），因此默认关闭；`--augment` 打开后模型会学到
    "锚点位置决定方向"的镜像对称性，只有在你的探头方向可能反转时才建议开启。
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[str] = []
    for sample in samples:
        for mode in modes:
            anchor = transform_structure(sample.anchor, mode)
            ux, uy = transform_offset(float(sample.offset_u[0]), float(sample.offset_u[1]), mode)
            xs.append(structure_features(anchor))
            ys.append(np.asarray([ux, uy], dtype=np.float64))
            groups.append(sample.stem)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(groups)


# ---------------------------------------------------------------------------
# 线性模型：先验 / 岭回归
# ---------------------------------------------------------------------------
@dataclass
class LinearRelation:
    """条件高斯模型 p(u|x) = N(A x + b, Sigma(x))。

    family="prior" 时忽略特征，等价于常数先验；family="ridge" 时是标准化特征上的岭回归。
    """

    family: str
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    W: np.ndarray              # (n_feat+1, 2)，最后一行为偏置
    resid_cov: np.ndarray      # (2,2) 标准化目标空间
    xtx_inv: np.ndarray        # (n_feat+1, n_feat+1)，标准化空间
    lam: float
    n_samples: int

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回标准化目标空间的 (均值, 协方差)。"""
        if self.family == "prior":
            # 先验模型没有条件化：标准化空间里均值恒为 0，方差即残差协方差
            return np.zeros(2), self.resid_cov.copy()
        xz = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std
        design = np.concatenate([xz, [1.0]])
        mean = design @ self.W
        leverage = float(design @ self.xtx_inv @ design)
        return mean, self.resid_cov * max(1.0 + leverage, 1.0)

    def to_json(self) -> dict:
        return {
            "family": self.family,
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_std": self.y_std.tolist(),
            "W": self.W.tolist(),
            "resid_cov": self.resid_cov.tolist(),
            "xtx_inv": self.xtx_inv.tolist(),
            "lam": self.lam,
            "n_samples": int(self.n_samples),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "LinearRelation":
        return cls(
            family=payload["family"],
            x_mean=np.asarray(payload["x_mean"], dtype=np.float64),
            x_std=np.asarray(payload["x_std"], dtype=np.float64),
            y_mean=np.asarray(payload["y_mean"], dtype=np.float64),
            y_std=np.asarray(payload["y_std"], dtype=np.float64),
            W=np.asarray(payload["W"], dtype=np.float64),
            resid_cov=np.asarray(payload["resid_cov"], dtype=np.float64),
            xtx_inv=np.asarray(payload["xtx_inv"], dtype=np.float64),
            lam=float(payload["lam"]),
            n_samples=int(payload["n_samples"]),
        )


def _ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> LinearRelation:
    n_feat = X.shape[1]
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0)
    y_std[y_std < 1e-8] = 1.0
    Xz = (X - x_mean) / x_std
    Yz = (Y - y_mean) / y_std
    design = np.concatenate([Xz, np.ones((Xz.shape[0], 1))], axis=1)
    reg = np.eye(n_feat + 1) * lam
    reg[-1, -1] = 0.0                      # 偏置不正则
    xtx = design.T @ design + reg
    xtx_inv = np.linalg.inv(xtx)
    W = xtx_inv @ design.T @ Yz
    resid = Yz - design @ W
    dof = max(1, design.shape[0] - (n_feat + 1))
    resid_cov = (resid.T @ resid) / dof
    resid_cov += np.eye(2) * 1e-6
    return LinearRelation(
        family="ridge", x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        W=W, resid_cov=resid_cov, xtx_inv=xtx_inv, lam=float(lam), n_samples=int(X.shape[0]),
    )


def _prior_fit(X: np.ndarray, Y: np.ndarray) -> LinearRelation:
    y_mean = Y.mean(axis=0)
    resid = Y - y_mean
    dof = max(1, Y.shape[0] - 1)
    resid_cov = (resid.T @ resid) / dof + np.eye(2) * 1e-6
    n_feat = X.shape[1]
    return LinearRelation(
        family="prior",
        x_mean=np.zeros(n_feat), x_std=np.ones(n_feat), y_mean=y_mean, y_std=np.ones(2),
        W=np.zeros((n_feat + 1, 2)), resid_cov=resid_cov,
        xtx_inv=np.zeros((n_feat + 1, n_feat + 1)), lam=0.0, n_samples=int(X.shape[0]),
    )


# ---------------------------------------------------------------------------
# 混合密度网络（样本量足够时的可选非线性模型）
# ---------------------------------------------------------------------------
class _MDNModule:
    """延迟导入 torch 的 MDN 包装（未安装 torch 时自动跳过）。"""

    @staticmethod
    def build(in_dim: int, hidden: int, components: int, seed: int = 0):
        import torch
        from torch import nn

        torch.manual_seed(seed)

        class MDN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden), nn.SiLU(),
                )
                self.pi = nn.Linear(hidden, components)
                self.mu = nn.Linear(hidden, components * 2)
                self.log_sigma = nn.Linear(hidden, components * 2)
                self.components = components

            def forward(self, x):
                h = self.trunk(x)
                log_pi = nn.functional.log_softmax(self.pi(h), dim=-1)
                mu = self.mu(h).view(-1, self.components, 2)
                log_sigma = self.log_sigma(h).view(-1, self.components, 2).clamp(-5.0, 2.0)
                return log_pi, mu, log_sigma

        return MDN()


@dataclass
class MDNRelation:
    """标准化特征/目标空间上的 MDN；预测时输出混合高斯的均值与协方差。"""

    state_dict: dict
    in_dim: int
    hidden: int
    components: int
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    n_samples: int

    def _module(self):
        model = _MDNModule.build(self.in_dim, self.hidden, self.components)
        model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        model = self._module()
        xz = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std
        with torch.no_grad():
            log_pi, mu, log_sigma = model(torch.as_tensor(xz, dtype=torch.float32).view(1, -1))
        weights = log_pi.exp()[0].numpy()
        mu = mu[0].numpy()
        sigma = np.exp(log_sigma[0].numpy())
        mean = (weights[:, None] * mu).sum(axis=0)
        var = (weights[:, None] * (sigma ** 2 + mu ** 2)).sum(axis=0) - mean ** 2
        return mean, np.diag(np.maximum(var, 1e-6))

    def to_meta(self) -> dict:
        return {
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "components": self.components,
            "n_samples": int(self.n_samples),
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_std": self.y_std.tolist(),
        }


def _train_mdn(
    X: np.ndarray,
    Y: np.ndarray,
    hidden: int = 16,
    components: int = 2,
    epochs: int = 400,
    lr: float = 0.01,
    weight_decay: float = 1e-3,
    patience: int = 60,
    seed: int = 0,
) -> MDNRelation:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0)
    y_std[y_std < 1e-8] = 1.0
    xt = torch.as_tensor((X - x_mean) / x_std, dtype=torch.float32)
    yt = torch.as_tensor((Y - y_mean) / y_std, dtype=torch.float32)

    model = _MDNModule.build(X.shape[1], hidden, components, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        log_pi, mu, log_sigma = model(xt)
        sigma = log_sigma.exp()
        log_prob = -0.5 * (((yt[:, None, :] - mu) / sigma) ** 2 + 2.0 * log_sigma
                           + math.log(2.0 * math.pi))
        loss = -torch.logsumexp(log_pi + log_prob.sum(dim=-1), dim=-1).mean()
        loss.backward()
        optimizer.step()
        value = float(loss.item())
        if value < best_loss - 1e-6:
            best_loss, bad = value, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return MDNRelation(
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        in_dim=int(X.shape[1]), hidden=int(hidden), components=int(components),
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, n_samples=int(X.shape[0]),
    )


# ---------------------------------------------------------------------------
# 预测结果
# ---------------------------------------------------------------------------
@dataclass
class RelationPrediction:
    """锚结构 -> 目标结构的完整预测（像素单位 + 方向语义）。"""

    anchor_cls: int
    target_cls: int
    family: str
    mean_u: np.ndarray
    cov_u: np.ndarray
    scale: float
    mean_px: np.ndarray
    cov_px: np.ndarray
    center_px: np.ndarray
    angle_deg: float
    distance_px: float
    sigma_long_px: float
    sigma_perp_px: float
    cone_half_deg: float
    roi: tuple[float, float, float, float]
    presence_rate: float
    outside_fov: bool = False

    @property
    def direction(self) -> str:
        return direction_label(self.angle_deg)

    @property
    def direction_zh(self) -> str:
        return direction_label_zh(self.angle_deg)

    def to_json(self) -> dict:
        return {
            "anchor": CLASS_NAMES[self.anchor_cls],
            "target": CLASS_NAMES[self.target_cls],
            "family": self.family,
            "angle_deg": round(wrap_angle_deg(self.angle_deg), 2),
            "direction": self.direction,
            "direction_zh": self.direction_zh,
            "distance_px": round(float(self.distance_px), 1),
            "sigma_long_px": round(float(self.sigma_long_px), 1),
            "sigma_perp_px": round(float(self.sigma_perp_px), 1),
            "cone_half_deg": round(float(self.cone_half_deg), 1),
            "predicted_center_px": [round(float(v), 1) for v in self.center_px],
            "offset_px": [round(float(v), 1) for v in self.mean_px],
            "offset_units": [round(float(v), 3) for v in self.mean_u],
            "roi_px": [round(float(v), 1) for v in self.roi],
            "target_presence_rate": round(float(self.presence_rate), 3),
            "outside_fov": bool(self.outside_fov),
        }


def roi_from_anchor(
    anchor: Structure,
    mean_px: np.ndarray,
    cov_px: np.ndarray,
    roi_scale: float = 1.0,
) -> tuple[float, float, float, float]:
    """由预测落点与不确定度给出搜索 ROI（像素，已裁剪到画面内）。

    预测落点可能整块落在画面外（`outside_fov=True`），此时裁剪后会把 ROI 压到最近的
    画面边缘上，并保证 `x1 < x2`、`y1 < y2`（至少 1 像素宽），避免调用方拿到反向矩形。
    """
    cx = anchor.cx * anchor.image_w + float(mean_px[0])
    cy = anchor.cy * anchor.image_h + float(mean_px[1])
    sx = math.sqrt(max(float(cov_px[0, 0]), 1.0))
    sy = math.sqrt(max(float(cov_px[1, 1]), 1.0))
    anchor_w = anchor.w * anchor.image_w
    anchor_h = anchor.h * anchor.image_h
    half_w = roi_scale * (0.45 * anchor_w + 2.0 * sx + 0.02 * anchor.image_w)
    half_h = roi_scale * (0.45 * anchor_h + 2.0 * sy + 0.02 * anchor.image_h)
    x_max = float(max(anchor.image_w - 1, 1))
    y_max = float(max(anchor.image_h - 1, 1))
    x1 = max(0.0, min(cx - half_w, x_max - 1.0))
    x2 = min(x_max, max(cx + half_w, x1 + 1.0))
    y1 = max(0.0, min(cy - half_h, y_max - 1.0))
    y2 = min(y_max, max(cy + half_h, y1 + 1.0))
    return x1, y1, x2, y2


def make_prediction(
    anchor: Structure,
    mean_u: np.ndarray,
    cov_u: np.ndarray,
    family: str,
    target_cls: int,
    presence_rate: float = float("nan"),
    cone_k: float = 1.0,
    roi_scale: float = 1.0,
) -> RelationPrediction:
    s = scale_px(anchor)
    mean_px = np.asarray(mean_u, dtype=np.float64) * s
    cov_px = np.asarray(cov_u, dtype=np.float64) * s * s
    center = np.asarray([anchor.cx * anchor.image_w, anchor.cy * anchor.image_h]) + mean_px
    distance = float(np.hypot(*mean_px))
    angle = math.degrees(math.atan2(float(mean_px[1]), float(mean_px[0])))
    if distance > 1e-6:
        unit = mean_px / distance
        perp = np.asarray([-unit[1], unit[0]])
        sigma_long = math.sqrt(max(float(unit @ cov_px @ unit), 0.0))
        sigma_perp = math.sqrt(max(float(perp @ cov_px @ perp), 0.0))
    else:
        sigma_long = sigma_perp = math.sqrt(max(float(np.trace(cov_px)), 0.0))
    cone = cone_k * math.degrees(math.atan2(sigma_perp, max(distance, 1e-6)))
    outside = not (0.0 <= center[0] < anchor.image_w and 0.0 <= center[1] < anchor.image_h)
    return RelationPrediction(
        anchor_cls=anchor.cls,
        target_cls=target_cls,
        family=family,
        mean_u=np.asarray(mean_u, dtype=np.float64),
        cov_u=np.asarray(cov_u, dtype=np.float64),
        scale=s,
        mean_px=mean_px,
        cov_px=cov_px,
        center_px=center,
        angle_deg=angle,
        distance_px=distance,
        sigma_long_px=sigma_long,
        sigma_perp_px=sigma_perp,
        cone_half_deg=max(3.0, min(cone, 89.0)),
        roi=roi_from_anchor(anchor, mean_px, cov_px, roi_scale=roi_scale),
        presence_rate=float(presence_rate),
        outside_fov=bool(outside),
    )


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def gaussian_nll(true_u: np.ndarray, mean_u: np.ndarray, cov_u: np.ndarray) -> float:
    cov = np.asarray(cov_u, dtype=np.float64)
    cov = cov + np.eye(2) * 1e-6
    diff = np.asarray(true_u, dtype=np.float64) - np.asarray(mean_u, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("inf")
    return float(0.5 * (diff @ np.linalg.solve(cov, diff) + logdet + 2.0 * math.log(2.0 * math.pi)))


def evaluate_predictions(
    samples: Sequence[PairSample],
    means_u: Sequence[np.ndarray],
    covs_u: Sequence[np.ndarray],
    family: str = "",
    cone_k: float = 1.0,
    roi_scale: float = 1.0,
) -> dict:
    """在（未增广的）成对样本上计算方向/距离/ROI/似然指标。"""
    if not samples:
        return {"n": 0}
    angular: list[float] = []
    radial: list[float] = []
    center_err: list[float] = []
    nlls: list[float] = []
    cone_hits = 0
    roi_hits = 0
    for sample, mean_u, cov_u in zip(samples, means_u, covs_u):
        prediction = make_prediction(
            sample.anchor, mean_u, cov_u, family, sample.target_cls,
            presence_rate=float("nan"), cone_k=cone_k, roi_scale=roi_scale,
        )
        true_px = sample.offset_u * prediction.scale
        pred_angle = math.degrees(math.atan2(float(prediction.mean_px[1]), float(prediction.mean_px[0])))
        true_angle = math.degrees(math.atan2(float(true_px[1]), float(true_px[0])))
        error = abs(wrap_angle_deg(pred_angle - true_angle))
        angular.append(error)
        radial.append(abs(float(np.hypot(*prediction.mean_px)) - float(np.hypot(*true_px))))
        center_err.append(float(np.hypot(*(prediction.center_px - (sample.anchor.center_px[0] + true_px[0],
                                                                   sample.anchor.center_px[1] + true_px[1])))))
        nlls.append(gaussian_nll(sample.offset_u, mean_u, cov_u))
        if error <= prediction.cone_half_deg:
            cone_hits += 1
        gt_center = (
            sample.anchor.cx * sample.anchor.image_w + true_px[0],
            sample.anchor.cy * sample.anchor.image_h + true_px[1],
        )
        x1, y1, x2, y2 = prediction.roi
        if x1 <= gt_center[0] <= x2 and y1 <= gt_center[1] <= y2:
            roi_hits += 1
    angular_arr = np.asarray(angular)
    return {
        "n": len(samples),
        "angle_mae_deg": round(float(angular_arr.mean()), 2),
        "angle_median_deg": round(float(np.median(angular_arr)), 2),
        "angle_p90_deg": round(float(np.percentile(angular_arr, 90)), 2),
        "within_15_deg": round(float((angular_arr <= 15).mean()), 3),
        "within_30_deg": round(float((angular_arr <= 30).mean()), 3),
        "within_45_deg": round(float((angular_arr <= 45).mean()), 3),
        "radial_mae_px": round(float(np.mean(radial)), 1),
        "center_mae_px": round(float(np.mean(center_err)), 1),
        "nll": round(float(np.mean(nlls)), 3),
        "cone_hit_rate": round(cone_hits / len(samples), 3),
        "roi_hit_rate": round(roi_hits / len(samples), 3),
    }


def _fit_family(
    family: str,
    X: np.ndarray,
    Y: np.ndarray,
    lam: float | None = None,
    mdn_kwargs: dict | None = None,
):
    if family == "prior":
        return _prior_fit(X, Y)
    if family == "ridge":
        return _ridge_fit(X, Y, lam if lam is not None else 1.0)
    if family == "mdn":
        return _train_mdn(X, Y, **(mdn_kwargs or {}))
    raise ValueError(f"unknown family: {family}")


def _predict_family(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把模型输出的标准化空间 (mean, cov) 换回 u 的单位。

    LinearRelation.predict 与 MDNRelation.predict 都返回**标准化目标空间**的量，
    因此这里必须统一反标准化，否则 MDN 的预测会退化成 ~0 偏移。
    """
    mean_z, cov_z = model.predict(x)
    mean_u = model.y_mean + model.y_std * np.asarray(mean_z, dtype=np.float64)
    cov_u = np.asarray(cov_z, dtype=np.float64) * np.outer(model.y_std, model.y_std)
    return mean_u, cov_u


def loo_predictions(
    samples: Sequence[PairSample],
    family: str,
    lam: float | None = None,
    mdn_kwargs: dict | None = None,
    modes: Sequence[str] = ("identity",),
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """按帧（stem）分组的留一预测，返回与输入**同序**的每对 (mean_u, cov_u)。

    分组保证同一帧的样本不会同时出现在训练/验证里。返回顺序必须与 samples 一一对应，
    否则 evaluate_predictions 的 zip 会把"这一帧的预测"记到"那一帧的真值"上。
    """
    index_by_stem: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        index_by_stem.setdefault(sample.stem, []).append(index)
    means: list[np.ndarray] = [np.zeros(2) for _ in samples]
    covs: list[np.ndarray] = [np.eye(2) for _ in samples]
    for stem, indices in index_by_stem.items():
        train = [s for s in samples if s.stem != stem]
        X, Y, _ = build_arrays(train, modes)
        model = _fit_family(family, X, Y, lam=lam, mdn_kwargs=mdn_kwargs)
        for index in indices:
            mean_u, cov_u = _predict_family(model, structure_features(samples[index].anchor))
            means[index] = mean_u
            covs[index] = cov_u
    return means, covs


def loo_cross_validate(
    samples: Sequence[PairSample],
    family: str,
    lam: float | None = None,
    mdn_kwargs: dict | None = None,
    modes: Sequence[str] = ("identity",),
) -> dict:
    """按帧（stem）分组的留一交叉验证。"""
    means, covs = loo_predictions(samples, family, lam=lam, mdn_kwargs=mdn_kwargs, modes=modes)
    return evaluate_predictions(samples, means, covs, family=family)


# ---------------------------------------------------------------------------
# 推理封装
# ---------------------------------------------------------------------------
class RelationPredictor:
    """加载 relation.json，按锚结构给出目标结构的方向/落点/ROI。"""

    def __init__(self, payload: dict, base_dir: Path | None = None) -> None:
        self.payload = payload
        self.base_dir = Path(base_dir) if base_dir else Path(".")
        self.models: dict[str, LinearRelation | MDNRelation] = {}
        self.meta: dict[str, dict] = {}
        for key, entry in payload.get("directions", {}).items():
            family = entry["family"]
            if family == "mdn":
                mdn_path = self.base_dir / entry["mdn"]["path"]
                import torch

                state = torch.load(mdn_path, map_location="cpu", weights_only=False)
                mdn_meta = entry["mdn"]
                self.models[key] = MDNRelation(
                    state_dict=state["state_dict"],
                    in_dim=int(mdn_meta["in_dim"]),
                    hidden=int(mdn_meta["hidden"]),
                    components=int(mdn_meta["components"]),
                    x_mean=np.asarray(mdn_meta["x_mean"], dtype=np.float64),
                    x_std=np.asarray(mdn_meta["x_std"], dtype=np.float64),
                    y_mean=np.asarray(mdn_meta["y_mean"], dtype=np.float64),
                    y_std=np.asarray(mdn_meta["y_std"], dtype=np.float64),
                    n_samples=int(mdn_meta.get("n_samples", 0)),
                )
            else:
                self.models[key] = LinearRelation.from_json(entry["linear"])
            self.meta[key] = entry

    # -- 构造 -----------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "RelationPredictor":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload, base_dir=path.parent)

    def supports(self, anchor_cls: int) -> bool:
        return any(key.startswith(CLASS_NAMES[anchor_cls] + "->") for key in self.models)

    def target_of(self, anchor_cls: int) -> int | None:
        for key in self.models:
            anchor_name, target_name = key.split("->")
            if anchor_name == CLASS_NAMES[anchor_cls]:
                return CLASS_NAMES.index(target_name)
        return None

    # -- 预测 -----------------------------------------------------------
    def predict(self, anchor: Structure) -> RelationPrediction | None:
        key = next((k for k in self.models if k.startswith(CLASS_NAMES[anchor.cls] + "->")), None)
        if key is None:
            return None
        model = self.models[key]
        mean_u, cov_u = _predict_family(model, structure_features(anchor))
        entry = self.meta[key]
        target_cls = CLASS_NAMES.index(key.split("->")[1])
        calibration = entry.get("calibration") or {}
        presence = entry.get("presence_rate")
        return make_prediction(
            anchor, mean_u, cov_u, entry["family"], target_cls,
            presence_rate=float(presence) if presence is not None else float("nan"),
            cone_k=float(calibration.get("cone_k", 1.0)),
            roi_scale=float(calibration.get("roi_scale", 1.0)),
        )

    def consistency(self, anchor: Structure, target: Structure) -> dict | None:
        """两结构都检测到时，用马氏距离检查这对组合是否符合位置先验。"""
        prediction = self.predict(anchor)
        if prediction is None:
            return None
        observed = np.asarray(
            [
                (target.cx - anchor.cx) * anchor.image_w,
                (target.cy - anchor.cy) * anchor.image_h,
            ]
        )
        diff = observed - prediction.mean_px
        cov = prediction.cov_px + np.eye(2) * 25.0
        try:
            mahalanobis = float(math.sqrt(max(float(diff @ np.linalg.solve(cov, diff)), 0.0)))
        except np.linalg.LinAlgError:
            return None
        return {
            "mahalanobis": round(mahalanobis, 2),
            "angle_error_deg": round(abs(wrap_angle_deg(
                math.degrees(math.atan2(float(observed[1]), float(observed[0]))) - prediction.angle_deg
            )), 1),
            "plausible": bool(mahalanobis <= 3.0),
        }


# ---------------------------------------------------------------------------
# 训练 / 选型 / 保存
# ---------------------------------------------------------------------------
def train_direction(
    samples: Sequence[PairSample],
    anchor_cls: int,
    target_cls: int,
    presence_rate: float,
    families: Sequence[str] = ("prior", "ridge", "mdn"),
    ridge_lambdas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    select_metric: str = "nll",
    seed: int = 0,
    mdn_min_samples: int = 40,
    augment: bool = False,
    verbose: bool = True,
) -> tuple[dict, dict]:
    """训练一个方向的关系模型：LOO 交叉验证选型 -> 校准不确定度 -> 全量重拟合。

    返回 (direction_entry, comparison)。
    """
    subset = [s for s in samples if s.anchor_cls == anchor_cls and s.target_cls == target_cls]
    stems = {s.stem for s in subset}
    # 留一交叉验证按帧分组：样本全来自同一帧时，留出后训练集为空，指标会静默变 NaN。
    if len(subset) < 3 or len(stems) < 2:
        raise SystemExit(
            f"{direction_key(anchor_cls, target_cls)} 只有 {len(subset)} 对样本 / {len(stems)} 帧，"
            "无法训练关系模型（留一验证至少需要 3 对样本且来自 2 帧以上）。"
        )
    modes = ("identity", "fliplr", "flipud", "rot180") if augment else ("identity",)
    X, Y, _ = build_arrays(subset, modes)

    candidates: list[tuple[str, float | None, dict]] = []
    if "prior" in families:
        candidates.append(("prior", None, {}))
    if "ridge" in families:
        for lam in ridge_lambdas:
            candidates.append(("ridge", lam, {}))
    if "mdn" in families:
        if len(subset) >= mdn_min_samples:
            for components in (2, 3):
                candidates.append(("mdn", None, {"components": components, "hidden": 16,
                                                 "epochs": 400, "patience": 60, "seed": seed}))
        elif verbose:
            print(f"      跳过 MDN（成对样本 {len(subset)} < {mdn_min_samples}，线性模型更稳）")

    comparison: dict[str, dict] = {}
    higher_is_better = select_metric in ("roi_hit_rate", "cone_hit_rate", "within_15_deg", "within_30_deg")
    best = None
    for family, lam, kwargs in candidates:
        label = family if family != "ridge" else f"ridge(lam={lam:g})"
        if family == "mdn":
            label = f"mdn(k={kwargs.get('components')})"
        metrics = loo_cross_validate(subset, family, lam=lam, mdn_kwargs=kwargs, modes=modes)
        comparison[label] = {"family": family, "lam": lam, "metrics": metrics}
        raw = metrics.get(select_metric, float("inf"))
        score = -float(raw) if higher_is_better else float(raw)
        if best is None or score < best[0] - 1e-9:
            best = (score, family, lam, kwargs, label, metrics)
        if verbose:
            print(f"      LOO {label:<18} angle_mae={metrics['angle_mae_deg']:5.2f}°  "
                  f"center_mae={metrics['center_mae_px']:6.1f}px  roi_hit={metrics['roi_hit_rate']:.2f}  "
                  f"nll={metrics['nll']:7.3f}")

    assert best is not None
    _, family, lam, kwargs, label, metrics = best
    if verbose:
        print(f"      -> 选中 {label}（按 LOO {select_metric}）")

    # 用选中模型的 LOO 预测校准"锥形角"和"ROI"的宽度，使覆盖率达到目标值
    means, covs = loo_predictions(subset, family, lam=lam, mdn_kwargs=kwargs, modes=modes)
    cone_k = 4.0
    for candidate in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        probe = evaluate_predictions(subset, means, covs, family=family, cone_k=candidate)
        if probe["cone_hit_rate"] >= 0.9:
            cone_k = candidate
            break
    roi_scale = 2.0
    for candidate in (1.0, 1.25, 1.5, 2.0):
        probe = evaluate_predictions(subset, means, covs, family=family, roi_scale=candidate)
        if probe["roi_hit_rate"] >= 0.95:
            roi_scale = candidate
            break
    metrics = evaluate_predictions(subset, means, covs, family=family, cone_k=cone_k, roi_scale=roi_scale)
    if verbose:
        print(f"      校准: cone_k={cone_k:g}（锥形命中 {metrics['cone_hit_rate']:.0%}），"
              f"roi_scale={roi_scale:g}（ROI 命中 {metrics['roi_hit_rate']:.0%}）")

    final_model = _fit_family(family, X, Y, lam=lam, mdn_kwargs=kwargs)
    entry: dict = {
        "anchor_cls": int(anchor_cls),
        "target_cls": int(target_cls),
        "family": family,
        "n_pairs": len(subset),
        "presence_rate": round(float(presence_rate), 4),
        "loo_metrics": metrics,
        "calibration": {"cone_k": cone_k, "roi_scale": roi_scale},
        "augmentation": list(modes),
        "selected": label,
        "linear": final_model.to_json() if isinstance(final_model, LinearRelation)
        else _ridge_fit(X, Y, lam if lam is not None else 1.0).to_json(),
    }
    if isinstance(final_model, MDNRelation):
        entry["mdn"] = {
            "path": f"relation_mdn_{anchor_cls}_{target_cls}.pt",
            **final_model.to_meta(),
        }
        entry["mdn_state"] = final_model.state_dict
    return entry, comparison


def train_all(
    data: Path,
    families: Sequence[str] = ("prior", "ridge", "mdn"),
    select_metric: str = "nll",
    seed: int = 0,
    augment: bool = False,
    ridge_lambdas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    verbose: bool = True,
    prepared: tuple[list[PairSample], dict[str, PresenceStats], dict] | None = None,
) -> tuple[dict, dict]:
    """训练所有方向的关系模型。

    `prepared` 是 `load_pairs()` 的返回值：调用方已经读过数据集时传进来，避免重复解析
    （同时保证 `--data xxx.yaml` 这种显式 yaml 路径不会被二次解析成 `xxx.yaml/data.yaml`）。
    """
    samples, presence, stats = prepared if prepared is not None else load_pairs(data)
    if not samples:
        raise SystemExit("没有找到同时含两个结构的标注帧，无法训练关系模型。")
    if verbose:
        print(f"      成对样本: {len(samples)} 对，来自 {len({s.stem for s in samples})} 张帧")
        for key, stat in presence.items():
            print(f"      P({key.split('->')[1]} | {key.split('->')[0]}) = "
                  f"{stat.n_both_images}/{stat.n_anchor_images} = {stat.rate:.3f}")

    payload: dict = {
        "version": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "classes": list(CLASS_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "target_normalization": "offset_px / sqrt(anchor_mask_area_px)",
        "augmentation": ["identity", "fliplr", "flipud", "rot180"] if augment else ["identity"],
        "selection": {"criterion": select_metric},
        "directions": {},
    }
    comparison: dict = {}
    mdn_states: dict[str, dict] = {}
    for anchor_cls, target_cls in DIRECTIONS:
        key = direction_key(anchor_cls, target_cls)
        if verbose:
            print(f"\n    [{key}]")
        try:
            entry, comp = train_direction(
                samples, anchor_cls, target_cls,
                presence_rate=presence[key].rate,
                families=families, select_metric=select_metric, seed=seed,
                ridge_lambdas=ridge_lambdas, augment=augment, verbose=verbose,
            )
        except SystemExit as exc:
            print(f"      跳过: {exc}")
            continue
        if "mdn_state" in entry:
            mdn_states[entry["mdn"]["path"]] = entry.pop("mdn_state")
        payload["directions"][key] = entry
        comparison[key] = comp

    payload["selection"]["comparison"] = comparison
    payload["data_stats"] = {
        "images": stats["images"],
        "images_with_labels": stats["images_with_labels"],
        "image_sizes": stats["image_sizes"][:5],
        "pairs": len(samples),
        "pair_frames": len({s.stem for s in samples}),
    }
    return payload, mdn_states


def save_relation(payload: dict, mdn_states: dict[str, dict], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, state in mdn_states.items():
        import torch

        torch.save({"state_dict": state}, out_dir / relative_path)
    path = out_dir / "relation.json"
    text = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return path

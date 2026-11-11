"""train_relation.py —— 训练"二尖瓣 ↔ 卵圆孔"的相对位置模型（方向 + 距离 + 不确定度）。

用法：
    python train_relation.py --data dataset

产物（默认写回数据集目录）：
    relation.json            线性模型参数、经验先验、LOO 交叉验证指标
    relation_mdn_*.pt        仅当 MDN 被选中时存在
    relation_report.md       可读的训练/评估报告

为什么用留一交叉验证（LOO）而不是固定验证集：成对样本只有几十个，固定留出 20% 会让
评估只依赖个位数样本；LOO 每次留一帧、按帧分组（同一帧的样本不会同时出现在训练/验证），
是这种规模下最稳的估计。最终模型再用全部成对样本重拟合。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from common import circular_mean_std_deg, resolve_data_yaml
from relation import (
    DIRECTIONS,
    direction_key,
    load_pairs,
    save_relation,
    train_all,
)


def _num(value: object, spec: str = ".2f") -> str:
    """格式化数值；None / NaN / Inf 一律显示 n/a（避免报告里出现 "nan"）。"""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return format(number, spec)


def _format_table(comparison: dict[str, dict]) -> str:
    lines = [
        "| 候选模型 | 角度 MAE | 角度中位 | ≤15° | ≤30° | 中心 MAE | ROI 命中 | NLL |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, entry in comparison.items():
        metrics = entry["metrics"]
        lines.append(
            f"| {label} | {_num(metrics['angle_mae_deg'])}° | {_num(metrics['angle_median_deg'])}° | "
            f"{_num(metrics['within_15_deg'], '.0%')} | {_num(metrics['within_30_deg'], '.0%')} | "
            f"{_num(metrics['center_mae_px'], '.0f')}px | {_num(metrics['roi_hit_rate'], '.0%')} | "
            f"{_num(metrics['nll'])} |"
        )
    return "\n".join(lines)


def _direction_prior_rows(samples) -> list[str]:
    """按方向分别统计经验先验（两个方向相反，混在一起做圆统计会得到无意义的双峰结果）。"""
    rows = [
        "## 经验先验（按方向，全部成对样本）",
        "",
        "| 方向 | 帧数 | 方向圆均值 | 圆标准差 | 归一化距离 \\|u\\| | 像素距离 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for anchor_cls, target_cls in DIRECTIONS:
        key = direction_key(anchor_cls, target_cls)
        subset = [s for s in samples if s.anchor_cls == anchor_cls and s.target_cls == target_cls]
        if not subset:
            continue
        angles = [float(np.degrees(np.arctan2(s.offset_u[1], s.offset_u[0]))) for s in subset]
        mean_angle, std_angle = circular_mean_std_deg(angles)
        norms = np.asarray([float(np.hypot(*s.offset_u)) for s in subset])
        px = np.asarray([
            float(np.hypot((s.target.cx - s.anchor.cx) * s.anchor.image_w,
                           (s.target.cy - s.anchor.cy) * s.anchor.image_h))
            for s in subset
        ])
        rows.append(
            f"| {key} | {len({s.stem for s in subset})} | {mean_angle:.1f}° | {std_angle:.1f}° | "
            f"{norms.mean():.2f} ± {norms.std():.2f} | {px.mean():.0f} ± {px.std():.0f} px |"
        )
    rows.append("")
    return rows


def _write_report(path: Path, payload: dict, samples, presence: dict) -> None:
    rows: list[str] = ["# 相对位置模型训练报告", ""]
    stats = payload.get("data_stats", {})
    rows.append(f"- 生成时间：{payload['created']}")
    rows.append(f"- 类别：`{payload['classes']}`")
    rows.append(f"- 目标归一化：`{payload['target_normalization']}`")
    rows.append(f"- 成对样本：{stats.get('pairs', 0)} 对，来自 {stats.get('pair_frames', 0)} 张帧"
                f"（有标注帧共 {stats.get('images_with_labels', 0)} 张，其余帧缺一个结构或为空）")
    rows.append(f"- 选型准则：LOO `{payload['selection']['criterion']}`")
    rows.append(f"- 几何增广：`{payload.get('augmentation', ['identity'])}`")
    rows.append("")

    if samples:
        rows += _direction_prior_rows(samples)

    rows += ["## 存在先验 P(目标 | 锚点)", ""]
    for key, stat in presence.items():
        anchor_name, target_name = key.split("->")
        rows.append(
            f"- P({target_name} | {anchor_name}) = {stat.n_both_images}/{stat.n_anchor_images} = "
            f"**{_num(stat.rate, '.3f')}**"
        )
    rows.append("")

    for anchor_cls, target_cls in DIRECTIONS:
        key = direction_key(anchor_cls, target_cls)
        entry = payload["directions"].get(key)
        if entry is None:
            continue
        rows += [f"## {key}", ""]
        rows.append(f"- 选中模型：**{entry['selected']}**（family=`{entry['family']}`）")
        rows.append(f"- 训练成对样本：{entry['n_pairs']}")
        rows.append(f"- P(目标存在) = {_num(entry['presence_rate'], '.3f')}")
        metrics = entry["loo_metrics"]
        rows.append(
            f"- LOO 指标：角度 MAE {_num(metrics['angle_mae_deg'])}°，"
            f"中位 {_num(metrics['angle_median_deg'])}°，"
            f"≤15° {_num(metrics['within_15_deg'], '.0%')}，"
            f"≤30° {_num(metrics['within_30_deg'], '.0%')}；"
            f"中心误差 {_num(metrics['center_mae_px'], '.0f')}px；"
            f"ROI 命中 {_num(metrics['roi_hit_rate'], '.0%')}；"
            f"锥形命中 {_num(metrics['cone_hit_rate'], '.0%')}；NLL {_num(metrics['nll'])}"
        )
        calibration = entry.get("calibration") or {}
        if calibration:
            rows.append(f"- 不确定度校准：`cone_k={calibration.get('cone_k', 1.0):g}`，"
                        f"`roi_scale={calibration.get('roi_scale', 1.0):g}`"
                        "（由 LOO 覆盖率反推，保证锥形≥90%、ROI≥95%）")
        rows.append("")
        rows.append(_format_table(payload["selection"]["comparison"][key]))
        rows.append("")

    rows += [
        "## 解读注意",
        "",
        "- 本数据集只有一个检查（同一台机器、同一次检查的连续帧），**LOO 指标不代表跨患者泛化**；",
        "  它衡量的是「同检查内换一帧」的稳定性。要评估跨患者泛化，需要多个检查号的独立数据。",
        "- `prior` 行就是「永远预测平均方向」的常数基线；只有当学习模型明显优于它时，条件化才有意义。",
        "- 默认不做左右/上下翻转增广：本数据是固定探头方向的单检查，「二尖瓣在右、卵圆孔在左」",
        "  是解剖事实；镜像样本在解剖上不存在，还会让常数先验的均值互相抵消。",
        "  只有在你的机器/探头方向可能反转时才加 `--augment`。",
        "- 样本量小时 `nll` 对协方差校准更敏感，`angle_mae_deg` 更直观；可用 `--select-metric` 切换。",
        "",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def _parse_lambdas(text: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    if not values:
        raise SystemExit("--ridge-lambdas 至少要给一个数值。")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("dataset"), help="数据集目录（含 data.yaml）")
    parser.add_argument("--out", type=Path, default=None, help="模型输出目录（默认=数据集目录）")
    parser.add_argument("--families", default="prior,ridge,mdn",
                        help="候选模型族，逗号分隔：prior,ridge,mdn")
    parser.add_argument("--select-metric", default="nll",
                        choices=["nll", "angle_mae_deg", "center_mae_px", "roi_hit_rate", "angle_median_deg"],
                        help="LOO 选型指标（roi_hit_rate 越大越好，其余越小越好）")
    parser.add_argument("--ridge-lambdas", default="0.001,0.01,0.1,1,10,100",
                        help="岭回归的正则强度候选（逗号分隔）")
    parser.add_argument("--mdn-min-samples", type=int, default=40,
                        help="成对样本少于该值时跳过 MDN（线性模型更稳）")
    parser.add_argument("--augment", action="store_true",
                        help="开启 fliplr/flipud/rot180 几何增广（仅当探头方向可能反转时使用）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data_yaml = resolve_data_yaml(args.data)
    out_dir = Path(args.out) if args.out else data_yaml.parent
    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    if "mdn" in families:
        try:
            import torch  # noqa: F401
        except ImportError:
            print("[警告] 未安装 torch，跳过 MDN 候选。")
            families = tuple(f for f in families if f != "mdn")
    if not families:
        raise SystemExit("至少需要一个模型族（prior / ridge / mdn）。")

    print(f"读取数据: {data_yaml}")
    prepared = load_pairs(data_yaml)
    samples, presence, _stats = prepared
    if not samples:
        raise SystemExit("没有成对样本：请确认标注里同时存在二尖瓣和卵圆孔。")

    print(f"成对样本 {len(samples)} 对（{len({s.stem for s in samples})} 帧），开始训练 ...\n")
    payload, mdn_states = train_all(
        data_yaml,
        families=families,
        select_metric=args.select_metric,
        seed=args.seed,
        augment=args.augment,
        ridge_lambdas=_parse_lambdas(args.ridge_lambdas),
        verbose=not args.quiet,
        prepared=prepared,
    )
    path = save_relation(payload, mdn_states, out_dir)

    report = out_dir / "relation_report.md"
    _write_report(report, payload, samples, presence)

    print(f"\n已保存: {path}")
    print(f"已保存: {report}")
    print("\n各方向结论：")
    for key, entry in payload["directions"].items():
        metrics = entry["loo_metrics"]
        print(f"  {key}: {entry['selected']}  "
              f"角度 MAE {_num(metrics['angle_mae_deg'], '.1f')}°，"
              f"≤15° {_num(metrics['within_15_deg'], '.0%')}，"
              f"ROI 命中 {_num(metrics['roi_hit_rate'], '.0%')}")
    print("\n下一步：")
    print("  python predict.py --weights runs/segment/echo_relation_seg/weights/best.pt "
          "--source <图像或视频或目录>")
    print("  python evaluate.py --weights runs/segment/echo_relation_seg/weights/best.pt --data dataset")


if __name__ == "__main__":
    sys.exit(main())

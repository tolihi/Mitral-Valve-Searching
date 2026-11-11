"""prepare_dataset.py —— 把 CVAT "Ultralytics YOLO Segmentation" 导出整理成可训练的
YOLO-seg 数据集（仅标签的 zip + 原始图像目录 -> images/labels + data.yaml）。

用法示例（Windows，注意引号）：
    python prepare_dataset.py ^
        --zip "exports\\project_annotations_ultralytics yolo segmentation 1.0.zip" ^
        --images "D:\\data\\echo\\<检查号>_<时间戳>" ^
        --out dataset --val-ratio 0.2

设计要点
--------
1. 本导出实际声明了 3 个类（"mitral valve" / "fossal" / "mitral and fossil"）。
   前两个映射为规范类 mitral_valve / fossa_ovalis；组合类默认丢弃并打印数量，
   可用 --combined both 把组合多边形同时写成两个类。
2. 只有被标注的帧才有 label 文件；其余图像作为"纯背景"负样本保留（--drop-empty 可关闭）。
3. 同一检查/视频的相邻帧高度相关，随机划分会高估指标。默认按"时间块 + 类别分层"划分：
   把帧按时间戳切成若干连续块，再分层（含双结构的块 / 其余块）抽块做验证集，
   既避免相邻帧泄漏，又保证验证集里两类（尤其是双结构）都有样本。
   只有一个检查号时这是唯一可行的划分方式；它仍然**不代表跨患者泛化**。
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from common import (
    CLASS_NAMES,
    IMAGE_SUFFIXES,
    canonical_class,
    class_index,
    exam_key,
    format_seg_line,
    image_files,
    parse_seg_line,
    save_json,
)

EXPECTED_CLASSES = list(CLASS_NAMES)


# ---------------------------------------------------------------------------
# 读取导出包
# ---------------------------------------------------------------------------
def read_export(
    zip_path: Path,
) -> tuple[dict[int, str], dict[str, list[tuple[int, np.ndarray]]], dict[str, tuple[str, bytes]]]:
    """返回 (原始类别号->名称, stem->[(原始类别号, 多边形)], stem->(后缀, zip 内图像字节))。"""
    raw_names: dict[int, str] = {}
    labels: dict[str, list[tuple[int, np.ndarray]]] = {}
    zip_images: dict[str, tuple[str, bytes]] = {}

    with zipfile.ZipFile(zip_path) as archive:
        names = list(archive.namelist())

        yaml_name = next((n for n in names if n.lower().endswith(("data.yaml", "data.yml"))), None)
        if yaml_name:
            spec = yaml.safe_load(archive.read(yaml_name).decode("utf-8", errors="replace")) or {}
            raw = spec.get("names", {}) or {}
            if isinstance(raw, list):
                raw_names = {i: str(v) for i, v in enumerate(raw)}
            else:
                for key, value in raw.items():
                    try:
                        raw_names[int(key)] = str(value)
                    except (TypeError, ValueError):
                        continue
        if not raw_names:
            raise SystemExit("导出包内没有可解析的 data.yaml/names，无法确定类别。")

        for name in names:
            if name.endswith("/"):
                continue
            suffix = Path(name).suffix.lower()
            parts = Path(name).parts
            if suffix == ".txt" and "labels" in parts:
                content = archive.read(name).decode("utf-8", errors="replace")
                instances: list[tuple[int, np.ndarray]] = []
                for line in content.splitlines():
                    parsed = parse_seg_line(line.strip())
                    if parsed is not None:
                        instances.append(parsed)
                labels[Path(name).stem] = instances
            elif suffix in IMAGE_SUFFIXES:
                zip_images[Path(name).stem] = (suffix, archive.read(name))

    return raw_names, labels, zip_images


def build_class_mapping(raw_names: dict[int, str], combined_policy: str) -> tuple[dict[int, list[int]], Counter]:
    """原始类别号 -> 规范类别号列表（组合类按策略映射为空或 [0, 1]）。"""
    mapping: dict[int, list[int]] = {}
    stats: Counter = Counter()
    for raw_id, raw_name in sorted(raw_names.items()):
        kind = canonical_class(raw_name)
        if kind == "combined":
            stats["combined_class"] += 1
            if combined_policy == "both":
                mapping[raw_id] = [0, 1]
                print(f"[警告] 原始类 {raw_id} ({raw_name!r}) 是组合类，按 --combined both 拆成两个类。")
            else:
                mapping[raw_id] = []
                print(f"[警告] 原始类 {raw_id} ({raw_name!r}) 是组合类，默认丢弃（可用 --combined both 保留）。")
        elif kind is None:
            stats["unknown_class"] += 1
            mapping[raw_id] = []
            print(f"[警告] 无法识别的类别 {raw_id} ({raw_name!r})，丢弃。")
        else:
            mapping[raw_id] = [class_index(kind)]
    return mapping, stats


# ---------------------------------------------------------------------------
# 划分策略
# ---------------------------------------------------------------------------
def combo_of(instances: list[tuple[int, np.ndarray]]) -> str:
    """图像内容标签：both / mitral_only / fossa_only / none。"""
    present = {cls for cls, _ in instances}
    if present == {0, 1}:
        return "both"
    if present == {0}:
        return "mitral_only"
    if present == {1}:
        return "fossa_only"
    return "none"


def _block_split_time(
    stems: list[str],
    val_ratio: float,
    combos: dict[str, str],
    seed: int,
) -> tuple[list[str], list[str], str]:
    """按时间块划分：相邻帧成块，块级分层抽样，避免相邻帧泄漏并保证类别覆盖。"""
    ordered = sorted(stems, key=lambda s: exam_key(s)[1])
    n = len(ordered)
    n_blocks = int(min(max(5, round(n / 12)), n))
    blocks = [list(b) for b in np.array_split(np.asarray(ordered, dtype=object), n_blocks)]
    block_has_both = [any(combos[s] == "both" for s in block) for block in blocks]
    block_has_fossa = [any(combos[s] in ("both", "fossa_only") for s in block) for block in blocks]

    k_val = int(max(1, round(n_blocks * val_ratio)))
    rng = random.Random(seed)
    both_idx = [i for i in range(n_blocks) if block_has_both[i]]
    other_idx = [i for i in range(n_blocks) if not block_has_both[i]]
    chosen: list[int] = []

    # 含双结构的块至少抽 1 块（如果验证集份额允许），其余从另一层抽，保证时间上分散
    if both_idx and k_val >= 1:
        k_both = min(len(both_idx), max(1, round(k_val * 0.4)))
        chosen += rng.sample(both_idx, k_both)
    remaining = [i for i in other_idx]
    rng.shuffle(remaining)
    chosen += remaining[: max(0, k_val - len(chosen))]
    if len(chosen) < k_val:
        extra = [i for i in both_idx if i not in chosen]
        rng.shuffle(extra)
        chosen += extra[: k_val - len(chosen)]
    chosen = sorted(set(chosen))

    val = [s for i in chosen for s in blocks[i]]
    val_set = set(val)
    train = [s for s in ordered if s not in val_set]
    note = (
        f"time-block+stratified（{n_blocks} 个连续时间块中取 {len(chosen)} 块做验证集；"
        "同检查内划分，不能代表跨患者泛化）"
    )
    return sorted(train), sorted(val), note


def _split_by_group(
    stems: list[str],
    val_ratio: float,
    combos: dict[str, str],
    groups: dict[str, str],
    seed: int,
) -> tuple[list[str], list[str], str]:
    """按检查号分组划分，并对"是否含双结构"做分层，避免验证集全无成对样本。"""
    group_to_stems: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        group_to_stems[groups[stem]].append(stem)
    group_has_both = {g: any(combos[s] == "both" for s in members) for g, members in group_to_stems.items()}
    n = len(stems)
    target_val = max(1, min(n - 1, int(round(n * val_ratio))))
    rng = random.Random(seed)

    both_groups = [g for g, has in group_has_both.items() if has]
    other_groups = [g for g, has in group_has_both.items() if not has]
    rng.shuffle(both_groups)
    rng.shuffle(other_groups)

    val: list[str] = []
    for group in both_groups[:1] + other_groups + both_groups[1:]:
        if len(val) >= target_val:
            break
        val.extend(group_to_stems[group])
    val_set = set(val)
    train = [s for s in stems if s not in val_set]
    if not train:
        raise SystemExit("按检查分组划分后训练集为空，请减小 --val-ratio 或改用 --split-mode time。")
    return sorted(train), sorted(val), f"group（按检查号分组，共 {len(group_to_stems)} 组）"


def split_records(
    stems: list[str],
    val_ratio: float,
    split_mode: str,
    seed: int,
    groups: dict[str, str],
    combos: dict[str, str],
) -> tuple[list[str], list[str], str]:
    """返回 (train_stems, val_stems, 划分方式说明)。"""
    if len(stems) < 2:
        raise SystemExit("至少需要 2 张图像才能划分训练/验证集。")

    unique_groups = sorted(set(groups.values()))
    if split_mode in ("auto", "group") and len(unique_groups) >= 2:
        return _split_by_group(stems, val_ratio, combos, groups, seed)
    if split_mode == "group":
        print("[警告] 只有一个检查分组，无法按检查划分；退化为按时间块划分。")

    if split_mode in ("auto", "time"):
        return _block_split_time(stems, val_ratio, combos, seed)

    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n_val = max(1, min(len(stems) - 1, int(round(len(stems) * val_ratio))))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val]), (
        "random（随机划分；相邻帧高度相关，指标会偏乐观，不能代表跨患者泛化）"
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", required=True, type=Path, help="CVAT Ultralytics YOLO Segmentation 导出 zip")
    parser.add_argument("--images", type=Path, default=None, help="原始 JPG/PNG 目录（zip 内已含图像时可省略）")
    parser.add_argument("--out", type=Path, default=Path("dataset"), help="输出数据集目录")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["auto", "group", "time", "random"], default="auto",
                        help="auto=有多个检查号时按检查分组，否则按时间块分层")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--combined", choices=["drop", "both"], default="drop",
                        help="组合类（'mitral and fossil'）的处理方式")
    parser.add_argument("--drop-empty", action="store_true", help="丢弃没有标注的图像（默认保留为背景负样本）")
    parser.add_argument("--force", action="store_true", help="输出目录非空时覆盖")
    args = parser.parse_args()

    if not args.zip.is_file():
        raise SystemExit(f"找不到 zip: {args.zip}")
    if not 0.05 <= args.val_ratio <= 0.5:
        raise SystemExit("--val-ratio 需在 [0.05, 0.5] 之间")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"输出目录非空: {out}\n如确认覆盖请加 --force，或换一个 --out。")

    print(f"[1/5] 读取导出包: {args.zip}")
    raw_names, raw_labels, zip_images = read_export(args.zip)
    print("      原始类别: " + ", ".join(f"{i}={n!r}" for i, n in sorted(raw_names.items())))
    mapping, _ = build_class_mapping(raw_names, args.combined)

    # -- 图像来源 ---------------------------------------------------------
    if args.images is not None:
        if not args.images.is_dir():
            raise SystemExit(f"图像目录不存在: {args.images}")
        source_images = {p.stem: p for p in image_files(args.images)}
        source_kind = str(args.images)
    elif zip_images:
        print(f"      导出包内含 {len(zip_images)} 张图像，将解压使用。")
        source_images = {}
        source_kind = "zip"
    else:
        raise SystemExit("导出包内没有图像，请用 --images 指定原始图像目录。")

    all_stems = set(raw_labels) | set(source_images)
    missing_images = sorted(set(raw_labels) - set(source_images))
    if missing_images:
        raise SystemExit(
            f"有 {len(missing_images)} 个标注找不到对应图像，例如: {missing_images[0]}\n"
            f"请确认 --images 指向包含这些帧的目录。"
        )
    print(f"[2/5] 配对: 标注 {len(raw_labels)} 张 / 图像 {len(source_images)} 张 / "
          f"未标注 {len(all_stems - set(raw_labels))} 张")

    # -- 清洗标注 ---------------------------------------------------------
    canonical: dict[str, list[tuple[int, np.ndarray]]] = {}
    invalid = Counter()
    clamped = 0
    for stem in sorted(all_stems):
        kept: list[tuple[int, np.ndarray]] = []
        for raw_cls, pts in raw_labels.get(stem, []):
            target_classes = mapping.get(raw_cls, [])
            if not target_classes:
                invalid["dropped_class"] += 1
                continue
            if len(pts) < 3:
                invalid["too_few_points"] += 1
                continue
            if not np.isfinite(pts).all():
                invalid["non_finite"] += 1
                continue
            if (pts < -0.01).any() or (pts > 1.01).any():
                clamped += 1
            pts = np.clip(pts, 0.0, 1.0)
            if np.ptp(pts[:, 0]) < 1e-4 and np.ptp(pts[:, 1]) < 1e-4:
                invalid["degenerate"] += 1
                continue
            for target_cls in target_classes:
                kept.append((target_cls, pts))
        canonical[stem] = kept

    if args.drop_empty:
        kept_stems = [s for s in sorted(all_stems) if canonical.get(s)]
    else:
        kept_stems = sorted(all_stems)
    if not kept_stems:
        raise SystemExit("清洗后没有可用图像。")

    combos = {stem: combo_of(canonical.get(stem, [])) for stem in kept_stems}
    n_both_total = sum(1 for v in combos.values() if v == "both")

    # -- 划分 -------------------------------------------------------------
    groups = {stem: exam_key(stem)[0] for stem in kept_stems}
    train_stems, val_stems, split_note = split_records(
        kept_stems, args.val_ratio, args.split_mode, args.seed, groups, combos
    )
    print(f"[3/5] 划分: train={len(train_stems)} / val={len(val_stems)} -> {split_note}")

    # -- 写出 -------------------------------------------------------------
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    if zip_images and args.images is None:
        raw_dir = out / "_zip_images"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for stem, (suffix, payload) in zip_images.items():
            target = raw_dir / f"{stem}{suffix}"
            target.write_bytes(payload)
            source_images[stem] = target

    per_split_stats: dict[str, dict] = {}
    for split, stems in (("train", train_stems), ("val", val_stems)):
        combo_count: Counter = Counter()
        inst_count: Counter = Counter()
        for stem in stems:
            image_path = source_images[stem]
            shutil.copy2(image_path, out / "images" / split / image_path.name)
            instances = canonical.get(stem, [])
            lines = [format_seg_line(cls, pts) for cls, pts in instances]
            (out / "labels" / split / f"{stem}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
            )
            combo_count[combos[stem]] += 1
            for cls in {c for c, _ in instances}:
                inst_count[CLASS_NAMES[cls]] += 1
        per_split_stats[split] = {
            "images": len(stems),
            "images_with_annotations": sum(1 for s in stems if canonical.get(s)),
            "combo": dict(combo_count),
            "images_with_both": combo_count.get("both", 0),
            "images_per_class": dict(inst_count),
        }

    names = {i: name for i, name in enumerate(EXPECTED_CLASSES)}
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": out.as_posix(),
                "train": "images/train",
                "val": "images/val",
                "nc": len(EXPECTED_CLASSES),
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    meta = {
        "source_zip": str(args.zip),
        "source_images": source_kind,
        "raw_class_names": {str(k): v for k, v in sorted(raw_names.items())},
        "class_mapping": {str(k): v for k, v in sorted(mapping.items())},
        "classes": list(EXPECTED_CLASSES),
        "split_mode": split_note,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "counts": {
            "raw_label_files": len(raw_labels),
            "images_on_disk": len(source_images),
            "unlabeled_images": len(all_stems - set(raw_labels)),
            "kept_images": len(kept_stems),
            "images_with_both_total": n_both_total,
            "clamped_polygons": clamped,
            "invalid": dict(invalid),
        },
        "splits": per_split_stats,
    }
    save_json(out / "dataset_meta.json", meta)

    print(f"[4/5] 已写出: {data_yaml}")
    for split in ("train", "val"):
        stats = per_split_stats[split]
        print(f"      {split}: {stats['images']} 张（有标注 {stats['images_with_annotations']}，"
              f"双结构 {stats['images_with_both']}，组合 {stats['combo']}）")
    if invalid or clamped:
        print(f"[5/5] 清洗: 丢弃 {dict(invalid)}，坐标裁剪 {clamped} 个多边形")
    else:
        print("[5/5] 清洗: 未发现异常标注")

    # -- 合理性提示 -------------------------------------------------------
    if per_split_stats["train"]["images_with_both"] < 8:
        print("[警告] 训练集中同时含二尖瓣与卵圆孔的图像不足 8 张，关系模型会很不稳定；"
              "建议改用 --split-mode random 或 --val-ratio 0.1。")
    if per_split_stats["val"]["images_with_both"] == 0:
        print("[警告] 验证集没有双结构图像，方向评估将缺少成对样本。")
    for cls in ("mitral_valve", "fossa_ovalis"):
        if cls not in per_split_stats["val"]["images_per_class"]:
            print(f"[警告] 验证集没有 {cls} 实例，该类分割指标无法评估。")

    print("\n下一步：")
    print(f"  1) python train_seg.py --data \"{out}\"")
    print(f"  2) python train_relation.py --data \"{out}\"")


if __name__ == "__main__":
    sys.exit(main())

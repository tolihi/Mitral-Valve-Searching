"""train_seg.py —— 用 YOLO11-seg 训练二尖瓣 / 卵圆孔实例分割模型。

用法：
    python train_seg.py --data dataset
    python train_seg.py --data dataset --model yolo11n-seg.pt --epochs 120 --batch 4 --device cpu

说明
----
* 类别固定为 0=mitral_valve、1=fossa_ovalis（顺序必须与 prepare_dataset.py 一致）。
* 超声图像只有单通道灰度信息、对比度低、边界模糊，因此增强以"几何安全 + 轻量光度"为主：
  - 只用 fliplr（左右翻转）与轻度旋转/缩放/平移，不用上下翻转（探头方向有解剖含义）；
  - 不做 mosaic 大强度拼接（默认 0.2，避免把两个结构拼到同一张图里破坏解剖关系）；
  - HSV 增强针对灰阶图实际影响很小，保留低强度即可。
* 泛化优先：默认 yolo11s-seg.pt 预训练权重、imgsz=1024（原始帧 1024x768）、AdamW + 余弦退火。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from common import CLASS_NAMES, image_files, resolve_data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("dataset"), help="数据集目录（含 data.yaml）或 yaml 路径")
    parser.add_argument("--model", default="yolo11s-seg.pt",
                        help="预训练权重；离线/CPU 可用 yolo11n-seg.pt 或本地权重路径")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8, help="显存不足时用 4/-1（-1=自动批大小）")
    parser.add_argument("--device", default="", help="''=自动，'0'=GPU0，'cpu'=CPU")
    parser.add_argument("--workers", type=int, default=4, help="Windows 上若卡在 dataloader 可设 0")
    parser.add_argument("--patience", type=int, default=40, help="验证指标多少轮不提升就早停")
    parser.add_argument("--freeze", type=int, default=0, help="冻结前 N 层（小数据可试 10）")
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "SGD", "auto"])
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--mosaic", type=float, default=0.2)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--degrees", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/segment"))
    parser.add_argument("--name", default="echo_relation_seg")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", action="store_true", help="缓存图像到内存（数据少时更快）")
    parser.add_argument("--exist-ok", action="store_true", help="允许复用同名输出目录")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据与参数，不训练")
    args = parser.parse_args()

    # ultralytics 8.4 起，相对 project 会被塞进 <runs_dir>/<task>/ 下面，
    # 导致 --project runs/smoke 变成 runs/segment/runs/smoke。这里先转绝对路径，行为可预期。
    project_dir = Path(args.project).expanduser().resolve()

    data_yaml = resolve_data_yaml(args.data)
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = spec.get("names", {})
    if isinstance(names, dict):
        # 按整数键排序（字符串键直接 sorted 会让 "10" 排在 "2" 前面）
        names = [names[key] for key in sorted(names, key=lambda k: int(k))]
    else:
        names = list(names)
    if names != list(CLASS_NAMES):
        raise SystemExit(
            f"数据集类别不匹配：data.yaml 里是 {names}，本脚本要求 {list(CLASS_NAMES)}。\n"
            f"请先用 prepare_dataset.py 生成数据集。"
        )
    root = Path(spec.get("path", data_yaml.parent))
    for split in ("train", "val"):
        split_dir = root / str(spec.get(split, f"images/{split}"))
        if not split_dir.is_dir():
            raise SystemExit(f"找不到 {split} 图像目录: {split_dir}")
        count = len(image_files(split_dir))
        print(f"      {split}: {split_dir} ({count} 张图像)")

    # --resume 必须从上次的 last.pt 续训（ultralytics 会从 ckpt 里恢复 args 与优化器状态）
    if args.resume:
        model_path = project_dir / args.name / "weights" / "last.pt"
        if not model_path.is_file():
            raise SystemExit(
                f"--resume 需要上次训练的权重：{model_path} 不存在。\n"
                f"要么去掉 --resume 重新训练，要么用 --project/--name 指向已有输出目录。"
            )
    else:
        model_path = Path(args.model)

    print("\n训练配置：")
    print(f"  模型={model_path}  epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}  "
          f"device={args.device or 'auto'}  freeze={args.freeze}")
    print(f"  输出目录={project_dir / args.name}")
    if args.dry_run:
        print("  --dry-run：仅检查，不启动训练。")
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("需要 ultralytics：pip install -r requirements.txt") from exc

    model = YOLO(str(model_path))
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        weight_decay=args.weight_decay,
        freeze=args.freeze or None,
        seed=args.seed,
        deterministic=True,
        amp=args.device != "cpu",
        cache=args.cache,
        rect=False,
        # 超声友好的增强：几何安全 + 轻量光度
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        copy_paste=0.1,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=2.0,
        perspective=0.0,
        fliplr=args.fliplr,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        erasing=0.1,
        overlap_mask=True,
        mask_ratio=4,
        plots=True,
        val=True,
        save=True,
        project=str(project_dir),
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
        verbose=True,
    )
    results = model.train(**train_kwargs)

    save_dir = Path(getattr(results, "save_dir", project_dir / args.name))
    meta = {
        "model": str(model_path),
        "data": str(data_yaml),
        "classes": list(CLASS_NAMES),
        "train_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in train_kwargs.items()},
        "save_dir": str(save_dir),
        "best": str(save_dir / "weights" / "best.pt"),
        "last": str(save_dir / "weights" / "last.pt"),
    }
    (save_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n训练结束。")
    print(f"  最优权重: {save_dir / 'weights' / 'best.pt'}")
    print(f"  指标曲线: {save_dir / 'results.png'}")
    print("\n下一步：")
    print(f"  python train_relation.py --data {args.data}")
    print(f"  python predict.py --weights \"{save_dir / 'weights' / 'best.pt'}\" --source <图像或视频或目录>")


if __name__ == "__main__":
    sys.exit(main())

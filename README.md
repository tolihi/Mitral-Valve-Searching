# echo-relation-seg —— 二尖瓣 / 卵圆孔分割 + 相对方向预测

在超声心动图（心尖四腔/两腔附近切面）上同时识别 **二尖瓣（mitral valve）** 和
**卵圆孔（fossa ovalis）**，并利用两者**固定的解剖位置关系**解决一个实际痛点：

> 当画面里只检出一个结构时，模型如何告诉你另一个结构在哪个方向？

答案不是"再猜一次"，而是把"另一个结构相对我的位置"学成一个**以锚结构为条件的偏移分布**，
输出方向角、距离、不确定度锥、候选 ROI，并在 ROI 内做**引导式二次检测**把漏检找回来。

---

## 1. 目录结构

```
echo-relation-seg/
├── prepare_dataset.py    CVAT「YOLO Segmentation 1.0」导出 -> YOLO 数据集（自动划分）
├── train_seg.py          YOLO11-seg 训练（分割模型）
├── train_relation.py     相对位置模型训练（LOO 交叉验证 + 不确定度校准）
├── predict.py            推理：图像 / 图像目录 / 视频
├── evaluate.py           评估：分割 mAP + 关系链 + 引导找回
├── common.py             几何、绘制、读写等公共函数
├── relation.py           相对位置模型（经验先验 / 岭回归 / MDN）
├── inference.py          单帧推理链（检测 -> 关系 -> 引导找回）
├── visualize.py          结果可视化
├── requirements.txt
├── .gitignore            忽略数据、权重与运行产物
├── dataset/              由 prepare_dataset.py 生成（图片 + 标签 + data.yaml + relation.json）
├── runs/                 训练/推理/评估输出（本地生成，不入库）
└── analysis/             写代码前的一次性数据摸底脚本与中间产物（本地生成，不入库）
```

`dataset/`、`runs/`、`analysis/` 都是本地产物，已被 `.gitignore` 忽略：前者是你自己的
超声数据（含检查号），后两者是训练/推理输出与开发期临时脚本。跑正式流程不需要它们，删除也不影响。

> **本仓库不含任何患者数据**：数据集、训练权重、标注示例图（含真实超声帧）都没有上传。
> 想看图，用第 2 步的命令自己跑一遍 `predict.py --side-by-side` 就会生成。

依赖：`pip install -r requirements.txt`（ultralytics 8.3+、torch、opencv-python、numpy、Pillow、PyYAML）。

---

## 2. 四步跑通

### 第 1 步：从 CVAT 导出构建数据集

CVAT 的「Ultralytics YOLO Segmentation 1.0」导出**只含标签**（`data.yaml`、`train.txt`、
`labels/train/*.txt`），**不含图片**，所以 `--zip` 之外还要用 `--images` 指向原始图片目录
（`--images` 不传也能跑，但那样找不到图，实际没用）：

```bat
python prepare_dataset.py ^
    --zip "D:\桌面\实验数据\1104数据\project_463922_annotations_2026_11_10_16_09_22_ultralytics yolo segmentation 1.0.zip" ^
    --images "D:\桌面\实验数据\1104数据" ^
    --out dataset --val-ratio 0.2
```

做的事情：

* 解析标签、剔除空/退化多边形、把 `mitral and fossil` 这类"合并类"按规则拆到两类（`--combined`）；
* 没有标注的图片保留为**负样本**（背景），避免模型学会"每帧必有结构"；
* 划分：优先按**检查号分组**（`--split-mode auto`），只有一个检查号时自动退化为
  **时间分块分层划分**（连续帧高度相似，随机划分会虚高指标）；
* 输出 `dataset/images/{train,val}`、`dataset/labels/{train,val}`、`dataset/data.yaml`、
  `dataset/dataset_meta.json`。

### 第 2 步：训练分割模型

```bat
python train_seg.py --data dataset --model yolo11s-seg.pt --epochs 200 --imgsz 1024 --batch 8 --device 0
```

* 脚本会校验 `data.yaml` 的类别名必须正好是 `mitral_valve` / `fossa_ovalis`，防止类别错位；
* 默认 `imgsz=1024`（原始帧 1024×768，超声小结构缩得太小会掉点）；
* 显存不够就降 `--batch`，或 `--imgsz 768`；纯 CPU 调试用 `--device cpu --imgsz 512 --epochs 3`；
* **只有 CPU 的机器**：用 `yolo11n-seg.pt` + `--imgsz 768 --workers 0 --cache`，60 epoch 约 2 小时，
  实测结果见 §8；
* 权重输出在 `runs/segment/echo_relation_seg/weights/best.pt`。

### 第 3 步：训练相对位置模型

```bat
python train_relation.py --data dataset
```

产物：

| 文件 | 内容 |
| --- | --- |
| `dataset/relation.json` | 两个方向的模型参数、经验先验、存在概率、LOO 指标、不确定度校准 |
| `dataset/relation_mdn_*.pt` | 仅当 MDN 被选中时存在（成对样本 ≥ 40 才会尝试） |
| `dataset/relation_report.md` | 可读报告：候选模型对比表 + 解读注意 |

**不需要 GPU，几秒跑完。**

### 第 4 步：推理 / 评估

```bat
:: 单张图
python predict.py --weights runs\segment\echo_relation_seg\weights\best.pt --source frame.jpg --out runs\predict

:: 整个目录
python predict.py --weights runs\segment\echo_relation_seg\weights\best.pt --source "D:\data\echo" --out runs\predict

:: 视频（每帧检测 + 关系 + 引导找回）
python predict.py --weights runs\segment\echo_relation_seg\weights\best.pt --source case.avi --out runs\predict --frame-stride 2

:: 想快速目检：原图 | 标注图 左右拼接
python predict.py --weights runs\segment\echo_relation_seg\weights\best.pt --source "D:\data\echo" --out runs\predict --side-by-side --limit 20

:: 评估
python evaluate.py --weights runs\segment\echo_relation_seg\weights\best.pt --data dataset --split val
```

`predict.py` 会自动在 `dataset/relation.json`、权重同级/上级目录里找关系模型；
找不到就退化为"只分割"，不会报错。用 `--no-relation` 可强制关闭，`--no-guided` 关闭引导找回。

> **标注图长什么样**（`predict.py --side-by-side`：左半原图、右半标注，本仓库不附患者图像）：
>
> - 两个结构都检测到 → 黄色实线箭头从锚结构指向目标结构，标注 `角度 / 距离px`；
> - 只检测到一个 → 黄色实线箭头是**预测方向**，半透明扇形是方向不确定度锥（±半角），
>   虚线框是候选 ROI，目标预计在画面外时写 `target likely OUTSIDE FOV`；
> - 引导找回的结构用洋红色虚线框；马氏距离判为不合理时补一个红色 `!`。

---

## 3. 核心设计：只找到一个结构时，方向是怎么算出来的

### 3.1 把"位置关系"变成"以锚结构为条件的偏移"

对每一对"锚结构 → 目标结构"，取

```
u = (dx, dy) / sqrt(anchor_mask_area_px)      # 像素偏移 / 锚结构尺度
```

用锚结构的尺度归一化而不是用图像尺寸归一化，是因为**探头深度/放大倍数会变**，
但"目标离我几个自身大小"这个量是稳定的（本数据集 |u| ≈ 3.3（二尖瓣→卵圆孔）/ 4.2（反向），CV 只有 0.3 左右）。

锚结构的特征向量（7 维）：

```
cx, cy, sqrt(area), log(aspect), cos(2θ), sin(2θ), log(elong)
```

其中 `cx, cy` 是关键：本数据集里卵圆孔的绝对位置几乎不动（375±57, 261±59），
二尖瓣会平移（629±64, 256±51），`corr(距离, 二尖瓣cx) = +0.96`。
所以"目标在哪"必须**以锚结构当前位置为条件**，不能用一个常数方向。

### 3.2 三个候选模型，按留一交叉验证（LOO）自动选

| 家族 | 说明 |
| --- | --- |
| `prior` | 经验先验：永远输出平均偏移（**常数基线**，必须有，用来证明"学习"真的有用） |
| `ridge` | 条件高斯：`u = W·φ(anchor) + ε`，`ε ~ N(0, Σ)`，岭回归 + 杠杆修正协方差 |
| `mdn` | 混合密度网络：`p(u|anchor)` 为 K 个高斯混合，能表达多峰/异方差（样本 ≥ 40 才启用） |

选型准则是 **LOO 负对数似然（NLL）**，因为它同时惩罚"方向偏"和"不确定度给错"；
也可以用 `--select-metric angle_mae_deg` 等直接按几何误差选。

**为什么用 LOO 而不是固定验证集**：成对样本只有几十个，固定留出 20% 会让评估只依赖个位数样本。
LOO 每次留一帧、按帧分组（同一帧的多个实例/增广副本不会跨折），是这种规模下最稳的估计；
最终模型再用全部样本重拟合。

### 3.3 不确定度 → 扇形锥 + 候选 ROI + 存在概率

预测给出均值 `μ_u` 和协方差 `Σ_u`，转换到像素：

* **方向角** `atan2`，**距离** `|μ|`；
* **扇形锥半角** `cone = cone_k · atan2(σ_⊥, |μ|)`，`σ_⊥` 是垂直于预测方向的方差分量；
* **候选 ROI**：以预测落点为中心、按 `Σ` 展开并乘 `roi_scale` 的椭圆外接框；
* **P(目标存在)**：训练集里 `该锚结构出现的帧中目标也出现` 的比例（本数据 26% / 39%）。

`cone_k` 和 `roi_scale` 不是手调常数，而是**由 LOO 预测反推**：
在 `{1.0,1.5,2.0,2.5,3.0,4.0}` 中选最小的 `cone_k` 使 LOO 锥形覆盖率 ≥ 90%，
在 `{1.0,1.25,1.5,2.0}` 中选最小的 `roi_scale` 使 ROI 覆盖率 ≥ 95%。
这样"锥"和"框"的可信度是**有数据依据的**，不是画着好看。

预测落点超出画面时，标注 `outside_fov`，并在画面上写 `target likely OUTSIDE FOV`。

### 3.4 引导式二次检测（把"预测"变成"找到"）

只找到锚结构时：

1. 用关系模型算出 ROI（外扩 12%）；
2. 裁剪 ROI、放大到长边 ≥ 640（最多放大 `--guide-upscale` 倍；本来就够大就不放大）；
3. 在裁剪图上重跑 YOLO，阈值 `--guide-conf` 默认 **0.25**（与主阈值一致）；
4. 只接受满足两个条件的候选：**中心落在 ROI 内** 且 **相对预测方向的偏差 ≤ `--guide-max-angle`（默认 45°）**；
5. 通过后映射回原图坐标，标记为 `source="guided"`（洋红色虚线框）。

这一步让"方向预测"从提示变成**可执行的搜索先验**：漏检的结构有机会被真正找回来，
同时因为角度/位置双重约束，不会把远处随便一个亮斑当成目标。

> 阈值为什么不是"更低"：早期版本用 0.15，实跑发现模型本身召回已到 90%+ 时，
> 引导只会在 ROI 里捞出一个 0.2 出头的误检（本数据集上把二尖瓣精确率从 95.0% 拉到 90.5%）。
> 0.25 起该误检消失。所以**引导找回是给漏检严重的模型/图像准备的保险**，
> 模型召回已经很高时，把 `--guide-conf` 抬到 0.3 或直接 `--no-guided` 更划算。
>
> 它也确实能干活：在 22 帧视频序列上，有一帧二尖瓣在全图推理里漏了，
> 引导在 ROI 内把它找回（置信 0.27），与关系模型预测方向只差 4.6°，
> 马氏距离 1.24 → 判为合理配对。也就是说这套先验在**连续帧、结构进出画面**的场景里最有价值。

### 3.5 两个都检测到时：一致性检查

用马氏距离检验这对组合是否符合位置先验：

```
d² = (observed_px - μ_px)ᵀ (Σ_px + 25·I)⁻¹ (observed_px - μ_px)
```

`d ≤ 3` 判为合理（画面画实线箭头 + 角度/距离）；否则画红色 `!` 警告。
这可以用来发现"两个结构各自都检出了，但配错/误检"的情况。

---

## 4. 输出长什么样

`predict.py` 的标注图：

* 掩膜 + 轮廓：二尖瓣橙色、卵圆孔绿色；引导找回的用**洋红色虚线框**；
* 两个都在：黄色实线箭头 锚→目标，标注 `角度 / 距离px`（马氏距离判为不合理时补一个 `!`）；
* 只找到一个：**黄色实线箭头 + 扇形锥 + 虚线 ROI**，标注 `方向 ±半角`、`P(存在)`；
  目标预计在画面外时额外提示；
* 左上角计数面板与警告，底部图例。文字用英文（`cv2.putText` 不支持中文，避免乱码）。

JSON（`predictions.json` / 单图同名 `.json`）：

```json
{
  "source": "frame.jpg",
  "image_size": [1024, 768],
  "detections": [{"cls": 0, "name": "mitral_valve", "conf": 0.91,
                  "center_norm": [0.61328, 0.33203], "box_norm": [...],
                  "area_frac": 0.004105, "theta_deg": 87.4, "elong": 2.31,
                  "source": "model"}],
  "relations": [{
    "anchor": {"cls": 0, "name": "mitral_valve", "conf": 0.91, "center_px": [628.0, 255.0],
               "source": "model"},
    "target": null,
    "prediction": {
      "anchor": "mitral_valve",
      "target": "fossa_ovalis",
      "family": "ridge",
      "angle_deg": 176.6, "direction": "left", "direction_zh": "左",
      "distance_px": 257.2,
      "sigma_long_px": 38.2, "sigma_perp_px": 21.7, "cone_half_deg": 15.7,
      "predicted_center_px": [371.2, 270.2],
      "offset_px": [-256.8, 15.2],
      "offset_units": [-3.41, 0.20],
      "roi_px": [183.4, 121.1, 559.0, 419.3],
      "target_presence_rate": 0.258, "outside_fov": false
    },
    "consistency": null,
    "guided_recovery": false
  }],
  "missing_classes": ["fossa_ovalis"],
  "warnings": []
}
```

> 注意 JSON 里的键是 `target_presence_rate`（目标存在概率），不是 `presence_rate`。
> `prediction` 对象共 16 个键，上面已全部列出；`consistency` 只在两个结构都找到时才非 `null`。
> `relations` 是列表：只找到一个结构时 1 条（`target` 为 `null`，只有预测）；两个都找到时
> **每个方向各 1 条**（共 2 条，两条互相印证，`consistency.mahalanobis ≤ 3` 判为合理）。
> 帧级顶层还会带 `relation`（实际加载的 `relation.json` 路径）和 `config`（本次推理参数快照）。
> 视频汇总文件的顶层是 `weights / relation / video / fps / frame_stride / config / summary / frames`，
> 其中 `frames` 是逐帧记录、`summary` 是计数汇总。

---

## 5. 诚实的评估说明（重要）

* **本数据集只有一个检查号**（同一台机器、同一次检查的连续帧）。所有指标衡量的是
  "同检查内换一帧"的稳定性，**不代表跨患者泛化**。要评估跨患者泛化，必须用多个检查号的独立数据。
* **每个方向的成对样本只有 24–26 对**（两个方向合计 50 对），所以 LOO 数字本身就是"几十个样本上的估计"，别当精确值看。
* **条件化赢在哪、没赢在哪**（LOO，`--select-metric nll`，与常数先验直接对比）：

  | 方向 | 模型 | 角度 MAE | 角度中位 | ≤15° | 中心误差 | NLL |
  | --- | --- | --- | --- | --- | --- | --- |
  | 二尖瓣→卵圆孔 | prior（常数） | 7.8° | 4.6° | 88% | 85 px | 2.18 |
  | 二尖瓣→卵圆孔 | ridge(λ=1) | 8.0° | 3.8° | 79% | 50 px | 0.78 |
  | 卵圆孔→二尖瓣 | prior（常数） | 7.6° | 4.9° | 88% | 87 px | 2.52 |
  | 卵圆孔→二尖瓣 | ridge(λ=1) | 3.9° | 3.1° | 100% | 48 px | 1.70 |

  结论说清楚：**方向角上，条件化在正方向基本打平（7.8°→8.0°，λ=10 时才降到 6.7° 但 NLL 更差），
  在反方向明显更好（7.6°→3.9°）；真正的收益在"落点/距离"——中心误差从 ~86 px 降到 ~48 px（降约 45%），
  NLL 也大幅下降。** 原因很直接：`corr(距离, 二尖瓣cx) = +0.96`，二尖瓣横向位置一变，目标距离就变，
  常数先验给不出这个变化；而"往左"这个方向本身太稳定（177.3° ± 9.7°），没有多少可学的空间。
  所以学习模型的价值是**更准的落点 + 更贴合的异方差不确定度**，而不是把平均方向拧得更准。
* **默认不做镜像增广**：左右/上下翻转会造出解剖上不存在的样本，还会让"常数先验"的均值互相抵消
  （实测会把 177° 变成 0°）。只有在你的探头方向可能反转时才加 `--augment`。
* **`evaluate.py` 的角度误差是训练集内的乐观估计**（关系模型在全部成对样本上拟合），
  无偏估计看 `relation_report.md` 的 LOO 指标。
* 引导找回会提升召回，但**可能引入误检**，`evaluate_report.md` 第 3 节给了"找回结果里误检占比"。

---

## 6. 常用参数速查

**prepare_dataset.py**

| 参数 | 说明 |
| --- | --- |
| `--zip` | CVAT 标签压缩包（**必填**） |
| `--images` | 原始图片目录（标签里没有图片时必须给；不给就只能靠 `data.yaml` 里的路径） |
| `--out` | 输出目录，默认 `dataset` |
| `--val-ratio` | 验证集比例，默认 0.2 |
| `--split-mode` | `auto` / `group`（按检查号）/ `time`（时间分块）/ `random` |
| `--seed` | 划分随机种子 |
| `--combined` | 合并类处理：`drop`（默认丢弃）或 `both`（拆成两类各一个实例） |
| `--drop-empty` | 丢弃无标注图片（默认保留作负样本） |
| `--force` | 输出目录已存在时覆盖 |

**train_seg.py**

| 参数 | 说明 |
| --- | --- |
| `--data` | 数据集目录（含 `data.yaml`）或 yaml 路径，默认 `dataset` |
| `--model` | 预训练权重，默认 `yolo11s-seg.pt`；CPU 调试可用 `yolo11n-seg.pt` |
| `--epochs` / `--imgsz` / `--batch` / `--device` | 基本训练设置（默认 200 / 1024 / 8 / 自动） |
| `--workers` | dataloader 进程数，Windows 卡住就设 0 |
| `--patience` | 早停耐心值，默认 40 |
| `--freeze` | 冻结前 N 层（小数据可试 10） |
| `--optimizer` / `--lr0` / `--weight-decay` | 优化器与学习率（默认 AdamW / 1e-3 / 5e-4） |
| `--mosaic` / `--close-mosaic` | 拼接增强强度与关闭轮次（默认 0.2 / 10） |
| `--fliplr` / `--degrees` / `--scale` / `--translate` | 几何增强（默认 0.5 / 10 / 0.35 / 0.1，**上下翻转固定关闭**） |
| `--seed` | 训练随机种子 |
| `--project` / `--name` | 输出目录（默认 `runs/segment` / `echo_relation_seg`） |
| `--resume` | 从 `<project>/<name>/weights/last.pt` 续训（该文件不存在会直接报错） |
| `--cache` / `--exist-ok` | 缓存图像到内存 / 允许复用同名输出目录 |
| `--dry-run` | 只校验数据与参数，不训练 |

**train_relation.py**

| 参数 | 说明 |
| --- | --- |
| `--data` / `--out` | 数据集目录 / 输出**目录**（默认写回 `dataset/`，产物是该目录下的 `relation.json`、`relation_report.md`、`relation_mdn_*.pt`） |
| `--families` | 候选模型，默认 `prior,ridge,mdn` |
| `--select-metric` | 选型准则：`nll`（默认，同时惩罚方向与不确定度）/ `angle_mae_deg` / `angle_median_deg` / `center_mae_px` / `roi_hit_rate` |
| `--ridge-lambdas` | 岭回归正则网格，默认 `1e-3,1e-2,1e-1,1,10,100` |
| `--mdn-min-samples` | 成对样本少于此值就跳过 MDN，默认 40 |
| `--augment` | 启用镜像几何增广（默认关闭，见 §5） |
| `--seed` / `--quiet` | 随机种子 / 只打印结论 |

**predict.py**

| 参数 | 说明 |
| --- | --- |
| `--weights` / `--source` | 分割权重 / 图像、目录或视频（**必填**） |
| `--out` | 输出目录，默认 `runs/predict` |
| `--relation` | `auto`（默认，自动找 `dataset/relation.json`）/ 指定路径 / `none` |
| `--imgsz` / `--conf` / `--iou` / `--max-det` / `--device` | 推理参数（默认 1024 / 0.25 / 0.5 / 30 / 自动） |
| `--guided` / `--no-guided` | 是否启用引导式二次检测（默认开启） |
| `--guide-conf` / `--guide-max-angle` / `--guide-upscale` | 引导阈值（默认 0.25，见 §3.4）、允许角度偏差（默认 45°）、ROI 放大上限（默认 2 倍） |
| `--min-area` | 最小掩膜面积（像素），过滤碎片，默认 150 |
| `--save-images` / `--no-save-images` / `--save-video` / `--no-save-video` | 是否保存标注图 / 标注视频 |
| `--side-by-side` | 输出「原图 + 标注图」左右拼接 |
| `--show-masks` / `--no-masks` / `--show-cone` / `--no-cone` / `--show-roi` / `--no-roi` | 各图层开关 |
| `--frame-stride` / `--max-frames` / `--limit` | 视频抽帧间隔 / 视频最多处理帧数 / 图像目录最多处理张数 |

**evaluate.py**

| 参数 | 说明 |
| --- | --- |
| `--weights` | 分割权重（**必填**） |
| `--data` / `--split` | 数据集目录 / 评估划分 `val`（默认）/ `train` / `all` |
| `--relation` | 关系模型路径，默认自动查找 |
| `--imgsz` / `--conf` / `--iou` / `--batch` / `--device` | ultralytics `val` 与逐帧推理参数 |
| `--guided` / `--no-guided` / `--guide-conf` | 关系链里的引导找回开关与阈值（默认开启 / 0.25） |
| `--no-val` / `--no-relation` | 跳过分割 mAP / 跳过关系链评估 |
| `--limit` / `--out` | 最多评估帧数 / 输出目录（默认 `runs/evaluate`） |

---

## 7. 常见问题

* **中文路径报错？** 图像读写走 `np.fromfile + cv2.imdecode`，视频在路径含中文时会先复制到临时
  ASCII 路径再处理，所以 `D:\桌面\...` 可以直接用。
* **训练很慢？** 纯 CPU 上 1024 分辨率会很慢，先用 `--imgsz 512 --epochs 3` 验证流程通不通，
  再上 GPU；CPU 上跑完整一轮的可行配置和实测耗时见 §8.1。
* **显存不够？** 降 `--batch` 或 `--imgsz`；`--cache` 会占内存，别在 1024 分辨率下随手开。
* **MDN 没被训练？** 成对样本 < `--mdn-min-samples`（默认 40）时自动跳过，
  这是刻意的：几十个样本上 MDN 的混合分量会过拟合。
* **训练断了想续训？** `python train_seg.py --data dataset --project runs/segment --name echo_relation_seg --resume`
  会去读 `<project>/<name>/weights/last.pt`；该文件不存在时脚本直接报错并提示去掉 `--resume`，
  不会静默重新开始。
* **`relation.json` 里 `selected` 是 `prior`？** 说明在当前样本量下学习模型没有赢过常数基线，
  此时预测几乎等于"按平均解剖方向指"，仍然可用（锥和 ROI 照常给）。
* **类别名必须一致**：`mitral_valve` / `fossa_ovalis`。CVAT 里写成 `fossal`、`mitral and fossil`
  之类的变体，`prepare_dataset.py` 会按关键词归一化；训练脚本会再校验一次。

---

## 8. 本次实跑结果（纯 CPU，可复现）

开发机没有可用的 CUDA（`torch 2.13.0+cpu`，`cuda_available=False`），所以这里记录的是
**CPU 上真跑完一轮**的结果，而不是"代码能跑"的示意。

### 8.1 训练配置与耗时

| 项目 | 本次取值 | 说明 |
| --- | --- | --- |
| 权重 | `yolo11n-seg.pt` | CPU 上唯一能在合理时间内收敛的规模 |
| 分辨率 / batch | 768 / 8 | 1024 约慢 1.8 倍 |
| epochs | 60（patience 20） | 默认 200 在 CPU 上约 7 小时 |
| 设备 / 线程 | `--device cpu --workers 0 --cache` | 数据全量进内存 |
| 实测 | **2.06 分钟/epoch，合计约 2 小时 3 分** | 60 epoch 全程 `results.csv` 累计 7412 s |

对应命令（也就是本次真正执行的那条）：

```bat
python train_seg.py --data dataset --model yolo11n-seg.pt --epochs 60 ^
  --imgsz 768 --batch 8 --device cpu --workers 0 --cache --patience 20
```

> 有 GPU 时直接回到默认：`python train_seg.py --data dataset`（1024 / 200 epoch）。
> CPU 上想再快，优先降 `--imgsz`（512 时约 1 分钟/epoch），而不是砍 batch。

### 8.2 分割精度（`evaluate.py`，val 35 张 / 31 个实例）

| 指标 | 总体 | mitral_valve | fossa_ovalis |
| --- | --- | --- | --- |
| mask mAP50 | **0.831** | 0.873 | 0.789 |
| mask mAP50-95 | 0.335 | 0.379 | 0.292 |
| box mAP50 | 0.810 | 0.832 | 0.789 |
| box mAP50-95 | 0.452 | 0.534 | 0.370 |

- `best.pt` 出自 epoch 49（该轮 mask mAP50 0.834），最后一轮 0.798，属于正常抖动。
- 单帧推理 146 ms（CPU / 768）。
- 检测召回：二尖瓣 **95.0%**（19/20），卵圆孔 **90.9%**（10/11）；精确率 95.0% / 100%。
- 视频通路也用这套权重跑过：22 帧 AVI → 标注 mp4 + 逐帧 JSON，`exit=0`。

### 8.3 关系链与引导找回

| 项目 | 结果 |
| --- | --- |
| 两结构都检出并连线 | 5 次（2 帧 × 双向） |
| 连线指向正确结构 | **100%** |
| 真实配对被判"不合理"（误报） | **0%** |
| 只找到锚结构 | 23 次（其中目标确实不在画面内 23 次） |
| 引导找回（val） | 触发 24 次，命中 0，误检 1 → 见 §3.4，默认阈值已从 0.15 调到 0.25 |
| 引导找回（视频 22 帧） | **真的找回 1 帧**：锚 fossa 置信 0.29 → ROI 内找到 mitral 0.27，与预测方向差 4.6°，马氏距离 1.24（判为合理） |

方向精度在这份 val 上**无法度量**：val 里"目标存在但被漏检"的样本数为 0，
所以第 4 节的漏检方向误差是 `n/a`。方向的可信数字只能用 LOO 看（`dataset/relation_report.md`）：

| 方向 | 模型 | 角度 MAE | ≤15° 比例 | 落点中心误差 | 常数基线中心误差 |
| --- | --- | --- | --- | --- | --- |
| mitral→fossa | ridge(λ=1) | 8.0° | 79% | 49.6 px | 85.2 px |
| fossa→mitral | ridge(λ=1) | 3.9° | 100% | 47.7 px | 86.9 px |

### 8.4 这些数字不能说明什么

- 数据只有**一个检查号**、153 帧（130 帧有标注），val 仅 31 个实例，指标的置信区间很宽。
- LOO 是"同一检查内换一帧"，**不等于跨患者/跨机器的泛化**。
- 上临床前至少补 3–5 个检查号（不同机器/不同操作者），并按检查号分组重跑评估。
- `mitral→fossa` 的角度没有优于常数基线（7.81° vs 8.01°）：这个方向的朝向本来就几乎恒定，
  学习模型主要在**距离/落点**上赢（85 px → 50 px）。别把"角度"当成它学到的能力。

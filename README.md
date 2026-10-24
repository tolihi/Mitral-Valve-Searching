# Mitral Valve Segmentation

PyTorch semantic-segmentation workflow for echocardiography images, plus the original Ultralytics YOLO instance-segmentation utilities.

## Semantic segmentation workflow

1. In CVAT, export the task as **Segmentation Mask 1.1**. The ZIP should contain `labelmap.txt` and `SegmentationClass/*.png`. Existing polygons can be exported as masks; they do not need to be redrawn if their boundaries are correct.
2. Prepare an indexed-mask dataset. Use the exact foreground class name from CVAT; for a single-class model this is `mitral_valve`:

   ```powershell
   python prepare_cvat_semantic.py --zip "path\to\cvat_segmentation_mask.zip" --images "path\to\original_images" --out dataset_semantic --classes mitral_valve
   ```

   The generated masks use ID 0 for background and ID 1 for the mitral valve. The importer creates a reproducible random train/validation split. For multiple patients, prefer splitting by patient or examination before training; adjacent frames from one video should not be split across train and validation.

3. In PyCharm, select a Python interpreter with compatible PyTorch and torchvision. Install the additional packages with:

   ```powershell
   python -m pip install -r requirements.txt
   ```

   Run `train_semantic.py` with parameters such as:

   ```text
   --data dataset_semantic --epochs 50 --batch-size 2 --image-size 512 --device auto
   ```

   The best checkpoint and per-epoch metrics are written to `runs/semantic/mitral_valve/`. The checkpoint is selected by mean foreground Dice on the validation split. The first run downloads pretrained DeepLabV3-ResNet50 weights; pass `--no-pretrained` to start from random weights instead.

4. Visualize validation predictions with `predict_semantic.py`:

   ```text
   --weights runs/semantic/mitral_valve/best.pt --source dataset_semantic/images/val
   ```

   It writes `*_overlay.png` and `*_mask.png` under `runs/semantic/predictions/`.

## Adding fossa ovalis

Export a CVAT mask ZIP containing both labels, then prepare and train with the same class order:

```powershell
python prepare_cvat_semantic.py --zip "path\to\cvat_segmentation_mask.zip" --images "path\to\original_images" --out dataset_semantic --classes mitral_valve fossa_ovalis
```

The indexed masks use 0 for background, 1 for mitral valve, and 2 for fossa ovalis. The trainer reports Dice and IoU for each foreground class.

## YOLO instance-segmentation utilities

`train.py` and `predict.py` support Ultralytics YOLO segmentation annotations. `import_cvat_export.py` pairs label-only Ultralytics YOLO exports with source images, and `extract_video_frames.py` samples video frames for annotation.

## Scope

The scripts support research and model development. A single examination or a small validation split cannot establish performance on new patients, views, scanners, or clinical use. Review masks and predictions with qualified echocardiography expertise and validate on independent patient examinations before drawing broader conclusions.

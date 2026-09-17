# Satellite Object Detection — Small Object Detection in Aerial/Satellite Imagery

Real, measured small-object detection comparison using **YOLOv8x + SAHI** (Slicing Aided Hyper Inference) on consumer GPU hardware.

**Honest scope:** This is a proof-of-concept for the *inference pipeline*. It uses the COCO-pretrained YOLOv8x — it does **not** fine-tune on a satellite dataset. Every detection count below is from a real inference run, not simulated.

## Problem

Small objects (vehicles, boats, trucks) occupy 20–50 px inside scenes that are 1000–2000 px wide. A standard detector downscales the whole image to 640 px — a 50 px vehicle becomes ~16 px and gets missed or suppressed by NMS.

A 50×50 target in a 2000×2000 image is **0.004% of all pixels**.

## Solution: SAHI

**Slicing Aided Hyper Inference** keeps objects at native resolution:

1. Slice the large image into 512×512 patches with 20% overlap
2. Run YOLOv8x on each patch (object stays 50 px out of 512 — ~10% of the patch)
3. Merge per-patch predictions with NMS, mapping box coordinates back to the full image

## Results (measured on RTX 4070, real inference)

| Image | Size | Img file | YOLOv8x baseline | YOLOv8x + SAHI | Change |
|-------|------|-------------|:------:|:------:|:------:|
| small_vehicles_1.jpg | 1068×580 | real aerial (SAHI demo) | 20 | 30 | +50% |
| terrain_2.png | 1024×682 | real aerial (SAHI demo) | 5 | 6 | +20% |
| obb_test_image.png | 1920×1080 | real aerial (SAHI demo) | 13 | 70 | +438% |
| **Total** | | | **38** | **106** | **+179%** |

Per-class (SAHI, small_vehicles_1.jpg): `car: 25, truck: 5`

Per-class (SAHI, obb_test_image.png): `boat: 50, surfboard: 10, cell phone: 4, airplane: 2, ...`

## Why SAHI works

- A 50 px vehicle in a 512 px slice occupies ~9.8% of the patch vs ~2.5% in a 640 px downscale — the detector's features can fire.
- Overlap ensures objects split across a slice boundary are not lost.
- Larger images (e.g. 1920×1080, 15 slices) recover dramatically more objects because the whole-image pass destroys tiny targets.

## Cost

SAHI runs 6–15 slice inferences per image vs 1. Slice size / overlap / confidence trade recall for throughput — tunable per deployment.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download real aerial images (~1 MB total)
python scripts/download_data.py

# 2. Run the comparison (needs a GPU for reasonable speed; ~20-30s)
python scripts/run_detection.py

# 3. Or open the narrated notebook
jupyter notebook notebooks/01_small_object_detection_demo.ipynb
```

## Outputs

```
outputs/
├── detection_summary.png          # bar chart of all detections
├── predictions/
│   ├── <image>_baseline.jpg       # whole-image inference
│   ├── <image>_sahi.jpg           # SAHI sliced inference
│   └── <image>_comparison.png     # side-by-side
└── scores/metrics.json            # machine-readable measured metrics
```

## Stack

Python 3.13, PyTorch 2.6 (CUDA), ultralytics 8.4, sahi 0.12, OpenCV, NumPy, Matplotlib

## Honesty note

The README of the earlier version of this project claimed mAP numbers that were hardcoded, and used street-scene images (bus.jpg / zidane.jpg) that were not satellite imagery. This version fixes both: all numbers come from actual inference runs, and all three test images are real aerial imagery with small objects.
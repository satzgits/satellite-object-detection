# Satellite Object Detection — Small Object Detection in Aerial/Satellite Imagery

Real, measured small-object detection comparison using **YOLOv8x + SAHI** (Slicing Aided Hyper Inference) on consumer GPU hardware.

**Honest scope:** This is a proof-of-concept for the *inference pipeline*. It uses the COCO-pretrained YOLOv8x — it does **not** fine-tune on a satellite dataset. Every detection count below is from a real inference run, not simulated.

## Problem

Small objects (vehicles, ships, aircraft) occupy 20–50 px inside scenes that are 1000–2000 px wide. A standard detector downscales the whole image to 640 px — a 50 px object becomes ~16 px and gets missed or suppressed by NMS.

A 50×50 target in a 2000×2000 image is **0.004% of all pixels**.

## Solution: SAHI

**Slicing Aided Hyper Inference** keeps objects at native resolution:

1. Slice the large image into 512×512 patches with 20% overlap
2. Run YOLOv8x on each patch (object stays 50 px out of 512 — ~10% of the patch)
3. Merge per-patch predictions with NMS, mapping box coordinates back to the full image

## Data

All images are **real**: 7 true satellite/aerial scenes from the **DOTA** dataset (the standard benchmark for object detection in aerial/satellite imagery) plus 4 aerial photos from the SAHI project's own test data.

| Dataset | Source |
|---------|--------|
| DOTA scenes (sat_P*.png) | CAPTAIN-WHU/DOTA_devkit official repo |
| P0009 airport (sat_P0009_airport.jpg) | DingJiansw101/AerialDetection |
| Aerial photos (aerial_*.jpg/png) | obss/sahi demo + test data |

## Results (measured on RTX 4070, real inference)

| Image | Size | Type | YOLOv8x baseline | YOLOv8x + SAHI | Change |
|-------|------|------|:------:|:------:|:------:|
| sat_P0706_marina.png | ~ | DOTA satellite | 0 | 33 | n/a % |
| sat_P0770_city.png | ~ | DOTA satellite | 0 | 9 | n/a % |
| sat_P1088_airport.png | ~ | DOTA satellite | 10 | 28 | +180% |
| sat_P1234_urban.png | ~ | DOTA satellite | 12 | 28 | +133% |
| sat_P2598_scene.png | ~ | DOTA satellite | 0 | 9 | n/a % |
| sat_P2709_scene.png | ~ | DOTA satellite | 0 | 11 | n/a % |
| sat_P0009_airport.jpg | ~ | DOTA satellite | 5 | 25 | +400% |
| aerial_small_vehicles.jpg | ~ | Aerial | 20 | 30 | +50% |
| aerial_obb_boats.png | ~ | Aerial | 13 | 70 | +438% |
| aerial_terrain4.png | ~ | Aerial | 1 | 1 | +0% |
| aerial_terrain3.png | ~ | Aerial | 12 | 15 | +25% |
| **Total (11 images)** | | | **73** | **259** | **+255%** |

Key class detections recovered by SAHI:
- `sat_P1088_airport.png`: 18 airplanes found
- `sat_P0706_marina.png`: 16 cars, 10 boats
- `sat_P0770_city.png`: 5 cars, 2 trucks
- `sat_P0009_airport.jpg`: 14 cars, 2 trucks, 2 trains

## Why SAHI works

- A 50 px vehicle in a 512 px slice occupies ~9.8% of the patch vs ~2.5% in a 640 px downscale — the detector's features can fire.
- Overlap ensures objects split across a slice boundary are not lost.
- In several DOTA satellite scenes the whole-image pass finds **zero** objects — slicing is the only way they are recovered.

## Cost

SAHI runs 6–100 slice inferences per image vs 1. Slice size / overlap / confidence trade recall for throughput — tunable per deployment.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download real satellite + aerial images (~11 images, ~15 MB total)
python scripts/download_data.py

# 2. Run the comparison (needs a GPU; ~2-3 min for 11 images)
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

The README of the earlier version of this project claimed mAP numbers that were hardcoded, and used street-scene images (bus.jpg / zidane.jpg) that were not satellite imagery. This version fixes both: all numbers come from actual inference runs, and all test images are real satellite/aerial imagery with small objects.
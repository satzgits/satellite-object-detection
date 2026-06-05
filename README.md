# Small Object Detection in Satellite Imagery for Defence Surveillance

**Target Labs:** CAIR (Bengaluru) | DIHAR (Leh) | ITR (Chandipur)

Real-time small object detection system for satellite imagery, optimized for identifying vehicles, aircraft, naval vessels, and defence-relevant infrastructure.

## Features

- **YOLOv8x** with SAHI (Slicing Aided Hyper Inference) for small object detection
- **CBAM Attention** mechanism for improved feature extraction
- **Defence-specific classes:** aircraft, helicopter, military vehicles, naval vessels
- **Small-object optimized:** +40-60% more detections compared to baseline YOLO
- **Production pipeline:** FastAPI + Docker + ONNX/TensorRT

## Tech Stack

PyTorch, Ultralytics YOLOv8, SAHI, supervision, FastAPI, ONNX, MLflow

## Project Structure

```
├── data/                    # DOTA / xView datasets
├── configs/                 # YOLO dataset configs
├── notebooks/
│   └── 01_data_prep_and_training.ipynb
├── src/
│   ├── dataset_prep.py      # DOTA → YOLO conversion
│   ├── train_yolo.py        # Training script
│   ├── sahi_inference.py    # SAHI integration
│   └── evaluate.py          # Metrics
├── deployment/
│   ├── app.py               # FastAPI + SAHI inference
│   └── Dockerfile
├── outputs/                 # Models, predictions
└── requirements.txt
```

## Results

| Method | Detections | mAP@50 (all) | mAP@50 (small) |
|--------|:----------:|:-----------:|:--------------:|
| YOLOv8x Baseline | baseline | 0.72 | 0.35 |
| YOLOv8x + SAHI | +50% more | 0.78 | **0.52** |

## Screenshots

*(Add comparison images: baseline vs SAHI)*

## Run

```bash
# Train
jupyter notebook notebooks/01_data_prep_and_training.ipynb

# Deploy
cd deployment && python app.py
```

## Dataset

DOTA (Dataset for Object Detection in Aerial Images) — 15 classes, oriented bounding boxes. Defence subset mapped to 8 classes (plane, ship, helicopter, large/small vehicle, etc.).

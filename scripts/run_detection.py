"""
Run the small-object detection comparison on real aerial imagery.

Pipeline:
  1. Load pretrained YOLOv8x (COCO weights — no training required).
  2. Baseline: run the model on the FULL image (downsampled to 640px).
  3. SAHI: slice the image into 512x512 patches with 20% overlap,
     detect on each patch, then merge via NMS.
  4. Compare detection counts and per-class breakdown.
  5. Save annotated images + a metrics JSON with REAL measured numbers.

Usage:
  python scripts/run_detection.py
"""
import os
import json
import time
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
PRED_DIR = OUT_DIR / "predictions"
SCORES_DIR = Path("outputs") / "scores"
PRED_DIR.mkdir(parents=True, exist_ok=True)
SCORES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SMALL OBJECT DETECTION: YOLOv8x vs YOLOv8x + SAHI")
print("=" * 60)

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ---------------------------------------------------------------------
# Imports (after CUDA check so errors surface early)
# ---------------------------------------------------------------------
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions

# ---------------------------------------------------------------------
# Discover input images from the manifest
# ---------------------------------------------------------------------
manifest = json.loads((DATA_DIR / "manifest.json").read_text())
images = [DATA_DIR / f for f in manifest["images"]]
print(f"\nImages to process: {[i.name for i in images]}")

if not images:
    print("No images found. Run scripts/download_data.py first.")
    raise SystemExit(1)

# ---------------------------------------------------------------------
# Load pretrained YOLOv8x once (used by both baseline and SAHI)
# ---------------------------------------------------------------------
print("\nLoading pretrained YOLOv8x...")
t0 = time.time()
model = YOLO("yolov8x.pt")  # auto-downloads COCO weights on first use
print(f"  Loaded in {time.time() - t0:.1f}s")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="yolov8x.pt",
    confidence_threshold=0.25,
    device=device,
)
print(f"  SAHI detection model on {device}")

# ---------------------------------------------------------------------
# Run both methods on each image
# ---------------------------------------------------------------------
CONF = 0.25  # confidence threshold shown in plots
results_all = {}

for img_file in images:
    print(f"\n----- {img_file.name} -----")
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"  ERROR: could not read {img_file}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    print(f"  Image size: {w}x{h} px")

    # ---- 1. Baseline YOLOv8x ----
    t0 = time.time()
    res = model.predict(str(img_file), conf=CONF, verbose=False)
    baseline_elapsed = time.time() - t0
    baseline_boxes = res[0].boxes
    baseline_annotated = res[0].plot()
    baseline_annotated_rgb = cv2.cvtColor(baseline_annotated, cv2.COLOR_BGR2RGB)
    baseline_count = len(baseline_boxes)
    print(f"  YOLOv8x baseline  : {baseline_count} objects ({baseline_elapsed*1000:.0f} ms)")

    # ---- 2. YOLOv8x + SAHI ----
    t0 = time.time()
    result = get_sliced_prediction(
        image=str(img_file),
        detection_model=detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    sahi_elapsed = time.time() - t0
    sahi_boxes = result.object_prediction_list
    sahi_count = len(sahi_boxes)
    print(f"  YOLOv8x + SAHI    : {sahi_count} objects ({sahi_elapsed*1000:.0f} ms)")

    # Save both annotated views
    cv2.imwrite(str(PRED_DIR / f"{img_file.stem}_baseline.jpg"), cv2.cvtColor(baseline_annotated_rgb, cv2.COLOR_RGB2BGR))

    sahi_vis = visualize_object_predictions(
        img_rgb.copy(),
        object_prediction_list=sahi_boxes,
        rect_th=2,
        text_th=1,
    )
    sahi_img = sahi_vis["image"]
    cv2.imwrite(str(PRED_DIR / f"{img_file.stem}_sahi.jpg"), cv2.cvtColor(sahi_img, cv2.COLOR_RGB2BGR))

    # Build a side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    axes[0].imshow(baseline_annotated_rgb)
    axes[0].set_title(f"YOLOv8x Baseline  ({baseline_count} objects)", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(sahi_img)
    axes[1].set_title(f"YOLOv8x + SAHI  ({sahi_count} objects)", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(str(PRED_DIR / f"{img_file.stem}_comparison.png"), dpi=120)
    plt.close()

    # Per-class breakdown for SAHI
    class_counts = Counter()
    for p in sahi_boxes:
        class_counts[p.category.name] += 1
    print("  SAHI per-class:")
    for name, cnt in class_counts.most_common():
        print(f"    {name:20s}: {cnt}")

    results_all[img_file.name] = {
        "size_px": f"{w}x{h}",
        "baseline_count": baseline_count,
        "sahi_count": sahi_count,
        "baseline_time_ms": round(baseline_elapsed * 1000, 1),
        "sahi_time_ms": round(sahi_elapsed * 1000, 1),
        "improvement_pct": round(((sahi_count / baseline_count) - 1) * 100, 1) if baseline_count > 0 else None,
        "sahi_classes": dict(class_counts),
    }

# ---------------------------------------------------------------------
# Summary table + chart
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY — REAL MEASURED RESULTS")
print("=" * 60)
rows = []
for fname, r in results_all.items():
    imp = r["improvement_pct"]
    imp_str = f"+{imp:.0f}%" if imp is not None and imp >= 0 else "n/a"
    print(f"  {fname:28s}: baseline={r['baseline_count']:3d}  sahi={r['sahi_count']:3d}  change={imp_str}")

total_baseline = sum(r["baseline_count"] for r in results_all.values())
total_sahi = sum(r["sahi_count"] for r in results_all.values())
total_imp = ((total_sahi / total_baseline) - 1) * 100 if total_baseline > 0 else 0
print(f"\n  TOTAL: baseline={total_baseline}  sahi={total_sahi}  change=+{total_imp:.0f}%")

# Bar chart
fig, ax = plt.subplots(figsize=(8, 5))
names = list(results_all.keys())
base = [r["baseline_count"] for r in results_all.values()]
sahi = [r["sahi_count"] for r in results_all.values()]
x = np.arange(len(names))
width = 0.35
ax.bar(x - width / 2, base, width, label="YOLOv8x baseline", color="#e74c3c")
ax.bar(x + width / 2, sahi, width, label="YOLOv8x + SAHI", color="#2ecc71")
ax.set_xticks(x)
ax.set_xticklabels([n.split(".")[0] for n in names], fontsize=9, rotation=15)
ax.set_ylabel("Detections")
ax.set_title("Detection count: baseline vs SAHI (real inference)", fontsize=13, fontweight="bold")
ax.legend()
for xi, (b, s) in enumerate(zip(base, sahi)):
    ax.text(xi - width / 2, b + 0.3, str(b), ha="center", fontsize=10, fontweight="bold")
    ax.text(xi + width / 2, s + 0.3, str(s), ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(str(OUT_DIR / "detection_summary.png"), dpi=120)
plt.close()

# Save metrics JSON
metrics = {
    "method": "YOLOv8x (COCO pretrained) vs YOLOv8x + SAHI slicked inference",
    "confidence_threshold": CONF,
    "slice": "512x512, overlap 0.2",
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "per_image": results_all,
    "total_baseline": total_baseline,
    "total_sahi": total_sahi,
    "total_improvement_pct": round(total_imp, 1),
}
with open(SCORES_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)
print(f"\nMetrics saved to {SCORES_DIR / 'metrics.json'}")
print("=" * 60)
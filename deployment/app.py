from fastapi import FastAPI, UploadFile, File
import cv2, numpy as np, uvicorn, os
from pathlib import Path
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

app = FastAPI(title="Satellite Object Detection API — DRDO Defence")

device = 'cuda:0' if __import__('torch').cuda.is_available() else 'cpu'
model_path = '../outputs/models/best_yolov8x.pt'
if not os.path.exists(model_path):
    model_path = 'yolov8x.pt'

model = YOLO(model_path)

sahi_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8', model_path=model_path,
    confidence_threshold=0.3, device=device,
)

DEFENCE_CLASSES = ['plane', 'ship', 'storage-tank', 'large-vehicle',
                   'small-vehicle', 'helicopter', 'bridge', 'harbor']

@app.post("/detect")
async def detect(file: UploadFile = File(...), use_sahi: bool = False):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if use_sahi:
        result = get_sliced_prediction(
            image=image_rgb, detection_model=sahi_model,
            slice_height=512, slice_width=512,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        )
        detections = [{
            "bbox": pred.bbox.to_voc_bbox(),
            "category": pred.category.name,
            "confidence": float(pred.score.value),
        } for pred in result.object_prediction_list]
    else:
        results = model(image_rgb)
        boxes = results[0].boxes
        detections = [{
            "bbox": box.xyxy[0].tolist(),
            "category": DEFENCE_CLASSES[int(box.cls[0])] if int(box.cls[0]) < len(DEFENCE_CLASSES) else "unknown",
            "confidence": float(box.conf[0]),
        } for box in boxes]

    return {"num_detections": len(detections), "detections": detections}

@app.get("/health")
def health():
    return {"status": "healthy", "gpu": ('cuda' in device), "device": device}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

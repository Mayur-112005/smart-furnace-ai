from ultralytics import YOLO

# Load base YOLOv8 nano model (smallest, best for CPU)
model = YOLO("yolov8n.pt")

# Train on our fire dataset
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="Fire-Dataset-for-YOLOv8-1/data.yaml",
    epochs=3,
    imgsz=320,      # smallest size - 3x faster
    batch=8,
    device="cpu",
    name="furnace_fire_detector_v3",
    patience=2,
    workers=0,
    fraction=0.2    # use only 20% of dataset - 800 images instead of 3996
)


print("\nTraining complete!")
print(f"Best model saved at: runs/detect/furnace_fire_detector/weights/best.pt")

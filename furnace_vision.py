from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Run on a fire/industrial image from the web
results = model(
    "fire.jpg",
    conf=0.25  # confidence threshold - detect anything above 25%
)

# Show what was detected
for result in results:
    boxes = result.boxes
    print(f"\nDetected {len(boxes)} objects:")
    for box in boxes:
        label = model.names[int(box.cls)]
        confidence = float(box.conf)
        print(f"  → {label}: {confidence:.0%} confidence")

# Save the annotated image
results[0].show()
results[0].save(filename="furnace_detection_result.jpg")
print("\nSaved result to furnace_detection_result.jpg")
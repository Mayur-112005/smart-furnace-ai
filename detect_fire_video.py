from ultralytics import YOLO
import cv2

# Load YOUR custom trained model
model = YOLO("runs/detect/furnace_fire_detector_v3/weights/best.pt")

# Open the downloaded furnace video
cap = cv2.VideoCapture("furnace_video.mp4")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Save output video with bounding boxes
out = cv2.VideoWriter(
    "furnace_detected.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (width, height)
)

frame_count = 0
print("Running fire detection on video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.25, verbose=False)
    annotated = results[0].plot()

    # Print detections every 30 frames
    if frame_count % 30 == 0:
        boxes = results[0].boxes
        if len(boxes) > 0:
            for box in boxes:
                label = model.names[int(box.cls)]
                conf = float(box.conf)
                print(f"  Frame {frame_count}: {label} detected at {conf:.0%} confidence")
        else:
            print(f"  Frame {frame_count}: no detection")

    out.write(annotated)
    frame_count += 1

cap.release()
out.release()
print(f"\nDone! Processed {frame_count} frames")
print("Output saved to furnace_detected.mp4")
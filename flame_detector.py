from ultralytics import YOLO
import cv2

# Load YOLOv8 nano model (smallest, fastest - good for CPU)
model = YOLO("yolov8n.pt")  # downloads automatically first run (~6MB)

# Open webcam (0 = default camera)
# We'll replace this with furnace camera feed later
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No webcam found - running on sample image instead")
    # Run on a sample image
    results = model("https://ultralytics.com/images/bus.jpg")
    results[0].show()
else:
    print("Webcam found! Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection on the frame
        results = model(frame, verbose=False)

        # Draw bounding boxes on frame
        annotated = results[0].plot()

        # Show the frame
        cv2.imshow("Furnace Detector", annotated)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
import paho.mqtt.client as mqtt
import json
import time
import cv2
from ultralytics import YOLO
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Load custom trained fire model
model = YOLO("runs/detect/furnace_fire_detector_v3/weights/best.pt")

# MQTT setup
BROKER = "localhost"
PORT   = 1883
TOPIC_ALERT = "plant/furnace1/fire_detection"

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.connect(BROKER, PORT)
print("Connected to MQTT broker")

# InfluxDB setup
INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = "_uKzAe62QagRrFOTHfa4jskRLicanl048jmP2vKa6MWoHKE51524Zs87WHnrkeC6o8blUoUac4RH6sBqLG5N2A=="
INFLUX_ORG    = "furnace_org"
INFLUX_BUCKET = "furnace_data"

influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)
print("Connected to InfluxDB")

# Open video
cap = cv2.VideoCapture("furnace_video.mp4")
print("Starting real-time fire detection...\n")

frame_count = 0
last_alert_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model(frame, conf=0.25, verbose=False)
    annotated = results[0].plot()
    boxes = results[0].boxes

    current_time = time.time()

    if len(boxes) > 0:
        max_conf = max([float(b.conf) for b in boxes])

        if max_conf >= 0.6:
            severity = "CRITICAL"
        elif max_conf >= 0.4:
            severity = "HIGH"
        else:
            severity = "MEDIUM"

        if current_time - last_alert_time >= 3:
            # Publish to MQTT
            alert = {
                "alert": True,
                "source": "YOLOv8_camera",
                "severity": severity,
                "confidence": round(max_conf, 2),
                "fires_detected": len(boxes),
                "timestamp": time.strftime("%H:%M:%S"),
                "message": f"🔥 FIRE DETECTED! Severity: {severity} | Confidence: {max_conf:.0%}"
            }
            mqtt_client.publish(TOPIC_ALERT, json.dumps(alert))

            # Write to InfluxDB
            point = (
                Point("fire_detections")
                .tag("source", "YOLOv8_camera")
                .tag("severity", severity)
                .field("confidence", round(max_conf, 2))
                .field("fires_detected", len(boxes))
            )
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

            print(f"[{alert['timestamp']}] 🔥 Severity: {severity} | Confidence: {max_conf:.0%} | Fires: {len(boxes)} → MQTT + InfluxDB")
            last_alert_time = current_time
    else:
        if frame_count % 30 == 0:
            # Write "no fire" to InfluxDB too for continuous graph
            point = (
                Point("fire_detections")
                .tag("source", "YOLOv8_camera")
                .tag("severity", "NONE")
                .field("confidence", 0.0)
                .field("fires_detected", 0)
            )
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
            print(f"[{time.strftime('%H:%M:%S')}] ✅ No fire — frame {frame_count}")

    cv2.imshow("Furnace Camera - Live Fire Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
mqtt_client.disconnect()
influx_client.close()
print("Stopped.")
import paho.mqtt.client as mqtt
import json
import time
import cv2
import numpy as np
from ultralytics import YOLO
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# ── Model paths ───────────────────────────────────────────────────────────────
FIRE_SMOKE_MODEL = "models/furnace_fire_smoke_best.pt"

model = YOLO(FIRE_SMOKE_MODEL)
print(f"Loaded model: {FIRE_SMOKE_MODEL}")
print(f"Classes: {model.names}")

# ── MQTT setup ────────────────────────────────────────────────────────────────
BROKER      = "localhost"
PORT        = 1883
TOPIC_ALERT = "plant/furnace1/fire_detection"
TOPIC_RACK  = "plant/alerts/rack_alarm"

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.connect(BROKER, PORT)
print("Connected to MQTT broker")

# ── InfluxDB setup ────────────────────────────────────────────────────────────
INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = "_uKzAe62QagRrFOTHfa4jskRLicanl048jmP2vKa6MWoHKE51524Zs87WHnrkeC6o8blUoUac4RH6sBqLG5N2A=="
INFLUX_ORG    = "furnace_org"
INFLUX_BUCKET = "furnace_data"

influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api     = influx_client.write_api(write_options=SYNCHRONOUS)
print("Connected to InfluxDB")

# ── Threat scoring logic ──────────────────────────────────────────────────────
def analyze_flame_color(frame, box):
    x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return "unknown", 1

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    blue_mask   = cv2.inRange(hsv, np.array([100, 50, 50]),   np.array([130, 255, 255]))
    white_mask  = cv2.inRange(hsv, np.array([0, 0, 200]),     np.array([180, 30, 255]))
    orange_mask = cv2.inRange(hsv, np.array([10, 100, 100]),  np.array([25, 255, 255]))
    red_mask1   = cv2.inRange(hsv, np.array([0, 100, 100]),   np.array([10, 255, 255]))
    red_mask2   = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    red_mask    = cv2.bitwise_or(red_mask1, red_mask2)

    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0:
        return "unknown", 1

    blue_pct   = cv2.countNonZero(blue_mask)   / total_pixels
    white_pct  = cv2.countNonZero(white_mask)  / total_pixels
    orange_pct = cv2.countNonZero(orange_mask) / total_pixels
    red_pct    = cv2.countNonZero(red_mask)    / total_pixels

    if blue_pct > 0.1:
        return "BLUE (>1400C)", 4
    elif white_pct > 0.2:
        return "WHITE (1200-1400C)", 3
    elif orange_pct > 0.2:
        return "ORANGE (800-1200C)", 2
    elif red_pct > 0.1:
        return "RED (<800C)", 1
    else:
        return "MIXED", 2


def get_box_size_score(box, frame_area):
    x1, y1, x2, y2 = box.xyxy[0]
    box_area = (x2 - x1) * (y2 - y1)
    ratio = float(box_area) / frame_area

    if ratio > 0.25:   return 4
    elif ratio > 0.10: return 3
    elif ratio > 0.04: return 2
    else:              return 1


def score_to_threat(score):
    if score >= 4:   return "CRITICAL"
    elif score >= 3: return "HIGH"
    elif score >= 2: return "MEDIUM"
    else:            return "LOW"


def get_recommendations(threat, detection_type):
    recs = {
        ("fire", "CRITICAL"): [
            "Shut down feed conveyor immediately",
            "Reduce electrode current by 15%",
            "Activate emergency cooling system",
            "Evacuate non-essential personnel",
            "Alert shift supervisor and safety officer"
        ],
        ("fire", "HIGH"): [
            "Reduce feed rate by 8% immediately",
            "Adjust electrode penetration depth",
            "Check cooling water flow rate",
            "Alert shift supervisor — possible burden hanging"
        ],
        ("fire", "MEDIUM"): [
            "Increase electrode current by 5%",
            "Inspect raw material feed consistency",
            "Check electrode current balance"
        ],
        ("fire", "LOW"): [
            "Monitor flame pattern for next 5 minutes",
            "Check electrode positioning"
        ],
        ("smoke", "CRITICAL"): [
            "CRITICAL smoke detected — possible combustion event",
            "Check furnace sealing and gas flow immediately",
            "Activate exhaust ventilation at maximum",
            "Alert safety team"
        ],
        ("smoke", "HIGH"): [
            "Abnormal smoke density detected",
            "Reduce feed rate by 10%",
            "Inspect gas extraction system",
            "Check electrode seal integrity"
        ],
        ("smoke", "MEDIUM"): [
            "Monitor smoke levels — may indicate uneven burning",
            "Check raw material moisture content",
            "Inspect burner nozzles"
        ],
        ("smoke", "LOW"): [
            "Minor smoke detected — monitor closely",
            "Check ventilation efficiency"
        ],
    }
    return recs.get((detection_type, threat), ["Monitor and observe"])


# ── Main detection loop ───────────────────────────────────────────────────────
cap = cv2.VideoCapture("furnace_video.mp4")
frame_h    = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_w    = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_area = frame_h * frame_w

# Resize for display — smaller = less lag on screen
DISPLAY_W = 640
DISPLAY_H = 360

print(f"\nStarting enhanced fire + smoke detection...")
print(f"Frame size: {int(frame_w)}x{int(frame_h)}")
print(f"Processing every 3rd frame to reduce CPU load\n")

frame_count     = 0
last_alert_time = 0
last_detections = []  # keep last result for skipped frame
last_annotated = None

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_count += 1

    # ── Skip frames — only run AI on every 3rd frame ──────────────────────────
    # Skip frames — only run AI on every 3rd frame
    if frame_count % 3 != 0:
        # Show last annotated frame instead of raw frame — no blinking!
        show = last_annotated if last_annotated is not None else frame
        display = cv2.resize(show, (DISPLAY_W, DISPLAY_H))
        cv2.imshow("Furnace AI — Fire + Smoke Detection", display)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # 30ms delay = ~33 FPS
            break
        continue

    # ── Run AI inference on this frame ────────────────────────────────────────
    results   = model(frame, conf=0.25, verbose=False, imgsz=320)  # smaller imgsz = faster
    boxes     = results[0].boxes
    names     = model.names
    annotated = results[0].plot()
    last_annotated = annotated  # save for skipped frames

    # Resize annotated frame for display
    display = cv2.resize(annotated, (DISPLAY_W, DISPLAY_H))
    cv2.imshow("Furnace AI — Fire + Smoke Detection", display)
    if cv2.waitKey(30) & 0xFF == ord('q'):  # 30ms delay on detection frames too
        break

    # ── Separate fire and smoke ───────────────────────────────────────────────
    fire_boxes  = [b for b in boxes if names[int(b.cls)].lower() in ["fire", "flame"]]
    smoke_boxes = [b for b in boxes if names[int(b.cls)].lower() == "smoke"]

    detections_summary = []
    current_time = time.time()

    # ── Analyze fire detections ───────────────────────────────────────────────
    for box in fire_boxes:
        conf                    = float(box.conf)
        color_name, color_score = analyze_flame_color(frame, box)
        size_score              = get_box_size_score(box, frame_area)
        conf_score              = int(conf * 4)
        combined                = (color_score * 0.4) + (size_score * 0.3) + (conf_score * 0.3)
        threat                  = score_to_threat(combined)

        detections_summary.append({
            "type":           "fire",
            "threat":         threat,
            "confidence":     round(conf, 2),
            "flame_color":    color_name,
            "color_score":    color_score,
            "size_score":     size_score,
            "combined_score": round(combined, 2)
        })

    # ── Analyze smoke detections ──────────────────────────────────────────────
    for box in smoke_boxes:
        conf       = float(box.conf)
        size_score = get_box_size_score(box, frame_area)
        conf_score = int(conf * 4)
        combined   = (size_score * 0.6) + (conf_score * 0.4)
        threat     = score_to_threat(combined)

        detections_summary.append({
            "type":           "smoke",
            "threat":         threat,
            "confidence":     round(conf, 2),
            "flame_color":    "N/A",
            "size_score":     size_score,
            "combined_score": round(combined, 2)
        })

    # ── Publish alerts every 3 seconds ───────────────────────────────────────
    if detections_summary and (current_time - last_alert_time >= 3):

        threat_order   = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        overall_threat = max(detections_summary, key=lambda d: threat_order[d["threat"]])["threat"]

        print(f"[{time.strftime('%H:%M:%S')}] {len(fire_boxes)} fire | {len(smoke_boxes)} smoke | Overall: {overall_threat}")
        for d in detections_summary:
            icon = "FIRE" if d["type"] == "fire" else "SMOKE"
            print(f"  [{icon}] Threat: {d['threat']} | Conf: {d['confidence']:.0%} | Color: {d['flame_color']} | Score: {d['combined_score']:.1f}")

        recs = get_recommendations(overall_threat, detections_summary[0]["type"])
        print(f"  AI Recommendations ({overall_threat}):")
        for i, r in enumerate(recs, 1):
            print(f"     {i}. {r}")

        # MQTT
        alert = {
            "alert":          True,
            "source":         "YOLOv8_enhanced",
            "overall_threat": overall_threat,
            "fires_detected": len(fire_boxes),
            "smoke_detected": len(smoke_boxes),
            "detections":     detections_summary,
            "timestamp":      time.strftime("%H:%M:%S"),
            "message":        f"DETECTED: {len(fire_boxes)} fire, {len(smoke_boxes)} smoke | Threat: {overall_threat}"
        }
        mqtt_client.publish(TOPIC_ALERT, json.dumps(alert))

        # InfluxDB
        for d in detections_summary:
            point = (
                Point("fire_smoke_detections")
                .tag("source",      "YOLOv8_enhanced")
                .tag("type",        d["type"])
                .tag("threat",      d["threat"])
                .tag("flame_color", d["flame_color"])
                .field("confidence",     d["confidence"])
                .field("combined_score", d["combined_score"])
                .field("size_score",     float(d["size_score"]))
            )
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

        print(f"  Published to MQTT + InfluxDB\n")
        last_alert_time = current_time

    elif not detections_summary and frame_count % 90 == 0:
        point = (
            Point("fire_smoke_detections")
            .tag("source", "YOLOv8_enhanced")
            .tag("type",   "none")
            .tag("threat", "NONE")
            .field("confidence",     0.0)
            .field("combined_score", 0.0)
            .field("size_score",     0.0)
        )
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        print(f"[{time.strftime('%H:%M:%S')}] No detection — frame {frame_count}")

cap.release()
cv2.destroyAllWindows()
mqtt_client.disconnect()
influx_client.close()
print("Stopped.")
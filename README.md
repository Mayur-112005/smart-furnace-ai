# 🔥 Smart Furnace AI Monitoring System

An AI-powered real-time monitoring system for industrial furnaces that combines IoT sensor data, computer vision, time-series database, and live dashboards to detect and predict furnace anomalies.

---

## 🏗️ System Architecture

```
Furnace (RGB + Thermal Camera + PLC Sensors)
         ↓
Edge AI Gateway (NVIDIA Jetson Orin)
         ↓ MQTT
Central AI Server
    ↓              ↓
Dashboard       Alarm Rack
(Grafana)    (Siren + Beacon)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| IoT Messaging | MQTT (Mosquitto + Paho) |
| Time-series DB | InfluxDB 2.9.1 |
| Dashboard | Grafana |
| Computer Vision | YOLOv8 (Ultralytics) |
| Automation | Node-RED |
| Edge AI (production) | NVIDIA Jetson Orin |
| Hardware Alarm (production) | ESP32 + Siren + Beacon |
| Language | Python 3.13 |

---

## 📁 Project Structure

```
smart-furnace-ai/
├── furnace_publisher.py          ← MQTT furnace sensor simulator
├── furnace_subscriber.py         ← MQTT → InfluxDB writer + alerts
├── flame_detector.py             ← basic YOLOv8 webcam test
├── furnace_vision.py             ← YOLOv8 on static image
├── detect_fire_video.py          ← YOLOv8 fire detection on video
├── yolo_mqtt_alarm.py            ← YOLOv8 → MQTT + InfluxDB pipeline
├── train_flame_detector.py       ← custom YOLOv8 model training
├── download_dataset.py           ← Roboflow dataset downloader
├── furnace_video.mp4             ← sample industrial fire video
├── fire.jpg                      ← sample furnace image
├── furnace_detection_result.jpg  ← YOLOv8 annotated output
├── Fire-Dataset-for-YOLOv8-1/   ← downloaded training dataset
├── runs/                         ← YOLOv8 training results + weights
└── .gitignore
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.13
- Node.js
- Mosquitto MQTT Broker
- InfluxDB 2.9.1
- Grafana
- Node-RED

### Python Dependencies
```bash
pip install paho-mqtt
pip install influxdb-client
pip install ultralytics
pip install roboflow
pip install yt-dlp
```

---

## 🔧 Configuration

### InfluxDB
```
URL:      http://localhost:8086
Org:      furnace_org
Bucket:   furnace_data
```

### MQTT Topics
```
plant/furnace1/sensor_data     ← sensor readings
plant/furnace1/fire_detection  ← YOLOv8 camera alerts
```

### Flame Status → Severity
| Flame Status | Severity |
|---|---|
| NORMAL | No alert |
| WEAK | MEDIUM |
| HIGH | HIGH |
| UNSTABLE | CRITICAL |

---

## 🚀 Startup Checklist

Every session, open these terminals in order:

**Terminal 1 — InfluxDB:**
```bash
influxd
```

**Terminal 2 — Sensor Publisher:**
```bash
cd C:\Users\MAYUR\OneDrive\Desktop\furnace_project
python furnace_publisher.py
```

**Terminal 3 — Subscriber (MQTT → InfluxDB):**
```bash
cd C:\Users\MAYUR\OneDrive\Desktop\furnace_project
python furnace_subscriber.py
```

**Terminal 4 — YOLOv8 Fire Detection:**
```bash
cd C:\Users\MAYUR\OneDrive\Desktop\furnace_project
python yolo_mqtt_alarm.py
```

**Terminal 5 — Node-RED:**
```bash
node-red
```

**Browser tabs:**
- Grafana Dashboard → `http://localhost:3000`
- Node-RED Flow → `http://localhost:1880`
- InfluxDB → `http://localhost:8086`

> **Note:** Mosquitto runs automatically as a Windows service — no manual start needed.

---

## 📊 What Each Component Does

| Component | What it does |
|---|---|
| `furnace_publisher.py` | Simulates PLC sensor readings — temperature, current, power, flame status — published to MQTT every 2 seconds |
| `furnace_subscriber.py` | Receives MQTT sensor data, prints severity alerts, writes every reading to InfluxDB |
| `yolo_mqtt_alarm.py` | Runs YOLOv8 on furnace video, detects fire in real time, publishes alerts to MQTT and writes detections to InfluxDB |
| `train_flame_detector.py` | Trains custom YOLOv8 model on fire dataset from Roboflow |
| `detect_fire_video.py` | Runs trained model on video and saves annotated output |
| Node-RED | Receives both MQTT streams and triggers alarms — in production connects to ESP32 physical alarm rack |
| Grafana | Live dashboard showing temperature, current, power and fire detection confidence |
| InfluxDB | Stores all time-series data — sensor readings and fire detections |

---

## 🤖 YOLOv8 Model

- **Dataset:** Fire Detection Dataset — 3,996 CCTV fire images (Roboflow Universe)
- **Model:** YOLOv8n (nano) — optimized for CPU
- **Trained model:** `runs/detect/furnace_fire_detector_v3/weights/best.pt`
- **Classes:** fire, smoke
- **Training:** 3 epochs, imgsz=320, CPU only

**To retrain:**
```bash
python train_flame_detector.py
```

---

## 👤 Author

**Prajapati Mayur Suryanath**
GitHub: [@Mayur-112005](https://github.com/Mayur-112005)

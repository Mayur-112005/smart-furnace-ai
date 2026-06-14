# 🔥 Smart Furnace AI Monitoring System

An AI-powered real-time monitoring system for industrial furnaces that combines IoT sensor data, computer vision, time-series database, live dashboards, and predictive AI to detect and forecast furnace anomalies.

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
| Predictive AI | Darts LSTM |
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
├── generate_historical_data.py   ← generates 7-day historical CSV
├── lstm_forecast.py              ← LSTM model training + 10-min forecast
├── furnace_historical.csv        ← 5040 readings, 7 days of data
├── lstm_forecast.png             ← forecast chart (actual vs predicted)
├── furnace_video.mp4             ← sample industrial fire video
├── furnace_detected.mp4          ← YOLOv8 annotated output video
├── fire.jpg                      ← sample furnace image
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
pip install darts[torch]
pip install opencv-python
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
plant/furnace1/sensor_data     ← PLC sensor readings (temp, current, power)
plant/furnace1/fire_detection  ← YOLOv8 camera fire alerts
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
| `yolo_mqtt_alarm.py` | Runs YOLOv8 on furnace video in real time, detects fire, publishes alerts to MQTT and writes detections to InfluxDB |
| `train_flame_detector.py` | Trains custom YOLOv8 model on fire dataset from Roboflow (3,996 CCTV fire images) |
| `detect_fire_video.py` | Runs trained model on video and saves annotated output with bounding boxes |
| `generate_historical_data.py` | Generates 7 days of realistic furnace sensor data as CSV for LSTM training |
| `lstm_forecast.py` | Trains Darts LSTM model on historical CSV and predicts furnace temperature 10 minutes ahead |
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

## 🔮 LSTM Predictive Model

- **Data:** 5,040 readings over 7 days (2-minute intervals)
- **Model:** BlockRNNModel (LSTM) via Darts library
- **Input:** Last 60 minutes (30 data points × 2 min)
- **Output:** Next 10 minutes (5 data points × 2 min)
- **Average error:** ~17°C on 900–1300°C range (~1.5% error)

**Sample forecast results:**

| Time ahead | Predicted | Actual | Error |
|---|---|---|---|
| +2 min | 1100.9°C | 1078.6°C | 22.3°C |
| +4 min | 1102.0°C | 1065.5°C | 36.5°C |
| +6 min | 1102.2°C | 1108.4°C | 6.2°C |
| +8 min | 1102.2°C | 1109.0°C | 6.8°C |
| +10 min | 1102.3°C | 1087.3°C | 15.0°C |

**To run forecast:**
```bash
python generate_historical_data.py
python lstm_forecast.py
```

---

## 🔔 Node-RED Alarm Flow

Two parallel MQTT streams both feed into Node-RED:

```
[plant/furnace1/sensor_data]    → [Function: severity check] → [Debug/Alarm]
[plant/furnace1/fire_detection] → [Debug/Alarm]
```

In production, the Debug nodes are replaced with an ESP32 GPIO trigger that activates the physical siren and strobe beacon on the alarm rack.

---

## 👤 Author

**Prajapati Mayur Suryanath**
GitHub: [@Mayur-112005](https://github.com/Mayur-112005)

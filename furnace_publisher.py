import paho.mqtt.client as mqtt
import time
import random
import json

# Connect to broker running on your own PC
BROKER = "localhost"
PORT   = 1883
TOPIC  = "plant/furnace1/sensor_data"

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER, PORT)

print("Publisher started. Sending data every 2 seconds...")
print("Press Ctrl+C to stop.\n")

while True:
    # Simulate furnace sensor readings
    data = {
        "temperature_c": round(random.uniform(900, 1200), 1),
        "current_ka":    round(random.uniform(60, 90), 2),
        "power_mw":       round(random.uniform(20, 40), 2),
        "flame_status": random.choice(["NORMAL", "NORMAL", "NORMAL", "WEAK", "HIGH", "UNSTABLE"]),
        "timestamp":      time.strftime("%H:%M:%S")
    }

    payload = json.dumps(data)
    client.publish(TOPIC, payload)

    print(f"Sent → {payload}")
    time.sleep(2)
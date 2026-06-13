import paho.mqtt.client as mqtt
import json

BROKER = "localhost"
PORT   = 1883
TOPIC  = "plant/furnace1/sensor_data"

SEVERITY = {
    "WEAK":     "MEDIUM",
    "HIGH":     "HIGH",
    "UNSTABLE": "CRITICAL"
}

def on_connect(client, userdata, flags, rc, properties):
    print("Connected to broker. Waiting for data...\n")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())

    flame = data["flame_status"].upper()

    print(f"[{data['timestamp']}] Temp: {data['temperature_c']}°C | "
          f"Current: {data['current_ka']} kA | "
          f"Power: {data['power_mw']} MW | "
          f"Flame: {flame}")

    if flame != "NORMAL":
        severity = SEVERITY.get(flame, "UNKNOWN")
        print(f"  *** ALERT: Abnormal flame detected! | Severity: {severity} ***")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT)
client.loop_forever()
import paho.mqtt.client as mqtt
import json
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# MQTT config
BROKER = "localhost"
PORT   = 1883
TOPIC  = "plant/furnace1/sensor_data"

# InfluxDB config
INFLUX_URL   = "http://localhost:8086"
INFLUX_TOKEN = "_uKzAe62QagRrFOTHfa4jskRLicanl048jmP2vKa6MWoHKE51524Zs87WHnrkeC6o8blUoUac4RH6sBqLG5N2A=="
INFLUX_ORG   = "furnace_org"
INFLUX_BUCKET = "furnace_data"

SEVERITY = {
    "WEAK":     "MEDIUM",
    "HIGH":     "HIGH",
    "UNSTABLE": "CRITICAL"
}

# Connect to InfluxDB
influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)
print("Connected to InfluxDB")

def on_connect(client, userdata, flags, rc, properties):
    print("Connected to MQTT broker. Waiting for data...\n")
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

    # Write to InfluxDB
    point = (
        Point("furnace_readings")
        .tag("furnace_id", "furnace1")
        .tag("flame_status", flame)
        .field("temperature_c", data["temperature_c"])
        .field("current_ka",    data["current_ka"])
        .field("power_mw",      data["power_mw"])
    )
    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT)
client.loop_forever()
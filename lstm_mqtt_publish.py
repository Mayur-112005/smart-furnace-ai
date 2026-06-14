import paho.mqtt.client as mqtt
import json
import time
import pandas as pd
from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.dataprocessing.transformers import Scaler
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# MQTT setup
BROKER = "localhost"
PORT   = 1883
TOPIC  = "plant/furnace1/temperature_forecast"

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.connect(BROKER, PORT)
print("Connected to MQTT broker")

# InfluxDB setup
INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = "_uKzAe62QagRrFOTHfa4jskRLicanl048jmP2vKa6MWoHKE51524Zs87WHnrkeC6o8blUoUac4RH6sBqLG5N2A=="  # paste your token
INFLUX_ORG    = "furnace_org"
INFLUX_BUCKET = "furnace_data"

influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)
print("Connected to InfluxDB")

# Load data and train model
print("\nLoading historical data...")
df = pd.read_csv("furnace_historical.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

series = TimeSeries.from_dataframe(df, time_col="timestamp", value_cols="temperature_c", freq="2min")

split = int(len(series) * 0.9)
train = series[:split]

scaler = Scaler()
train_scaled = scaler.fit_transform(train)

print("Training LSTM model...")
model = BlockRNNModel(
    model="LSTM",
    input_chunk_length=30,
    output_chunk_length=5,
    n_epochs=10,
    hidden_dim=32,
    n_rnn_layers=2,
    batch_size=32,
    random_state=42,
    pl_trainer_kwargs={"enable_progress_bar": True}
)
model.fit(train_scaled)
print("Model ready!\n")

# Run forecast loop — publishes new forecast every 30 seconds
print("Publishing forecasts every 30 seconds. Press Ctrl+C to stop.\n")
while True:
    forecast_scaled = model.predict(n=5, series=train_scaled)
    forecast = scaler.inverse_transform(forecast_scaled)
    forecast_values = forecast.values().flatten()

    # Build forecast message
    predictions = []
    for i, val in enumerate(forecast_values):
        predictions.append({
            "minutes_ahead": (i + 1) * 2,
            "predicted_temp": round(float(val), 1)
        })

    payload = {
        "source": "LSTM_model",
        "timestamp": time.strftime("%H:%M:%S"),
        "forecast": predictions
    }

    # Publish to MQTT
    mqtt_client.publish(TOPIC, json.dumps(payload))

    # Write each prediction to InfluxDB
    for pred in predictions:
        point = (
            Point("temperature_forecast")
            .tag("source", "LSTM")
            .field("predicted_temp", pred["predicted_temp"])
            .field("minutes_ahead", pred["minutes_ahead"])
        )
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

    print(f"[{payload['timestamp']}] Forecast published:")
    for pred in predictions:
        print(f"  +{pred['minutes_ahead']} min → {pred['predicted_temp']}°C")
    print()

    time.sleep(30)
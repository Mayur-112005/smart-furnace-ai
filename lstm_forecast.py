import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.dataprocessing.transformers import Scaler

print("Loading historical furnace data...")
df = pd.read_csv("furnace_historical.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Create Darts TimeSeries from temperature column
series = TimeSeries.from_dataframe(
    df,
    time_col="timestamp",
    value_cols="temperature_c",
    freq="2min"
)

print(f"Total data points: {len(series)}")
print(f"Date range: {series.start_time()} → {series.end_time()}")

# Split into train (90%) and test (10%)
split = int(len(series) * 0.9)
train, test = series[:split], series[split:]
print(f"Train: {len(train)} points | Test: {len(test)} points")

# Scale data (LSTM works better with normalized data)
scaler = Scaler()
train_scaled = scaler.fit_transform(train)
test_scaled  = scaler.transform(test)

# Build LSTM model
print("\nBuilding LSTM model...")
model = BlockRNNModel(
    model="LSTM",
    input_chunk_length=30,   # look at last 60 minutes (30 x 2min)
    output_chunk_length=5,   # predict next 10 minutes (5 x 2min)
    n_epochs=10,
    hidden_dim=32,
    n_rnn_layers=2,
    dropout=0.1,
    batch_size=32,
    random_state=42,
    pl_trainer_kwargs={"enable_progress_bar": True}
)

print("Training LSTM model (this takes 5-10 minutes on CPU)...")
model.fit(train_scaled)
print("Training complete!")

# Predict next 10 minutes (5 steps x 2 min)
print("\nGenerating forecast...")
forecast_scaled = model.predict(n=5, series=train_scaled)
forecast = scaler.inverse_transform(forecast_scaled)

# Get actual values for comparison
actual_last = series[split:split+5]

# Print results
print("\n--- 10-Minute Temperature Forecast ---")
forecast_values = forecast.values().flatten()
actual_values   = actual_last.values().flatten()

for i, (pred, actual) in enumerate(zip(forecast_values, actual_values)):
    t = (i + 1) * 2
    print(f"  +{t:2d} min → Predicted: {pred:.1f}°C | Actual: {actual:.1f}°C | Error: {abs(pred-actual):.1f}°C")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(train[-60:].time_index, train[-60:].values(), label="Historical (last 2hrs)", color="blue")
plt.plot(actual_last.time_index, actual_last.values(), label="Actual", color="green", linewidth=2)
plt.plot(forecast.time_index, forecast.values(), label="LSTM Forecast", color="red", linestyle="--", linewidth=2, marker="o")
plt.title("Furnace Temperature — LSTM 10-Minute Forecast")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lstm_forecast.png", dpi=150)
plt.show()
print("\nForecast chart saved to lstm_forecast.png")
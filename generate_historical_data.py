import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate 7 days of furnace data — one reading every 2 minutes
# That's 7 * 24 * 30 = 5040 readings

print("Generating 7 days of historical furnace data...")

np.random.seed(42)
start_time = datetime(2026, 6, 1, 0, 0, 0)
readings = []

base_temp = 1050  # base furnace temperature

for i in range(5040):
    timestamp = start_time + timedelta(minutes=2 * i)

    # Simulate realistic temperature patterns:
    # - Daily cycle (hotter during day shift)
    # - Random fluctuations
    # - Occasional spikes (abnormal events)
    hour = timestamp.hour
    day_factor = 50 * np.sin(np.pi * hour / 12)  # day/night cycle
    noise = np.random.normal(0, 15)               # random noise
    spike = 80 if np.random.random() < 0.02 else 0  # 2% chance of spike

    temperature = base_temp + day_factor + noise + spike
    temperature = round(max(900, min(1300, temperature)), 1)

    current = round(np.random.uniform(60, 90) + noise * 0.1, 2)
    power   = round(np.random.uniform(20, 40) + noise * 0.05, 2)

    readings.append({
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature_c": temperature,
        "current_ka": current,
        "power_mw": power
    })

df = pd.DataFrame(readings)
df.to_csv("furnace_historical.csv", index=False)
print(f"Saved {len(df)} readings to furnace_historical.csv")
print(df.head(10))
print(f"\nTemperature range: {df['temperature_c'].min()}°C to {df['temperature_c'].max()}°C")
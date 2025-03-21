import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure consistent outputs for reproducibility
np.random.seed(42)

# Generate a date range spanning 3 months with hourly intervals
date_range = pd.date_range(start="2024-01-01", end="2024-03-31 23:00", freq="h")
n_hours = len(date_range)  # Total number of hours in this period

# === Temporal Features ===
# Create a DataFrame to store timestamps and derived time-related features
df = pd.DataFrame({
    "timestamp": date_range,
    "hour": date_range.hour,  # Extract hour of the day (0-23)
    "day_of_week": date_range.dayofweek,  # Monday=0, Sunday=6
    "month": date_range.month,  # Month of the year
    "holiday": np.random.choice([0, 1], size=n_hours, p=[0.95, 0.05])  # Randomly mark 5% of days as holidays
})

# === Weather Features ===
# Temperature follows both a daily cycle (sinusoidal) and a seasonal cycle
daily_cycle = 5 * np.sin(2 * np.pi * df["hour"] / 24)  # Daily temperature variation
seasonal_cycle = 10 * np.sin(2 * np.pi * (date_range.dayofyear - 1) / 365)  # Seasonal temperature variation

# Base temperature fluctuates around 20°C, modified by daily & seasonal patterns
temperature_base = 20 + daily_cycle + seasonal_cycle
df["temperature"] = temperature_base + np.random.normal(0, 2, n_hours)  # Adding noise

# Cloud cover changes smoothly over time using an autoregressive model
cloud_cover = np.zeros(n_hours)
cloud_cover[0] = np.random.uniform(0, 1)  # Initialize with a random value
for i in range(1, n_hours):
    cloud_cover[i] = cloud_cover[i-1] * 0.9 + np.random.uniform(0, 0.1)  # Smoothed transition
df["cloud_cover"] = np.clip(cloud_cover, 0, 1)  # Ensure values stay between 0 and 1

# === Solar Power Generation ===
# Solar irradiance follows a sinusoidal pattern during daylight hours
solar_irradiance = np.where(
    (df["hour"] >= 6) & (df["hour"] <= 18),  # Sunlight hours
    1000 * np.sin(np.pi * (df["hour"] - 6) / 12),  # Peak around noon
    0  # No sunlight at night
)

# Solar panel efficiency fluctuates between 18% and 22%
panel_efficiency = np.random.uniform(0.18, 0.22, n_hours)

# Solar PV output depends on irradiance and panel efficiency, active between 8 AM - 5 PM
df["solar_pv_output"] = np.where(
    (df["hour"] >= 8) & (df["hour"] <= 17),  # Solar active period
    solar_irradiance * panel_efficiency * 100 * np.sin(np.pi * (df["hour"] - 8) / 9),
    0
)

# === Load Demand Estimation ===
# Base load is higher during business hours (8 AM - 8 PM) and lower at night
df["load_demand"] = np.where(
    (df["hour"] >= 8) & (df["hour"] <= 20),
    100 * (1 - 0.2 * (df["day_of_week"] >= 5)),  # Weekend reduction
    30 + 10 * (df["hour"] < 5)  # Higher demand early morning due to heating
)

# Modify demand based on temperature (higher temp increases cooling demand)
df["load_demand"] += np.exp((df["temperature"] - 25) / 10)

# Add slight random variations to simulate real-world fluctuations
df["load_demand"] *= (1 + np.random.normal(0, 0.05, n_hours))

# === Battery Storage & Diesel Generator Simulation ===
# Battery parameters
battery_capacity = 200  # Max storage capacity in kWh
max_charge_rate = 50  # Max charge/discharge rate per hour
battery_level = battery_capacity * 0.5  # Start with 50% charge

# Initialize columns for energy storage system
df["battery_charge"] = np.zeros(n_hours)  # Energy charged into battery
df["battery_discharge"] = np.zeros(n_hours)  # Energy drawn from battery
df["battery_soc"] = np.zeros(n_hours)  # Battery state of charge (%)
df["diesel_generator_usage"] = np.zeros(n_hours)  # Diesel backup (if needed)

# Simulate battery operation over time
for i in range(1, n_hours):
    net_demand = df.at[i, "load_demand"] - df.at[i, "solar_pv_output"]  # Net energy required

    if net_demand > 0:  # Demand exceeds solar supply (deficit)
        if battery_level > 0:  # Battery available to discharge
            discharge = min(net_demand, max_charge_rate, battery_level)
            df.at[i, "battery_discharge"] = discharge
            battery_level -= discharge
        else:  # No battery left, fallback to diesel generator
            df.at[i, "diesel_generator_usage"] = net_demand
    else:  # Excess solar power available (surplus)
        charge = min(-net_demand, max_charge_rate, battery_capacity - battery_level)
        df.at[i, "battery_charge"] = charge
        battery_level += charge

    # Ensure battery stays within 0-100% range
    battery_level = max(0, min(battery_level, battery_capacity))
    df.at[i, "battery_soc"] = (battery_level / battery_capacity) * 100  # Convert to percentage


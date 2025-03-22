import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
import joblib
import xgboost as xgb
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

# Streamlit app configuration
st.set_page_config(page_title="Energy Prediction Dashboard", layout="wide")

# Title
st.title("Energy Prediction Dashboard")

# GitHub repo (raw file links)
GITHUB_REPO = "https://raw.githubusercontent.com/wilberkamaa/energy_demand_forecast/master/"
MODEL_FILES = {
    "rf_demand": "random_forest_model.pkl",
    "xgb_demand": "xgboost_model.json",
    "rf_solar": "rf_solar_model.pkl",
    "xgb_solar": "xgb_solar_model.json",
    "rf_diesel": "rf_diesel_model.pkl",
    "xgb_diesel": "xgb_diesel_model.json"
}

@st.cache_resource
def load_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        if not os.path.exists(filename):
            url = GITHUB_REPO + filename
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                st.write(f"✅ Downloaded {filename}")
            else:
                st.write(f"❌ Failed to download {filename}. Check the URL.")
        try:
       
            if "xgb" in name:
                xgb_model = xgb.Booster()
                xgb_model.load_model(filename)
                models[name] = xgb_model
                st.write(f"✅ Loaded XGBoost model from {filename}")
            else:
                models[name] = joblib.load(filename)
                st.write(f"✅ Loaded {filename}")
        except Exception as e:
            st.write(f"❌ Error loading {filename}: {e}")
    return models

@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/wilberkamaa/energy_demand_forecast/refs/heads/master/energy_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df["hour_demand"] = df["hour"] * df["demand_prev_day"]
    return df

# Load models and data
models = load_models()
rf_demand = models["rf_demand"]
xgb_demand = models["xgb_demand"]
rf_solar = models["rf_solar"]
xgb_solar = models["xgb_solar"]
rf_diesel = models["rf_diesel"]
xgb_diesel = models["xgb_diesel"]
df = load_data()

# Sidebar for interactive controls
st.sidebar.header("Controls")
# Slider for selecting the end time
min_timestamp = df["timestamp"].min().to_pydatetime()
max_timestamp = df["timestamp"].max().to_pydatetime()
end_time_selected = st.sidebar.slider(
    "Select End Time",
    min_value=min_timestamp,
    max_value=max_timestamp,
    value=max_timestamp,
    format="YYYY-MM-DD HH:mm",
    step=timedelta(hours=1)
)
# Convert back to pandas.Timestamp for filtering
end_time = pd.to_datetime(end_time_selected)
# Slider for selecting the date range (hours back from end_time)
date_range = st.sidebar.slider(
    "Select Date Range (hours back from end time)",
    min_value=24,
    max_value=720,
    value=168,
    step=24
)
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost", "Both"])
show_weather = st.sidebar.checkbox("Show Weather Data", value=False)

# Filter data based on date range
start_time = end_time - timedelta(hours=date_range)
df_filtered = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)].copy()
df_filtered.drop(columns=["timestamp"], inplace=True)

# Check if df_filtered is empty
if df_filtered.empty:
    st.error(f"No data available for the selected time range: {start_time} to {end_time}. Please adjust the 'Select End Time' or 'Date Range' sliders.")
else:
    with st.expander("View Data Sample"):
        st.write("First 5 rows of the dataset:")
        st.dataframe(df_filtered.head())
        st.write("Available columns:", list(df_filtered.columns))

    # Define feature sets
    demand_features = [
        "hour", "day_of_week", "month",
        "demand_prev_hour", "demand_prev_day", "demand_rolling_7d", "demand_rolling_24h",
        "hour_demand"
    ]
    solar_features = [
        "hour", "day_of_week", "month", "holiday",
        "temperature", "cloud_cover"
    ]
    diesel_features = [
        "hour", "day_of_week", "month", "holiday",
        "load_demand", "demand_prev_hour", "demand_prev_day", "demand_rolling_7d", "demand_rolling_24h",
        "solar_pv_output", "battery_soc"
    ]

    # Predict load_demand
    X_demand = df_filtered[demand_features]
    y_true = df_filtered["load_demand"]
    rf_pred_demand = rf_demand.predict(X_demand)
    dmat_demand = xgb.DMatrix(X_demand)
    xgb_pred_demand = xgb_demand.predict(dmat_demand)

    # Predict solar_pv_output for the latest time step
    latest_solar_features = df_filtered[solar_features].iloc[-1:].copy()
    rf_pred_solar = rf_solar.predict(latest_solar_features)[0]
    xgb_pred_solar = xgb_solar.predict(xgb.DMatrix(latest_solar_features))[0]
    solar_pred = (rf_pred_solar + xgb_pred_solar) / 2 if model_choice == "Both" else (rf_pred_solar if model_choice == "Random Forest" else xgb_pred_solar)

    # Predict diesel_generator_usage (using predicted solar_pv_output)
    latest_diesel_features = df_filtered[diesel_features].iloc[-1:].copy()
    latest_diesel_features["solar_pv_output"] = solar_pred
    recent_soc = df_filtered["battery_soc"][df_filtered["battery_soc"] > 0]
    latest_diesel_features["battery_soc"] = recent_soc.iloc[-1] if not recent_soc.empty else 20
    rf_pred_diesel = rf_diesel.predict(latest_diesel_features)[0]
    xgb_pred_diesel = xgb_diesel.predict(xgb.DMatrix(latest_diesel_features))[0]
    diesel_pred = (rf_pred_diesel + xgb_pred_diesel) / 2 if model_choice == "Both" else (rf_pred_diesel if model_choice == "Random Forest" else xgb_pred_diesel)

    # Derive battery_soc
    latest_demand = y_true.iloc[-1]
    battery_soc = min(100, max(20, 50 + (solar_pred * 0.5 - latest_demand * 0.3 - diesel_pred * 0.1)))

    # Overview Panel
    st.header("Overview")
    col1, col2, col3, col4 = st.columns(4)  # Updated to 4 columns
    with col1:
        st.metric("Current Load Demand", f"{y_true.iloc[-1]:.1f} kW", f"{y_true.iloc[-1] - y_true.iloc[-2]:.1f} kW")
    with col2:
        st.metric("Solar PV Output", f"{solar_pred:.1f} kW")
    with col3:
        st.metric("Battery SOC", f"{battery_soc:.1f}%")
    with col4:
        st.metric("Diesel Generator Usage", f"{diesel_pred:.1f} kW")

    # Time-Series Trends
    st.header("Time-Series Trends")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=df_filtered.index, y="load_demand", data=df_filtered, label="Actual Demand", ax=ax)
    if model_choice in ["Random Forest", "Both"]:
        sns.lineplot(x=df_filtered.index, y=rf_pred_demand, label="RF Prediction", ax=ax)
    if model_choice in ["XGBoost", "Both"]:
        sns.lineplot(x=df_filtered.index, y=xgb_pred_demand, label="XGB Prediction", ax=ax)
    if show_weather:
        ax2 = ax.twinx()
        sns.lineplot(x=df_filtered.index, y="temperature", data=df_filtered, color="gray", label="Temperature", ax=ax2)
    ax.set_ylabel("Demand (kW)")
    ax.legend()
    st.pyplot(fig)

    # Forecast Panel
    st.header("24-Hour Forecast")
    future_hours = pd.date_range(start=end_time, periods=25, freq="H")[1:]
    X_future = pd.DataFrame({
        "hour": future_hours.hour,
        "day_of_week": future_hours.dayofweek,
        "month": future_hours.month,
        "demand_prev_hour": np.roll(y_true[-24:], -1)[:24],
        "demand_prev_day": y_true[-24:].values,
        "demand_rolling_7d": df_filtered["demand_rolling_7d"].iloc[-24:].values,
        "demand_rolling_24h": df_filtered["demand_rolling_24h"].iloc[-24:].values,
        "hour_demand": future_hours.hour * y_true[-24:].values
    })
    rf_future_demand = rf_demand.predict(X_future)
    dmat_future = xgb.DMatrix(X_future)
    xgb_future_demand = xgb_demand.predict(dmat_future)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(future_hours, rf_future_demand, label="RF Forecast", color="blue")
    ax.plot(future_hours, xgb_future_demand, label="XGB Forecast", color="orange")
    ax.set_ylabel("Predicted Demand (kW)")
    ax.legend()
    st.pyplot(fig)

    # Feature Importance
    def plot_feature_importance(rf_model, xgb_model, input_data, features):
        st.header("📊 Feature Importance")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(x=rf_model.feature_importances_, y=features, ax=ax1)
        ax1.set_title("Random Forest Feature Importance")
        ax1.set_xlabel("Importance")
        ax1.set_ylabel("Features")
        try:
            dmatrix = xgb.DMatrix(input_data[features], feature_names=features)
            explainer = shap.Explainer(xgb_model)
            shap_values = explainer(dmatrix)
            shap_importance = np.abs(shap_values.values).mean(axis=0)
            sns.barplot(x=shap_importance, y=features, ax=ax2)
            ax2.set_title("XGBoost Feature Importance (SHAP)")
            ax2.set_xlabel("Mean |SHAP Value|")
            ax2.set_ylabel("Features")
        except Exception as e:
            st.error(f"SHAP importance failed: {e}")
            ax2.text(0.5, 0.5, "SHAP Plot Failed", ha="center", va="center", transform=ax2.transAxes)
        plt.tight_layout()
        st.pyplot(fig)

    plot_feature_importance(rf_demand, xgb_demand, df_filtered, demand_features)

    # Energy Balance
    st.header("Energy Balance")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(df_filtered.index, df_filtered["solar_pv_output"], 
                 df_filtered["battery_discharge"], df_filtered["diesel_generator_usage"],
                 labels=["Solar", "Battery", "Diesel"])
    ax.plot(df_filtered.index, df_filtered["load_demand"], label="Demand", color="black")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    st.pyplot(fig)

    # Model Performance
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Random Forest")
        st.write(f"RMSE: {mean_squared_error(y_true, rf_pred_demand, squared=False):.2f}")
        st.write(f"MAE: {mean_absolute_error(y_true, rf_pred_demand):.2f}")
        st.write(f"R²: {r2_score(y_true, rf_pred_demand):.2f}")
    with col2:
        st.subheader("XGBoost")
        st.write(f"RMSE: {mean_squared_error(y_true, xgb_pred_demand, squared=False):.2f}")
        st.write(f"MAE: {mean_absolute_error(y_true, xgb_pred_demand):.2f}")
        st.write(f"R²: {r2_score(y_true, xgb_pred_demand):.2f}")

# Instructions
st.sidebar.markdown("Adjust the controls to customize the view. Use the sliders to select the end time and date range.")

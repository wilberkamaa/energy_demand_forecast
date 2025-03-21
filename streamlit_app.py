import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb  # Explicitly import xgboost
import requests
import os
import warnings

# Suppress XGBoost warnings about version mismatch (temporary until model is updated)
warnings.filterwarnings("ignore", category=UserWarning)

# 📌 GitHub repo (raw file links)
GITHUB_REPO = "https://raw.githubusercontent.com/wilberkamaa/energy_demand_forecast/master/"
MODEL_FILES = {
    "rf_model": "random_forest_model.pkl",
    "xgb_model": "xgboost_model.pkl"
}

@st.cache_resource
def load_models():
    models = {}

    for name, filename in MODEL_FILES.items():
        # Download if the file doesn't exist locally
        if not os.path.exists(filename):
            url = GITHUB_REPO + filename
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"✅ Downloaded {filename}")
            else:
                print(f"❌ Failed to download {filename}. Check the URL.")

        # Load the model
        try:
            if name == "xgb_model":
                # Load XGBoost model and wrap it to ensure compatibility
                xgb_model = joblib.load(filename)
                # If it's an older serialized Booster, convert to current format
                if isinstance(xgb_model, xgb.Booster):
                    models[name] = xgb_model
                else:
                    # Assume it's an XGBRegressor or similar and extract Booster
                    models[name] = xgb_model.get_booster()
                print(f"✅ Loaded XGBoost model from {filename}")
            else:
                # Load Random Forest model
                models[name] = joblib.load(filename)
                print(f"✅ Loaded {filename}")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")

    return models["rf_model"], models["xgb_model"]

# 📌 Load the models
rf_model, xgb_model = load_models()

# 📌 Load data for visualization
df = pd.read_csv("https://raw.githubusercontent.com/wilberkamaa/energy_demand_forecast/refs/heads/master/energy_data.csv")
df.drop(columns=["timestamp"], inplace=True)

# 🔹 Sidebar - User Inputs
st.sidebar.header("Adjust Prediction Inputs")
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 3)
month = st.sidebar.slider("Month", 1, 12, 6)
demand_prev_hour = st.sidebar.number_input("Demand Previous Hour", value=300.0)
demand_prev_day = st.sidebar.number_input("Demand Previous Day", value=350.0)
demand_rolling_7d = st.sidebar.number_input("7-Day Rolling Demand", value=320.0)
demand_rolling_24h = st.sidebar.number_input("24-Hour Rolling Demand", value=310.0)

# 🔹 Prepare input features
input_data = pd.DataFrame({
    "hour": [hour],
    "day_of_week": [day_of_week],
    "month": [month],
    "demand_prev_hour": [demand_prev_hour],
    "demand_prev_day": [demand_prev_day],
    "demand_rolling_7d": [demand_rolling_7d],
    "demand_rolling_24h": [demand_rolling_24h]
})
input_data["hour_demand"] = input_data["hour"] * input_data["demand_prev_day"]

# 🔹 Select model
model_choice = st.sidebar.radio("Choose Model", ["Random Forest", "XGBoost"])

# 📌 Make prediction
if model_choice == "Random Forest":
    prediction = rf_model.predict(input_data)[0]
else:
    # For XGBoost, use the Booster predict method
    dmatrix = xgb.DMatrix(input_data)
    prediction = xgb_model.predict(dmatrix)[0]

st.sidebar.subheader(f"Predicted Demand: {prediction:.2f}")

# 🔹 Plot past demand vs prediction
st.subheader("🔍 Load Demand Forecasting")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index[-100:], df["load_demand"].values[-100:], label="Actual Demand", color="blue", linestyle="dashed")
ax.axhline(prediction, color="red", linestyle="dotted", label=f"Predicted Demand ({model_choice})")
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Load Demand")
st.pyplot(fig)

# 📌 SHAP Explanation (Only for tree models)
if st.checkbox("Show Feature Importance (SHAP)"):
    st.subheader("📊 SHAP Feature Importance")
    if model_choice == "Random Forest":
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(input_data)
        shap.summary_plot(shap_values, input_data)
    else:
        # XGBoost SHAP with Booster
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(input_data)
        shap.summary_plot(shap_values, input_data)
    st.pyplot(plt.gcf())  # Use gcf() to get the current figure

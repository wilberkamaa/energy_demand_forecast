## Energy Load Demand Forecasting Dashboard

### Overview

This project predicts energy load demand using XGBoost and visualizes the results in a Streamlit dashboard. The model is trained on historical energy consumption data, incorporating factors like time, weather variations, and past demand trends.

**Features**

- Machine Learning Model: Uses XGBoost for demand forecasting.

- Data Visualization: Displays load predictions using Streamlit.

- Synthetic Data Support: Simulates demand scenarios when real-world data is unavailable.

- User Interaction: Allows dynamic input adjustments.


The model expects a CSV file with the following columns:

- timestamp – Date and time of energy measurement

- load_demand – Energy consumption value

- temperature – Ambient temperature (if available)

- weather_conditions – Categorical/weather impact data

### Model Training

The model is trained using historical energy demand data. It learns patterns and predicts future demand based on input features. If new data is available, the model can be retrained dynamically.

**Future Enhancements**

- Improve model accuracy with additional external factors.

- Deploy the model using a cloud-based API.

- Add real-time data streaming for live forecasting.


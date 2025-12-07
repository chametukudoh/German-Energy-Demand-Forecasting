import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="German Energy Demand Forecasting", page_icon="🔋", layout="wide")

MODEL_CANDIDATES = [
    Path("energy_forecast_model.pkl"),  # user-provided override
    Path("models/stacked_ensemble.joblib"),  # CLI-trained model
]


@st.cache_resource
def load_model():
    """Load the trained model from the first available candidate path."""
    for path in MODEL_CANDIDATES:
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f), path
    return None, None


def build_feature_row(date: datetime, hour: int) -> dict:
    """Construct a single feature row with basic calendar features and cyclical encodings."""
    dow = date.weekday()
    month = date.month
    is_weekend = 1 if dow >= 5 else 0
    features = {
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "is_weekend": is_weekend,
    }
    # Add cyclical encodings to better align with training features
    features.update(
        {
            "sin_hour": np.sin(2 * np.pi * hour / 24),
            "cos_hour": np.cos(2 * np.pi * hour / 24),
            "sin_day_of_week": np.sin(2 * np.pi * dow / 7),
            "cos_day_of_week": np.cos(2 * np.pi * dow / 7),
            "sin_month": np.sin(2 * np.pi * month / 12),
            "cos_month": np.cos(2 * np.pi * month / 12),
        }
    )
    # Placeholder defaults for any additional model features (e.g., temperature, renewables, lags)
    defaults = {
        "temperature_C": 0.0,
        "wind_generation_MW": 0.0,
        "solar_generation_MW": 0.0,
    }
    features.update({k: defaults.get(k, 0.0) for k in defaults})
    return features


model, model_path = load_model()

st.title("🔋 German Energy Demand Forecasting")
st.markdown("### Predicting Electricity Load for Germany's Energy Grid")

if model is None:
    st.error(
        "No model file found. Place `energy_forecast_model.pkl` in the project root "
        "or use `models/stacked_ensemble.joblib` from `train.py`."
    )
    st.stop()

st.sidebar.header("Input Parameters")
date_input = st.sidebar.date_input("Select Date", datetime.now())
hour_input = st.sidebar.slider("Hour of Day", 0, 23, 12)

st.sidebar.markdown(f"**Loaded model:** `{model_path}`")

features = build_feature_row(date_input, hour_input)

if st.sidebar.button("Predict"):
    X_input = pd.DataFrame([features])
    try:
        prediction = float(model.predict(X_input)[0])
        st.success(f"Predicted Load: {prediction:,.0f} MW")
        st.info(f"Confidence Interval: {prediction * 0.95:,.0f} - {prediction * 1.05:,.0f} MW")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed. Ensure model features match input schema. Details: {exc}")

st.subheader("24-Hour Forecast")
forecast_rows = []
for h in range(24):
    feats = build_feature_row(date_input, h)
    X_h = pd.DataFrame([feats])
    try:
        pred = float(model.predict(X_h)[0])
    except Exception:
        pred = np.nan
    forecast_rows.append({"Hour": h, "Predicted Load (MW)": pred})

df_forecast = pd.DataFrame(forecast_rows)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_forecast["Hour"],
        y=df_forecast["Predicted Load (MW)"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#1f77b4", width=3),
    )
)
fig.update_layout(
    title="24-Hour Load Forecast",
    xaxis_title="Hour of Day",
    yaxis_title="Load (MW)",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Average Load", f"{df_forecast['Predicted Load (MW)'].mean():,.0f} MW")
col2.metric("Peak Load", f"{df_forecast['Predicted Load (MW)'].max():,.0f} MW")
col3.metric("Min Load", f"{df_forecast['Predicted Load (MW)'].min():,.0f} MW")

st.markdown("---")
st.markdown("Built by Chamberlain Etukudoh | Data Scientist")
st.markdown("[LinkedIn](https://linkedin.com/in/chamberlain-etukudoh-770b7948)")

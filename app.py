import pickle
from datetime import datetime
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from feature_engineering import build_feature_row

try:
    import joblib  # type: ignore
except Exception:  # noqa: BLE001
    joblib = None

try:
    import xgboost as xgb  # type: ignore
except Exception as exc:  # noqa: BLE001
    xgb = None
    xgb_import_error = exc
else:
    xgb_import_error = None

try:
    import mlflow.pyfunc  # type: ignore
except Exception as exc:  # noqa: BLE001
    mlflow = None
    mlflow_import_error = exc
else:
    mlflow = mlflow
    mlflow_import_error = None


st.set_page_config(page_title="German Energy Demand Forecasting", page_icon="🔋", layout="wide")

APP_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    APP_DIR / "models" / "xgboost_optimized_model",  # tuned XGB from MLflow (pyfunc)
    APP_DIR / "energy_forecast_model.pkl",  # user-provided override (pickle)
    APP_DIR / "models" / "stacked_ensemble.joblib",  # CLI-trained model (pickle/joblib)
]


class _BoosterPredictor:
    def __init__(self, booster):
        self._booster = booster

    def predict(self, X: pd.DataFrame):
        dmatrix = xgb.DMatrix(X)  # type: ignore[union-attr]
        return self._booster.predict(dmatrix)

    def get_booster(self):
        return self._booster


@st.cache_resource
def load_model():
    """Load the trained model from the first available candidate path."""
    report = []
    report.append(f"cwd: {os.getcwd()}")
    report.append(f"app_dir: {APP_DIR}")
    report.append(f"xgboost_import: {'ok' if xgb is not None else f'failed ({xgb_import_error})'}")
    report.append(f"mlflow_import: {'ok' if mlflow is not None else f'failed ({mlflow_import_error})'}")

    for path in MODEL_CANDIDATES:
        report.append(f"candidate: {path} (exists={path.exists()})")
        if not path.exists():
            continue
        # MLflow pyfunc directory
        if path.is_dir() and (path / "MLmodel").exists() and mlflow is not None:
            try:
                model = mlflow.pyfunc.load_model(path.as_posix())
                report.append(f"loaded: mlflow.pyfunc ({path})")
                return model, path, report
            except Exception as exc:  # noqa: BLE001
                report.append(f"load_failed: mlflow.pyfunc ({path}) -> {exc}")
        # MLflow-exported XGBoost model without requiring MLflow (Streamlit Cloud-friendly)
        if path.is_dir() and (path / "model.xgb").exists() and xgb is not None:
            model_file = path / "model.xgb"
            try:
                booster = xgb.Booster()
                booster.load_model(model_file.as_posix())
                report.append(f"loaded: xgb.Booster.load_model ({model_file})")
                return _BoosterPredictor(booster), model_file, report
            except Exception as exc:  # noqa: BLE001
                report.append(f"load_failed: xgb.Booster.load_model ({model_file}) -> {exc}")

            try:
                tmp_dir = Path(tempfile.gettempdir())
                tmp_model = tmp_dir / f"{model_file.stem}-{model_file.stat().st_size}.ubj"
                if not tmp_model.exists():
                    tmp_model.write_bytes(model_file.read_bytes())
                booster = xgb.Booster()
                booster.load_model(tmp_model.as_posix())
                report.append(f"loaded: xgb.Booster.load_model (forced .ubj via {tmp_model})")
                return _BoosterPredictor(booster), model_file, report
            except Exception as exc2:  # noqa: BLE001
                report.append(f"load_failed: xgb.Booster.load_model (forced .ubj) -> {exc2}")

            try:
                tmp_dir = Path(tempfile.gettempdir())
                tmp_model = tmp_dir / f"{model_file.stem}-{model_file.stat().st_size}.json"
                if not tmp_model.exists():
                    tmp_model.write_bytes(model_file.read_bytes())
                booster = xgb.Booster()
                booster.load_model(tmp_model.as_posix())
                report.append(f"loaded: xgb.Booster.load_model (forced .json via {tmp_model})")
                return _BoosterPredictor(booster), model_file, report
            except Exception as exc3:  # noqa: BLE001
                report.append(f"load_failed: xgb.Booster.load_model (forced .json) -> {exc3}")
        # Pickle / joblib
        if path.is_file():
            if path.suffix.lower() == ".joblib" and joblib is not None:
                try:
                    report.append(f"loaded: joblib ({path})")
                    return joblib.load(path), path, report
                except Exception as exc:  # noqa: BLE001
                    report.append(f"load_failed: joblib ({path}) -> {exc}")
            try:
                with open(path, "rb") as f:
                    report.append(f"loaded: pickle ({path})")
                    return pickle.load(f), path, report
            except Exception as exc:  # noqa: BLE001
                report.append(f"load_failed: pickle ({path}) -> {exc}")
    return None, None, report


model, model_path, load_report = load_model()
expected_features = None
# Try to infer expected feature order from the loaded model
if model is not None:
    for attr in ("feature_names_in_", "feature_names"):
        if hasattr(model, attr):
            expected_features = list(getattr(model, attr))
            break
    if expected_features is None and hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            if booster.feature_names:
                expected_features = list(booster.feature_names)
        except Exception:
            pass
    if expected_features is None and hasattr(model, "metadata"):
        try:
            sig = model.metadata.get_input_schema()
            if sig and getattr(sig, "input_names", None):
                expected_features = list(sig.input_names())
        except Exception:
            pass

st.title("🔋 German Energy Demand Forecasting")
st.markdown("### Predicting Electricity Load for Germany's Energy Grid")

if model is None:
    st.error(
        "No model file found. Place `energy_forecast_model.pkl` in the project root "
        "or place a trained artifact under `models/` (e.g. `models/stacked_ensemble.joblib`)."
    )
    st.caption("Debug info (what the app tried to load):")
    st.code("\n".join(load_report))
    st.stop()

st.sidebar.header("Input Parameters")
date_input = st.sidebar.date_input("Select Date", datetime.now())
hour_input = st.sidebar.slider("Hour of Day", 0, 23, 12)
temp_input = st.sidebar.number_input("Temperature (°C)", value=0.0)
wind_input = st.sidebar.number_input("Wind generation (MW)", value=0.0, min_value=0.0)
solar_input = st.sidebar.number_input("Solar generation (MW)", value=0.0, min_value=0.0)

st.sidebar.markdown(f"**Loaded model:** `{model_path}`")
st.sidebar.warning(
    "Demo limitation: lag and rolling-load features use zero placeholders until a recent-load feed is connected."
)

features = build_feature_row(date_input, hour_input)
# enrich with weather/renewables inputs
features.update(
    {
        "temperature_C": temp_input,
        "wind_generation_MW": wind_input,
        "solar_generation_MW": solar_input,
    }
)

# Interaction features (must match training order - these come before lag features)
renewable_total = features["wind_generation_MW"] + features["solar_generation_MW"]
features["temp_hour"] = temp_input * features["hour"]
features["renewable_total"] = renewable_total
features["renewable_ratio"] = features["wind_generation_MW"] / (features["solar_generation_MW"] + 1)

# Lag/rolling placeholders (if you have recent load history, wire it here)
placeholders = {
    "load_lag_1": 0.0,
    "load_lag_24": 0.0,
    "load_lag_168": 0.0,
    "load_rolling_mean_24": 0.0,
    "load_rolling_std_24": 0.0,
    "temp_rolling_mean_24": temp_input,
    "temp_squared": temp_input**2,
    "temp_deviation": temp_input - temp_input,
}
features.update(placeholders)

if st.sidebar.button("Predict"):
    X_input = pd.DataFrame([features])
    if expected_features:
        X_input = X_input.reindex(columns=expected_features, fill_value=0.0)
    try:
        prediction = float(model.predict(X_input)[0])
        st.success(f"Predicted Load: {prediction:,.0f} MW")
        st.caption("A calibrated prediction interval is not available in this prototype.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed. Ensure model features match input schema. Details: {exc}")

st.subheader("24-Hour Forecast")
forecast_rows = []
for h in range(24):
    feats = build_feature_row(date_input, h)
    # Add weather/renewables inputs
    feats.update({
        "temperature_C": temp_input,
        "wind_generation_MW": wind_input,
        "solar_generation_MW": solar_input,
    })
    # Interaction features (same order as above)
    renewable_total_h = feats["wind_generation_MW"] + feats["solar_generation_MW"]
    feats["temp_hour"] = temp_input * feats["hour"]
    feats["renewable_total"] = renewable_total_h
    feats["renewable_ratio"] = feats["wind_generation_MW"] / (feats["solar_generation_MW"] + 1)
    # Lag/rolling placeholders
    feats.update({
        "load_lag_1": 0.0,
        "load_lag_24": 0.0,
        "load_lag_168": 0.0,
        "load_rolling_mean_24": 0.0,
        "load_rolling_std_24": 0.0,
        "temp_rolling_mean_24": temp_input,
        "temp_squared": temp_input**2,
        "temp_deviation": temp_input - temp_input,
    })
    X_h = pd.DataFrame([feats])
    if expected_features:
        X_h = X_h.reindex(columns=expected_features, fill_value=0.0)
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

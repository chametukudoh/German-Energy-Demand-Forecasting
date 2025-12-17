# German Energy Demand Forecasting

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://german-energy-demand-forecasting-6j78ybwhfjkpkpgmwk5tsb.streamlit.app/)

Interactive forecasting, notebooks, and CLI pipelines for hourly German electricity load using historical load, weather, renewables, and calendar features from `german_energy_load_2022_2024.csv` (2022–2024 hourly).

## 🎯 Business Problem
Germany’s grid needs accurate short-term demand forecasts to balance renewables and keep the Energiewende on track. This project builds tree-based models and ensembles to predict hourly load and surface insights on seasonality and driver importance.

## 📊 Data
- Source: Prepared CSV (`german_energy_load_2022_2024.csv`) derived from ENTSO-E-style transparency data.
- Horizon: 2022–2024 hourly.
- Features: calendar (hour/day/month, weekend), cyclical encodings, lags (1h/24h/168h), rolling stats (24h/168h), temperature, wind, solar.

## 🤖 Model Performance (from MLflow runs)
| Model           | MAE (MW) | RMSE (MW) | R²    |
|-----------------|---------:|----------:|:------|
| Random Forest   | ~1,080   | ~1,297    | 0.976 |
| XGBoost         | ~1,087   | ~1,307    | 0.976 |
| Ensemble (stack)| -        | -         | -     |

Notes: RF slightly edges XGB on this dataset; XGB is more regularized (smaller train–test gap). The repo includes an MLflow `pyfunc` artifact under `models/xgboost_optimized_model/` for the Streamlit app.

## 🚀 Live Demo
Streamlit app: `https://german-energy-demand-forecasting-6j78ybwhfjkpkpgmwk5tsb.streamlit.app/`

Local run:
```bash
streamlit run app.py
```

Docker:
```bash
docker build -t energy-forecast-streamlit .
docker run --rm -p 8501:8501 energy-forecast-streamlit
```
Open `http://localhost:8501`.

Docker Compose:
```bash
docker compose up --build
```
Open `http://localhost:8501`.

Deployment notes (Streamlit Community Cloud):
- Python is pinned via `runtime.txt` to `python-3.11` to avoid building heavy deps on newer interpreters.
- System dependency `libgomp1` is installed via `packages.txt` for XGBoost/LightGBM OpenMP runtime support.
  - The app loads `models/xgboost_optimized_model/model.xgb` directly (no MLflow dependency) to avoid `pyarrow` builds on Streamlit Cloud.

## 📈 Key Features
- Hourly load prediction with 24-hour forecast plot and summary metrics.
- Feature engineering: cyclical time encodings, lags, rolling stats, weather/renewables.
- Optuna tuning workflow for XGBoost.
- MLflow tracking for parameters, metrics, and artifacts.
- Two notebooks: quick MLflow walkthrough and rich EDA/modeling narrative.

## 💻 Technical Stack
- Python 3.11 recommended (3.10/3.12 OK)
- ML: scikit-learn, XGBoost, LightGBM (CLI ensemble)
- Data: pandas, numpy
- Viz: matplotlib, seaborn, plotly
- Tracking: MLflow
- App: Streamlit

## 🛠️ Setup
```bash
git clone https://github.com/chametukudoh/German-Energy-Demand-Forecasting.git
cd German-Energy-Demand-Forecasting
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
For training/EDA dependencies:
```bash
python -m pip install -r requirements-dev.txt
```

## Models
- Inference artifacts used by the Streamlit app:
  - `models/xgboost_optimized_model/` (MLflow pyfunc)
  - Optional override: `energy_forecast_model.pkl` (root)

## Notebooks
- `notebooks/energy_forecasting_mlflow.ipynb` – compact, self-contained walkthrough with MLflow logging (RF/XGB baselines, Optuna-tuned XGB).
- `notebooks/german_energy_load_forecasting.ipynb` – detailed EDA, seasonal/temporal plots, RF/XGB baselines, Optuna tuning, comparisons, and MLflow runs.

## MLflow UI
From project root:
```bash
mlflow ui --backend-store-uri mlruns
```
Open http://localhost:5000 to browse runs and artifacts.

## App Inputs / Model Expectations
- Minimal inputs: date and hour; app builds calendar/cyclical features and uses default zeros for weather/renewables. For best results, deploy a model trained with matching feature schema (e.g., `energy_forecast_model.pkl` or `models/stacked_ensemble.joblib`).

## 📧 Contact
**Chamberlain Etukudoh**  
Data Scientist | Open to opportunities in 🇨🇦 🇳🇱 🇩🇪  
LinkedIn: [chamberlain-etukudoh](https://linkedin.com/in/chamberlain-etukudoh-770b7948)  
Email: chamberlainet@gmail.com

## 📄 License
MIT License

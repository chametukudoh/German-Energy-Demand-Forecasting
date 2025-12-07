# Energy Demand Forecasting

End-to-end ML pipelines for hourly electricity demand using historical load, weather, renewables, and calendar features from `german_energy_load_2022_2024.csv`. Includes CLI training, two notebooks, Optuna tuning, and MLflow experiment tracking.

## Structure
- `train.py` – CLI to train/evaluate base models and a stacked ensemble; saves metrics and artifacts.
- `src/config.py`, `src/data_loader.py`, `src/features.py`, `src/modeling.py`, `src/evaluation.py` – core pipeline pieces.
- `models/` – created at runtime for saved models, metrics, and prediction samples.
- `notebooks/energy_forecasting_mlflow.ipynb` – compact, self-contained walkthrough with MLflow logging (EDA, features, RF/XGB baselines, Optuna-tuned XGB).
- `notebooks/german_energy_load_forecasting.ipynb` – richer narrative notebook with extensive EDA, feature engineering, RF and XGB baselines, Optuna tuning, plots, and MLflow runs.
- `mlruns/` – MLflow local backend store (created by notebooks/CLI runs).

## Setup
1) Use Python 3.11 (or 3.10/3.12). Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
2) Optional: start MLflow UI to inspect runs logged by notebooks:
   ```bash
   mlflow ui --backend-store-uri mlruns
   ```

## CLI training
- Quick smoke test:
  ```bash
  python train.py --fast-dev-run --sample-frac 0.15
  ```
- Full training:
  ```bash
  python train.py
  ```
Artifacts go to `models/`:
- `stacked_ensemble.joblib` – imputer + stacking pipeline.
- `metrics.json` – validation metrics for each base model and test metrics for the stack.
- `validation_predictions.csv` – timestamps, actual load, stacked predictions on validation window.

## Notebook usage
- Open either notebook and run top-to-bottom. They are self-contained and assume the CSV is at the project root (use `../german_energy_load_2022_2024.csv` inside `notebooks/`).
- Toggle `FAST_DEV_RUN` / `N_TRIALS` in the notebooks to shorten runs.
- MLflow runs are logged under `mlruns/`; use the same path when launching the UI.

### MLflow usage
From the project root:
```bash
mlflow ui --backend-store-uri mlruns
```
Then open http://localhost:5000 to browse experiments and artifacts.

## Features engineered
- Calendar: hour/day/month, weekend flag, cyclical encodings (`sin_`, `cos_` terms).
- Load history: lags at 1h/24h/168h, rolling mean/std at 24h and 168h.
- External signals: temperature, wind, solar, plus provided calendar columns.

## Models and results
- Base learners: RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, LGBMRegressor; stacked ensemble in `train.py`.
- Notebook baselines (from MLflow runs): RF and XGB both achieve high R² (~0.976) on test; RF slightly better RMSE (~1297 MW) vs XGB baseline (~1307 MW), with XGB more regularized (smaller train–test gap). Optuna-tuned XGB is available in the notebook for further gains.

## Notes
- Splits are chronological: last 90 days for test, preceding 30 days for validation (CLI). Notebooks use time-aware splits consistent with their narratives.
- Run commands from the project root (or adjust paths) so the CSV is found.

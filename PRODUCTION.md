# Production Operations Guide
## German Energy Demand Forecasting System

---

## 📋 Table of Contents
1. [Production Monitoring](#production-monitoring)
2. [Retraining Strategy](#retraining-strategy)
3. [Drift Detection](#drift-detection)
4. [Failure Modes & Mitigations](#failure-modes--mitigations)
5. [Model Validation & Deployment](#model-validation--deployment)
6. [Incident Response](#incident-response)

---

## 🔍 Production Monitoring

### Technical Monitoring (Real-Time)

#### Inference Performance
| Metric | Target | Alert Threshold | Action |
|--------|--------|-----------------|--------|
| **Latency (p50)** | <50ms | >100ms | Scale compute resources |
| **Latency (p95)** | <100ms | >200ms | Investigate bottlenecks |
| **Throughput** | >100 req/s | <50 req/s | Check resource utilization |
| **Error rate** | 0% | >0.1% | Immediate investigation |

#### Prediction Quality Checks
| Metric | Expected Range | Alert Threshold | Significance |
|--------|----------------|-----------------|--------------|
| **Prediction mean** | 40-50 GW | <35 GW or >55 GW | Distribution shift detected |
| **Prediction std dev** | 8-10 GW | <5 GW or >15 GW | Model uncertainty issue |
| **Null predictions** | 0 | >0 | Critical: Model failure |
| **Prediction range** | 25-65 GW | Outside range | Out-of-distribution input |

#### Feature Distribution Monitoring
```python
# Weekly KS test for feature drift
features_to_monitor = ['temperature_C', 'wind_generation_MW', 'solar_generation_MW']

for feature in features_to_monitor:
    ks_statistic, p_value = ks_2samp(training_data[feature], production_data[feature])
    if p_value < 0.01:
        alert(f"Feature drift detected: {feature}")
```

**Alert Triggers:**
- Temperature distribution shifts beyond ±5°C from training mean
- Wind/solar generation >2σ from training distribution
- Kolmogorov-Smirnov test p-value <0.01 (weekly check)

---

### Business KPIs (Daily/Weekly Reporting)

#### Forecast Accuracy Metrics
| KPI | Target | Warning | Critical | Business Impact |
|-----|--------|---------|----------|-----------------|
| **MAPE (7-day rolling)** | <2.5% | >3.0% | >4.0% | Balancing cost increase >€1M/month |
| **RMSE (7-day rolling)** | <1,300 MW | >1,500 MW | >2,000 MW | Reserve capacity inefficiency |
| **Forecast bias** | ±100 MW | ±200 MW | ±500 MW | Systematic under/over forecasting |
| **95%ile absolute error** | <3,000 MW | >3,500 MW | >5,000 MW | Emergency reserve activation |
| **Correlation (r) with actuals** | >0.95 | <0.93 | <0.90 | Model losing predictive power |

#### Operational Metrics
```
Daily Report Template:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
German Energy Forecast - Daily Summary
Date: 2024-12-01
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Accuracy Metrics:
  MAPE (24h):       2.3% ✅
  MAPE (7-day):     2.5% ✅
  RMSE (24h):       1,245 MW ✅
  Max Error:        2,890 MW (Hour 18)
  Forecast Bias:    +87 MW (slight over-prediction)

Predictions Served: 24
  Successful:       24 (100%)
  Failed:           0
  Avg Latency:      38ms

Feature Health:
  Temperature:      ✅ Within range
  Wind Gen:         ✅ Normal
  Solar Gen:        ⚠️  Higher than usual (+15%)

Alerts: 0 critical, 1 warning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔄 Retraining Strategy

### Scheduled Retraining

**Cadence:** Monthly batch retraining with rolling 6-month window

**Schedule:**
- **Day 1 of month:** Extract latest data (6 months)
- **Day 2-3:** Feature engineering, model training, hyperparameter tuning
- **Day 4-5:** Validation (backtesting + shadow mode)
- **Day 6-7:** Gradual deployment (10% → 50% → 100%)
- **Day 8+:** Monitor performance differential

**Data Requirements:**
- Minimum 180 days of historical data
- Maximum 365 days (diminishing returns beyond 1 year)
- Must include at least one full seasonal cycle (winter + summer)

---

### Trigger-Based Retraining

Retrain immediately if **any** of the following occurs:

| Trigger | Threshold | Validation Required | Priority |
|---------|-----------|---------------------|----------|
| **Accuracy degradation** | MAPE >4% for 3 consecutive days | 4-week backtest | 🔴 High |
| **Feature drift** | KS test p<0.01 for any feature | 2-week backtest | 🟡 Medium |
| **Data accumulation** | 10,000+ new samples since last training | 2-week backtest | 🟢 Low |
| **Seasonal transition** | March 15-April 15, Sept 15-Oct 15 | 4-week backtest | 🟡 Medium |
| **Systematic bias** | Forecast bias >500 MW for 7 days | 4-week backtest | 🔴 High |
| **Distribution shift** | Mean prediction outside 40-50 GW for 5 days | Full validation | 🔴 High |

**Emergency Retraining Protocol:**
1. **Trigger identified** → Automated alert to ML team
2. **Root cause analysis** (1-2 hours): Data quality issue? Weather anomaly? Model drift?
3. **Decision:** Retrain vs. adjust features vs. rollback to previous model
4. **If retraining:** Expedited training (24-48 hours)
5. **Shadow mode validation:** 48 hours minimum
6. **Deployment:** Gradual rollout with rollback plan

---

### Retraining Workflow

```bash
# Step 1: Data extraction
python scripts/extract_data.py --start-date 2024-06-01 --end-date 2024-12-01

# Step 2: Feature engineering
python scripts/feature_engineering.py --input data/raw/load_2024.csv --output data/processed/

# Step 3: Model training with Optuna
python scripts/train_xgboost.py --trials 50 --cv-folds 5

# Step 4: Validation
python scripts/validate_model.py --backtest-weeks 4 --model models/candidate_model.xgb

# Step 5: Deploy to shadow mode
python scripts/deploy.py --mode shadow --model models/candidate_model.xgb

# Step 6: Compare shadow vs. production
python scripts/compare_models.py --duration 7days

# Step 7: Gradual rollout
python scripts/deploy.py --mode production --traffic 0.1  # 10%
# Monitor for 24h, then increase
python scripts/deploy.py --mode production --traffic 0.5  # 50%
# Monitor for 24h, then full
python scripts/deploy.py --mode production --traffic 1.0  # 100%
```

---

## 📊 Drift Detection

### Feature Drift

**Method:** Kolmogorov-Smirnov (KS) two-sample test

**Implementation:**
```python
from scipy.stats import ks_2samp
import pandas as pd

def detect_feature_drift(training_data, production_data, features, threshold=0.01):
    """
    Detect distribution shifts between training and production data.

    Returns:
        dict: {feature: {'ks_stat': float, 'p_value': float, 'drift': bool}}
    """
    results = {}
    for feature in features:
        ks_stat, p_value = ks_2samp(
            training_data[feature].dropna(),
            production_data[feature].dropna()
        )
        results[feature] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'drift_detected': p_value < threshold
        }
    return results

# Run weekly
drift_report = detect_feature_drift(
    training_df,
    last_7_days_production_df,
    features=['temperature_C', 'wind_generation_MW', 'solar_generation_MW']
)
```

**Thresholds:**
- **p-value <0.01:** Significant drift → trigger retraining review
- **p-value <0.05:** Moderate drift → increase monitoring frequency
- **p-value ≥0.05:** No drift → normal operations

---

### Prediction Drift

**Residual Auto-correlation Check:**
```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# Monthly check: residuals should be white noise
residuals = actual_load - predicted_load
lb_test = acorr_ljungbox(residuals, lags=24, return_df=True)

if (lb_test['lb_pvalue'] < 0.05).any():
    alert("Residual auto-correlation detected - model may be missing temporal patterns")
```

**Calibration Check:**
```python
# Monthly: Predictions should be unbiased
mean_error = (predicted_load - actual_load).mean()
if abs(mean_error) > 200:  # MW
    alert(f"Systematic bias detected: {mean_error:.0f} MW")
```

---

### Concept Drift

**Indicators:**
1. **Gradual accuracy decline:** MAPE increases 0.1%+ per week for 4 consecutive weeks
2. **Feature importance shifts:** Top-3 features change rank order
3. **Seasonal pattern changes:** Peak hours shift from historical norms
4. **Weather sensitivity changes:** Temperature coefficient changes >20%

**Response:** Scheduled retraining within 2 weeks

---

## ⚠️ Failure Modes & Mitigations

### 1. Extreme Weather Events

**Symptoms:**
- Forecast errors spike >5,000 MW
- Temperature outside training range (e.g., heat wave >40°C, cold snap <-20°C)
- Unprecedented wind/solar generation levels

**Examples:**
- Winter Storm Uri (2021): Extreme cold → demand surge
- European heat wave (2022): AC demand spike + solar over-generation

**Mitigation:**
```python
# Fallback logic in production
def predict_with_fallback(model, features):
    # Check for out-of-distribution inputs
    if features['temperature_C'] < -20 or features['temperature_C'] > 40:
        # Use persistence + temperature correction
        return persistence_forecast(features) + temperature_adjustment(features)

    # Normal prediction
    return model.predict(features)
```

**Operational Response:**
- Switch to manual forecasting for extreme weather (TSO override)
- Post-event: Retrain model including extreme weather data
- Update alert thresholds to detect extreme weather early

---

### 2. Holiday & Special Events

**Symptoms:**
- 10-15% systematic under-prediction on holidays
- Errors concentrated on specific calendar days (Christmas, New Year, Easter)

**Root Cause:**
- Load patterns differ significantly from typical weekdays/weekends
- Current model lacks explicit holiday features

**Mitigation:**
```python
# Pre-computed holiday adjustments
HOLIDAY_ADJUSTMENTS = {
    'christmas_day': -0.12,      # 12% load reduction
    'new_years_day': -0.10,      # 10% load reduction
    'good_friday': -0.08,        # 8% load reduction
    'easter_monday': -0.07,
    # ... other holidays
}

def adjust_for_holidays(prediction, date):
    if date in HOLIDAY_ADJUSTMENTS:
        return prediction * (1 + HOLIDAY_ADJUSTMENTS[date])
    return prediction
```

**Long-term Fix:**
- Add binary holiday features to model
- Retrain with labeled holiday data
- Validate improvement vs. adjustment approach

---

### 3. Grid Outages & Missing Lag Features

**Symptoms:**
- Actual load data unavailable for lag feature calculation
- Lag features become stale/invalid
- Model predictions degrade (MAPE increases 1-2%)

**Root Cause:**
- Communication failures with TSO data feeds
- Grid sensor outages
- Data pipeline failures

**Mitigation:**
```python
# Graceful degradation for missing lags
def compute_lag_features_with_fallback(current_data, historical_data):
    try:
        # Try to get actual lags
        lag_1h = historical_data.loc[current_time - 1h, 'load_MW']
        lag_24h = historical_data.loc[current_time - 24h, 'load_MW']
        lag_168h = historical_data.loc[current_time - 168h, 'load_MW']
    except KeyError:
        # Fallback to historical averages
        lag_1h = historical_data.groupby('hour')['load_MW'].mean()[current_hour]
        lag_24h = historical_data.groupby(['day_of_week', 'hour'])['load_MW'].mean()[(current_dow, current_hour)]
        lag_168h = lag_24h  # Use same-day-of-week average

        alert("Using historical averages for lag features due to missing data")

    return lag_1h, lag_24h, lag_168h
```

**Operational Response:**
- Automated fallback to historical averages
- Alert TSO data team for feed restoration
- Increased monitoring during outage period

---

### 4. Model File Corruption

**Symptoms:**
- Model fails to load on app startup
- All 3 fallback methods (MLflow pyfunc, XGB Booster, pickle) fail
- Error: "corrupted file" or "version mismatch"

**Root Cause:**
- Disk corruption
- Incomplete model upload during deployment
- Version incompatibility (XGBoost version mismatch)

**Mitigation:**
```python
# Multi-tier backup strategy
MODEL_BACKUP_LOCATIONS = [
    "/app/models/xgboost_optimized_model/",  # Primary
    "s3://energy-forecast-models/prod/current/",  # Cloud backup
    "/app/models/backup/xgboost_last_known_good/",  # Local rollback
]

def load_model_with_backups():
    for location in MODEL_BACKUP_LOCATIONS:
        try:
            model = load_model(location)
            log(f"Model loaded from: {location}")
            return model
        except Exception as e:
            log(f"Failed to load from {location}: {e}")

    # Critical: All backups failed
    alert("CRITICAL: All model backups failed to load")
    raise ModelLoadError("No valid model available")
```

**Prevention:**
- Automated model checksums (MD5/SHA256) validation
- Cloud backup after every deployment
- Version pinning in requirements.txt
- Pre-deployment smoke tests

---

### 5. Daylight Saving Time Transitions

**Symptoms:**
- Prediction errors on DST transition days (spring: 23-hour day, fall: 25-hour day)
- Missing hour (spring) or duplicate hour (fall) causes lag misalignment

**Root Cause:**
- Time-based indexing breaks when hour 2:00-3:00 is skipped (spring) or repeated (fall)

**Mitigation:**
```python
# Handle DST transitions during feature engineering
def handle_dst_transitions(df):
    """
    Spring (March): 2:00 AM → 3:00 AM (23-hour day)
      - Fill missing hour with linear interpolation

    Fall (October): 2:00 AM appears twice (25-hour day)
      - Average the two 2:00 AM observations
    """
    # Detect DST transitions
    df = df.sort_index()
    hour_diff = df.index.to_series().diff()

    # Spring: Missing hour
    missing_hours = hour_diff > pd.Timedelta(hours=1)
    for idx in df[missing_hours].index:
        # Linear interpolation
        prev_load = df.loc[idx - pd.Timedelta(hours=2), 'load_MW']
        next_load = df.loc[idx, 'load_MW']
        interpolated = (prev_load + next_load) / 2
        df.loc[idx - pd.Timedelta(hours=1), 'load_MW'] = interpolated

    # Fall: Duplicate hour
    duplicate_hours = df.index.duplicated(keep=False)
    df = df[~duplicate_hours | df.index.duplicated(keep='first')]

    return df
```

**Impact:** Removes 2 outlier predictions per year

---

## 🚀 Model Validation & Deployment

### Pre-Deployment Validation Checklist

Before deploying a new model to production:

- [ ] **Backtesting:** 4-week backtest shows MAPE improvement ≥0.3% over current production model
- [ ] **No regression:** No individual day with MAPE >5% in backtest period
- [ ] **Feature compatibility:** New model uses same feature schema as production (or migration plan exists)
- [ ] **Latency test:** p95 inference latency <100ms on production hardware
- [ ] **Smoke tests:** 100 random predictions complete without errors
- [ ] **Edge case testing:**
  - [ ] Extreme temperatures (-20°C, +40°C)
  - [ ] Zero renewables (calm, night)
  - [ ] Maximum renewables (windy, sunny)
  - [ ] Holiday dates
- [ ] **Drift detection:** Residuals pass auto-correlation test (Ljung-Box p>0.05)
- [ ] **Version control:** Model artifact, training script, hyperparameters committed to Git
- [ ] **Documentation:** Update PRODUCTION.md with any new dependencies or configuration changes

---

### Shadow Mode Testing

**Purpose:** Run new model alongside production model without affecting user-facing predictions

**Duration:** Minimum 7 days (ideally 14 days to capture 2 full weeks)

**Metrics to Compare:**
| Metric | Requirement |
|--------|-------------|
| MAPE | New model ≤ production model |
| RMSE | New model ≤ production model |
| Latency (p95) | New model <100ms |
| Error rate | New model = 0% |
| Worst-case error | New model <5,000 MW |

**Implementation:**
```python
# Shadow mode: predict with both models, serve production, log both
production_prediction = production_model.predict(features)
shadow_prediction = shadow_model.predict(features)

# Serve production prediction
response = production_prediction

# Log both for comparison
log_prediction(
    timestamp=now,
    production_pred=production_prediction,
    shadow_pred=shadow_prediction,
    features=features,
    model_version_prod=production_model.version,
    model_version_shadow=shadow_model.version
)
```

**Decision Criteria:**
- ✅ **Deploy if:** Shadow model MAPE ≤ production AND no critical errors
- ⚠️ **Extended testing if:** Shadow model MAPE within 0.1% of production (needs more data)
- ❌ **Reject if:** Shadow model MAPE >0.2% worse OR any critical errors

---

### Gradual Rollout Strategy

**Traffic Split Approach:**

| Phase | Duration | Traffic % to New Model | Rollback Trigger |
|-------|----------|------------------------|------------------|
| **Phase 1** | 24 hours | 10% | MAPE >0.5% worse than production |
| **Phase 2** | 24 hours | 50% | MAPE >0.3% worse than production |
| **Phase 3** | 48 hours | 100% | MAPE >0.2% worse than production |

**Rollback Protocol:**
```bash
# Immediate rollback if trigger hit
if new_model_mape > production_mape + 0.5:
    deploy.rollback(model="production_v42")
    alert("ROLLBACK: New model underperforming, reverted to production_v42")
```

**Success Criteria for Full Deployment:**
- New model serves 100% traffic for 7 days
- MAPE ≤ previous production model
- No critical alerts triggered
- TSO feedback positive or neutral

---

## 🚨 Incident Response

### Incident Severity Levels

| Level | Definition | Response Time | Example |
|-------|------------|---------------|---------|
| **P0 - Critical** | Predictions unavailable; system down | <15 minutes | Model load failure, app crash |
| **P1 - High** | Degraded accuracy; MAPE >5% | <1 hour | Extreme weather event, data feed failure |
| **P2 - Medium** | Minor degradation; MAPE 3-5% | <4 hours | Feature drift, minor data quality issues |
| **P3 - Low** | Monitoring alert; no user impact | <24 hours | Latency spike resolved, non-critical log errors |

---

### Incident Response Playbook

#### P0: System Down (Model Unavailable)

**Symptoms:** Predictions returning errors; app unavailable

**Immediate Actions (0-15 min):**
1. ✅ Check model file integrity (all 3 fallback methods)
2. ✅ Verify app container health
3. ✅ Check cloud infrastructure status
4. ✅ Rollback to last-known-good model backup
5. ✅ Notify TSO stakeholders: "Forecasting system degraded, investigating"

**Resolution (15-60 min):**
1. Restore from S3 backup
2. Redeploy container
3. Validate predictions resume
4. Notify stakeholders: "System restored"

**Post-Incident (24h):**
1. Root cause analysis
2. Document failure mode
3. Update backup procedures

---

#### P1: High Forecast Error (MAPE >5%)

**Symptoms:** Forecast errors spike; TSO reports large deviations

**Immediate Actions (0-60 min):**
1. ✅ Check for extreme weather (temperature, wind, solar anomalies)
2. ✅ Verify data feed quality (missing values, stale data)
3. ✅ Review recent predictions for patterns
4. ✅ Switch to fallback model (persistence + adjustment) if needed
5. ✅ Notify TSO: "High error event detected, using fallback forecasts"

**Resolution (1-8h):**
1. If extreme weather: Activate manual forecasting protocol
2. If data issue: Repair data feed, backfill missing values
3. If model drift: Trigger emergency retraining
4. Validate forecasts return to acceptable range

**Post-Incident (48h):**
1. Analyze event root cause
2. Update extreme weather thresholds
3. Consider model retraining with event data

---

### On-Call Escalation

**Tier 1 - ML Engineer (Primary):**
- Responds to all monitoring alerts
- Handles P2/P3 incidents
- Escalates P0/P1 to Tier 2

**Tier 2 - Senior ML Engineer / Data Scientist:**
- Handles P0/P1 incidents
- Makes rollback/retraining decisions
- Escalates to stakeholders if prolonged outage

**Tier 3 - Engineering Manager / TSO Liaison:**
- Stakeholder communication
- Business impact assessment
- Post-incident review oversight

---

## 📞 Contact & Escalation

**ML Team:**
- Primary: Chamberlain Etukudoh (chamberlainet@gmail.com)
- Slack: #energy-forecast-alerts
- PagerDuty: energy-forecast-oncall

**Stakeholders:**
- TSO Operations: [TSO contact]
- Business Owner: [Product Manager]

**Escalation Path:**
P3 → Slack alert → ML Engineer
P2 → Email + Slack → ML Engineer (4h SLA)
P1 → PagerDuty → Senior ML Engineer (1h SLA)
P0 → PagerDuty → Senior ML Engineer + Manager (15min SLA)

---

## 📚 Additional Resources

- [Model Training Notebook](notebooks/german_energy_load_forecasting.ipynb)
- [MLflow Experiment Tracking](http://localhost:5000)
- [Streamlit App](https://german-energy-demand-forecasting-6j78ybwhfjkpkpgmwk5tsb.streamlit.app/)
- [GitHub Repository](https://github.com/chametukudoh/German-Energy-Demand-Forecasting)

**Document Version:** 1.0
**Last Updated:** 2024-12-01
**Owner:** Chamberlain Etukudoh

"""Feature helpers shared by the Streamlit app and smoke tests."""

from math import cos, pi, sin


def build_feature_row(date, hour: int) -> dict:
    """Construct calendar features matching the tuned XGBoost schema."""
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")

    dow = date.weekday()
    month = date.month

    return {
        "temperature_C": 0.0,
        "wind_generation_MW": 0.0,
        "solar_generation_MW": 0.0,
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "is_weekend": int(dow >= 5),
        "year": date.year,
        "hour_sin": sin(2 * pi * hour / 24),
        "hour_cos": cos(2 * pi * hour / 24),
        "day_sin": sin(2 * pi * dow / 7),
        "day_cos": cos(2 * pi * dow / 7),
        "month_sin": sin(2 * pi * month / 12),
        "month_cos": cos(2 * pi * month / 12),
        "quarter": ((month - 1) // 3) + 1,
        "day_of_year": date.timetuple().tm_yday,
        "week_of_year": date.isocalendar().week,
        "is_working_hours": int(8 <= hour <= 18),
        "is_night": int(hour <= 5 or hour >= 22),
        "is_peak_morning": int(7 <= hour <= 10),
        "is_peak_evening": int(17 <= hour <= 20),
    }

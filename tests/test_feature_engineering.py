from datetime import datetime
from math import isclose

import pytest

from feature_engineering import build_feature_row


def test_build_feature_row_marks_weekend_and_peak_hours():
    features = build_feature_row(datetime(2026, 6, 20), 9)

    assert features["is_weekend"] == 1
    assert features["is_working_hours"] == 1
    assert features["is_peak_morning"] == 1
    assert features["quarter"] == 2


def test_cyclical_hour_encoding_wraps_at_midnight():
    features = build_feature_row(datetime(2026, 6, 19), 0)

    assert isclose(features["hour_sin"], 0.0, abs_tol=1e-12)
    assert isclose(features["hour_cos"], 1.0, abs_tol=1e-12)


def test_rejects_invalid_hour():
    with pytest.raises(ValueError, match="hour must be between 0 and 23"):
        build_feature_row(datetime(2026, 6, 19), 24)

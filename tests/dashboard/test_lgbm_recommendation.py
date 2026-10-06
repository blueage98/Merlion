#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Tests for ForecastModel.recommend_lgbm_params(), which suggests LGBMForecaster's maxlags/max_forecast_steps
in the Forecasting tab's confirmation popup before training starts.
"""
import sys

import numpy as np
import pandas as pd
import pytest

from merlion.dashboard.models.forecast import ForecastModel

# recommend_lgbm_params() resamples the data with TemporalResample, which is broken on Python 3.14
# by the AggregationPolicy/Enum bug documented in test_anomaly_model.py.
PY314_RESAMPLE_BUG = pytest.mark.xfail(
    condition=sys.version_info >= (3, 14),
    reason="Python 3.14 Enum does not treat functools.partial values as members, breaking TimeSeries.align()",
    strict=True,
)


def _seasonal_df(period=24, n_periods=40, noise=0.3, seed=0, freq="h"):
    rng = np.random.default_rng(seed)
    n = period * n_periods
    t = np.arange(n)
    values = 10 * np.sin(2 * np.pi * t / period) + noise * rng.normal(size=n)
    return pd.DataFrame({"value": values}, index=pd.date_range("2023-01-01", periods=n, freq=freq))


@PY314_RESAMPLE_BUG
def test_recommends_seasonal_period_as_maxlags():
    rec = ForecastModel.recommend_lgbm_params(_seasonal_df(period=24), "value")

    assert rec["period"] == 24
    assert rec["maxlags"] == 24
    assert rec["period_acf"] >= ForecastModel.lgbm_acf_threshold
    # a sine wave decorrelates after ~1/6 of its period (cos(2*pi*h/24) < 0.5 once h > 4)
    assert 1 <= rec["max_forecast_steps"] <= 4
    assert rec["granularity"] == 3600
    assert rec["n_points"] == 24 * 40


@PY314_RESAMPLE_BUG
def test_forecast_horizon_follows_acf_decay():
    # AR(1) with phi = 0.95 has ACF(h) = 0.95 ** h, which drops below 0.5 at h = 14
    rng = np.random.default_rng(0)
    x = np.zeros(5000)
    for i in range(1, len(x)):
        x[i] = 0.95 * x[i - 1] + rng.normal()
    df = pd.DataFrame({"value": x}, index=pd.date_range("2023-01-01", periods=len(x), freq="min"))

    rec = ForecastModel.recommend_lgbm_params(df, "value")

    assert 9 <= rec["max_forecast_steps"] <= 19
    assert rec["maxlags"] >= rec["max_forecast_steps"]


@PY314_RESAMPLE_BUG
def test_white_noise_gets_minimal_settings():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"value": rng.normal(size=500)}, index=pd.date_range("2023-01-01", periods=500, freq="h"))

    rec = ForecastModel.recommend_lgbm_params(df, "value")

    assert rec["period"] is None
    assert rec["max_forecast_steps"] == 1
    assert rec["maxlags"] == 20


@PY314_RESAMPLE_BUG
def test_irregular_timestamps_are_resampled_first():
    df = _seasonal_df(period=24)
    df = df.drop(df.index[[5, 6, 7, 100, 101, 300]])

    rec = ForecastModel.recommend_lgbm_params(df, "value")

    assert rec["granularity"] == 3600
    assert rec["n_points"] == 24 * 40
    assert rec["period"] == 24


@PY314_RESAMPLE_BUG
def test_maxlags_search_is_capped(monkeypatch):
    monkeypatch.setattr(ForecastModel, "lgbm_max_lags", 30)
    # the seasonal period (48) is beyond the cap, so it cannot be recommended
    rec = ForecastModel.recommend_lgbm_params(_seasonal_df(period=48), "value")

    assert rec["max_lag_searched"] == 30
    assert rec["maxlags"] <= 30
    assert rec["period"] is None


@PY314_RESAMPLE_BUG
def test_target_column_given_as_string_index():
    df = _seasonal_df(period=24).rename(columns={"value": 0})

    rec = ForecastModel.recommend_lgbm_params(df, "0")

    assert rec["period"] == 24


def test_missing_target_column_raises():
    with pytest.raises((AssertionError, ValueError)):
        ForecastModel.recommend_lgbm_params(_seasonal_df(), "not_a_column")


@PY314_RESAMPLE_BUG
def test_too_short_training_data_raises():
    with pytest.raises(AssertionError, match="too short"):
        ForecastModel.recommend_lgbm_params(_seasonal_df(period=4, n_periods=5), "value")


def test_steps_to_duration():
    assert ForecastModel.steps_to_duration(258, 60.0) == "0 days 04:18:00"
    assert ForecastModel.steps_to_duration(3, "MS") == "3 x MS"

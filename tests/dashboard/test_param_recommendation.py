#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Tests for the per-algorithm hyperparameter recommendations (merlion/dashboard/models/recommend.py) shown in the
Forecasting tab before training. Each test uses synthetic data whose structure (period, trend, unit roots, order)
is known, and checks that the recommended values recover it.
"""
import sys
import time

import numpy as np
import pandas as pd
import pytest

from merlion.dashboard.models.forecast import ForecastModel
from merlion.dashboard.models.recommend import SeriesStats, ndiffs, stepwise_arma_search

# Recommendations resample the data with TemporalResample, which is broken on Python 3.14
# by the AggregationPolicy/Enum bug documented in test_anomaly_model.py.
PY314_RESAMPLE_BUG = pytest.mark.xfail(
    condition=sys.version_info >= (3, 14),
    reason="Python 3.14 Enum does not treat functools.partial values as members, breaking TimeSeries.align()",
    strict=True,
)


def _df(values, freq="h", **columns):
    index = pd.date_range("2023-01-01", periods=len(values), freq=freq)
    return pd.DataFrame({"value": values, **columns}, index=index)


def _sine(period=24, n_periods=40, amplitude=10, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(period * n_periods)
    return amplitude * np.sin(2 * np.pi * t / period) + noise * rng.normal(size=len(t))


def _ar(coefs, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(len(coefs), n):
        x[i] = sum(c * x[i - k - 1] for k, c in enumerate(coefs)) + rng.normal()
    return x


def _white_noise(n=500, seed=0):
    return np.random.default_rng(seed).normal(size=n)


def _multiplicative(period=12, n_periods=40, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(period * n_periods)
    return (10 + 0.05 * t) * (1 + 0.3 * np.sin(2 * np.pi * t / period)) + 0.1 * rng.normal(size=len(t))


def _params(rec):
    return {p.name: p.value for p in rec.params}


# --- shared statistics ------------------------------------------------------------------------------------------------


@PY314_RESAMPLE_BUG
def test_series_stats_detects_period_and_seasonal_strength():
    stats = SeriesStats(_df(_sine(period=24)), "value")
    assert 24 in stats.periods
    assert stats.dominant_period() == 24
    assert stats.seasonal_strength(24) > 0.9
    assert stats.trend_strength(24) < 0.5


@PY314_RESAMPLE_BUG
def test_series_stats_white_noise_has_no_structure():
    stats = SeriesStats(_df(_white_noise()), "value")
    assert stats.periods == []
    assert stats.dominant_period() is None
    assert stats.trend_strength() < 0.2
    assert stats.seasonal_strength(24) < 0.5


@PY314_RESAMPLE_BUG
def test_series_stats_trend_and_multiplicative_seasonality():
    stats = SeriesStats(_df(_multiplicative()), "value")
    assert stats.trend_strength(12) > 0.5
    rho, pval, n_cycles = stats.amplitude_level_correlation(12)
    assert rho > 0.5 and pval < 0.05 and n_cycles >= 30
    assert stats.is_multiplicative(12, 0.5)[0]
    # an additive seasonality around a trend is not multiplicative
    t = np.arange(480)
    additive = SeriesStats(_df(10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12)), "value")
    assert not additive.is_multiplicative(12, 0.5)[0]


@PY314_RESAMPLE_BUG
def test_multiplicative_needs_positive_data():
    stats = SeriesStats(_df(_multiplicative() - 100), "value")
    mul, why = stats.is_multiplicative(12, 0.5)
    assert not mul and "<= 0" in why


@PY314_RESAMPLE_BUG
def test_max_forecast_steps_follows_acf_decay():
    # AR(1) with phi = 0.95 has ACF(h) = 0.95 ** h, which drops below 0.5 at h = 14
    rec = ForecastModel.recommend_params("Arima", _df(_ar([0.95], n=5000), freq="min"), "value")
    assert 9 <= _params(rec)["max_forecast_steps"] <= 19


# --- registry & tree models -------------------------------------------------------------------------------------------


def test_tuned_algorithms_are_registered():
    assert set(ForecastModel.tuned_algorithms) == {
        "DefaultForecaster",
        "Arima",
        "LGBMForecaster",
        "ETS",
        "Prophet",
        "Sarima",
        "VectorAR",
        "RandomForestForecaster",
        "ExtraTreesForecaster",
    }
    assert "AutoETS" not in ForecastModel.tuned_algorithms
    assert "AutoProphet" not in ForecastModel.tuned_algorithms
    with pytest.raises(ValueError):
        ForecastModel.recommend_params("AutoETS", _df(_white_noise()), "value")


@PY314_RESAMPLE_BUG
def test_lgbm_recommendation_is_unchanged():
    df = _df(_sine(period=24))
    expected = ForecastModel.recommend_lgbm_params(df, "value")
    rec = ForecastModel.recommend_params("LGBMForecaster", df, "value")
    assert _params(rec) == {"maxlags": expected["maxlags"], "max_forecast_steps": expected["max_forecast_steps"]}
    assert rec.n_points == expected["n_points"] and rec.granularity == expected["granularity"]


@PY314_RESAMPLE_BUG
@pytest.mark.parametrize("algorithm", ["RandomForestForecaster", "ExtraTreesForecaster"])
def test_tree_models_use_seasonal_period_as_maxlags(algorithm):
    rec = ForecastModel.recommend_params(algorithm, _df(_sine(period=24)), "value")
    assert _params(rec)["maxlags"] == 24
    assert "Dominant seasonal period" in rec.params[0].basis


def test_too_short_data_raises():
    with pytest.raises(AssertionError, match="too short"):
        ForecastModel.recommend_params("ETS", _df(_white_noise(n=30)), "value")


# --- Arima ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x, expected_d",
    [
        (np.cumsum(_white_noise(n=2000)), 1),
        (_ar([0.6, 0.3]), 0),
        (np.cumsum(np.cumsum(_white_noise(n=2000))), 2),
    ],
    ids=["random_walk", "stationary_ar2", "double_cumsum"],
)
def test_ndiffs(x, expected_d):
    d, pvals = ndiffs(x)
    assert d == expected_d
    assert len(pvals) >= d


def test_stepwise_search_respects_fit_budget():
    x = _ar([0.6, 0.3], n=1000)
    (p, q), aicc, n_fits = stepwise_arma_search(x, d=0, max_fits=30)
    assert p >= 1 and np.isfinite(aicc) and n_fits <= 30
    (_, _), _, n_fits = stepwise_arma_search(x, d=0, max_fits=5)
    assert n_fits <= 5


@PY314_RESAMPLE_BUG
def test_arima_recommendation():
    rec = ForecastModel.recommend_params("Arima", _df(np.cumsum(_white_noise(n=2000))), "value")
    order = _params(rec)["order"]
    assert order[1] == 1
    basis = rec.params[0].basis
    assert "KPSS" in basis and "ADF" in basis and "AICc" in basis

    rec = ForecastModel.recommend_params("Arima", _df(_ar([0.6, 0.3])), "value")
    order = _params(rec)["order"]
    assert order[1] == 0 and order[0] >= 1


@PY314_RESAMPLE_BUG
def test_arima_horizon_after_differencing_follows_the_differences():
    # a random walk's differences are white noise: once differenced, ARIMA has nothing to forecast but the level
    rec = ForecastModel.recommend_params("Arima", _df(np.cumsum(_white_noise(n=2000))), "value")
    horizon = {p.name: p for p in rec.params}["max_forecast_steps"]
    assert horizon.value <= 3
    assert "1 difference(s)" in horizon.basis and "flat at the current level" in horizon.basis
    # the shared rule (ACF of the series itself) would give a much longer horizon for the same data
    assert SeriesStats(_df(np.cumsum(_white_noise(n=2000))), "value").horizon(ForecastModel.acf_threshold) > 50


@PY314_RESAMPLE_BUG
def test_arima_horizon_without_differencing_uses_shared_rule():
    df = _df(_ar([0.6, 0.3]))
    rec = ForecastModel.recommend_params("Arima", df, "value")
    assert _params(rec)["order"][1] == 0
    assert _params(rec)["max_forecast_steps"] == SeriesStats(df, "value").horizon(ForecastModel.acf_threshold)


@PY314_RESAMPLE_BUG
def test_arima_notes_unmodeled_seasonality():
    rec = ForecastModel.recommend_params("Arima", _df(_sine(period=24)), "value")
    assert any("seasonal period of 24 steps" in n and "Use Sarima" in n for n in rec.notes)
    df = _df(_sine(period=288, n_periods=10, amplitude=5), freq="5min")
    rec = ForecastModel.recommend_params("Arima", df, "value")
    assert any("coarser granularity" in n and "Prophet" in n for n in rec.notes)
    rec = ForecastModel.recommend_params("Arima", _df(_ar([0.6, 0.3])), "value")
    assert not any("seasonal period" in n for n in rec.notes)


@PY314_RESAMPLE_BUG
def test_arima_search_on_long_data_is_bounded():
    rec_start = time.time()
    rec = ForecastModel.recommend_params("Arima", _df(np.cumsum(_white_noise(n=20000)), freq="min"), "value")
    assert time.time() - rec_start < 20
    assert rec.n_points == 20000
    assert rec.ic_points == ForecastModel.ic_max_points
    assert any(str(ForecastModel.ic_max_points) in note for note in rec.notes)


# --- Sarima -----------------------------------------------------------------------------------------------------------


@PY314_RESAMPLE_BUG
def test_sarima_daily_cycle_of_hourly_data():
    start = time.time()
    rec = ForecastModel.recommend_params("Sarima", _df(_sine(period=24)), "value")
    assert time.time() - start < 30
    seasonal_order = _params(rec)["seasonal_order"]
    assert seasonal_order[3] == 24
    assert seasonal_order[1] == 1
    assert all(v in (0, 1) for v in seasonal_order[:3])
    basis = {p.name: p.basis for p in rec.params}["seasonal_order"]
    assert "seasonal strength" in basis and "AICc" in basis
    # with a seasonal difference, the forecast repeats the last cycle: at least one cycle is covered
    horizon = {p.name: p for p in rec.params}["max_forecast_steps"]
    assert horizon.value >= 24 and "repeats the last seasonal cycle" in horizon.basis
    assert not any("seasonal period" in n for n in rec.notes)


@PY314_RESAMPLE_BUG
def test_sarima_period_longer_than_cap_has_no_seasonal_part():
    # the daily cycle of 5-minute data (288 steps) is longer than sarima_max_period
    df = _df(_sine(period=288, n_periods=10, amplitude=5), freq="5min")
    rec = ForecastModel.recommend_params("Sarima", df, "value")
    params = {p.name: p for p in rec.params}
    assert params["seasonal_order"].value == [0, 0, 0, 0]
    # the detected period may be off by one step (e.g. 289) because of the noise
    assert "longer than 24 steps" in params["seasonal_order"].basis
    assert "coarser granularity" in params["seasonal_order"].basis
    assert len(params["order"].value) == 3
    assert any("which Sarima does not model" in n for n in rec.notes)


@PY314_RESAMPLE_BUG
def test_sarima_without_seasonality():
    rec = ForecastModel.recommend_params("Sarima", _df(_ar([0.6, 0.3])), "value")
    params = {p.name: p for p in rec.params}
    assert params["seasonal_order"].value == [0, 0, 0, 0]
    assert "No significant seasonal period" in params["seasonal_order"].basis


# --- ETS --------------------------------------------------------------------------------------------------------------


@PY314_RESAMPLE_BUG
def test_ets_trend_and_multiplicative_seasonality():
    rec = ForecastModel.recommend_params("ETS", _df(_multiplicative()), "value")
    assert _params(rec) | {"max_forecast_steps": None} == {
        "error": "add",
        "trend": "add",
        "damped_trend": True,
        "seasonal": "mul",
        "seasonal_periods": 12,
        "max_forecast_steps": None,
    }


@PY314_RESAMPLE_BUG
def test_ets_white_noise():
    params = _params(ForecastModel.recommend_params("ETS", _df(_white_noise()), "value"))
    assert params["trend"] == "None"
    assert params["seasonal"] == "None"
    assert params["seasonal_periods"] is None


@PY314_RESAMPLE_BUG
def test_ets_long_period_is_recommended_with_warning():
    df = _df(_sine(period=288, n_periods=10, amplitude=5), freq="5min")
    rec = ForecastModel.recommend_params("ETS", df, "value")
    params = {p.name: p for p in rec.params}
    assert params["seasonal_periods"].value > ForecastModel.ets_slow_period
    assert "can take very long" in params["seasonal_periods"].basis
    assert any("can take very long" in note for note in rec.notes)
    # short periods have no warning
    rec = ForecastModel.recommend_params("ETS", _df(_multiplicative()), "value")
    assert rec.notes == []


@PY314_RESAMPLE_BUG
def test_ets_trains_with_recommended_settings(set_progress):
    df = _df(_multiplicative())
    train_df, test_df = df.iloc[:400], df.iloc[400:]
    rec = ForecastModel.recommend_params("ETS", train_df, "value")
    params = {p.name: {"None": None}.get(p.value, p.value) for p in rec.params}
    model, train_metrics, test_metrics, _ = ForecastModel().train(
        "ETS", train_df, test_df, "value", [], [], params, set_progress
    )
    assert model.config.seasonal_periods == 12
    assert np.isfinite(test_metrics["RMSE"])


# --- Prophet ----------------------------------------------------------------------------------------------------------


@PY314_RESAMPLE_BUG
def test_prophet_calendar_seasonalities():
    # 8 weeks of hourly data with a daily cycle
    rec = ForecastModel.recommend_params("Prophet", _df(_sine(period=24, n_periods=56)), "value")
    params = _params(rec)
    assert params["daily_seasonality"] is True
    assert params["yearly_seasonality"] is False
    assert params["seasonality_mode"] == "additive"
    basis = {p.name: p.basis for p in rec.params}
    assert "less than two yearly cycles" in basis["yearly_seasonality"]


@PY314_RESAMPLE_BUG
def test_prophet_daily_seasonality_not_observable_on_daily_data():
    rec = ForecastModel.recommend_params("Prophet", _df(_white_noise(n=800), freq="D"), "value")
    params = _params(rec)
    assert params["daily_seasonality"] is False
    assert params["weekly_seasonality"] is False


# --- VectorAR ---------------------------------------------------------------------------------------------------------


@PY314_RESAMPLE_BUG
def test_vector_ar_order_of_var2_process():
    rng = np.random.default_rng(0)
    a1 = np.array([[0.5, 0.1], [0.2, 0.3]])
    a2 = np.array([[-0.3, 0.0], [0.1, -0.2]])
    y = np.zeros((2000, 2))
    for i in range(2, len(y)):
        y[i] = a1 @ y[i - 1] + a2 @ y[i - 2] + rng.normal(size=2)
    rec = ForecastModel.recommend_params("VectorAR", _df(y[:, 0], feature=y[:, 1]), "value", ["feature"])
    assert _params(rec)["maxlags"] == 2
    assert "AIC" in rec.params[0].basis and "HQIC" in rec.params[0].basis


@PY314_RESAMPLE_BUG
def test_vector_ar_univariate_order():
    rec = ForecastModel.recommend_params("VectorAR", _df(_ar([0.4, 0.2, 0.3])), "value")
    assert 2 <= _params(rec)["maxlags"] <= 4
    assert "univariate" in rec.params[0].basis


@PY314_RESAMPLE_BUG
def test_vector_ar_with_constant_feature_omits_maxlags():
    # a constant feature makes the VAR covariance matrix singular: only maxlags is left out
    x = _ar([0.5])
    rec = ForecastModel.recommend_params("VectorAR", _df(x, feature=np.zeros(len(x))), "value", ["feature"])
    assert [p.name for p in rec.params] == ["max_forecast_steps"]
    assert any("maxlags is not recommended" in note for note in rec.notes)


# --- DefaultForecaster ------------------------------------------------------------------------------------------------


@PY314_RESAMPLE_BUG
def test_default_forecaster_granularity(set_progress):
    df = _df(_sine(period=288, n_periods=10, amplitude=5), freq="5min")
    train_df, test_df = df.iloc[:2500], df.iloc[2500:]
    rec = ForecastModel.recommend_params("DefaultForecaster", train_df, "value")
    params = _params(rec)
    assert params["granularity"] == "5min"
    _, _, test_metrics, _ = ForecastModel().train(
        "DefaultForecaster", train_df, test_df, "value", [], [], params, set_progress
    )
    assert np.isfinite(test_metrics["RMSE"])

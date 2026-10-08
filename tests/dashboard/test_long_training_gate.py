#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Tests for the approval asked before a long training in the Forecasting tab: the training time estimates
(merlion/dashboard/models/train_time.py) and the request -> (popup ->) train logic, without a Dash server.
"""

import dash
import numpy as np
import pandas as pd
import pytest

from merlion.dashboard.callbacks.forecast import handle_train_request
from merlion.dashboard.models.forecast import ForecastModel
from merlion.dashboard.models.train_time import TrainTimeEstimate, estimate_train_seconds, format_duration
from merlion.dashboard.pages.forecast import create_long_train_modal

LONG = TrainTimeEstimate(
    seconds=ForecastModel.train_confirm_seconds + 60, basis="ETS with a long period", speed_factor=1
)
SHORT = TrainTimeEstimate(seconds=5, basis="", speed_factor=1)
REQUEST = {"time": 1.0, "overrides": {"seasonal_periods": 493}}


# --- request -> (popup ->) train ---------------------------------------------------------------------------------------


def test_short_training_starts_right_away():
    is_open, content, pending, trigger = handle_train_request(
        "forecasting-train-request", REQUEST, None, lambda r: SHORT
    )
    assert is_open is False and pending is None
    assert trigger == REQUEST


def test_long_training_asks_first():
    is_open, content, pending, trigger = handle_train_request(
        "forecasting-train-request", REQUEST, None, lambda r: LONG
    )
    assert is_open is True
    assert pending == REQUEST
    assert trigger is dash.no_update
    text = " ".join(str(c.children) for c in content)
    assert format_duration(LONG.seconds) in text and "ETS with a long period" in text


def test_approved_long_training_starts():
    is_open, content, pending, trigger = handle_train_request(
        "forecasting-long-train-proceed-btn", None, REQUEST, lambda r: LONG
    )
    assert is_open is False and pending is None
    assert trigger["overrides"] == REQUEST["overrides"]
    assert trigger["time"] != REQUEST["time"]  # a new trigger, so that training starts even for the same request


def test_cancelled_long_training_does_not_start():
    is_open, content, pending, trigger = handle_train_request(
        "forecasting-long-train-cancel-btn", None, REQUEST, lambda r: LONG
    )
    assert is_open is False and pending is None
    assert trigger is dash.no_update


def test_training_starts_if_the_estimate_fails():
    def fail(request):
        raise ValueError("no estimate")

    is_open, content, pending, trigger = handle_train_request("forecasting-train-request", REQUEST, None, fail)
    assert is_open is False and trigger == REQUEST


def test_unknown_algorithm_has_no_estimate():
    assert estimate_train_seconds("SomeNewForecaster", {}, 1000) is None
    is_open, *_, trigger = handle_train_request("forecasting-train-request", REQUEST, None, lambda r: None)
    assert is_open is False and trigger == REQUEST


def test_layout_has_the_long_training_popup():
    layout = str(create_long_train_modal())
    for component_id in [
        "forecasting-long-train-modal",
        "forecasting-long-train-content",
        "forecasting-long-train-proceed-btn",
        "forecasting-long-train-cancel-btn",
    ]:
        assert component_id in layout


# --- estimates ---------------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def calibrated_machine(monkeypatch):
    # Estimates as on the calibration machine, so that they do not depend on the machine running the tests
    monkeypatch.setattr("merlion.dashboard.models.train_time.machine_speed_factor", lambda: 1.0)


def _ets(m, n=2000):
    params = dict(error="add", trend="add", damped_trend=True, seasonal="add" if m else None, seasonal_periods=m)
    return estimate_train_seconds("ETS", params, n).seconds


def test_ets_estimate_grows_steeply_with_the_seasonal_period():
    assert _ets(None) < 1
    assert _ets(24) < 10
    assert _ets(493) > ForecastModel.train_confirm_seconds  # the SKAB / machine temperature case
    assert _ets(24) < _ets(48) < _ets(96) < _ets(192) < _ets(493)


def test_estimates_grow_with_the_data_length():
    assert _ets(48, n=1000) < _ets(48, n=2000) < _ets(48, n=8000)
    arima = [estimate_train_seconds("Arima", {"order": [24, 1, 0]}, n).seconds for n in (1000, 4000)]
    assert arima[0] < arima[1]


@pytest.mark.parametrize(
    "algorithm, params",
    [
        ("Arima", {"order": [4, 1, 2]}),
        ("Arima", {"order": [48, 1, 0]}),
        ("Prophet", {}),
        ("LGBMForecaster", {"maxlags": 24, "max_forecast_steps": 48}),
        ("RandomForestForecaster", {"maxlags": 24, "max_forecast_steps": 48}),
        ("ExtraTreesForecaster", {"maxlags": 24, "max_forecast_steps": 48}),
        ("AutoETS", {}),
        ("AutoProphet", {}),
        ("VectorAR", {"maxlags": 20}),
        ("DefaultForecaster", {}),
    ],
)
def test_usual_settings_do_not_ask(algorithm, params):
    est = estimate_train_seconds(algorithm, params, 2000, n_variables=2 if algorithm == "VectorAR" else 1)
    assert 0 < est.seconds < ForecastModel.train_confirm_seconds
    assert est.basis


def test_sarima_with_a_long_seasonal_period_asks():
    est = estimate_train_seconds("Sarima", {"order": [4, 1, 2], "seasonal_order": [1, 1, 1, 288]}, 2000)
    assert est.seconds > ForecastModel.train_confirm_seconds


def test_format_duration():
    assert format_duration(42) == "42 s"
    assert format_duration(330) == "5 min 30 s"
    assert format_duration(4000) == "1 h 7 min"


def test_estimate_from_the_training_data():
    t = np.arange(2016)
    df = pd.DataFrame(
        {"value": np.sin(2 * np.pi * t / 288)}, index=pd.date_range("2023-01-01", periods=len(t), freq="5min")
    )
    params = dict(error="add", trend=None, damped_trend=False, seasonal="add", seasonal_periods=493)
    est = ForecastModel.estimate_train_time("ETS", df, "value", [], [], params)
    assert est.seconds == pytest.approx(estimate_train_seconds("ETS", params, 2016).seconds)

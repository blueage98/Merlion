#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Tests for the model recommendation (merlion/dashboard/models/model_select.py): each candidate algorithm, with its
recommended hyperparameters, is scored on the last 20% of the training data.
"""

import numpy as np
import pandas as pd
import pytest

from merlion.dashboard.models.forecast import ForecastModel
from merlion.dashboard.models.model_select import recommend_model, recommended_params


@pytest.fixture(autouse=True)
def calibrated_machine(monkeypatch):
    monkeypatch.setattr("merlion.dashboard.models.train_time.machine_speed_factor", lambda: 1.0)


def _seasonal(n=960, period=24, seed=0):
    t = np.arange(n)
    x = 50 + 10 * np.sin(2 * np.pi * t / period) + np.random.default_rng(seed).normal(size=n)
    return pd.DataFrame({"value": x}, index=pd.date_range("2023-01-01", periods=n, freq="h"))


def test_recommends_a_model_that_follows_the_cycle():
    rec = ForecastModel.recommend_model(_seasonal(), "value", horizon=24, algorithms=["ETS", "Arima", "Prophet"])
    assert rec.algorithm in ("ETS", "Arima", "Prophet")
    assert rec.n_valid == int(0.2 * 960)
    maes = [c.valid_mae for c in rec.candidates]
    assert maes == sorted(maes) and np.isfinite(maes[0])
    best = rec.candidates[0]
    assert best.algorithm == rec.algorithm and best.status == "ok"
    # the recommended hyperparameters are used, with max_forecast_steps set to the horizon
    assert rec.params["max_forecast_steps"] == 24
    # a cycle of 1 day, forecast 1 day at a time: well below the amplitude of 10
    assert best.valid_mae < 3


def test_recommended_params_leave_out_max_forecast_steps():
    params = recommended_params("ETS", _seasonal(), "value", ForecastModel)
    assert params["seasonal_periods"] == 24 and "max_forecast_steps" not in params
    assert recommended_params("Prophet", _seasonal(), "value", ForecastModel) == {}  # no recommender: defaults


def test_slow_candidates_are_left_out(monkeypatch):
    monkeypatch.setattr(ForecastModel, "train_confirm_seconds", 1e-6)
    rec = ForecastModel.recommend_model(_seasonal(), "value", horizon=24, algorithms=["ETS", "Arima"])
    assert rec.algorithm is None
    assert all(c.status.startswith("left out: estimated training time") for c in rec.candidates)


def test_failing_candidates_are_ranked_last():
    def make_model(algorithm, params):
        if algorithm == "Arima":
            raise ValueError("broken")
        from merlion.models.factory import ModelFactory

        return ModelFactory.create(algorithm, **params)

    rec = recommend_model(_seasonal(), "value", 24, ForecastModel, algorithms=["Arima", "ETS"], make_model=make_model)
    assert rec.algorithm == "ETS"
    assert rec.candidates[-1].algorithm == "Arima" and rec.candidates[-1].status.startswith("failed: ValueError")

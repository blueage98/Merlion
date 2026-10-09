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
from merlion.dashboard.models.model_select import DEFAULT_CANDIDATES, recommend_model, recommended_params


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


# --- Multivariate data: feature and exogenous variables ----------------------


def _leading(n=600, lead=1, seed=0, noise=0.1):
    """A target that follows the feature variable ``x`` ``lead`` steps later, and an unrelated feature ``z``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n + lead)
    df = pd.DataFrame(
        {"value": x[:n] + noise * rng.normal(size=n), "x": x[lead:], "z": rng.normal(size=n)},
        index=pd.date_range("2023-01-01", periods=n, freq="h"),
    )
    return df


def _with_exog(n=400, seed=0):
    """A target driven by an exogenous variable known in advance (e.g. a planned set point)."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    return pd.DataFrame(
        {"value": 50 + 5 * e + 0.5 * rng.normal(size=n), "e": e},
        index=pd.date_range("2023-01-01", periods=n, freq="h"),
    )


class _Recorder:
    """make_model that keeps the models it builds."""

    def __init__(self):
        self.models = {}

    def __call__(self, algorithm, params):
        from merlion.models.factory import ModelFactory

        self.models[algorithm] = ModelFactory.create(algorithm, **params)
        return self.models[algorithm]


def _basis(rec, name):
    return next(p.basis for p in rec.params if p.name == name)


def test_tree_recommendation_considers_the_feature_variables():
    df = _leading(lead=30)
    univariate = ForecastModel.recommend_params("RandomForestForecaster", df, "value")
    multivariate = ForecastModel.recommend_params("RandomForestForecaster", df, "value", ["x", "z"])
    # x tells the target 30 steps ahead: maxlags covers it, which the target alone does not suggest
    assert univariate.get("maxlags") < 30 and multivariate.get("maxlags") == 30
    assert "feature variable x" in _basis(multivariate, "maxlags")
    assert "feature" not in _basis(univariate, "maxlags")
    # a feature leading by 1 step is within the lags: same value, but the basis says the features were considered
    df = _leading(lead=1)
    univariate = ForecastModel.recommend_params("LGBMForecaster", df, "value")
    multivariate = ForecastModel.recommend_params("LGBMForecaster", df, "value", ["x", "z"])
    assert univariate.get("maxlags") == multivariate.get("maxlags")
    assert "feature variables were considered" in _basis(multivariate, "maxlags")
    assert "3 variables" in _basis(multivariate, "maxlags")


def test_ets_and_arima_recommendations_ignore_the_feature_variables():
    df = _seasonal().assign(x=np.random.default_rng(1).normal(size=960), z=np.arange(960.0))
    for algorithm in ("ETS", "Arima"):
        assert recommended_params(algorithm, df, "value", ForecastModel, ["x", "z"]) == recommended_params(
            algorithm, df, "value", ForecastModel
        )


def test_estimated_time_counts_the_variables():
    df = _leading()
    kwargs = dict(algorithms=["RandomForestForecaster"], params_fn=lambda *args: {"maxlags": 24})
    uni = recommend_model(df, "value", 1, ForecastModel, **kwargs).candidates[0]
    multi = recommend_model(df, "value", 1, ForecastModel, feature_columns=["x", "z"], **kwargs).candidates[0]
    assert multi.estimated_seconds > 2 * uni.estimated_seconds


def test_vector_ar_is_a_default_candidate_only_with_feature_variables():
    def fail(algorithm, params):
        raise ValueError("not evaluated")

    kwargs = dict(make_model=fail, params_fn=lambda *args: {})
    df = _leading()
    with_features = recommend_model(df, "value", 1, ForecastModel, feature_columns=["x", "z"], **kwargs)
    without = recommend_model(df, "value", 1, ForecastModel, **kwargs)
    assert "VectorAR" in [c.algorithm for c in with_features.candidates]
    assert [c.algorithm for c in without.candidates] == DEFAULT_CANDIDATES
    # an explicit list of candidates is used as is
    given = recommend_model(df, "value", 1, ForecastModel, feature_columns=["x"], algorithms=["ETS"], **kwargs)
    assert [c.algorithm for c in given.candidates] == ["ETS"]


def test_feature_variables_are_used_in_the_validation():
    df = _leading(lead=1)
    kwargs = dict(algorithms=["RandomForestForecaster"], params_fn=lambda *args: {"maxlags": 4})
    uni = recommend_model(df, "value", 2, ForecastModel, **kwargs)
    multi = recommend_model(df, "value", 2, ForecastModel, feature_columns=["x"], **kwargs)
    # the target is the feature variable one step earlier: unpredictable from its own past (MAE ~0.8), while the
    # first of the 2 steps forecast at a time is known from the feature variable
    assert multi.candidates[0].valid_mae < 0.7 * uni.candidates[0].valid_mae
    assert multi.n_points == uni.n_points and multi.n_valid == uni.n_valid


def test_arima_gets_the_exogenous_variables_separately():
    df = _with_exog()
    recorder = _Recorder()
    kwargs = dict(algorithms=["Arima"], params_fn=lambda *args: {"order": [1, 0, 0]}, make_model=recorder)
    with_exog = recommend_model(df, "value", 24, ForecastModel, exog_columns=["e"], **kwargs)
    model = recorder.models["Arima"]
    assert model.exog_dim == 1 and model.dim == 1
    without = recommend_model(df.loc[:, ["value"]], "value", 24, ForecastModel, **kwargs)
    # the exogenous values of the validation part are used: the error is not that of 5 * e (~4)
    assert with_exog.candidates[0].valid_mae < 1 < without.candidates[0].valid_mae
    assert "target_seq_index" not in with_exog.params


def test_ets_gets_the_exogenous_variables_as_columns():
    recorder = _Recorder()
    rec = recommend_model(
        _with_exog(), "value", 24, ForecastModel, exog_columns=["e"], make_model=recorder, algorithms=["ETS"]
    )
    assert rec.algorithm == "ETS" and rec.candidates[0].status == "ok"
    assert recorder.models["ETS"].dim == 2 and recorder.models["ETS"].target_seq_index == 0


def test_forecast_model_recommend_model_keywords():
    df = _with_exog().assign(x=np.random.default_rng(2).normal(size=400))
    rec = ForecastModel.recommend_model(
        train_df=df, target_column="value", horizon=12, feature_columns=["x"], exog_columns=["e"], algorithms=["ETS"]
    )
    assert rec.algorithm == "ETS" and rec.params["max_forecast_steps"] == 12
    assert {"algorithm", "params", "candidates", "n_valid", "n_points", "notes"} <= set(vars(rec))

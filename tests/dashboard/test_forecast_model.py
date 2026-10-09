#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Tests for merlion/dashboard/models/forecast.py::ForecastModel, mirroring
test_anomaly_model.py's coverage of the same three interface points
(ModelFactory, TimeSeries, ModelBase/Config).
"""

import numpy as np
import pandas as pd
import pytest

from merlion.models.base import ModelBase
from merlion.dashboard.models.anomaly import AnomalyModel
from merlion.dashboard.models.forecast import ForecastModel

# --- Happy path -------------------------------------------------------------


def test_train_returns_model_and_metrics(forecast_train_test_df, set_progress):
    train_df, test_df = forecast_train_test_df
    forecast_model = ForecastModel()

    model, train_metrics, test_metrics, figure = forecast_model.train(
        algorithm="Arima",
        train_df=train_df,
        test_df=test_df,
        target_column="value",
        feature_columns=[],
        exog_columns=[],
        params={"max_forecast_steps": 20},
        set_progress=set_progress,
    )

    assert isinstance(model, ModelBase)
    assert train_metrics is not None and "sMAPE" in train_metrics
    assert test_metrics is not None and "sMAPE" in test_metrics
    assert figure is not None
    assert set_progress.calls


def test_save_and_load_model_round_trip(forecast_train_test_df, set_progress, file_manager):
    train_df, test_df = forecast_train_test_df
    forecast_model = ForecastModel()

    model, _, _, _ = forecast_model.train(
        algorithm="Arima",
        train_df=train_df,
        test_df=test_df,
        target_column="value",
        feature_columns=[],
        exog_columns=[],
        params={"max_forecast_steps": 20},
        set_progress=set_progress,
    )

    # save_model/load_model are shared ModelMixin statics, reused as-is by both models
    ForecastModel.save_model(file_manager.model_directory, model, "Arima")
    loaded = ForecastModel.load_model(file_manager.model_directory, "Arima")

    assert isinstance(loaded, ModelBase)
    assert type(loaded) is type(model)


# --- Failure paths -----------------------------------------------------------


def test_train_with_unregistered_algorithm_raises(forecast_train_test_df, set_progress):
    train_df, test_df = forecast_train_test_df
    forecast_model = ForecastModel()

    with pytest.raises(ValueError):
        forecast_model.train(
            algorithm="NotARealAlgorithm",
            train_df=train_df,
            test_df=test_df,
            target_column="value",
            feature_columns=[],
            exog_columns=[],
            params={"max_forecast_steps": 20},
            set_progress=set_progress,
        )


def test_train_with_missing_target_column_raises(forecast_train_test_df, set_progress):
    """Unlike AnomalyModel._check() (a clean AssertionError), ForecastModel.train()
    tries int(target_column) as a fallback before asserting -- for a non-numeric,
    unknown column name this raises ValueError instead of AssertionError. This test
    documents that actual (inconsistent) behavior rather than an idealized one."""
    train_df, test_df = forecast_train_test_df
    forecast_model = ForecastModel()

    with pytest.raises(ValueError):
        forecast_model.train(
            algorithm="Arima",
            train_df=train_df,
            test_df=test_df,
            target_column="does_not_exist",
            feature_columns=[],
            exog_columns=[],
            params={"max_forecast_steps": 20},
            set_progress=set_progress,
        )


def test_load_model_from_missing_directory_raises(file_manager):
    # Uses IsolationForest rather than Arima: importing Arima pulls in
    # merlion.models.automl.seasonality, which hits the same Python-3.14/Enum
    # incompatibility documented on the happy-path tests below (see module docstring
    # note in test_train_returns_model_and_metrics' xfail reason). IsolationForest's
    # import chain doesn't touch that module, so it isolates the load-failure check.
    with pytest.raises(FileNotFoundError):
        AnomalyModel.load_model(file_manager.model_directory, "IsolationForest")


# --- Column arrangement shared by training and model recommendation ----------


def _multivariate_df(n=40):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame(rng.normal(size=(n, 4)), index=idx, columns=["x", "value", "temp", "price"])


def test_arrange_columns_puts_target_first_and_exog_last():
    layout = ForecastModel.arrange_columns(_multivariate_df(), "value", ["x"], ["price"])
    assert layout.columns == ["value", "x", "price"]
    assert layout.target_seq_index == 0


def test_exog_is_split_off_for_models_supporting_it():
    df = _multivariate_df()
    layout = ForecastModel.arrange_columns(df, "value", ["x"], ["price", "temp"])
    data, exog = layout.split(df, supports_exog=True)
    assert list(data.columns) == ["value", "x"] and list(exog.columns) == ["price", "temp"]
    assert layout.model_columns(True) == ["value", "x"]


def test_exog_stays_a_column_for_models_not_supporting_it():
    df = _multivariate_df()
    layout = ForecastModel.arrange_columns(df, "value", ["x"], ["price"])
    data, exog = layout.split(df, supports_exog=False)
    assert list(data.columns) == ["value", "x", "price"] and exog is None
    # without exogenous variables, nothing is split off either way
    layout = ForecastModel.arrange_columns(df, "value", ["x"], [])
    assert layout.split(df, supports_exog=True)[1] is None


def test_arrange_columns_converts_string_names_of_integer_columns():
    df = _multivariate_df().set_axis([0, 1, 2, 3], axis=1)
    layout = ForecastModel.arrange_columns(df, "1", ["0"], ["3"])
    assert layout.columns == [1, 0, 3] and layout.target_seq_index == 0
    with pytest.raises(AssertionError):
        ForecastModel.arrange_columns(df, "1", [], ["7"])


@pytest.mark.parametrize("algorithm", ["Arima", "ETS"])
def test_train_with_feature_and_exog_columns(algorithm, set_progress):
    df = _multivariate_df(170)
    model, _, test_metrics, _ = ForecastModel().train(
        algorithm=algorithm,
        train_df=df.iloc[:150],
        test_df=df.iloc[150:],
        target_column="value",
        feature_columns=["x"],
        exog_columns=["price"],
        params={"max_forecast_steps": 20},
        set_progress=set_progress,
    )
    assert model.target_seq_index == 0 and "sMAPE" in test_metrics
    # Arima takes the exogenous variable separately, ETS as an ordinary column
    assert (model.exog_dim if algorithm == "Arima" else model.dim) == (1 if algorithm == "Arima" else 3)


def test_test_data_is_cut_at_the_forecast_horizon_in_time_not_rows(set_progress):
    """
    With 1-second data that has some 2-second gaps, max_forecast_steps rows of test data span more time than the
    horizon (max_forecast_steps steps of 1 second). The test data must be cut by time, or forecasting the test time
    stamps fails (seen in the Auto mode on SKAB data).
    """
    rng = np.random.default_rng(0)
    gaps = np.where(rng.random(1200) < 0.1, 2, 1)
    index = pd.Timestamp("2020-02-08 13:30:47") + pd.to_timedelta(np.cumsum(gaps), unit="s")
    df = pd.DataFrame({"value": np.sin(np.arange(1200) / 20) + rng.normal(scale=0.1, size=1200)}, index=index)
    train_df, test_df = df.iloc[:1000], df.iloc[1000:]
    steps = 150
    model, _, test_metrics, _ = ForecastModel().train(
        "ETS", train_df, test_df, "value", [], [], {"max_forecast_steps": steps}, set_progress
    )
    assert all(np.isfinite(v) for v in test_metrics.values())
    # 150 rows of the test data span more than 150 seconds, so fewer rows are scored
    assert test_df.index[steps] > train_df.index[-1] + pd.Timedelta(seconds=steps)

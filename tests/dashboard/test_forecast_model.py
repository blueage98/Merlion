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
import sys

import pytest

from merlion.models.base import ModelBase
from merlion.dashboard.models.anomaly import AnomalyModel
from merlion.dashboard.models.forecast import ForecastModel

# On Python 3.14, these still fail via the AggregationPolicy/Enum bug documented in
# test_anomaly_model.py (unrelated to the bug below, and not yet re-verified as fixed
# since it's a merlion-core/Python-3.14 incompatibility, not a pandas one).
#
# On Python 3.12, these previously failed with `pandas.KeyError: -1`, root-caused to
# merlion/models/forecast/sarima.py:112 and :131 and ets.py:145 all doing
# `series[-1]` (a bare bracket, label-based lookup) on a Series with a DatetimeIndex,
# intending positional "last element" access. Older pandas silently fell back to
# positional indexing for this; pandas 3.0.3 (installed here) does not, so `-1` is
# looked up as a literal index label and raises KeyError. Fixed by using
# `.iloc[-1]` (unambiguously positional) at all three sites -- confirmed passing
# on Python 3.12 after the fix.
PY314_ENUM_PARTIAL_BUG = sys.version_info >= (3, 14)
FORECAST_TRAIN_BUG_REASON = (
    "Python 3.14 Enum does not treat functools.partial values as members "
    "(AggregationPolicy.__members__ is empty), breaking TimeSeries.align() -- "
    "see test_anomaly_model.py. The separate pandas `series[-1]` KeyError bug "
    "(sarima.py:112,131, ets.py:145) was fixed by switching to `.iloc[-1]`."
)


# --- Happy path -------------------------------------------------------------


@pytest.mark.xfail(condition=PY314_ENUM_PARTIAL_BUG, reason=FORECAST_TRAIN_BUG_REASON, strict=True)
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


@pytest.mark.xfail(condition=PY314_ENUM_PARTIAL_BUG, reason=FORECAST_TRAIN_BUG_REASON, strict=True)
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

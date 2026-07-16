#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Tests for merlion/dashboard/models/anomaly.py::AnomalyModel, focused on the
three interface points it uses to talk to merlion core: ModelFactory,
TimeSeries, and ModelBase/Config (see analayze_codebase.md section 5-1).
"""
import sys

import pytest

from merlion.models.base import ModelBase
from merlion.dashboard.models.anomaly import AnomalyModel

# On Python 3.14, `Enum` members whose values are `functools.partial` objects are
# no longer recognized as members (`AggregationPolicy.__members__` comes back empty --
# reproduced independently of the dashboard: a bare `class Foo(Enum): A = partial(...)`
# already has `Foo.__members__ == {}` on this interpreter). merlion's
# `merlion/utils/resample.py::AggregationPolicy` hits this, which breaks
# `TemporalResample`/`TimeSeries.align()` and therefore every real `model.train()`
# call. This is a merlion-core/Python-version incompatibility, not a bug in these
# tests or in the dashboard<->core interface being tested here.
#
# Re-verified on Python 3.12 (2026-07-16): the bug does NOT reproduce there --
# `Foo.__members__` comes back with all members, and these two tests pass for real
# (previously they were unconditionally marked xfail while this was only tested on
# 3.14; that made them XPASS -- a false "still broken" signal -- once run on 3.12).
PY314_ENUM_PARTIAL_BUG = sys.version_info >= (3, 14)
PY314_ENUM_PARTIAL_BUG_REASON = (
    "Python 3.14 Enum does not treat functools.partial values as members "
    "(AggregationPolicy.__members__ is empty), breaking TimeSeries.align() "
    "and thus all real model.train() calls -- a merlion-core/Python 3.14 "
    "incompatibility, not a dashboard interface bug. Confirmed fixed on Python 3.12."
)


# --- Happy path -------------------------------------------------------------


@pytest.mark.xfail(condition=PY314_ENUM_PARTIAL_BUG, reason=PY314_ENUM_PARTIAL_BUG_REASON, strict=True)
def test_train_returns_model_and_metrics(train_test_df, set_progress):
    train_df, test_df = train_test_df
    anomaly_model = AnomalyModel()

    model, train_metrics, test_metrics, figure = anomaly_model.train(
        algorithm="IsolationForest",
        train_df=train_df,
        test_df=test_df,
        columns=["value"],
        label_column="label",
        params={},
        threshold_params=None,
        set_progress=set_progress,
    )

    # ModelFactory produced a real core model instance
    assert isinstance(model, ModelBase)
    # TimeSeries conversion + evaluate.anomaly.TSADMetric ran end to end
    assert train_metrics is not None and "F1" in train_metrics
    assert test_metrics is not None and "F1" in test_metrics
    assert figure is not None
    assert set_progress.calls  # progress callback was actually invoked


@pytest.mark.xfail(condition=PY314_ENUM_PARTIAL_BUG, reason=PY314_ENUM_PARTIAL_BUG_REASON, strict=True)
def test_save_and_load_model_round_trip(train_test_df, set_progress, file_manager):
    train_df, test_df = train_test_df
    anomaly_model = AnomalyModel()

    model, _, _, _ = anomaly_model.train(
        algorithm="IsolationForest",
        train_df=train_df,
        test_df=test_df,
        columns=["value"],
        label_column=None,
        params={},
        threshold_params=None,
        set_progress=set_progress,
    )

    AnomalyModel.save_model(file_manager.model_directory, model, "IsolationForest")
    loaded = AnomalyModel.load_model(file_manager.model_directory, "IsolationForest")

    assert isinstance(loaded, ModelBase)
    assert type(loaded) is type(model)


# --- Failure paths -----------------------------------------------------------


def test_train_with_unregistered_algorithm_raises(train_test_df, set_progress):
    """ModelFactory.get_model_class() must reject algorithm names it doesn't know,
    and AnomalyModel.train() must let that error propagate rather than swallow it."""
    train_df, test_df = train_test_df
    anomaly_model = AnomalyModel()

    with pytest.raises(ValueError):
        anomaly_model.train(
            algorithm="NotARealAlgorithm",
            train_df=train_df,
            test_df=test_df,
            columns=["value"],
            label_column=None,
            params={},
            threshold_params=None,
            set_progress=set_progress,
        )


def test_train_with_missing_column_raises(train_test_df, set_progress):
    """AnomalyModel._check() must reject a column that isn't in the DataFrame,
    before any TimeSeries conversion or model training is attempted.

    NOTE: for a non-numeric, unknown column name, _check() (anomaly.py:88-91) tries
    `int(columns[i])` as a fallback before asserting membership, so this actually
    raises ValueError rather than the AssertionError one might expect from an
    `assert` statement -- this test documents that actual (surprising) behavior."""
    train_df, test_df = train_test_df
    anomaly_model = AnomalyModel()

    with pytest.raises(ValueError):
        anomaly_model.train(
            algorithm="IsolationForest",
            train_df=train_df,
            test_df=test_df,
            columns=["does_not_exist"],
            label_column=None,
            params={},
            threshold_params=None,
            set_progress=set_progress,
        )


def test_load_model_from_missing_directory_raises(file_manager):
    """ModelBase.load() reads config.json from disk; loading an algorithm that was
    never trained/saved must fail loudly (FileNotFoundError), not silently."""
    with pytest.raises(FileNotFoundError):
        AnomalyModel.load_model(file_manager.model_directory, "IsolationForest")

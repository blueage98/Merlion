#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Shared fixtures for dashboard <-> merlion core interface tests.

These tests exercise `merlion/dashboard/models/*.py` directly (no Dash server,
no callbacks) against the three real interface points identified in
analayze_codebase.md: ModelFactory, TimeSeries, and ModelBase/Config.
"""
import numpy as np
import pandas as pd
import pytest

from merlion.dashboard.utils.file_manager import FileManager


@pytest.fixture
def file_manager(tmp_path, monkeypatch):
    """A fresh FileManager pointed at a per-test tmp_path.

    FileManager is a process-wide singleton (see analayze_codebase.md, weakness #4),
    but __init__ re-runs and resets all folder attributes every time a new instance's
    __init__ runs, so a fresh instance per test avoids cross-test state leakage.

    NOTE: `FileManager(str(tmp_path))` does NOT work here -- it's a real bug this
    fixture had to work around. `SingletonClass.__new__(cls)` takes no args, but
    Python's `type.__call__` always calls `cls.__new__(cls, *args, **kwargs)` first,
    so `FileManager(directory)` raises `TypeError: __new__() takes 1 positional
    argument but 2 were given` before `__init__` ever runs. In production this is
    never hit because the app only ever calls the no-arg `FileManager()`. We bypass
    the broken __new__ dispatch directly instead (see analayze_codebase.md weakness list).
    """
    monkeypatch.delattr(FileManager, "instance", raising=False)
    fm = object.__new__(FileManager)
    fm.__init__(directory=str(tmp_path))
    FileManager.instance = fm
    return fm


def _make_df(n, freq, seed, with_label):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq=freq)
    values = rng.normal(size=n)
    values[-5:] += 10  # inject a clear anomalous spike near the end
    df = pd.DataFrame({"value": values}, index=idx)
    if with_label:
        labels = np.zeros(n, dtype=int)
        labels[-5:] = 1
        df["label"] = labels
    return df


@pytest.fixture
def train_test_df():
    """A small labeled univariate train/test split, valid for AnomalyModel.train()."""
    train_df = _make_df(n=150, freq="h", seed=1, with_label=True)
    test_df = _make_df(n=50, freq="h", seed=2, with_label=True)
    return train_df, test_df


@pytest.fixture
def forecast_train_test_df():
    """A contiguous univariate train/test split, valid for ForecastModel.train().

    Forecasting requires the test period to immediately follow the train period in
    time (ForecastModel.train() calls model.forecast() starting right after the
    training data ends -- see merlion/models/forecast/base.py:152-157, which asserts
    test time_stamps fall within [train_end, train_end + max_forecast_steps]). An
    earlier version of this fixture generated train_df/test_df independently from the
    same start date, which produced non-contiguous, overlapping ranges and tripped
    that assertion -- a bug in this fixture, not in merlion.
    """
    rng = np.random.default_rng(1)
    n_train, n_test = 150, 20
    idx = pd.date_range("2023-01-01", periods=n_train + n_test, freq="h")
    values = rng.normal(size=n_train + n_test)
    values[-5:] += 10  # inject a clear anomalous spike near the end
    df = pd.DataFrame({"value": values}, index=idx)
    return df.iloc[:n_train], df.iloc[n_train:]


@pytest.fixture
def set_progress():
    """No-op stand-in for the Dash `set_progress` callback the dashboard passes in."""
    calls = []

    def _set_progress(*args, **kwargs):
        calls.append((args, kwargs))

    _set_progress.calls = calls
    return _set_progress

#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Estimates of how long training a forecasting algorithm takes, used in the Forecasting tab to ask for approval before
a long training (`ForecastModel.train_confirm_seconds`).

The cost model of each algorithm comes from measured ``model.train()`` times (2026-10-08, one thread, synthetic hourly
data with a daily cycle) over the data length and the parameters that drive the cost:

- ETS: the seasonal period. ETS estimates one initial state per step of the period, and the time grows steeply with
  it (2 s for 48 steps, 44 s for 144, 100 s for 192 and ~32 min for 463 on 2000 points), so the measured times are
  interpolated log-log and extrapolated with the slope of the longest periods. The periods above 192 were measured on
  the SKAB and NAB machine temperature sensor data of the manufacturing benchmark.
- Arima/Sarima: the dimension of the state of the SARIMAX model, ``max(p + P m, q + Q m + 1) + d + D m``.
- Tree models: the data length, ``maxlags`` and the number of variables (power laws fitted to the measurements,
  within 7% of them; with several variables, the models take the lags of all of them and forecast all of them, which
  multiplied the time by 5.6 (LGBM), 8.0 (random forest) and 7.1 (extra trees) for 8 variables, measured 2026-10-09).
- Prophet, AutoETS, AutoProphet, VectorAR, DefaultForecaster: the data length.

The times are then scaled by the speed of the machine running the dashboard relative to the calibration machine:
a fixed reference fit is timed once per process (`machine_speed_factor`). The estimates are meant to tell a few
seconds from minutes and hours, and can be off by a factor of 2 or so.
"""

from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Optional
import warnings

import numpy as np

#: Time of the reference fit (`_reference_fit`) on the calibration machine, in seconds.
REFERENCE_SECONDS = 0.41

#: ETS training time on 2000 points by seasonal period (0: no seasonality), in seconds.
ETS_BY_PERIOD = [
    (1, 0.04),
    (12, 0.49),
    (24, 0.94),
    (48, 1.99),
    (96, 4.09),
    (144, 44.5),
    (192, 100.3),
    (237, 175.3),
    (463, 1897.0),
    (493, 2380.0),
]
#: ETS training time relative to 2000 points, by data length (seasonal period 48).
ETS_BY_LENGTH = [(500, 0.36), (1000, 0.56), (2000, 1.0), (4000, 1.9), (8000, 7.55)]


@dataclass
class TrainTimeEstimate:
    """An estimated training time, with what drives it."""

    seconds: float
    #: what drives the cost, e.g. "ETS with seasonal_periods = 493 on 2016 points"
    basis: str
    #: time on this machine relative to the calibration machine
    speed_factor: float


def format_duration(seconds) -> str:
    """A duration in seconds as e.g. "42 s", "5 min 30 s" or "1 h 7 min"."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60} s"
    return f"{seconds // 3600} h {round(seconds % 3600 / 60)} min"


def _reference_fit():
    import statsmodels.api as sm

    x = np.cumsum(np.random.default_rng(0).normal(size=2000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm.tsa.SARIMAX(x, order=(2, 1, 2)).fit(disp=0)


@lru_cache(maxsize=1)
def machine_speed_factor() -> float:
    """
    Time of the reference fit on this machine relative to the calibration machine (> 1: slower), timed once per
    process (best of 3, ~1 s in total).
    """
    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        _reference_fit()
        best = min(best, time.perf_counter() - start)
    return float(np.clip(best / REFERENCE_SECONDS, 0.25, 100))


def _loglog(table, x) -> float:
    """Interpolates ``table`` (sorted ``(x, y)`` pairs) log-log at ``x``, extrapolating with the end slopes."""
    xs, ys = np.log([p[0] for p in table]), np.log([p[1] for p in table])
    lx = np.log(max(x, 1e-9))
    if lx <= xs[0]:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return float(np.exp(ys[0] + slope * (lx - xs[0])))
    if lx >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return float(np.exp(ys[-1] + slope * (lx - xs[-1])))
    return float(np.exp(np.interp(lx, xs, ys)))


def _int_tuple(value, n, default):
    if value is None or isinstance(value, str) and value.strip() in ("", "None"):
        return list(default)
    if isinstance(value, str):
        value = [v for v in value.strip("()[] ").split(",") if v.strip()]
    values = [int(v) for v in value]
    return values if len(values) == n else list(default)


def _period(params) -> int:
    seasonal = params.get("seasonal", "add")
    m = params.get("seasonal_periods")
    if seasonal in (None, "None") or m in (None, "None", ""):
        return 0
    return int(m)


def _ets(params, n):
    m = _period(params)
    seconds = _loglog(ETS_BY_PERIOD, max(m, 1)) * _loglog(ETS_BY_LENGTH, n)
    what = f"seasonal_periods = {m}" if m else "no seasonality"
    return seconds, f"ETS with {what} on {n} points (the time grows steeply with the seasonal period)"


def _sarimax(algorithm, params, n):
    p, d, q = _int_tuple(params.get("order"), 3, (4, 1, 2))
    if algorithm == "Arima":
        P = D = Q = m = 0
    else:
        P, D, Q, m = _int_tuple(params.get("seasonal_order"), 4, (2, 0, 1, 24))
    k = max(p + P * m, q + Q * m + 1) + d + D * m
    if algorithm == "Arima":
        # (24, 1, 0): 1.2 s, (48, 1, 0): 7 s, (60, 1, 0): 12.6 s; (30, 0, 10): 3.5 s; small orders ~0.4-0.9 s
        seconds = max(0.4, 1.16 * (k / 25) ** 2.6) * (1 + q / 10)
        what = f"Arima with order {(p, d, q)}"
    else:
        # seasonal_order (1, 1, 1, 12): 7 s, (1, 1, 1, 24): 28 s, (1, 1, 1, 48): 204 s (order (4, 1, 2))
        seconds = 7.0 * (k / 29) ** 2.7
        what = f"Sarima with order {(p, d, q)} and seasonal_order {(P, D, Q, m)}"
    return seconds * (n / 2000), f"{what} on {n} points (a state of {k} dimensions)"


def _trees(algorithm, params, n, n_variables=1):
    lags = int(params.get("maxlags") or 24)
    base, n_exp, lag_exp, var_exp = {
        "LGBMForecaster": (0.23, 0.39, 0.21, 0.83),
        "RandomForestForecaster": (1.32, 1.21, 0.89, 1.0),
        "ExtraTreesForecaster": (0.73, 1.07, 0.87, 0.95),
    }[algorithm]
    seconds = base * (n / 2000) ** n_exp * (lags / 24) ** lag_exp * max(int(n_variables), 1) ** var_exp
    what = f"{algorithm} with maxlags = {lags} on {n} points"
    return seconds, what if n_variables <= 1 else f"{what} of {n_variables} variables"


def _by_length(algorithm, params, n, n_variables):
    base, n_exp = {
        "Prophet": (0.27, 0.67),
        "AutoETS": (7.05, 1.0),
        "AutoProphet": (1.05, 0.77),
        "DefaultForecaster": (6.97, 0.88),
        "VectorAR": (0.03, 1.0),
    }[algorithm]
    seconds = base * (n / 2000) ** n_exp
    if algorithm == "VectorAR":
        lags = int(params.get("maxlags") or 20)
        seconds *= max(1.0, n_variables * lags / 100)
    return seconds, f"{algorithm} on {n} points"


def estimate_train_seconds(algorithm, params, n_points, n_variables=1) -> Optional[TrainTimeEstimate]:
    """
    Estimated time to train ``algorithm`` with ``params`` on ``n_points`` points (after resampling) of
    ``n_variables`` variables on this machine, or ``None`` for an algorithm without a cost model.
    """
    params = params or {}
    n = max(int(n_points), 1)
    if algorithm == "ETS":
        seconds, basis = _ets(params, n)
    elif algorithm in ("Arima", "Sarima"):
        seconds, basis = _sarimax(algorithm, params, n)
    elif algorithm in ("LGBMForecaster", "RandomForestForecaster", "ExtraTreesForecaster"):
        seconds, basis = _trees(algorithm, params, n, n_variables)
    elif algorithm in ("Prophet", "AutoETS", "AutoProphet", "DefaultForecaster", "VectorAR"):
        seconds, basis = _by_length(algorithm, params, n, n_variables)
    else:
        return None
    factor = machine_speed_factor()
    return TrainTimeEstimate(seconds=seconds * factor, basis=basis, speed_factor=factor)

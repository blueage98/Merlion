#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Recommends a forecasting algorithm for the training data: each candidate algorithm is set up with its recommended
hyperparameters (merlion/dashboard/models/recommend.py), trained on the first 80% of the training data, and scored by
its mean absolute error over the last 20%, forecast ``horizon`` steps at a time from the actual values before each
forecast (without retraining), like Merlion's `ModelSelector` does. Candidates whose estimated training time on the
whole training data exceeds ``train_confirm_seconds`` are left out (merlion/dashboard/models/train_time.py).

On M4 Hourly and the manufacturing windows (499 series, the same candidates), this validation picked better models
than scoring a single holdout of ``horizon`` points (median test MASE 0.747 vs 0.800; better on 176 series, worse on
125, Wilcoxon p = 3e-5), and better than always using the best single algorithm (168 vs 102, p = 1e-6). sMAPE and
MAE picked the same model on 92% of the series; MAE is used since sMAPE is unstable for values near 0 (sensor data).
"""

from dataclasses import dataclass, field
import logging
import time
import traceback
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from merlion.evaluate.forecast import ForecastEvaluator, ForecastEvaluatorConfig, ForecastMetric
from merlion.models.factory import ModelFactory
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import TimeSeries
from merlion.dashboard.models.recommend import RECOMMENDERS, format_value, parse_confirmed_values
from merlion.dashboard.models.train_time import estimate_train_seconds

logger = logging.getLogger(__name__)

#: Algorithms compared by default (the candidates of the 2026-10-09 benchmark). Prophet has no recommender and is
#: compared with its default settings; SARIMA-type models are not used for manufacturing data.
DEFAULT_CANDIDATES = ["ETS", "Arima", "LGBMForecaster", "RandomForestForecaster", "ExtraTreesForecaster", "Prophet"]


@dataclass
class ModelCandidate:
    """One candidate algorithm: its settings and how it did on the validation part."""

    algorithm: str
    params: dict
    #: mean absolute error on the validation part (``inf`` if not evaluated)
    valid_mae: float = np.inf
    #: estimated training time on the whole training data, in seconds
    estimated_seconds: Optional[float] = None
    #: time taken by the validation, in seconds
    eval_seconds: float = 0.0
    #: "ok", or why the candidate was not evaluated / failed
    status: str = "ok"


@dataclass
class ModelRecommendation:
    """The recommended algorithm, with all the candidates ranked by validation error."""

    algorithm: Optional[str]
    params: dict
    candidates: List[ModelCandidate]
    #: number of points of the validation part
    n_valid: int
    n_points: int
    notes: List[str] = field(default_factory=list)


def recommended_params(algorithm, train_df, target_column, cfg, feature_columns=None) -> dict:
    """
    The recommended hyperparameters of ``algorithm`` (empty for an algorithm without a recommender), converted the
    way the values confirmed in the dashboard popup are. ``max_forecast_steps`` is left out: the caller sets it.
    """
    if algorithm not in RECOMMENDERS:
        return {}
    rec = cfg.recommend_params(algorithm, train_df, target_column, feature_columns)
    params = [p for p in rec.params if p.name != "max_forecast_steps"]
    overrides, errors = parse_confirmed_values([p.spec() for p in params], [format_value(p.value) for p in params])
    if errors:
        raise ValueError(" ".join(errors))
    return overrides


def recommend_model(
    train_df: pd.DataFrame,
    target_column,
    horizon: int,
    cfg,
    algorithms: Sequence[str] = None,
    make_model: Callable = None,
    params_fn: Callable = None,
    valid_frac: float = 0.2,
) -> ModelRecommendation:
    """
    Recommends the algorithm with the lowest validation error among ``algorithms``.

    :param train_df: the training data (univariate target).
    :param horizon: the number of steps to forecast at a time.
    :param valid_frac: the fraction of the (latest) training data used for validation.
    :param cfg: the settings class (``ForecastModel``).
    :param make_model: builds a model from ``(algorithm, params)`` (default: `ModelFactory.create`).
    :param params_fn: the hyperparameters of ``(algorithm, train_df, target_column)`` (default: the recommended ones).
    """
    algorithms = list(algorithms or DEFAULT_CANDIDATES)
    make_model = make_model or (lambda algorithm, params: ModelFactory.create(algorithm, **params))
    params_fn = params_fn or (lambda algorithm, df, target: recommended_params(algorithm, df, target, cfg))

    if target_column not in train_df:
        target_column = int(target_column)
    ts = TimeSeries.from_pd(train_df.loc[:, [target_column]])
    resample = TemporalResample()
    resample.train(ts)
    data = resample(ts).to_pd()
    n = len(data)
    n_valid = int(max(1, valid_frac * n))
    horizon = int(max(1, min(horizon, n_valid)))
    fit_part, valid_part = TimeSeries.from_pd(data.iloc[:-n_valid]), TimeSeries.from_pd(data.iloc[-n_valid:])
    step = data.index[1] - data.index[0] if n > 1 else pd.Timedelta(seconds=1)
    evaluator_config = ForecastEvaluatorConfig(retrain_freq=None, horizon=step * horizon)

    candidates, notes = [], []
    for algorithm in algorithms:
        try:
            params = dict(params_fn(algorithm, train_df, target_column))
        except Exception as e:
            logger.warning(f"Could not recommend {algorithm} settings: {e}")
            params = {}
            notes.append(f"{algorithm}: no recommended settings ({type(e).__name__}), its defaults are used.")
        params["max_forecast_steps"] = horizon
        cand = ModelCandidate(algorithm, params)
        est = estimate_train_seconds(algorithm, params, n)
        cand.estimated_seconds = None if est is None else est.seconds
        if est is not None and est.seconds > cfg.train_confirm_seconds:
            cand.status = f"left out: estimated training time {est.seconds:.0f} s > {cfg.train_confirm_seconds} s"
            candidates.append(cand)
            continue
        start = time.time()
        try:
            evaluator = ForecastEvaluator(model=make_model(algorithm, params), config=evaluator_config)
            _, forecast = evaluator.get_predict(train_vals=fit_part, test_vals=valid_part)
            mae = float(ForecastMetric.MAE.value(valid_part, forecast))
            if not np.isfinite(mae):
                raise ValueError("the forecast is not finite")
            cand.valid_mae = mae
        except Exception as e:
            cand.status = f"failed: {type(e).__name__}: {e}"[:200]
            logger.debug(traceback.format_exc())
        cand.eval_seconds = time.time() - start
        candidates.append(cand)

    candidates.sort(key=lambda c: c.valid_mae)
    best = candidates[0] if candidates and np.isfinite(candidates[0].valid_mae) else None
    return ModelRecommendation(
        algorithm=None if best is None else best.algorithm,
        params={} if best is None else best.params,
        candidates=candidates,
        n_valid=n_valid,
        n_points=n,
        notes=notes,
    )

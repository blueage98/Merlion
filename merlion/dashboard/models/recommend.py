#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Recommended hyperparameters for the forecasting algorithms, shown in the Forecasting tab before training.

Each algorithm's parameters are estimated from the training data with a method that matches what the parameter
means in the model:

- autocorrelation structure (how far back to look / how far ahead to trust): ACF decay and seasonal periods,
- structural properties (unit roots, trend, seasonality, multiplicative effects): KPSS tests, STL strengths and the
  relation between the seasonal amplitude and the level,
- model orders: information criteria (AICc/BIC) over candidate models fitted on the latest part of the data.

Each recommender takes a :class:`SeriesStats` (statistics shared across algorithms, computed lazily and cached) and
the settings class (``ForecastModel``, whose class variables hold the thresholds and time limits), and returns a
:class:`Recommendation`. Recommenders are registered in ``RECOMMENDERS``.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from merlion.models.automl.seasonality import SeasonalityLayer, PeriodicityStrategy
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import TimeSeries
from merlion.dashboard.models.train_time import estimate_train_seconds, format_duration


@dataclass
class ParamRecommendation:
    """A recommended value for one hyperparameter, with the reasoning behind it."""

    name: str
    value: Any
    #: "int" (positive integer), "choice" (one of ``choices``), "int_tuple" (``tuple_len`` integers >= 0) or "str"
    kind: str
    basis: str
    choices: Optional[List[str]] = None
    tuple_len: Optional[int] = None
    #: whether the value may be left empty (``None``)
    optional: bool = False
    #: whether the value is a number of time steps (shown as a duration as well)
    is_steps: bool = False

    def spec(self) -> dict:
        """The JSON-serializable description of the input field, used to validate the confirmed value."""
        d = asdict(self)
        d.pop("value")
        d.pop("basis")
        return d


@dataclass
class Recommendation:
    """The recommended hyperparameters of an algorithm, plus the data they are computed from."""

    algorithm: str
    params: List[ParamRecommendation]
    n_points: int
    granularity: Any
    #: number of (latest) points the model-fitting searches were run on, if any
    ic_points: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def get(self, name):
        return next((p.value for p in self.params if p.name == name), None)


def steps_to_duration(steps, granularity):
    """Converts a number of time steps at the given granularity into a human-readable duration."""
    if isinstance(granularity, (int, float)):
        return str(pd.Timedelta(seconds=steps * granularity))
    return f"{steps} x {granularity}"


def format_value(value) -> str:
    """Formats a parameter value the way the algorithm setting table expects it."""
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(str(v) for v in value) + ")"
    return str(value)


class SeriesStats:
    """
    Statistics of the target variable shared by the recommenders. The data is resampled the same way the models do
    (``TemporalResample``) before computing anything. Expensive statistics are computed on first use and cached.
    """

    min_points = 40

    def __init__(self, train_df, target_column, max_lags=2000, stl_max_points=10000):
        if target_column not in train_df:
            target_column = int(target_column)
        assert target_column in train_df, f"The target variable {target_column} is not in the time series."
        self.train_df = train_df
        self.target_column = target_column

        ts = TimeSeries.from_pd(train_df.loc[:, [target_column]])
        resample = TemporalResample()
        resample.train(ts)
        self.granularity = resample.granularity
        self.series = resample(ts).to_pd().iloc[:, 0].astype(float)
        self.x = self.series.values
        assert len(self.x) >= self.min_points, (
            f"The training data is too short ({len(self.x)} points) to recommend settings. "
            f"At least {self.min_points} points are needed."
        )
        self.max_lag = min(len(self.x) // 4, max_lags)
        self.stl_max_points = stl_max_points
        self._cache = {}

    def _cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def n_points(self):
        return len(self.x)

    @property
    def granularity_seconds(self) -> float:
        """The granularity in seconds (estimated from the timestamps if it is not a fixed duration, e.g. monthly)."""
        if isinstance(self.granularity, (int, float)):
            return float(self.granularity)
        return float(np.median(np.diff(self.series.index.values).astype("timedelta64[s]").astype(float)))

    @property
    def acf(self) -> np.ndarray:
        """ACF of the target up to ``max_lag``."""
        return self._cached("acf", lambda: np.nan_to_num(sm.tsa.acf(self.x, nlags=self.max_lag, fft=True)))

    @property
    def periods(self) -> List[int]:
        """Statistically significant seasonal periods (at most ``max_lag``)."""

        def detect():
            if np.std(self.x) == 0:
                return []
            periods = SeasonalityLayer.detect_seasonality(
                self.x, max_lag=self.max_lag, pval=0.01, periodicity_strategy=PeriodicityStrategy.All
            )
            # a period of 1 step is not a seasonality
            return [p for p in periods if p > 1]

        return self._cached("periods", detect)

    def horizon(self, threshold) -> int:
        """
        The predictability horizon: the last lag before the ACF first drops below ``threshold``, i.e. how far
        the recent history still says something about the future. ``max_lag`` if the ACF never drops below it.
        """
        below = np.flatnonzero(self.acf[1:] < threshold)
        return max(int(below[0]) if len(below) > 0 else self.max_lag, 1)

    def dominant_period(self, min_period=1, threshold=None, require_trough=True, max_period=None) -> Optional[int]:
        """
        The significant period with the highest ACF among those > ``min_period`` (and ACF >= ``threshold``, and
        <= ``max_period``).

        With ``require_trough``, the ACF must also be higher at the period than at half of it: a seasonal cycle has a
        trough half a period earlier, while short lags of a slowly changing series (e.g. 17 steps of a 288-step
        cycle) have a high ACF just because consecutive values are close.
        """
        acf = self.acf
        max_period = self.max_lag if max_period is None else min(max_period, self.max_lag)
        candidates = [p for p in self.periods if min_period < p <= max_period and acf[p] > 0]
        if threshold is not None:
            candidates = [p for p in candidates if acf[p] >= threshold]
        if require_trough:
            candidates = [p for p in candidates if acf[p] > acf[p // 2]]
        return max(candidates, key=lambda p: acf[p]) if candidates else None

    def tail(self, n) -> np.ndarray:
        return self.x[-n:]

    #: Longest period STL is run with. Longer periods are first averaged into blocks of consecutive points.
    stl_max_period = 100

    def _stl(self, period):
        """
        STL decomposition of (the latest part of) the data. Its cost grows with both the length and the period, so
        for a period longer than ``stl_max_period``, consecutive points are averaged into blocks to bring the period
        down to at most ``stl_max_period`` steps. The strengths and correlations computed from it are ratios of
        variances and are hardly affected by the coarser resolution.

        :return: ``(data, STL result, period)``, where the data and the period are after averaging, if any.
        """

        def fit():
            x = self.tail(max(self.stl_max_points, 10 * period))
            block = int(np.ceil(period / self.stl_max_period))
            if block > 1:
                x = x[len(x) % block :].reshape(-1, block).mean(axis=1)
            p = int(round(period / block))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return x, STL(x, period=p, robust=False).fit(), p

        return self._cached(("stl", period), fit)

    @staticmethod
    def _strength(component, resid):
        var_r = np.var(resid)
        var_total = np.var(component + resid)
        return 0.0 if var_total == 0 else float(max(0.0, 1 - var_r / var_total))

    def seasonal_strength(self, period) -> float:
        """Strength of seasonality, 1 - Var(R) / Var(S + R), from an STL decomposition (Wang, Smith & Hyndman)."""
        _, res, _ = self._stl(period)
        return self._strength(res.seasonal, res.resid)

    def trend_strength(self, period=None) -> float:
        """
        Strength of trend, 1 - Var(R) / Var(T + R). With a seasonal period, T and R come from an STL decomposition.
        Without one, T is a centered moving average over 5% of the data and R = x - T.
        """
        if period is not None:
            _, res, _ = self._stl(period)
            return self._strength(res.trend, res.resid)

        def compute():
            x = self.tail(self.stl_max_points)
            w = max(5, len(x) // 20)
            trend = pd.Series(x).rolling(w, center=True, min_periods=1).mean().values
            return self._strength(trend, x - trend)

        return self._cached("trend_strength", compute)

    def amplitude_level_correlation(self, period):
        """
        Spearman correlation between the level (mean of the STL trend) and the seasonal amplitude (std of the
        detrended data) of each complete cycle. A strong positive correlation means the seasonality is
        multiplicative. Returns ``(rho, p-value, number of cycles)``; ``(0, 1, n)`` if there are fewer than 4 cycles.
        """

        def compute():
            x, res, period_ = self._stl(period)
            n_cycles = len(x) // period_
            if n_cycles < 4:
                return 0.0, 1.0, n_cycles
            start = len(x) - n_cycles * period_
            level = res.trend[start:].reshape(n_cycles, period_).mean(axis=1)
            amplitude = (x - res.trend)[start:].reshape(n_cycles, period_).std(axis=1)
            if np.std(level) == 0 or np.std(amplitude) == 0:
                return 0.0, 1.0, n_cycles
            rho, pval = scipy.stats.spearmanr(level, amplitude)
            return float(np.nan_to_num(rho)), float(np.nan_to_num(pval, nan=1.0)), n_cycles

        return self._cached(("amp_level", period), compute)

    def is_multiplicative(self, period, min_corr) -> Tuple[bool, str]:
        """Whether the seasonality with the given period is multiplicative, and why."""
        rho, pval, n_cycles = self.amplitude_level_correlation(period)
        positive = bool(np.min(self.x) > 0)
        mul = positive and rho >= min_corr and pval < 0.05
        if not positive:
            why = "the data has values <= 0, so multiplicative seasonality is not possible"
        elif n_cycles < 4:
            why = f"only {n_cycles} complete cycles, too few to relate the seasonal amplitude to the level"
        else:
            relation = "grows" if mul else "does not grow"
            why = (
                f"the seasonal amplitude {relation} with the level (Spearman correlation over {n_cycles} cycles "
                f"= {rho:.2f}, p = {pval:.3f}; multiplicative if >= {min_corr} and p < 0.05)"
            )
        return mul, why

    def multivariate(self, feature_columns) -> pd.DataFrame:
        """The target and feature variables, resampled together."""
        columns = [self.target_column]
        for c in feature_columns or []:
            c = c if c in self.train_df else int(c)
            assert c in self.train_df, f"The feature variable {c} is not in the time series."
            columns.append(c)
        ts = TimeSeries.from_pd(self.train_df.loc[:, columns])
        resample = TemporalResample(granularity=self.granularity)
        resample.train(ts)
        return resample(ts).to_pd().astype(float)


# ----------------------------------------------------------------------------------------------------------------------
# Shared rules
# ----------------------------------------------------------------------------------------------------------------------


def recommend_max_forecast_steps(stats: SeriesStats, cfg) -> ParamRecommendation:
    """``max_forecast_steps``: the predictability horizon from the ACF decay, the same for every algorithm."""
    threshold = cfg.acf_threshold
    steps = stats.horizon(threshold)
    if steps >= stats.max_lag:
        basis = f"The autocorrelation (ACF) stays >= {threshold} over the whole searched range."
    else:
        basis = (
            f"The autocorrelation (ACF) stays >= {threshold} for {steps} steps. Beyond that, the recent history "
            f"says little about the future and the forecast errors compound."
        )
    return ParamRecommendation("max_forecast_steps", steps, "int", basis, is_steps=True)


def lgbm_params(stats: SeriesStats, threshold) -> dict:
    """
    ``maxlags`` and ``max_forecast_steps`` for the autoregressive tree models (LGBM, random forest, extra trees).

    - ``max_forecast_steps``: the predictability horizon (see :meth:`SeriesStats.horizon`).
    - ``maxlags``: the dominant seasonal period, so that the model sees one full cycle. This is the significant
      seasonality with the highest ACF among those beyond ``max_forecast_steps`` (shorter lags have a high ACF just
      because the series changes slowly), and its ACF must be at least ``threshold``. Like the models' own default,
      maxlags is at least ``max_forecast_steps``.
    """
    acf = stats.acf
    max_forecast_steps = stats.horizon(threshold)
    # Unchanged rule from before the other algorithms were added: periods within the horizon are already excluded
    period = stats.dominant_period(min_period=max_forecast_steps, threshold=threshold, require_trough=False)
    maxlags = max(period or 1, max_forecast_steps, min(20, stats.max_lag))
    return {
        "maxlags": int(maxlags),
        "max_forecast_steps": int(max_forecast_steps),
        "period": None if period is None else int(period),
        "period_acf": None if period is None else round(float(acf[period]), 3),
        "granularity": stats.granularity,
        "n_points": stats.n_points,
        "max_lag_searched": int(stats.max_lag),
        "acf_threshold": threshold,
    }


def ndiffs(y, alpha=0.05, max_d=2):
    """
    Number of differences needed to make ``y`` stationary: difference while the KPSS test (null hypothesis:
    level stationarity) rejects at level ``alpha``, at most ``max_d`` times. Returns ``(d, [KPSS p-values])``.
    """
    pvals = []
    d = 0
    while True:
        if np.std(y) == 0:
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pval = float(kpss(y, regression="c", nlags="auto")[1])
        pvals.append(pval)
        if pval >= alpha or d >= max_d:
            break
        y = np.diff(y)
        d += 1
    return d, pvals


def _aicc(y, order, seasonal_order=(0, 0, 0, 0), simple_differencing=False):
    """
    AICc of a SARIMA model, fitted like the Sarima/Arima models do (no trend term). ``inf`` if fitting fails.
    ``simple_differencing`` differences the data before fitting, which is several times faster for seasonal models
    (the state no longer includes the m seasonal lags) and ranks the candidates the same way.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.tsa.SARIMAX(
                y, order=order, seasonal_order=seasonal_order, simple_differencing=simple_differencing
            )
            res = model.fit(disp=0, maxiter=50)
        aicc = float(res.aicc)
        return aicc if np.isfinite(aicc) else np.inf
    except Exception:
        return np.inf


def stepwise_arma_search(y, d, max_p=5, max_q=5, max_fits=30):
    """
    Hyndman-Khandakar stepwise search for the ARIMA(p, d, q) order with the lowest AICc: start from (2, 2), (0, 0),
    (1, 0) and (0, 1), then repeatedly move to the best neighbor (p and/or q changed by 1) until no neighbor is
    better or ``max_fits`` models have been fitted. Candidates that fail to fit are skipped.

    :return: ``((p, q), aicc, number of models fitted)``. The AICc is ``inf`` if every candidate failed.
    """
    scores = {}

    def score(p, q):
        if (p, q) not in scores and len(scores) < max_fits:
            scores[(p, q)] = _aicc(y, (p, d, q))
        return scores.get((p, q), np.inf)

    starts = [(min(2, max_p), min(2, max_q)), (0, 0), (min(1, max_p), 0), (0, min(1, max_q))]
    best = min(dict.fromkeys(starts), key=lambda pq: score(*pq))
    while len(scores) < max_fits:
        p, q = best
        neighbors = [
            (p + dp, q + dq)
            for dp in (-1, 0, 1)
            for dq in (-1, 0, 1)
            if (dp, dq) != (0, 0) and 0 <= p + dp <= max_p and 0 <= q + dq <= max_q
        ]
        candidate = min(neighbors, key=lambda pq: score(*pq))
        if score(*candidate) >= score(*best):
            break
        best = candidate
    return best, scores[best], len(scores)


def _arima_order(y, d, kpss_pvals, cfg, prefix=""):
    """Recommends (p, d, q) for ``y`` (already seasonally differenced if needed), given its d from :func:`ndiffs`."""
    (p, q), aicc, n_fits = stepwise_arma_search(y, d, max_fits=cfg.ic_max_fits)
    if not np.isfinite(aicc):
        return None
    if not kpss_pvals:
        d_basis = f"d = {d}: the series is constant."
    elif d == 0:
        d_basis = f"d = 0: the KPSS test does not reject stationarity (p = {kpss_pvals[0]:.3f})."
    else:
        d_basis = f"d = {d}: the KPSS test rejects stationarity (p = {kpss_pvals[0]:.3f} < 0.05)"
        if len(kpss_pvals) > d and kpss_pvals[d] >= 0.05:
            d_basis += f", but not after {d} difference(s) (p = {kpss_pvals[d]:.3f})."
        else:
            d_basis += f", and differencing is capped at {d}."
    z = np.diff(y, n=d) if d > 0 else y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = float(adfuller(z, autolag="AIC")[1]) if np.std(z) > 0 else float("nan")
    basis = (
        f"{prefix}{d_basis} ADF p-value after differencing = {adf_p:.3f}. "
        f"p = {p}, q = {q}: lowest AICc ({aicc:.1f}) among {n_fits} models (stepwise search, p, q <= 5, at most "
        f"{cfg.ic_max_fits} models) "
        f"fitted on the latest {len(y)} points."
    )
    return (p, d, q), basis


def recommend_arima_horizon(stats: SeriesStats, cfg, y, d, D=0, m=None) -> ParamRecommendation:
    """
    ``max_forecast_steps`` for (S)ARIMA. Without differencing, this is the shared rule (ACF decay of the series).
    With differencing, the model forecasts the differenced series, whose autocorrelation is what the ARMA part can
    use: once it is no longer significant, the forecasted differences are ~0 and the forecast is flat at the current
    level (or, with a seasonal difference, repeats the last cycle). So the horizon is the last lag before the ACF of
    the differenced series first drops inside the 95% significance bound, and at least one cycle with a seasonal
    difference.

    :param y: the (latest) data the order was chosen on; ``d``/``D``/``m``: the differencing of the model.
    """
    if d + D == 0:
        return recommend_max_forecast_steps(stats, cfg)
    z = np.asarray(y, dtype=float)
    if D:
        z = z[m:] - z[:-m]
    z = np.diff(z, n=d) if d > 0 else z
    bound = 1.96 / np.sqrt(len(z))
    nlags = min(len(z) // 4, stats.max_lag)
    acf = np.nan_to_num(sm.tsa.acf(z, nlags=nlags, fft=True)) if np.std(z) > 0 else np.zeros(nlags + 1)
    inside = np.flatnonzero(np.abs(acf[1:]) < bound)
    memory = int(inside[0]) if len(inside) > 0 else nlags
    steps = max(memory, 1)
    differencing = [f"{d} difference(s)"] if d else []
    differencing += [f"a seasonal difference ({m} steps)"] if D else []
    differencing = " and ".join(differencing)
    basis = (
        f"The model forecasts the series after {differencing}, whose autocorrelation is significant "
        f"(|ACF| > {bound:.3f}) for {memory} step(s). Beyond that, the forecasted differences are ~0"
    )
    if D:
        steps = max(steps, m)
        basis += f" and the forecast repeats the last seasonal cycle, so it covers at least one cycle ({m} steps)."
    else:
        basis += ", so the forecast stays flat at the current level and says nothing more about the future."
    return ParamRecommendation("max_forecast_steps", int(steps), "int", basis, is_steps=True)


def _unmodeled_seasonality_note(stats: SeriesStats, cfg, m, algorithm) -> Optional[str]:
    """A note on a significant seasonal period that the (S)ARIMA recommendation does not model, with alternatives."""
    if m is None:
        return None
    duration = steps_to_duration(m, stats.granularity)
    note = (
        f"The data has a seasonal period of {m} steps ({duration}, ACF = {stats.acf[m]:.2f}), which "
        f"{algorithm} does not model here, so its forecast cannot follow this cycle. "
    )
    if m <= cfg.sarima_max_period:
        return note + "Use Sarima, which recommends a seasonal part for it."
    return note + (
        f"SARIMA is too slow to fit for periods longer than {cfg.sarima_max_period} steps. Resample the data at a "
        f"coarser granularity (so that the period is at most {cfg.sarima_max_period} steps), or use an algorithm "
        f"that models it: LGBMForecaster/RandomForestForecaster (maxlags >= {m}) or Prophet."
    )


# ----------------------------------------------------------------------------------------------------------------------
# Recommenders
# ----------------------------------------------------------------------------------------------------------------------


def recommend_tree(algorithm, stats: SeriesStats, cfg, feature_columns=None) -> Recommendation:
    """LGBMForecaster, RandomForestForecaster, ExtraTreesForecaster: ``maxlags`` and ``max_forecast_steps``."""
    rec = lgbm_params(stats, cfg.acf_threshold)
    threshold, granularity, period = rec["acf_threshold"], rec["granularity"], rec["period"]
    if period is not None and rec["maxlags"] == period:
        maxlags_basis = (
            f"Dominant seasonal period ({steps_to_duration(period, granularity)}, ACF = {rec['period_acf']}), "
            f"so the model sees one full cycle."
        )
    else:
        maxlags_basis = (
            f"No seasonal period with ACF >= {threshold} beyond the forecast horizon (searched up to "
            f"{rec['max_lag_searched']} steps), so it covers the forecast horizon."
        )
    params = [
        ParamRecommendation("maxlags", rec["maxlags"], "int", maxlags_basis, is_steps=True),
        recommend_max_forecast_steps(stats, cfg),
    ]
    return Recommendation(algorithm, params, stats.n_points, granularity)


def _arima_forecast(y, order, steps) -> Optional[np.ndarray]:
    """
    Fits ARIMA the way the Arima model does (SARIMAX without a trend term, without enforcing stationarity or
    invertibility) and forecasts ``steps`` ahead. ``None`` if fitting fails or the forecast is not finite.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.tsa.SARIMAX(y, order=order, enforce_stationarity=False, enforce_invertibility=False)
            forecast = np.asarray(model.fit(disp=0).forecast(steps), dtype=float)
        return forecast if np.all(np.isfinite(forecast)) else None
    except Exception:
        return None


def _long_ar_order(y, d, m, cfg) -> int:
    """
    AR order of the long-AR candidate for seasonal period ``m``: the AR order with the lowest AIC among those up to
    ``2 m`` (at most ``arima_max_ar``) on the (differenced) data, and at least ``m``, so that the AR part reaches one
    full cycle back and the forecast can follow the cycle.
    """
    z = np.diff(y, n=d) if d > 0 else y
    max_lag = max(m, min(2 * m, cfg.arima_max_ar, len(z) // 10))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lags = ar_select_order(z, maxlag=max_lag, ic="aic", trend="c" if d == 0 else "n").ar_lags
    except Exception:
        lags = None
    return int(max(max(lags or [0]), m))


def _unmodeled_arima_period_note(stats: SeriesStats, cfg, m, algorithm) -> str:
    duration = steps_to_duration(m, stats.granularity)
    return (
        f"The data has a seasonal period of {m} steps ({duration}, ACF = {stats.acf[m]:.2f}), which {algorithm} does "
        f"not model: an AR part reaching one cycle back is too slow to fit for periods longer than "
        f"{cfg.arima_max_period} steps, so its forecast cannot follow this cycle. Resample the data at a coarser "
        f"granularity (so that the period is at most {cfg.arima_max_period} steps), or use an algorithm that models it: "
        f"LGBMForecaster/RandomForestForecaster (maxlags >= {m})."
    )


def recommend_arima(algorithm, stats: SeriesStats, cfg, feature_columns=None) -> Recommendation:
    """
    Arima: ``order`` chosen among candidates by their forecast error on a holdout, fitted the way the Arima model is.

    The candidates are:

    - the short order: d from the KPSS test, (p, q) from a stepwise AICc search (and the same search with d = 1 if the
      KPSS test gives d = 0),
    - if there is a seasonal period m of at most ``arima_max_period`` steps, a long AR order (p >= m, q = 0) with
      d = 0 and/or 1, so that the forecast follows the cycle. Arima has no seasonal part, so a long AR part is how it
      can model a cycle.

    Each candidate is fitted without the latest ``arima_holdout_cycles`` cycles (or ``arima_holdout_steps`` steps) of
    the data, like the Arima model fits (no trend term, stationarity not enforced), and the one with the lowest mean
    absolute error on those points is recommended. This also rules out orders whose fit diverges, and orders with
    d = 0 whose forecast decays to 0 (there is no constant term).
    """
    y = stats.tail(cfg.ic_max_points)
    d, kpss_pvals = ndiffs(y)
    m = stats.dominant_period(max_period=cfg.arima_max_period)
    m_all = stats.dominant_period()

    # Candidate orders, with how they were chosen
    candidates = {}
    short = _arima_order(y, d, kpss_pvals, cfg)
    if short is not None:
        candidates[tuple(short[0])] = ("short order", short[1])
    if d == 0:
        (p1, q1), aicc, n_fits = stepwise_arma_search(y, 1, max_fits=cfg.ic_max_fits)
        if np.isfinite(aicc):
            basis = (
                f"d = 1 (alternative to the KPSS test's d = 0). p = {p1}, q = {q1}: lowest AICc ({aicc:.1f}) among "
                f"{n_fits} models (stepwise search)."
            )
            candidates.setdefault((p1, 1, q1), ("short order with d = 1", basis))
    if m is not None:
        for dd in sorted({d, 1}):
            p_long = _long_ar_order(y, dd, m, cfg)
            basis = (
                f"d = {dd}. p = {p_long}: AR order with the lowest AIC up to {min(2 * m, cfg.arima_max_ar)} lags and at "
                f"least the seasonal period m = {m} ({steps_to_duration(m, stats.granularity)}, ACF = "
                f"{stats.acf[m]:.2f}), so that the forecast follows the cycle. q = 0."
            )
            candidates.setdefault((p_long, dd, 0), (f"long AR order (d = {dd})", basis))

    # Compare the candidates on a holdout, if the data is long enough for it
    holdout = cfg.arima_holdout_cycles * m if m is not None else cfg.arima_holdout_steps
    errors = {}
    if len(candidates) > 1 and len(y) - holdout >= max(4 * holdout, 10 * max(o[0] for o in candidates), 50):
        for order in candidates:
            forecast = _arima_forecast(y[:-holdout], order, holdout)
            errors[order] = np.inf if forecast is None else float(np.mean(np.abs(y[-holdout:] - forecast)))

    params = []
    chosen = None
    if errors and np.isfinite(min(errors.values())):
        chosen = min(errors, key=errors.get)
        ranking = ", ".join(
            f"{candidates[o][0]} ({', '.join(map(str, o))}): " + ("failed" if not np.isfinite(e) else f"{e:.3g}")
            for o, e in sorted(errors.items(), key=lambda oe: oe[1])
        )
        basis = (
            f"{candidates[chosen][1]} Chosen for the lowest mean absolute error when forecasting the latest {holdout} "
            f"points from the {len(y) - holdout} before them, fitted the way Arima fits (no trend term, stationarity "
            f"not enforced). Errors: {ranking}."
        )
    elif short is not None:
        chosen, basis = tuple(short[0]), short[1]
    if chosen is not None:
        params.append(ParamRecommendation("order", list(chosen), "int_tuple", basis, tuple_len=3))

    # max_forecast_steps: with a long AR part, the forecast repeats the cycle, so it covers at least one cycle
    d_chosen = chosen[1] if chosen is not None else d
    horizon = recommend_arima_horizon(stats, cfg, y, d_chosen)
    if chosen is not None and m is not None and chosen[0] >= m and horizon.value < m:
        horizon = ParamRecommendation(
            "max_forecast_steps",
            int(m),
            "int",
            f"The AR part reaches one seasonal cycle ({m} steps) back, so the forecast follows the cycle for at least "
            f"one cycle ahead.",
            is_steps=True,
        )
    params.append(horizon)

    notes = [f"The order search used the latest {len(y)} points."]
    if m_all is not None and m_all > cfg.arima_max_period:
        notes.insert(0, _unmodeled_arima_period_note(stats, cfg, m_all, algorithm))
    return Recommendation(algorithm, params, stats.n_points, stats.granularity, ic_points=len(y), notes=notes)


def recommend_sarima(algorithm, stats: SeriesStats, cfg, feature_columns=None) -> Recommendation:
    """
    Sarima: the seasonal period m is the dominant significant period, the seasonal difference D is 1 if the STL
    seasonal strength exceeds ``sarima_seasonal_strength``, the order (p, d, q) is chosen like Arima's on the
    seasonally differenced data, and then (P, Q) in {0, 1}^2 by the AICc of the full model, fitted on the latest
    ``sarima_cycles`` cycles (the fits get slower with m, which is why m is capped at ``sarima_max_period``).
    """
    m = stats.dominant_period()
    params = []
    D = 0
    if m is None or m > cfg.sarima_max_period:
        y = stats.tail(cfg.ic_max_points)
        if m is None:
            why = "No significant seasonal period, so the seasonal part is left out."
        else:
            why = (
                f"The dominant seasonal period ({m} steps = {steps_to_duration(m, stats.granularity)}) is longer than "
                f"{cfg.sarima_max_period} steps, which makes SARIMA too slow to fit, so the seasonal part is left "
                f"out. Resample the data at a coarser granularity or use another algorithm (e.g. ETS, LGBMForecaster) "
                f"to model this seasonality."
            )
        seasonal_order, seasonal_basis = [0, 0, 0, 0], why
        d, kpss_pvals = ndiffs(y)
        order = _arima_order(y, d, kpss_pvals, cfg)
    else:
        y = stats.tail(max(cfg.ic_max_points, 10 * m))
        strength = stats.seasonal_strength(m)
        D = int(strength > cfg.sarima_seasonal_strength)
        ys = y[m:] - y[:-m] if D else y
        d, kpss_pvals = ndiffs(ys)
        order = _arima_order(ys, d, kpss_pvals, cfg, prefix="After the seasonal difference: " if D else "")
        p, q = (order[0][0], order[0][2]) if order is not None else (1, 1)
        y_recent = y[-cfg.sarima_cycles * m :]
        scores = {
            (P, Q): _aicc(y_recent, (p, d, q), (P, D, Q, m), simple_differencing=True) for P in (0, 1) for Q in (0, 1)
        }
        P, Q = min(scores, key=scores.get)
        if not np.isfinite(scores[(P, Q)]):
            seasonal_order = None
        else:
            seasonal_order = [P, D, Q, m]
            seasonal_basis = (
                f"m = {m}: dominant seasonal period ({steps_to_duration(m, stats.granularity)}, "
                f"ACF = {stats.acf[m]:.3f}). D = {D}: STL seasonal strength = {strength:.2f} "
                f"({'>' if D else '<='} {cfg.sarima_seasonal_strength}). P = {P}, Q = {Q}: lowest AICc "
                f"({scores[(P, Q)]:.1f}) among (P, Q) in {{0, 1}}^2 with order ({p}, {d}, {q}), fitted on the latest "
                f"{len(y_recent)} points ({len(y_recent) // m} cycles)."
            )
    if order is not None:
        params.append(ParamRecommendation("order", list(order[0]), "int_tuple", order[1], tuple_len=3))
    if seasonal_order is not None:
        params.append(ParamRecommendation("seasonal_order", seasonal_order, "int_tuple", seasonal_basis, tuple_len=4))
    params.append(recommend_arima_horizon(stats, cfg, y, d, D=D, m=m))
    notes = [f"The order search used the latest {len(y)} points."]
    if m is not None and m > cfg.sarima_max_period:
        notes.insert(0, _unmodeled_seasonality_note(stats, cfg, m, algorithm))
    return Recommendation(algorithm, params, stats.n_points, stats.granularity, ic_points=len(y), notes=notes)


def _ets_period(stats: SeriesStats, cfg):
    """
    The seasonal period for ETS, how it was chosen, and notes on the periods left out. The candidates are the
    significant periods whose ACF is at least ``ets_min_acf`` (weaker cycles do not improve the forecast, while they
    can make training very slow) and that the data covers at least twice. They are tried from the highest ACF down,
    and the first one whose estimated training time is within ``train_confirm_seconds`` is chosen, since ETS training
    time grows steeply with the period (it estimates one initial state per step of the period).

    :return: ``(period or None, basis, notes)``
    """
    acf = stats.acf
    candidates = [
        p
        for p in stats.periods
        if 1 < p <= stats.max_lag and acf[p] >= cfg.ets_min_acf and acf[p] > acf[p // 2] and stats.n_points >= 2 * p
    ]
    candidates.sort(key=lambda p: acf[p], reverse=True)

    def slow_note(skipped):
        # The strongest period left out, and the others (often near multiples of it) in short
        (p, seconds), others = skipped[0], [s[0] for s in skipped[1:]]
        note = (
            f"The seasonal period of {p} steps ({steps_to_duration(p, stats.granularity)}, ACF = {acf[p]:.2f}) is "
            f"not recommended: ETS would take about {format_duration(seconds)} to train with it (more than "
            f"{format_duration(cfg.train_confirm_seconds)}). To model it, resample the data at a coarser granularity "
            f"or use LGBMForecaster (maxlags >= {p})."
        )
        if others:
            note += f" Other periods left out for the same reason: {', '.join(map(str, others))} steps."
        return [note]

    skipped = []
    for p in candidates:
        est = estimate_train_seconds("ETS", {"seasonal": "add", "seasonal_periods": p}, stats.n_points)
        if est.seconds <= cfg.train_confirm_seconds:
            basis = (
                f"Seasonal period with the highest ACF among those >= {cfg.ets_min_acf} "
                f"({steps_to_duration(p, stats.granularity)}, ACF = {acf[p]:.3f}); STL seasonal strength = "
                f"{stats.seasonal_strength(p):.2f}. Estimated training time: {format_duration(est.seconds)}."
            )
            if skipped:
                basis += " Stronger periods were left out because they would train too long (see the notes)."
            return p, basis, slow_note(skipped) if skipped else []
        skipped.append((p, est.seconds))

    if skipped:
        return None, "No seasonal period that ETS can be trained with in time (see the notes).", slow_note(skipped)
    dominant = stats.dominant_period()
    if dominant is not None and acf[dominant] < cfg.ets_min_acf:
        basis = (
            f"No seasonal period strong enough to model: the dominant significant period "
            f"({steps_to_duration(dominant, stats.granularity)}) has ACF = {acf[dominant]:.3f} < {cfg.ets_min_acf}, "
            f"too weak for a seasonal component to improve the forecast."
        )
    else:
        basis = "No significant seasonal period."
    return None, basis, []


def recommend_ets(algorithm, stats: SeriesStats, cfg, feature_columns=None) -> Recommendation:
    """
    ETS: each component is matched to one property of the data. The seasonal period is the strongest significant
    period with an ACF of at least ``ets_min_acf`` that ETS can be trained with in ``train_confirm_seconds`` (see
    `_ets_period`); the seasonality is multiplicative if its amplitude grows with the level; there is a (damped)
    trend if the STL trend strength is at least ``ets_trend_strength``.
    """
    m, periods_basis, notes = _ets_period(stats, cfg)
    if m is None:
        seasonal, seasonal_basis = "None", "No seasonal period (see seasonal_periods)."
    else:
        mul, why = stats.is_multiplicative(m, cfg.multiplicative_min_corr)
        seasonal = "mul" if mul else "add"
        seasonal_basis = f"{'Multiplicative' if mul else 'Additive'}: {why}."

    strength = stats.trend_strength(m)
    has_trend = strength >= cfg.ets_trend_strength
    trend_basis = f"Trend strength = {strength:.2f} ({'>=' if has_trend else '<'} {cfg.ets_trend_strength})" + (
        f", from an STL decomposition with period {m}." if m is not None else ", from a moving-average trend."
    )
    damped_basis = (
        "A damped trend levels off over long horizons, which is usually more accurate than extrapolating it."
        if has_trend
        else "There is no trend to damp."
    )
    params = [
        ParamRecommendation(
            "error",
            "add",
            "choice",
            "Additive errors: multiplicative errors need strictly positive data with a level-dependent variance and "
            "are numerically less stable.",
            choices=["add", "mul"],
        ),
        ParamRecommendation(
            "trend", "add" if has_trend else "None", "choice", trend_basis, choices=["add", "mul", "None"]
        ),
        ParamRecommendation("damped_trend", has_trend, "choice", damped_basis, choices=["True", "False"]),
        ParamRecommendation("seasonal", seasonal, "choice", seasonal_basis, choices=["add", "mul", "None"]),
        ParamRecommendation("seasonal_periods", m, "int", periods_basis, optional=True, is_steps=True),
        recommend_max_forecast_steps(stats, cfg),
    ]
    return Recommendation(algorithm, params, stats.n_points, stats.granularity, notes=notes)


#: Prophet's calendar seasonalities: (parameter, cycle name, period in seconds, aggregation bin in seconds)
_PROPHET_SEASONALITIES = [
    ("yearly_seasonality", "yearly", 365.25 * 86400, 7 * 86400),
    ("weekly_seasonality", "weekly", 7 * 86400, 86400),
    ("daily_seasonality", "daily", 86400, 3600),
]


def _calendar_seasonality(stats: SeriesStats, name, period, bin_size, min_acf):
    """
    Whether a calendar seasonality (e.g. daily) is present, and why. It must be observable (the data spans at least
    two cycles and the granularity is below half a cycle) and significant: after averaging the data into bins (one
    hour for daily, one day for weekly, one week for yearly, so that shorter seasonalities average out) and removing
    the trend with a moving average over one cycle, the ACF near the cycle length (+/- 10%) must exceed both the 95%
    significance bound and ``min_acf``.
    """
    granularity = stats.granularity_seconds
    span = (stats.series.index[-1] - stats.series.index[0]).total_seconds()
    if span < 2 * period:
        return False, f"The data spans {steps_to_duration(1, span)}, less than two {name} cycles."
    if granularity >= period / 2:
        return False, f"The granularity is too coarse to observe a {name} cycle."

    bin_size = max(bin_size, granularity)
    a = stats.series.resample(pd.Timedelta(seconds=bin_size)).mean().interpolate().values
    lag = int(round(period / bin_size))
    lo, hi = max(1, int(np.floor(0.9 * lag))), int(np.ceil(1.1 * lag))
    if lag < 2 or len(a) <= hi + 1:
        return False, f"Too few points to test a {name} cycle."
    a = a - pd.Series(a).rolling(lag, center=True, min_periods=1).mean().values
    acf = np.nan_to_num(sm.tsa.acf(a, nlags=hi, fft=True))
    peak = float(acf[lo : hi + 1].max())
    bound = max(1.96 / np.sqrt(len(a)), min_acf)
    present = peak > bound
    return present, (
        f"ACF at one {name} cycle = {peak:.2f} ({'>' if present else '<='} {bound:.2f}, the larger of the 95% "
        f"significance bound and {min_acf}), after averaging over {steps_to_duration(1, bin_size)} bins and detrending."
    )


def recommend_prophet(algorithm, stats: SeriesStats, cfg, feature_columns=None) -> Recommendation:
    """Prophet: each calendar seasonality must be observable and significant; the mode follows the ETS rule."""
    params = []
    for name, cycle, period, bin_size in _PROPHET_SEASONALITIES:
        present, basis = _calendar_seasonality(stats, cycle, period, bin_size, cfg.prophet_min_acf)
        params.append(ParamRecommendation(name, present, "choice", basis, choices=["True", "False", "auto"]))

    m = stats.dominant_period()
    if m is None:
        mode, mode_basis = "additive", "Additive: no significant seasonal period."
    else:
        mul, why = stats.is_multiplicative(m, cfg.multiplicative_min_corr)
        mode = "multiplicative" if mul else "additive"
        mode_basis = f"{mode.capitalize()}: for the dominant period ({steps_to_duration(m, stats.granularity)}), {why}."
    params.append(
        ParamRecommendation("seasonality_mode", mode, "choice", mode_basis, choices=["additive", "multiplicative"])
    )
    params.append(recommend_max_forecast_steps(stats, cfg))
    return Recommendation(algorithm, params, stats.n_points, stats.granularity)


def recommend_vector_ar(algorithm, stats: SeriesStats, cfg, feature_columns=None) -> Recommendation:
    """
    VectorAR: ``maxlags`` is the VAR order with the lowest BIC (the target and the feature variables). BIC is
    preferred over AIC because the number of VAR parameters grows with the square of the number of variables.
    Without feature variables, it is the AR order with the lowest BIC.
    """
    feature_columns = list(feature_columns or [])
    if feature_columns:
        y = stats.multivariate(feature_columns).values[-cfg.ic_max_points :]
        k = y.shape[1]
        max_order = max(1, min(cfg.var_max_lags, len(y) // (10 * k)))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                selected = sm.tsa.VAR(y).select_order(maxlags=max_order).selected_orders
        except (np.linalg.LinAlgError, ValueError) as e:
            # e.g. a constant or collinear feature variable makes the covariance matrix singular
            notes = [
                f"maxlags is not recommended: selecting the VAR order failed ({type(e).__name__}: {e}). "
                f"Check for constant or collinear feature variables."
            ]
            params = [recommend_max_forecast_steps(stats, cfg)]
            return Recommendation(algorithm, params, stats.n_points, stats.granularity, notes=notes)
        maxlags = max(int(selected["bic"]), 1)
        basis = (
            f"Lowest BIC among VAR orders 0 to {max_order} for {k} variables (orders selected by AIC = "
            f"{selected['aic']}, BIC = {selected['bic']}, HQIC = {selected['hqic']}), fitted on the latest {len(y)} "
            f"points. BIC favors fewer lags, since each lag adds {k * k} parameters."
        )
    else:
        y = stats.tail(cfg.ic_max_points)
        max_order = max(1, min(cfg.var_max_lags, len(y) // 10))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lags = ar_select_order(y, maxlag=max_order, ic="bic", trend="c").ar_lags
        maxlags = max(max(lags or [0]), 1)
        basis = (
            f"Lowest BIC among AR orders 0 to {max_order} (univariate, since no feature variables are selected), "
            f"fitted on the latest {len(y)} points."
        )
    params = [
        ParamRecommendation("maxlags", int(maxlags), "int", basis, is_steps=True),
        recommend_max_forecast_steps(stats, cfg),
    ]
    notes = [f"The order search used the latest {len(y)} points."]
    return Recommendation(algorithm, params, stats.n_points, stats.granularity, ic_points=len(y), notes=notes)


def recommend_default(algorithm, stats: SeriesStats, cfg, feature_columns=None) -> Recommendation:
    """DefaultForecaster: ``max_forecast_steps`` and the inferred ``granularity``."""
    if isinstance(stats.granularity, (int, float)):
        granularity = pd.tseries.frequencies.to_offset(pd.Timedelta(seconds=stats.granularity)).freqstr
    else:
        granularity = str(stats.granularity)
    params = [
        recommend_max_forecast_steps(stats, cfg),
        ParamRecommendation(
            "granularity",
            granularity,
            "str",
            "Granularity inferred from the timestamps of the training data "
            f"({steps_to_duration(1, stats.granularity)}).",
        ),
    ]
    return Recommendation(algorithm, params, stats.n_points, stats.granularity)


RECOMMENDERS: Dict[str, Callable[..., Recommendation]] = {
    "DefaultForecaster": recommend_default,
    "Arima": recommend_arima,
    "LGBMForecaster": recommend_tree,
    "ETS": recommend_ets,
    "Prophet": recommend_prophet,
    "Sarima": recommend_sarima,
    "VectorAR": recommend_vector_ar,
    "RandomForestForecaster": recommend_tree,
    "ExtraTreesForecaster": recommend_tree,
}


def parse_confirmed_values(specs: Sequence[dict], values: Sequence[Any]):
    """
    Validates the values confirmed in the popup against the specs of the recommended parameters
    (:meth:`ParamRecommendation.spec`), and converts them to the types the model configs expect.

    :return: ``(overrides, errors)``: the converted values by parameter name, and a list of error messages.
    """
    overrides, errors = {}, []
    for spec, value in zip(specs, values):
        name, kind = spec["name"], spec["kind"]
        if isinstance(value, str):
            value = value.strip()
        if value is None or value == "" or value == "None" and kind != "choice":
            if spec.get("optional"):
                overrides[name] = None
            else:
                errors.append(f"{name} is required.")
            continue

        if kind == "int":
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, str) and value.lstrip("-").isdigit():
                value = int(value)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                errors.append(f"{name} must be a positive integer.")
            else:
                overrides[name] = value
        elif kind == "choice":
            value = str(value)
            if value not in spec["choices"]:
                errors.append(f"{name} must be one of {', '.join(spec['choices'])}.")
            else:
                overrides[name] = {"None": None, "True": True, "False": False}.get(value, value)
        elif kind == "int_tuple":
            n = spec["tuple_len"]
            items = str(value).strip("()[] ").split(",")
            items = [s.strip() for s in items if s.strip() != ""]
            if len(items) != n or not all(s.isdigit() for s in items):
                errors.append(f"{name} must be {n} integers >= 0, e.g. ({', '.join(['1'] * n)}).")
            else:
                overrides[name] = [int(s) for s in items]
        else:
            overrides[name] = str(value)
    return overrides, errors

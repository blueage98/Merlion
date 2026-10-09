#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from dataclasses import dataclass
import logging
import sys
from typing import Optional, Tuple

import pandas as pd

from merlion.models.factory import ModelFactory
from merlion.evaluate.forecast import ForecastEvaluator, ForecastMetric
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import TimeSeries
from merlion.dashboard.models.recommend import (
    RECOMMENDERS,
    Recommendation,
    SeriesStats,
    lgbm_params,
    steps_to_duration,
)
from merlion.dashboard.models.train_time import TrainTimeEstimate, estimate_train_seconds
from merlion.dashboard.models.model_select import ModelRecommendation, recommend_model
from merlion.dashboard.models.utils import ModelMixin, DataMixin
from merlion.dashboard.utils.log import DashLogger

dash_logger = DashLogger(stream=sys.stdout)


@dataclass
class ColumnLayout:
    """
    How the variables are arranged for a forecasting model: the target first, then the feature variables, then the
    exogenous variables. Used both to train a model (`ForecastModel.train`) and to validate the candidate algorithms
    (merlion/dashboard/models/model_select.py), so that both see the data the same way.
    """

    target_column: object
    feature_columns: list
    exog_columns: list

    @property
    def columns(self) -> list:
        return [self.target_column] + self.feature_columns + self.exog_columns

    @property
    def target_seq_index(self) -> int:
        return self.columns.index(self.target_column)

    def model_columns(self, supports_exog) -> list:
        """The columns of the model's training data: the exogenous variables are left out if the model takes them
        separately (``exog_data``), and are ordinary input variables otherwise."""
        if supports_exog and self.exog_columns:
            return [self.target_column] + self.feature_columns
        return self.columns

    def split(self, df: pd.DataFrame, supports_exog) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """``(data, exog)``: the model's training data and its exogenous data (``None`` if it takes none)."""
        if supports_exog and self.exog_columns:
            return df.loc[:, self.model_columns(True)], df.loc[:, self.exog_columns]
        return df.loc[:, self.columns], None


class ForecastModel(ModelMixin, DataMixin):
    algorithms = [
        "DefaultForecaster",
        "Arima",
        "LGBMForecaster",
        "ETS",
        "AutoETS",
        "Prophet",
        "AutoProphet",
        "Sarima",
        "VectorAR",
        "RandomForestForecaster",
        "ExtraTreesForecaster",
    ]

    # Algorithms whose hyperparameters are recommended (and confirmed by the user) before training.
    tuned_algorithms = list(RECOMMENDERS)

    # Settings of the recommenders (see merlion/dashboard/models/recommend.py).
    # ACF level below which the recent history is considered too weak to forecast from (max_forecast_steps).
    acf_threshold = 0.5
    # Same threshold, under the name used by recommend_lgbm_params() before the other algorithms were added.
    lgbm_acf_threshold = acf_threshold
    # Upper bound on the lags searched for seasonal periods, which is also the maximum recommended maxlags.
    lgbm_max_lags = 2000
    # Number of latest points the model-fitting searches (ARIMA/SARIMA orders, VAR order) are run on.
    ic_max_points = 2000
    # Maximum number of ARIMA models fitted by the stepwise order search (high orders take ~0.5s each to fit).
    ic_max_fits = 15
    # Longest seasonal period ARIMA models with a long AR part (p >= period). Fitting an AR part of 48-60 lags takes a
    # few seconds on 2000 points; much longer periods (e.g. the daily cycle of 5-minute data) are not modeled.
    arima_max_period = 48
    # Maximum AR order of the long-AR candidate (searched up to twice the period, capped here).
    arima_max_ar = 60
    # Length of the holdout the ARIMA candidates are compared on, in seasonal cycles (or in default steps without a
    # seasonal period).
    arima_holdout_cycles = 2
    arima_holdout_steps = 24
    # With a seasonal period, a short ARIMA order is only chosen if its holdout error is this much lower than that
    # of the long AR order (short orders that won the holdout narrowly generalized poorly on M4 Hourly).
    arima_short_margin = 0.2
    # Longest seasonal period for which SARIMA gets a seasonal part. Fitting gets slower with the period: searching
    # the seasonal orders takes ~4s for 24 (e.g. daily cycle of hourly data), ~15s for 48 and over a minute for 100.
    sarima_max_period = 24
    # Number of latest seasonal cycles the seasonal orders (P, Q) of SARIMA are searched on.
    sarima_cycles = 20
    # STL seasonal strength above which SARIMA gets a seasonal difference (D = 1), as in auto.arima.
    sarima_seasonal_strength = 0.64
    # STL trend strength from which ETS gets a (damped) trend.
    ets_trend_strength = 0.5
    # Minimum ACF of a seasonal period for ETS to model it. On the manufacturing benchmark (NAB machine temperature,
    # SKAB sensors), the significant periods were either weak (ACF < 0.1: modeling them did not improve the forecast
    # on average, and periods of 216-493 steps took 3-40 min to train) or clear (ACF >= 0.4: MASE improved by ~0.18).
    ets_min_acf = 0.3
    # Spearman correlation between the seasonal amplitude and the level from which the seasonality is multiplicative.
    multiplicative_min_corr = 0.5
    # Maximum VAR order searched for VectorAR's maxlags.
    var_max_lags = 20
    # Estimated training time (in seconds) above which training only starts once the user approves it. The service
    # runs on limited resources, so a long training is not started without asking.
    train_confirm_seconds = 300

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(dash_logger)

    @staticmethod
    def get_available_algorithms():
        return ForecastModel.algorithms

    @staticmethod
    def _series_stats(train_df, target_column):
        return SeriesStats(train_df, target_column, max_lags=ForecastModel.lgbm_max_lags)

    @staticmethod
    def recommend_params(algorithm, train_df, target_column, feature_columns=None) -> Recommendation:
        """
        Recommends the hyperparameters of the given algorithm from the training data, with the reasoning behind
        each value. See merlion/dashboard/models/recommend.py for the method used for each algorithm.

        :raises ValueError: if the algorithm has no recommender, or no parameter could be recommended.
        """
        if algorithm not in RECOMMENDERS:
            raise ValueError(f"No recommended settings for {algorithm}.")
        stats = ForecastModel._series_stats(train_df, target_column)
        rec = RECOMMENDERS[algorithm](algorithm, stats, ForecastModel, feature_columns=feature_columns)
        if not rec.params:
            raise ValueError(f"Could not recommend any setting for {algorithm}.")
        return rec

    @staticmethod
    def arrange_columns(train_df, target_column, feature_columns=None, exog_columns=None) -> ColumnLayout:
        """
        The arrangement of the variables for training (see `ColumnLayout`). Column names are converted to ``int`` if
        they are not in the data as given (the dashboard passes them as strings).

        :raises ValueError: if the target variable is neither in the data nor an integer.
        :raises AssertionError: if the target or an exogenous variable is not in the data.
        """
        if target_column not in train_df:
            target_column = int(target_column)
        assert target_column in train_df, f"The target variable {target_column} is not in the time series."
        try:
            feature_columns = [int(c) if c not in train_df else c for c in feature_columns or []]
        except ValueError:
            feature_columns = []
        try:
            exog_columns = [int(c) if c not in train_df else c for c in exog_columns or []]
        except ValueError:
            exog_columns = []
        for exog_column in exog_columns:
            assert exog_column in train_df, f"Exogenous variable {exog_column} is not in the time series."
        return ColumnLayout(target_column, feature_columns, exog_columns)

    @staticmethod
    def recommend_model(
        train_df, target_column, horizon, feature_columns=None, exog_columns=None, algorithms=None
    ) -> ModelRecommendation:
        """
        Recommends a forecasting algorithm (with its recommended hyperparameters) for the training data, by the error
        of each candidate on the last 20% of the data, forecast ``horizon`` steps at a time. The candidates are
        trained on the variables arranged the way `train` does (target, feature and exogenous variables). See
        merlion/dashboard/models/model_select.py.
        """
        return recommend_model(
            train_df,
            target_column,
            horizon,
            ForecastModel,
            feature_columns=feature_columns,
            exog_columns=exog_columns,
            algorithms=algorithms,
        )

    @staticmethod
    def estimate_train_time(
        algorithm, train_df, target_column, feature_columns, exog_columns, params
    ) -> TrainTimeEstimate:
        """
        Estimates how long training ``algorithm`` with ``params`` on the training data takes on this machine. The
        data is resampled the way the models do, and the variables are those the model is trained on. See
        merlion/dashboard/models/train_time.py for the cost model.
        """
        if target_column not in train_df:
            target_column = int(target_column)
        columns = [target_column] + [c if c in train_df else int(c) for c in list(feature_columns) + list(exog_columns)]
        ts = TimeSeries.from_pd(train_df.loc[:, columns])
        resample = TemporalResample()
        resample.train(ts)
        n_points = len(resample(ts).to_pd())
        return estimate_train_seconds(algorithm, params, n_points, n_variables=len(columns))

    @staticmethod
    def recommend_lgbm_params(train_df, target_column):
        """
        Recommends ``maxlags`` and ``max_forecast_steps`` for LGBMForecaster from the autocorrelation (ACF) of the
        target variable, after resampling the training data the same way the model does (``TemporalResample``).

        - ``max_forecast_steps``: LGBMForecaster forecasts autoregressively, so its errors compound with the horizon.
          We stop right before the ACF first drops below ``acf_threshold``, i.e. once the recent history no
          longer says much about the value being forecasted.
        - ``maxlags``: the dominant seasonal period, so that the model sees one full cycle. This is the significant
          seasonality with the highest ACF among those beyond ``max_forecast_steps`` (shorter lags have a high ACF
          just because the series changes slowly), and its ACF must be at least ``acf_threshold``. Like
          LGBMForecaster's own default, maxlags is at least ``max_forecast_steps``. The search is capped at
          ``lgbm_max_lags`` (and a quarter of the training data) to bound the number of features.

        :return: A dict with the recommended ``maxlags`` and ``max_forecast_steps``, plus the statistics they are
            based on (``period``, ``period_acf``, ``granularity``, ``n_points``, ``max_lag_searched``, ``acf_threshold``).
        """
        return lgbm_params(ForecastModel._series_stats(train_df, target_column), ForecastModel.acf_threshold)

    @staticmethod
    def steps_to_duration(steps, granularity):
        """Converts a number of time steps at the given granularity into a human-readable duration."""
        return steps_to_duration(steps, granularity)

    @staticmethod
    def _compute_metrics(evaluator, ts, predictions):
        return {
            m: round(evaluator.evaluate(ground_truth=ts, predict=predictions, metric=ForecastMetric[m]), 5)
            for m in ["MAE", "MARRE", "RMSE", "sMAPE", "RMSPE"]
        }

    @staticmethod
    def _forecast_horizon_end(model, max_forecast_steps):
        """
        The last time stamp a trained model can forecast, computed the way `ForecasterBase.resample_time_stamps` does:
        ``max_forecast_steps`` steps of the training granularity after the end of the training data. ``None`` if the
        model does not expose its granularity.
        """
        dt, t0 = getattr(model, "timedelta", None), getattr(model, "last_train_time", None)
        if dt is None or t0 is None:
            return None
        offset = getattr(model, "timedelta_offset", pd.Timedelta(0))
        steps = pd.date_range(start=t0, periods=max_forecast_steps + 1, freq=dt) + offset
        steps = steps[1:] if steps[0] == t0 else steps[:-1]
        return steps[-1]

    def train(self, algorithm, train_df, test_df, target_column, feature_columns, exog_columns, params, set_progress):
        # Arrange the columns so that the target column is first, and exogenous columns are last
        layout = ForecastModel.arrange_columns(train_df, target_column, feature_columns, exog_columns)
        train_df = train_df.loc[:, layout.columns]
        test_df = test_df.loc[:, layout.columns]

        # Get the target_seq_index & initialize the model
        params["target_seq_index"] = layout.target_seq_index
        model_class = ModelFactory.get_model_class(algorithm)
        model = model_class(model_class.config_class(**params))

        # Handle exogenous regressors if they are supported by the model
        train_df, train_exog = layout.split(train_df, model.supports_exog)
        test_df, test_exog = layout.split(test_df, model.supports_exog)
        exog_ts = None if train_exog is None else TimeSeries.from_pd(pd.concat((train_exog, test_exog)))

        self.logger.info(f"Training the forecasting model: {algorithm}...")
        set_progress(("2", "10"))
        train_ts = TimeSeries.from_pd(train_df)
        predictions = model.train(train_ts, exog_data=exog_ts)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        self.logger.info("Computing training performance metrics...")
        set_progress(("6", "10"))
        evaluator = ForecastEvaluator(model, config=ForecastEvaluator.config_class())
        train_metrics = ForecastModel._compute_metrics(evaluator, train_ts, predictions)
        set_progress(("7", "10"))

        test_ts = TimeSeries.from_pd(test_df)
        if "max_forecast_steps" in params and params["max_forecast_steps"] is not None:
            # Keep the test points the model can forecast. The horizon is max_forecast_steps steps of the training
            # granularity, which is not max_forecast_steps rows of irregularly sampled data (e.g. 1-second data with
            # some 2-second gaps spans more time than the horizon).
            t_end = ForecastModel._forecast_horizon_end(model, int(params["max_forecast_steps"]))
            if t_end is not None:
                test_ts, _ = test_ts.bisect(t=t_end, t_in_left=True)
            else:
                n = min(len(test_ts) - 1, int(params["max_forecast_steps"]))
                test_ts, _ = test_ts.bisect(t=test_ts.time_stamps[n])

        self.logger.info("Computing test performance metrics...")
        test_pred, test_err = model.forecast(time_stamps=test_ts.time_stamps, exog_data=exog_ts)
        test_metrics = ForecastModel._compute_metrics(evaluator, test_ts, test_pred)
        set_progress(("8", "10"))

        self.logger.info("Plotting forecasting results...")
        figure = model.plot_forecast_plotly(
            time_series=test_ts, time_series_prev=train_ts, exog_data=exog_ts, plot_forecast_uncertainty=True
        )
        figure.update_layout(width=None, height=500)
        self.logger.info("Finished.")
        set_progress(("10", "10"))

        return model, train_metrics, test_metrics, figure

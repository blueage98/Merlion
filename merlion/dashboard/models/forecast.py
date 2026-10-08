#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import sys

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
from merlion.dashboard.models.utils import ModelMixin, DataMixin
from merlion.dashboard.utils.log import DashLogger

dash_logger = DashLogger(stream=sys.stdout)


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
    # Minimum ACF at one calendar cycle for Prophet's yearly/weekly/daily seasonality, besides being significant.
    prophet_min_acf = 0.1
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

    def train(self, algorithm, train_df, test_df, target_column, feature_columns, exog_columns, params, set_progress):
        if target_column not in train_df:
            target_column = int(target_column)
        assert target_column in train_df, f"The target variable {target_column} is not in the time series."
        try:
            feature_columns = [int(c) if c not in train_df else c for c in feature_columns]
        except ValueError:
            feature_columns = []
        try:
            exog_columns = [int(c) if c not in train_df else c for c in exog_columns]
        except ValueError:
            exog_columns = []
        for exog_column in exog_columns:
            assert exog_column in train_df, f"Exogenous variable {exog_column} is not in the time series."

        # Re-arrange dataframe so that the target column is first, and exogenous columns are last
        columns = [target_column] + feature_columns + exog_columns
        train_df = train_df.loc[:, columns]
        test_df = test_df.loc[:, columns]

        # Get the target_seq_index & initialize the model
        params["target_seq_index"] = columns.index(target_column)
        model_class = ModelFactory.get_model_class(algorithm)
        model = model_class(model_class.config_class(**params))

        # Handle exogenous regressors if they are supported by the model
        if model.supports_exog and len(exog_columns) > 0:
            exog_ts = TimeSeries.from_pd(pd.concat((train_df.loc[:, exog_columns], test_df.loc[:, exog_columns])))
            train_df = train_df.loc[:, [target_column] + feature_columns]
            test_df = test_df.loc[:, [target_column] + feature_columns]
        else:
            exog_ts = None

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

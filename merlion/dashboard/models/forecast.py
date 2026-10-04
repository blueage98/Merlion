#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from merlion.models.automl.seasonality import SeasonalityLayer, PeriodicityStrategy
from merlion.models.factory import ModelFactory
from merlion.evaluate.forecast import ForecastEvaluator, ForecastMetric
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import TimeSeries
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

    # Algorithms whose maxlags/max_forecast_steps are recommended (and confirmed by the user) before training.
    tuned_algorithms = ["LGBMForecaster"]
    # Autocorrelation level below which the recent history is considered too weak to forecast from.
    lgbm_acf_threshold = 0.5
    # Upper bound on the recommended maxlags, which is also the number of lag features LightGBM is trained on.
    lgbm_max_lags = 2000

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(dash_logger)

    @staticmethod
    def get_available_algorithms():
        return ForecastModel.algorithms

    @staticmethod
    def recommend_lgbm_params(train_df, target_column):
        """
        Recommends ``maxlags`` and ``max_forecast_steps`` for LGBMForecaster from the autocorrelation (ACF) of the
        target variable, after resampling the training data the same way the model does (``TemporalResample``).

        - ``max_forecast_steps``: LGBMForecaster forecasts autoregressively, so its errors compound with the horizon.
          We stop right before the ACF first drops below ``lgbm_acf_threshold``, i.e. once the recent history no
          longer says much about the value being forecasted.
        - ``maxlags``: the dominant seasonal period, so that the model sees one full cycle. This is the significant
          seasonality with the highest ACF among those beyond ``max_forecast_steps`` (shorter lags have a high ACF
          just because the series changes slowly), and its ACF must be at least ``lgbm_acf_threshold``. Like
          LGBMForecaster's own default, maxlags is at least ``max_forecast_steps``. The search is capped at
          ``lgbm_max_lags`` (and a quarter of the training data) to bound the number of features.

        :return: A dict with the recommended ``maxlags`` and ``max_forecast_steps``, plus the statistics they are
            based on (``period``, ``period_acf``, ``granularity``, ``n_points``, ``max_lag_searched``, ``acf_threshold``).
        """
        if target_column not in train_df:
            target_column = int(target_column)
        assert target_column in train_df, f"The target variable {target_column} is not in the time series."

        ts = TimeSeries.from_pd(train_df.loc[:, [target_column]])
        resample = TemporalResample()
        resample.train(ts)
        x = resample(ts).to_pd().iloc[:, 0].values.astype(float)
        assert len(x) >= 40, f"The training data is too short ({len(x)} points) to recommend LGBMForecaster settings."

        threshold = ForecastModel.lgbm_acf_threshold
        max_lag = min(len(x) // 4, ForecastModel.lgbm_max_lags)
        acf = np.nan_to_num(sm.tsa.acf(x, nlags=max_lag, fft=True))

        below = np.flatnonzero(acf[1:] < threshold)
        max_forecast_steps = int(below[0]) if len(below) > 0 else max_lag
        max_forecast_steps = max(max_forecast_steps, 1)

        period = None
        if np.std(x) > 0:
            periods = SeasonalityLayer.detect_seasonality(
                x, max_lag=max_lag, pval=0.01, periodicity_strategy=PeriodicityStrategy.All
            )
            strong = [p for p in periods if max_forecast_steps < p <= max_lag and acf[p] >= threshold]
            period = max(strong, key=lambda p: acf[p]) if strong else None

        maxlags = max(period or 1, max_forecast_steps, min(20, max_lag))
        return {
            "maxlags": int(maxlags),
            "max_forecast_steps": int(max_forecast_steps),
            "period": None if period is None else int(period),
            "period_acf": None if period is None else round(float(acf[period]), 3),
            "granularity": resample.granularity,
            "n_points": len(x),
            "max_lag_searched": int(max_lag),
            "acf_threshold": threshold,
        }

    @staticmethod
    def steps_to_duration(steps, granularity):
        """Converts a number of time steps at the given granularity into a human-readable duration."""
        if isinstance(granularity, (int, float)):
            return str(pd.Timedelta(seconds=steps * granularity))
        return f"{steps} x {granularity}"

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

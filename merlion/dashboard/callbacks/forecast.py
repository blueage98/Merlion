#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import os
import time
import traceback

import dash
import numpy as np
from dash import ALL, Input, Output, State, dcc, html, callback
from merlion.dashboard.utils.file_manager import FileManager
from merlion.dashboard.models.forecast import ForecastModel
from merlion.dashboard.models.recommend import format_value, parse_confirmed_values
from merlion.dashboard.models.train_time import format_duration
from merlion.dashboard.pages.forecast import (
    analysis_mode_description,
    analysis_mode_text,
    create_confirm_inputs,
    create_model_selection_content,
)
from merlion.dashboard.pages.utils import create_param_table, create_metric_table, create_empty_figure
from merlion.transform.resample import TemporalResample
from merlion.utils.time_series import TimeSeries

logger = logging.getLogger(__name__)
file_manager = FileManager()


@callback(
    Output("forecasting-select-file", "options"),
    Output("forecasting-select-target", "value"),
    Output("forecasting-select-features", "value"),
    Output("forecasting-select-exog", "value"),
    Input("forecasting-select-file-parent", "n_clicks"),
    Input("forecasting-select-file", "value"),
    [
        State("forecasting-select-target", "value"),
        State("forecasting-select-features", "value"),
        State("forecasting-select-exog", "value"),
    ],
)
def update_select_file_dropdown(n_clicks, filename, target, features, exog):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_ids = {p["prop_id"].split(".")[0]: p["value"] for p in ctx.triggered}
        if "forecasting-select-file-parent" in prop_ids:
            files = file_manager.uploaded_files()
            for f in files:
                options.append({"label": f, "value": f})
        if "forecasting-select-file" in prop_ids:
            target, features, exog = None, None, None
    return options, target, features, exog


@callback(Output("forecasting-select-test-file", "options"), Input("forecasting-select-test-file-parent", "n_clicks"))
def update_select_test_file_dropdown(n_clicks):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "forecasting-select-test-file-parent":
        files = file_manager.uploaded_files()
        for filename in files:
            options.append({"label": filename, "value": filename})
    return options


@callback(
    Output("forecasting-select-target", "options"),
    Input("forecasting-select-target-parent", "n_clicks"),
    [
        State("forecasting-select-file", "value"),
        State("forecasting-select-features", "value"),
        State("forecasting-select-exog", "value"),
    ],
)
def select_target(n_clicks, filename, feat_names, exog_names):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "forecasting-select-target-parent":
        if filename is not None:
            file_path = os.path.join(file_manager.data_directory, filename)
            df = ForecastModel().load_data(file_path, nrows=2)
            forbidden = (feat_names or []) + (exog_names or [])
            options += [{"label": s, "value": s} for s in df.columns if s not in forbidden]
    return options


@callback(
    Output("forecasting-select-features", "options"),
    Input("forecasting-select-features-parent", "n_clicks"),
    [
        State("forecasting-select-file", "value"),
        State("forecasting-select-target", "value"),
        State("forecasting-select-exog", "value"),
    ],
)
def select_features(n_clicks, filename, target_name, exog_names):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "forecasting-select-features-parent":
        if filename is not None and target_name is not None:
            file_path = os.path.join(file_manager.data_directory, filename)
            df = ForecastModel().load_data(file_path, nrows=2)
            options += [{"label": s, "value": s} for s in df.columns if s not in [target_name] + (exog_names or [])]
    return options


@callback(
    Output("forecasting-select-exog", "options"),
    Input("forecasting-select-exog-parent", "n_clicks"),
    [
        State("forecasting-select-file", "value"),
        State("forecasting-select-target", "value"),
        State("forecasting-select-features", "value"),
    ],
)
def select_exog(n_clicks, filename, target_name, feat_names):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "forecasting-select-exog-parent":
        if filename is not None and target_name is not None:
            file_path = os.path.join(file_manager.data_directory, filename)
            df = ForecastModel().load_data(file_path, nrows=2)
            options += [{"label": s, "value": s} for s in df.columns if s not in [target_name] + (feat_names or [])]
    return options


@callback(
    Output("forecasting-select-algorithm", "options"),
    Input("forecasting-select-algorithm-parent", "n_clicks"),
    [State("forecasting-select-target", "value")],
)
def select_algorithm_parent(n_clicks, selected_target):
    options = []
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "forecasting-select-algorithm-parent":
        algorithms = ForecastModel.get_available_algorithms()
        options += [{"label": s, "value": s} for s in algorithms]
    return options


def handle_select_algorithm(algorithm, auto_result, param_info=None):
    """
    The logic of `select_algorithm`, separated from Dash so that it can be tested: the algorithm setting table of
    ``algorithm``, with the default values, or with the selected hyperparameters if ``algorithm`` is the one selected
    by the last Auto training (``auto_result``, see `run_auto_selection`).

    :param param_info: a function returning the parameter info of an algorithm (default:
        `ForecastModel.get_parameter_info`).
    """
    if algorithm is None:
        return create_param_table()
    params = (param_info or ForecastModel.get_parameter_info)(algorithm)
    if auto_result and auto_result.get("algorithm") == algorithm:
        # Same format as the values confirmed in the recommendation popup
        params = {name: {"default": value["default"]} for name, value in params.items()}
        for name, value in auto_result["params"].items():
            params[name] = {"default": format_value(value)}
    return create_param_table(params)


@callback(
    Output("forecasting-param-table", "children"),
    Input("forecasting-select-algorithm", "value"),
    State("forecasting-auto-result", "data"),
    prevent_initial_call=True,
)
def select_algorithm(algorithm, auto_result):
    param_table = create_param_table()
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "forecasting-select-algorithm":
        param_table = handle_select_algorithm(algorithm, auto_result)
    return param_table


def analysis_mode(feature_cols, exog_cols) -> dict:
    """
    The analysis type of the Auto mode: univariate if no other feature and no exogenous variable is selected,
    multivariate otherwise.

    :return: ``{"kind": "univariate" | "multivariate", "n_features": ..., "n_exog": ...}``
    """
    n_features, n_exog = len(feature_cols or []), len(exog_cols or [])
    kind = "univariate" if n_features == 0 and n_exog == 0 else "multivariate"
    return {"kind": kind, "n_features": n_features, "n_exog": n_exog}


def auto_mode_label(auto, feature_cols, exog_cols) -> str:
    """The analysis type shown under the Auto checkbox if Auto is checked, empty otherwise."""
    return analysis_mode_text(analysis_mode(feature_cols, exog_cols)) if auto else ""


def auto_mode_content(auto, feature_cols, exog_cols) -> list:
    """The line under the Auto checkbox: the analysis type and what the Auto mode does for it, empty without Auto."""
    if not auto:
        return []
    mode = analysis_mode(feature_cols, exog_cols)
    return [html.B(f"Auto 모드: {analysis_mode_text(mode)}"), html.Br(), html.Span(analysis_mode_description(mode))]


@callback(
    Output("forecasting-auto-mode", "children"),
    Input("forecasting-auto-train", "value"),
    Input("forecasting-select-features", "value"),
    Input("forecasting-select-exog", "value"),
)
def update_auto_mode(auto, feature_cols, exog_cols):
    return auto_mode_content(auto, feature_cols, exog_cols)


def algorithm_dropdown_disabled(auto) -> bool:
    """The algorithm dropdown is not used in the Auto mode, so it is disabled while Auto is checked."""
    return bool(auto)


@callback(Output("forecasting-select-algorithm", "disabled"), Input("forecasting-auto-train", "value"))
def toggle_auto_train(auto):
    return algorithm_dropdown_disabled(auto)


def handle_auto_result(result):
    """
    The logic of `show_auto_result`, separated from Dash so that it can be tested.

    :param result: the model selection of the last training (see `run_auto_selection`), ``None`` after a manual one.
    :return: ``(algorithm dropdown value, algorithm dropdown options, Model Selection card content)``. After an Auto
        training, the dropdown shows the selected algorithm, and changing its value makes `select_algorithm` fill the
        setting table with the selected hyperparameters.
    """
    content = create_model_selection_content(result)
    if not result:
        return dash.no_update, dash.no_update, content
    # Set the options too, since they are only loaded when the dropdown is clicked
    options = [{"label": s, "value": s} for s in ForecastModel.get_available_algorithms()]
    return result["algorithm"], options, content


@callback(
    Output("forecasting-select-algorithm", "value"),
    Output("forecasting-select-algorithm", "options", allow_duplicate=True),
    Output("forecasting-model-selection", "children"),
    Input("forecasting-auto-result", "data"),
    prevent_initial_call=True,
)
def show_auto_result(result):
    return handle_auto_result(result)


def _load_train_test(filename, train_percentage, test_filename, file_mode):
    df = ForecastModel().load_data(os.path.join(file_manager.data_directory, filename))
    assert len(df) > 20, f"The input time series length ({len(df)}) is too small."
    if file_mode == "single":
        n = int(int(train_percentage) * len(df) / 100)
        train_df, test_df = df.iloc[:n], df.iloc[n:]
    else:
        assert test_filename, "The test file is empty!"
        test_df = ForecastModel().load_data(os.path.join(file_manager.data_directory, test_filename))
        train_df = df
    return train_df, test_df


def _current_params(table):
    return {p["Parameter"]: p["Value"] for p in table["props"]["data"] if p["Parameter"]}


def _create_recommendation_content(rec, table):
    """The table of the recommended settings (`Recommendation`), with the current values and the reasoning."""
    current = _current_params(table)
    granularity = rec.granularity
    header = html.Tr([html.Th(c) for c in ["Parameter", "Current", "Recommended", "Duration", "Basis"]])
    body = [
        html.Tr(
            [
                html.Td(p.name),
                html.Td(current.get(p.name, "")),
                html.Td(format_value(p.value)),
                html.Td(
                    ForecastModel.steps_to_duration(p.value, granularity)
                    if p.is_steps and isinstance(p.value, int)
                    else ""
                ),
                html.Td(p.basis),
            ]
        )
        for p in rec.params
    ]
    content = [
        html.P(
            f"Computed from the training data: {rec.n_points} points after resampling at a granularity of "
            f"{ForecastModel.steps_to_duration(1, granularity)}."
        ),
        html.Table([header] + body, className="table table-sm"),
    ]
    content += [html.P(note) for note in rec.notes]
    if any(p.name == "max_forecast_steps" for p in rec.params):
        content.append(html.P("The test metrics are computed on the first max_forecast_steps points of the test data."))
    return content


def handle_train_settings(
    prop_id, algorithm, table, load_train_df, target_col, feature_cols, specs, values, recommend=None, auto=False
):
    """
    The logic of `confirm_train_settings`, separated from Dash so that it can be tested.

    :param prop_id: the id of the button that was clicked.
    :param load_train_df: a function returning the training data (only called if a recommendation is needed).
    :param specs: the specs of the recommended parameters shown in the popup.
    :param values: the values of the popup's input fields, in the same order as ``specs``.
    :param recommend: the function computing the `Recommendation` (default: `ForecastModel.recommend_params`).
    :param auto: whether the Auto checkbox is checked. Then Train requests an Auto training (``"auto": True``), which
        selects the algorithm and its hyperparameters itself, so the popup is not opened.
    :return: ``(popup is open, popup content, popup inputs, specs, error message, train request, param table)``,
        with `dash.no_update` for the outputs that do not change. The train request goes through
        `handle_train_request`, which starts training or first asks for approval if it would take long.
    """
    no_update = dash.no_update
    recommend = recommend or ForecastModel.recommend_params

    if prop_id == "forecasting-train-btn":
        if auto:
            trigger = {"time": time.time(), "overrides": {}, "auto": True}
            return False, no_update, no_update, no_update, "", trigger, no_update
        if algorithm in ForecastModel.tuned_algorithms:
            try:
                rec = recommend(algorithm, load_train_df(), target_col, feature_cols)
                content = _create_recommendation_content(rec, table)
                inputs = create_confirm_inputs(rec.params)
                specs = [p.spec() for p in rec.params]
                return True, content, inputs, specs, "", no_update, no_update
            except Exception:
                # Start training anyway, so that any data error is reported in the exception modal as usual.
                logger.warning(f"Could not recommend {algorithm} settings:\n{traceback.format_exc()}")
        trigger = {"time": time.time(), "overrides": {}}
        return False, no_update, no_update, no_update, "", trigger, no_update

    if prop_id == "forecasting-param-confirm-btn":
        overrides, errors = parse_confirmed_values(specs or [], values or [])
        if errors:
            return True, no_update, no_update, no_update, " ".join(errors), no_update, no_update
        # Show the confirmed values in the algorithm setting table as well
        params = {name: {"default": value} for name, value in _current_params(table).items()}
        for name, value in overrides.items():
            params[name] = {"default": format_value(value)}
        trigger = {"time": time.time(), "overrides": overrides}
        return False, no_update, no_update, no_update, "", trigger, create_param_table(params)

    return False, no_update, no_update, no_update, "", no_update, no_update


@callback(
    Output("forecasting-param-confirm-modal", "is_open"),
    Output("forecasting-param-confirm-content", "children"),
    Output("forecasting-confirm-inputs", "children"),
    Output("forecasting-confirm-specs", "data"),
    Output("forecasting-param-confirm-error", "children"),
    Output("forecasting-train-request", "data"),
    Output("forecasting-param-table", "children", allow_duplicate=True),
    [
        Input("forecasting-train-btn", "n_clicks"),
        Input("forecasting-param-confirm-btn", "n_clicks"),
        Input("forecasting-param-cancel-btn", "n_clicks"),
    ],
    [
        State("forecasting-select-file", "value"),
        State("forecasting-select-target", "value"),
        State("forecasting-select-features", "value"),
        State("forecasting-select-algorithm", "value"),
        State("forecasting-param-table", "children"),
        State("forecasting-training-slider", "value"),
        State("forecasting-select-test-file", "value"),
        State("forecasting-file-radio", "value"),
        State("forecasting-confirm-specs", "data"),
        State({"type": "forecasting-confirm-param", "name": ALL}, "value"),
        State({"type": "forecasting-confirm-param", "name": ALL}, "id"),
        State("forecasting-auto-train", "value"),
    ],
    prevent_initial_call=True,
)
def confirm_train_settings(
    train_clicks,
    confirm_clicks,
    cancel_clicks,
    filename,
    target_col,
    feature_cols,
    algorithm,
    table,
    train_percentage,
    test_filename,
    file_mode,
    specs,
    values,
    ids,
    auto,
):
    """
    Runs before training. For the algorithms in `ForecastModel.tuned_algorithms`, opens a popup with the recommended
    settings and only starts training once they are confirmed. Other algorithms, and the Auto mode, start right away.
    """
    # Match the input values to the specs by parameter name, since Dash does not guarantee their order
    by_name = {i["name"]: v for i, v in zip(ids or [], values or [])}
    values = [by_name.get(spec["name"]) for spec in specs or []]
    return handle_train_settings(
        prop_id=dash.callback_context.triggered_id,
        algorithm=algorithm,
        table=table,
        load_train_df=lambda: _load_train_test(filename, train_percentage, test_filename, file_mode)[0],
        target_col=target_col,
        feature_cols=feature_cols or [],
        specs=specs,
        values=values,
        auto=bool(auto),
    )


def _long_train_content(estimate):
    """The content of the popup asking whether to start a long training (`TrainTimeEstimate`)."""
    return [
        html.P(
            f"Training is estimated to take about {format_duration(estimate.seconds)}, which is longer than "
            f"{format_duration(ForecastModel.train_confirm_seconds)}."
        ),
        html.P(f"Main cost: {estimate.basis}"),
        html.P(
            f"The estimate accounts for the speed of this machine ({estimate.speed_factor:.1f}x the time of the "
            f"machine it was calibrated on) and can be off by a factor of 2 or so."
        ),
        html.P("Start training anyway? To train faster, cancel and change the settings (e.g. as suggested above)."),
    ]


def handle_train_request(prop_id, request, pending, estimate):
    """
    The logic of `gate_long_training`, separated from Dash so that it can be tested. A train request starts training
    right away, unless its estimated training time exceeds `ForecastModel.train_confirm_seconds`: then a popup asks
    for approval first, and training only starts if the user approves.

    :param prop_id: the id of the component that triggered the callback.
    :param request: the train request (``{"time": ..., "overrides": {...}}``, plus ``"auto": True`` in the Auto mode,
        which starts right away since the model selection already leaves out the candidates that would take long).
    :param pending: the request waiting for approval, if any.
    :param estimate: a function of the request returning its `TrainTimeEstimate` (or ``None`` if unknown).
    :return: ``(popup is open, popup content, pending request, train trigger)``, with `dash.no_update` for the
        outputs that do not change.
    """
    no_update = dash.no_update
    if prop_id == "forecasting-train-request" and request:
        if request.get("auto"):
            return False, no_update, None, request
        try:
            est = estimate(request)
        except Exception:
            # The estimate is a safeguard: if it fails, train as before (any data error is reported by training)
            logger.warning(f"Could not estimate the training time:\n{traceback.format_exc()}")
            est = None
        if est is not None and est.seconds > ForecastModel.train_confirm_seconds:
            return True, _long_train_content(est), request, no_update
        return False, no_update, None, request
    if prop_id == "forecasting-long-train-proceed-btn" and pending:
        return False, no_update, None, {**pending, "time": time.time()}
    if prop_id in ("forecasting-long-train-proceed-btn", "forecasting-long-train-cancel-btn"):
        return False, no_update, None, no_update
    return no_update, no_update, no_update, no_update


@callback(
    Output("forecasting-long-train-modal", "is_open"),
    Output("forecasting-long-train-content", "children"),
    Output("forecasting-pending-train", "data"),
    Output("forecasting-train-trigger", "data"),
    [
        Input("forecasting-train-request", "data"),
        Input("forecasting-long-train-proceed-btn", "n_clicks"),
        Input("forecasting-long-train-cancel-btn", "n_clicks"),
    ],
    [
        State("forecasting-pending-train", "data"),
        State("forecasting-select-file", "value"),
        State("forecasting-select-target", "value"),
        State("forecasting-select-features", "value"),
        State("forecasting-select-exog", "value"),
        State("forecasting-select-algorithm", "value"),
        State("forecasting-param-table", "children"),
        State("forecasting-training-slider", "value"),
        State("forecasting-select-test-file", "value"),
        State("forecasting-file-radio", "value"),
    ],
    prevent_initial_call=True,
)
def gate_long_training(
    request,
    proceed_clicks,
    cancel_clicks,
    pending,
    filename,
    target_col,
    feature_cols,
    exog_cols,
    algorithm,
    table,
    train_percentage,
    test_filename,
    file_mode,
):
    """Starts training for a train request, after asking for approval if the training is estimated to take long."""

    def estimate(req):
        train_df = _load_train_test(filename, train_percentage, test_filename, file_mode)[0]
        params = ForecastModel.parse_parameters(
            param_info=ForecastModel.get_parameter_info(algorithm), params=_current_params(table)
        )
        params.update(req.get("overrides", {}))
        return ForecastModel.estimate_train_time(
            algorithm, train_df, target_col, feature_cols or [], exog_cols or [], params
        )

    return handle_train_request(dash.callback_context.triggered_id, request, pending, estimate)


def _column(df, column):
    # Like ForecastModel, columns of files without a header are integers but selected as strings
    return column if column in df else int(column)


def auto_horizon(train_df, test_df, target_col) -> int:
    """
    The number of steps to forecast in the Auto mode: the number of test points after resampling them at the
    granularity of the training data, as the trained model does. For irregularly sampled data, this differs from the
    number of test rows.
    """
    target_col = _column(train_df, target_col)
    resample = TemporalResample()
    resample.train(TimeSeries.from_pd(train_df.loc[:, [target_col]]))
    return len(resample(TimeSeries.from_pd(test_df.loc[:, [target_col]])).to_pd())


def _finite_or_none(value):
    # Stores are serialized as JSON, which has no infinity
    return float(value) if value is not None and np.isfinite(value) else None


def run_auto_selection(train_df, test_df, target_col, feature_cols, exog_cols, recommend=None) -> dict:
    """
    Selects the algorithm and its hyperparameters for the Auto mode from the training data, forecasting as many steps
    as the test data has (see `auto_horizon`). For a univariate analysis, only the target variable is given to the
    model selection; for a multivariate one, the selected features and exogenous variables are given too.

    :param recommend: the model selection (default: `ForecastModel.recommend_model`).
    :return: the content of the ``forecasting-auto-result`` Store: ``algorithm``, ``params``, ``candidates`` (each
        with ``algorithm``, ``params``, ``valid_mae``, ``estimated_seconds``, ``eval_seconds``, ``status``),
        ``n_valid``, ``n_points``, ``horizon``, ``notes`` and the analysis type ``mode`` (see `analysis_mode`).
    :raises RuntimeError: if no algorithm could be selected, with the status of each candidate.
    """
    recommend = recommend or ForecastModel.recommend_model
    horizon = auto_horizon(train_df, test_df, target_col)
    mode = analysis_mode(feature_cols, exog_cols)
    if mode["kind"] == "univariate":
        rec = recommend(train_df, target_col, horizon)
    else:
        rec = recommend(
            train_df, target_col, horizon, feature_columns=list(feature_cols or []), exog_columns=list(exog_cols or [])
        )
    if rec.algorithm is None:
        reasons = "\n".join(f"- {c.algorithm}: {c.status}" for c in rec.candidates)
        raise RuntimeError(f"No forecasting algorithm could be selected. The candidates:\n{reasons}")
    return {
        "algorithm": rec.algorithm,
        "params": dict(rec.params),
        "candidates": [
            {
                "algorithm": c.algorithm,
                "params": dict(c.params),
                "valid_mae": _finite_or_none(c.valid_mae),
                "estimated_seconds": _finite_or_none(c.estimated_seconds),
                "eval_seconds": _finite_or_none(c.eval_seconds),
                "status": c.status,
            }
            for c in rec.candidates
        ],
        "n_valid": int(rec.n_valid),
        "n_points": int(rec.n_points),
        "horizon": int(horizon),
        "notes": list(rec.notes),
        "mode": mode,
    }


@callback(
    Output("forecasting-training-metrics", "children"),
    Output("forecasting-test-metrics", "children"),
    Output("forecasting-plots", "children"),
    Output("forecasting-exception-modal", "is_open"),
    Output("forecasting-exception-modal-content", "children"),
    Output("forecasting-auto-result", "data"),
    [Input("forecasting-train-trigger", "data"), Input("forecasting-exception-modal-close", "n_clicks")],
    [
        State("forecasting-select-file", "value"),
        State("forecasting-select-target", "value"),
        State("forecasting-select-features", "value"),
        State("forecasting-select-exog", "value"),
        State("forecasting-select-algorithm", "value"),
        State("forecasting-param-table", "children"),
        State("forecasting-training-slider", "value"),
        State("forecasting-select-test-file", "value"),
        State("forecasting-file-radio", "value"),
    ],
    running=[
        (Output("forecasting-train-btn", "disabled"), True, False),
        (Output("forecasting-cancel-btn", "disabled"), False, True),
    ],
    cancel=[Input("forecasting-cancel-btn", "n_clicks")],
    background=True,
    manager=file_manager.get_long_callback_manager(),
    progress=[Output("forecasting-progressbar", "value"), Output("forecasting-progressbar", "max")],
)
def click_train_test(
    set_progress,
    trigger,
    modal_close,
    filename,
    target_col,
    feature_cols,
    exog_cols,
    algorithm,
    table,
    train_percentage,
    test_filename,
    file_mode,
):
    ctx = dash.callback_context
    modal_is_open = False
    modal_content = ""
    train_metric_table = create_metric_table()
    test_metric_table = create_metric_table()
    figure = create_empty_figure()
    # The model selection of an Auto training, None for a manual training
    auto_result = None
    set_progress(("0", "10"))

    try:
        if ctx.triggered and trigger:
            prop_id = ctx.triggered_id
            if prop_id == "forecasting-train-trigger":
                assert filename, "The training data file is empty!"
                assert target_col, "Please select a target variable/metric for forecasting."
                auto = bool(trigger.get("auto"))
                assert auto or algorithm, "Please select a forecasting algorithm."
                feature_cols = feature_cols or []
                exog_cols = exog_cols or []
                train_df, test_df = _load_train_test(filename, train_percentage, test_filename, file_mode)

                if auto:
                    # Select the algorithm and its hyperparameters (progress 0 -> 2), then train with them (2 -> 10)
                    set_progress(("1", "10"))
                    auto_result = run_auto_selection(train_df, test_df, target_col, feature_cols, exog_cols)
                    set_progress(("2", "10"))
                    algorithm = auto_result["algorithm"]
                    params = dict(auto_result["params"])
                else:
                    params = ForecastModel.parse_parameters(
                        param_info=ForecastModel.get_parameter_info(algorithm),
                        params={p["Parameter"]: p["Value"] for p in table["props"]["data"] if p["Parameter"]},
                    )
                    params.update(trigger.get("overrides", {}))
                model, train_metrics, test_metrics, figure = ForecastModel().train(
                    algorithm, train_df, test_df, target_col, feature_cols, exog_cols, params, set_progress
                )
                ForecastModel.save_model(file_manager.model_directory, model, algorithm)
                train_metric_table = create_metric_table(train_metrics)
                test_metric_table = create_metric_table(test_metrics)
                figure = dcc.Graph(figure=figure)

    except Exception:
        error = traceback.format_exc()
        modal_is_open = True
        modal_content = error
        logger.error(error)
        auto_result = None

    return train_metric_table, test_metric_table, figure, modal_is_open, modal_content, auto_result


@callback(
    Output("forecasting-slider-collapse", "is_open"),
    Output("forecasting-test-file-collapse", "is_open"),
    Input("forecasting-file-radio", "value"),
)
def set_file_mode(value):
    if value == "single":
        return True, False
    else:
        return False, True

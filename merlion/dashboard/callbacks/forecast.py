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
from dash import ALL, Input, Output, State, dcc, html, callback
from merlion.dashboard.utils.file_manager import FileManager
from merlion.dashboard.models.forecast import ForecastModel
from merlion.dashboard.models.recommend import format_value, parse_confirmed_values
from merlion.dashboard.pages.forecast import create_confirm_inputs
from merlion.dashboard.pages.utils import create_param_table, create_metric_table, create_empty_figure

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


@callback(
    Output("forecasting-param-table", "children"),
    Input("forecasting-select-algorithm", "value"),
    prevent_initial_call=True,
)
def select_algorithm(algorithm):
    param_table = create_param_table()
    ctx = dash.callback_context
    prop_id = ctx.triggered_id
    if prop_id == "forecasting-select-algorithm":
        param_info = ForecastModel.get_parameter_info(algorithm)
        param_table = create_param_table(param_info)
    return param_table


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
    prop_id, algorithm, table, load_train_df, target_col, feature_cols, specs, values, recommend=None
):
    """
    The logic of `confirm_train_settings`, separated from Dash so that it can be tested.

    :param prop_id: the id of the button that was clicked.
    :param load_train_df: a function returning the training data (only called if a recommendation is needed).
    :param specs: the specs of the recommended parameters shown in the popup.
    :param values: the values of the popup's input fields, in the same order as ``specs``.
    :param recommend: the function computing the `Recommendation` (default: `ForecastModel.recommend_params`).
    :return: ``(popup is open, popup content, popup inputs, specs, error message, train trigger, param table)``,
        with `dash.no_update` for the outputs that do not change.
    """
    no_update = dash.no_update
    recommend = recommend or ForecastModel.recommend_params

    if prop_id == "forecasting-train-btn":
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
    Output("forecasting-train-trigger", "data"),
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
):
    """
    Runs before training. For the algorithms in `ForecastModel.tuned_algorithms`, opens a popup with the recommended
    settings and only starts training once they are confirmed. Other algorithms start right away.
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
    )


@callback(
    Output("forecasting-training-metrics", "children"),
    Output("forecasting-test-metrics", "children"),
    Output("forecasting-plots", "children"),
    Output("forecasting-exception-modal", "is_open"),
    Output("forecasting-exception-modal-content", "children"),
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
    set_progress(("0", "10"))

    try:
        if ctx.triggered and trigger:
            prop_id = ctx.triggered_id
            if prop_id == "forecasting-train-trigger":
                assert filename, "The training data file is empty!"
                assert target_col, "Please select a target variable/metric for forecasting."
                assert algorithm, "Please select a forecasting algorithm."
                feature_cols = feature_cols or []
                exog_cols = exog_cols or []
                train_df, test_df = _load_train_test(filename, train_percentage, test_filename, file_mode)

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

    return train_metric_table, test_metric_table, figure, modal_is_open, modal_content


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

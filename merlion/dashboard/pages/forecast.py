#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from merlion.dashboard.pages.utils import create_modal, create_param_table, create_metric_table, create_empty_figure


def create_control_panel() -> html.Div:
    return html.Div(
        id="control-card",
        children=[
            html.Br(),
            html.P("Select Training Data File"),
            html.Div(
                id="forecasting-select-file-parent",
                children=[
                    dbc.RadioItems(
                        id="forecasting-file-radio",
                        options=[
                            {"label": "Single data file", "value": "single"},
                            {"label": "Separate train/test files", "value": "separate"},
                        ],
                        value="single",
                        inline=True,
                    ),
                    dcc.Dropdown(id="forecasting-select-file", options=[], style={"width": "100%"}),
                ],
            ),
            dbc.Collapse(
                html.Div(
                    id="control-card",
                    children=[
                        html.Br(),
                        html.P("Training Data Percentage"),
                        dcc.Slider(
                            id="forecasting-training-slider",
                            min=5,
                            max=95,
                            step=1,
                            marks={t * 10: str(t * 10) for t in range(1, 10)},
                            value=80,
                        ),
                    ],
                ),
                id="forecasting-slider-collapse",
                is_open=True,
            ),
            dbc.Collapse(
                html.Div(
                    id="control-card",
                    children=[
                        html.Br(),
                        html.P("Select Test Data File"),
                        html.Div(
                            id="forecasting-select-test-file-parent",
                            children=[
                                dcc.Dropdown(id="forecasting-select-test-file", options=[], style={"width": "100%"})
                            ],
                        ),
                    ],
                ),
                id="forecasting-test-file-collapse",
                is_open=False,
            ),
            html.Br(),
            html.P("Select Target Column"),
            html.Div(
                id="forecasting-select-target-parent",
                children=[dcc.Dropdown(id="forecasting-select-target", options=[], style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Select Other Features (Optional)"),
            html.Div(
                id="forecasting-select-features-parent",
                children=[
                    dcc.Dropdown(id="forecasting-select-features", options=[], multi=True, style={"width": "100%"})
                ],
            ),
            html.Br(),
            html.P("Select Exogenous Variables (Optional; Known A Priori)"),
            html.Div(
                id="forecasting-select-exog-parent",
                children=[dcc.Dropdown(id="forecasting-select-exog", options=[], multi=True, style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Select Forecasting Algorithm"),
            html.Div(
                id="forecasting-select-algorithm-parent",
                children=[dcc.Dropdown(id="forecasting-select-algorithm", options=[], style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Algorithm Setting"),
            html.Div(id="forecasting-param-table", children=[create_param_table()]),
            html.Progress(id="forecasting-progressbar", style={"width": "100%", "color": "#1AB9FF"}),
            html.Br(),
            html.Div(
                children=[
                    html.Button(id="forecasting-train-btn", children="Train", n_clicks=0),
                    html.Button(id="forecasting-cancel-btn", children="Cancel", style={"margin-left": "15px"}),
                    # Auto: select the algorithm and its hyperparameters from the training data, then train
                    dbc.Checkbox(
                        id="forecasting-auto-train",
                        label="Auto",
                        value=False,
                        style={"display": "inline-block", "margin-left": "15px"},
                    ),
                ],
                style={"textAlign": "center"},
            ),
            # The analysis type of the Auto mode (univariate / multivariate) and what it means, empty if Auto is
            # unchecked. On its own line: next to the buttons, it wrapped under them in the narrow control panel.
            html.Div(id="forecasting-auto-mode", style={"margin-top": "8px", "font-size": "13px"}),
            html.Br(),
            create_modal(
                modal_id="forecasting-exception-modal",
                header="An Exception Occurred",
                content="An exception occurred. Please click OK to continue.",
                content_id="forecasting-exception-modal-content",
                button_id="forecasting-exception-modal-close",
            ),
            create_param_confirm_modal(),
            create_long_train_modal(),
            # Requests training: set right away for most algorithms, or once the recommended settings are confirmed.
            dcc.Store(id="forecasting-train-request"),
            # A request whose estimated training time is long, waiting for the user's approval
            dcc.Store(id="forecasting-pending-train"),
            # Starts training: set from a request right away, or once a long training is approved.
            dcc.Store(id="forecasting-train-trigger"),
            # The model selection of the last Auto training (None after a manual training)
            dcc.Store(id="forecasting-auto-result"),
        ],
    )


def create_param_confirm_modal() -> html.Div:
    """
    Popup showing the recommended settings of the selected algorithm, which must be confirmed before training starts.
    The input fields depend on the algorithm: they are created by `create_confirm_inputs` when the popup opens.
    """
    return html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Recommended Settings")),
                    dbc.ModalBody(
                        [
                            html.Div(id="forecasting-param-confirm-content"),
                            html.Br(),
                            html.P("Settings used for training (edit if needed):"),
                            html.Div(id="forecasting-confirm-inputs"),
                            # The specs of the recommended parameters, used to validate the confirmed values
                            dcc.Store(id="forecasting-confirm-specs"),
                            html.Div(
                                id="forecasting-param-confirm-error", style={"color": "red", "margin-top": "10px"}
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Confirm", id="forecasting-param-confirm-btn", n_clicks=0),
                            dbc.Button("Cancel", id="forecasting-param-cancel-btn", n_clicks=0, color="secondary"),
                        ]
                    ),
                ],
                id="forecasting-param-confirm-modal",
                is_open=False,
                backdrop="static",
                size="lg",
            )
        ]
    )


def create_long_train_modal() -> html.Div:
    """
    Popup asking whether to start a training whose estimated time exceeds `ForecastModel.train_confirm_seconds`.
    Training only starts if the user approves.
    """
    return html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Long Training")),
                    dbc.ModalBody(html.Div(id="forecasting-long-train-content")),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Train anyway", id="forecasting-long-train-proceed-btn", n_clicks=0),
                            dbc.Button("Cancel", id="forecasting-long-train-cancel-btn", n_clicks=0, color="secondary"),
                        ]
                    ),
                ],
                id="forecasting-long-train-modal",
                is_open=False,
                backdrop="static",
            )
        ]
    )


def create_right_column() -> html.Div:
    return html.Div(
        id="right-column-data",
        children=[
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Model Selection"),
                    html.Hr(),
                    html.Div(id="forecasting-model-selection", children=create_model_selection_content(None)),
                ],
            ),
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Forecasting Results"),
                    html.Hr(),
                    html.Div(id="forecasting-plots", children=[create_empty_figure()]),
                ],
            ),
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Testing Metrics"),
                    html.Hr(),
                    html.Div(id="forecasting-test-metrics", children=[create_metric_table()]),
                ],
            ),
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Training Metrics"),
                    html.Hr(),
                    html.Div(id="forecasting-training-metrics", children=[create_metric_table()]),
                ],
            ),
        ],
    )


def create_forecasting_layout() -> html.Div:
    return html.Div(
        id="forecasting_views",
        children=[
            # Left column
            html.Div(id="left-column-data", className="three columns", children=[create_control_panel()]),
            # Right column
            html.Div(className="nine columns", children=create_right_column()),
        ],
    )


def create_confirm_input(spec: dict, value) -> html.Div:
    """
    The input field of one recommended parameter in the confirmation popup, pre-filled with ``value``: a number
    input for integers, a dropdown for choices, and a text input for tuples and strings.

    :param spec: the parameter's spec, see `merlion.dashboard.models.recommend.ParamRecommendation.spec`.
    """
    component_id = {"type": "forecasting-confirm-param", "name": spec["name"]}
    style = {"width": "100%"}
    if spec["kind"] == "int":
        component = dcc.Input(id=component_id, type="number", min=1, step=1, value=value, style=style)
    elif spec["kind"] == "choice":
        options = [{"label": c, "value": c} for c in spec["choices"]]
        component = dcc.Dropdown(id=component_id, options=options, value=str(value), clearable=False, style=style)
    else:
        component = dcc.Input(id=component_id, type="text", value=value, style=style)
    return html.Div([html.Label(spec["name"]), component], style={"margin-top": "10px"})


def create_confirm_inputs(params) -> list:
    """The input fields of all the recommended parameters (`ParamRecommendation`s), in order."""
    from merlion.dashboard.models.recommend import format_value

    inputs = []
    for p in params:
        if p.kind == "int_tuple" or p.kind == "str":
            value = format_value(p.value)
        else:
            value = p.value
        inputs.append(create_confirm_input(p.spec(), value))
    return inputs


def analysis_mode_text(mode: dict) -> str:
    """
    The analysis type as shown next to the Auto checkbox, e.g. "단변량" or "다변량 (특징 2, 외생 1)".

    :param mode: the analysis type, see `merlion.dashboard.callbacks.forecast.analysis_mode`.
    """
    if mode["kind"] == "univariate":
        return "단변량"
    return f"다변량 (특징 {mode['n_features']}, 외생 {mode['n_exog']})"


def analysis_mode_description(mode: dict) -> str:
    """What the Auto mode does for the analysis type, shown under the analysis type."""
    if mode["kind"] == "univariate":
        return "대상 변수만으로 알고리즘과 하이퍼파라미터를 고릅니다."
    return (
        "선택한 특징 변수와 외생 변수를 함께 써서 알고리즘과 하이퍼파라미터를 고릅니다 "
        "(특징 변수가 있으면 VectorAR도 후보에 포함)."
    )


def recommendation_basis_text(mode: dict) -> str:
    """The analysis type the hyperparameters were recommended for, e.g. "다변량 기준 추천 (특징 2, 외생 1)"."""
    if mode["kind"] == "univariate":
        return "단변량 기준 추천"
    return f"다변량 기준 추천 (특징 {mode['n_features']}, 외생 {mode['n_exog']})"


def _format_seconds(seconds) -> str:
    return "-" if seconds is None else f"{seconds:.1f} s"


def create_model_selection_content(result) -> list:
    """
    The content of the Model Selection card: a hint if there is no Auto training result, otherwise the candidates
    ranked by validation error, the validation part and the notes of the model selection.

    :param result: the content of the ``forecasting-auto-result`` Store, see
        `merlion.dashboard.callbacks.forecast.run_auto_selection`.
    """
    if not result:
        return [html.P("Auto 모드로 학습하면 후보 비교 결과가 표시됩니다.", style={"color": "grey"})]

    header = html.Tr([html.Th(c) for c in ["Rank", "Algorithm", "Valid MAE", "Status", "Est. train time", "Eval time"]])
    body, rank = [], 0
    for cand in result["candidates"]:
        # Only the evaluated candidates are ranked; they come first, sorted by validation error
        mae = cand["valid_mae"]
        rank = rank + 1 if mae is not None else rank
        body.append(
            html.Tr(
                [
                    html.Td(str(rank) if mae is not None else "-"),
                    html.Td(cand["algorithm"]),
                    html.Td(f"{mae:.5g}" if mae is not None else "-"),
                    html.Td(cand["status"]),
                    html.Td(_format_seconds(cand["estimated_seconds"])),
                    html.Td(_format_seconds(cand["eval_seconds"])),
                ]
            )
        )
    steps = result["params"].get("max_forecast_steps", result["horizon"])
    content = [
        html.P(f"Selected: {result['algorithm']} ({recommendation_basis_text(result['mode'])})"),
        html.Table([header] + body, className="table table-sm"),
        html.P(
            f"Validation: the last {result['n_valid']} of {result['n_points']} training points (after resampling), "
            f"forecast {steps} steps at a time. The test data has {result['horizon']} steps."
        ),
    ]
    content += [html.P(note) for note in result["notes"]]
    return content

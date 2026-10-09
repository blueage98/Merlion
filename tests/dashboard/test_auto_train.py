#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Tests for the Auto mode of the Forecasting tab, which selects the algorithm and its hyperparameters from the training
data before training: the layout, the request flow, the model selection and how its result is shown (without a Dash
server; the model selection itself is replaced by a fake one).
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional

import dash
import numpy as np
import pandas as pd
import pytest

from merlion.dashboard.callbacks.forecast import (
    algorithm_dropdown_disabled,
    analysis_mode,
    auto_horizon,
    auto_mode_content,
    auto_mode_label,
    handle_auto_result,
    handle_select_algorithm,
    handle_train_request,
    handle_train_settings,
    run_auto_selection,
)
from merlion.dashboard.pages.forecast import create_forecasting_layout, create_model_selection_content
from merlion.dashboard.pages.utils import create_param_table


@dataclass
class FakeCandidate:
    algorithm: str
    params: dict
    valid_mae: float = np.inf
    estimated_seconds: Optional[float] = None
    eval_seconds: float = 0.0
    status: str = "ok"


@dataclass
class FakeRecommendation:
    algorithm: Optional[str]
    params: dict
    candidates: List[FakeCandidate]
    n_valid: int = 30
    n_points: int = 150
    notes: List[str] = field(default_factory=list)


CANDIDATES = [
    FakeCandidate("ExtraTreesForecaster", {"maxlags": 24, "max_forecast_steps": 20}, 0.5, 3.0, 1.2),
    FakeCandidate("Arima", {"order": [4, 1, 2], "max_forecast_steps": 20}, 0.8, 1.0, 0.4),
    FakeCandidate(
        "ETS", {"seasonal_periods": 493, "max_forecast_steps": 20}, estimated_seconds=1500.0, status="left out: slow"
    ),
]
REC = FakeRecommendation("ExtraTreesForecaster", CANDIDATES[0].params, CANDIDATES, notes=["Prophet: a note"])


def _recorder(rec=REC):
    calls = []

    def recommend(*args, **kwargs):
        calls.append((args, kwargs))
        return rec

    recommend.calls = calls
    return recommend


def _hourly(n, start="2023-01-01", columns=("value",)):
    idx = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({c: np.arange(n, dtype=float) for c in columns}, index=idx)


def _texts(component):
    """All the strings in a Dash component tree."""
    if isinstance(component, str):
        return [component]
    if isinstance(component, (list, tuple)):
        return [t for c in component for t in _texts(c)]
    return _texts(getattr(component, "children", None) or [])


# --- layout ------------------------------------------------------------------------------------------------------------


def test_layout_has_the_auto_components():
    layout = str(create_forecasting_layout())
    for component_id in [
        "forecasting-auto-train",
        "forecasting-auto-mode",
        "forecasting-auto-result",
        "forecasting-model-selection",
    ]:
        assert component_id in layout


def test_auto_is_unchecked_at_first():
    def find(component, component_id):
        if getattr(component, "id", None) == component_id:
            return component
        children = getattr(component, "children", None)
        for child in children if isinstance(children, list) else [children]:
            if child is not None and not isinstance(child, str):
                found = find(child, component_id)
                if found is not None:
                    return found

    checkbox = find(create_forecasting_layout(), "forecasting-auto-train")
    assert checkbox.value is False and checkbox.label == "Auto"


def test_empty_model_selection_shows_a_hint():
    content = create_model_selection_content(None)
    assert len(content) == 1
    assert "Auto 모드로 학습하면 후보 비교 결과가 표시됩니다" in _texts(content)[0]


def _result(feature_cols=(), exog_cols=()):
    train_df, test_df = _hourly(150), _hourly(20, start="2023-01-07 06:00")
    return run_auto_selection(train_df, test_df, "value", list(feature_cols), list(exog_cols), recommend=_recorder())


def test_model_selection_shows_the_ranked_candidates():
    content = create_model_selection_content(_result())
    table = content[1]
    rows = table.children[1:]
    assert len(rows) == 3
    cells = [[td.children for td in row.children] for row in rows]
    assert cells[0][:3] == ["1", "ExtraTreesForecaster", "0.5"]
    assert cells[1][:2] == ["2", "Arima"]
    assert cells[2][:4] == ["-", "ETS", "-", "left out: slow"]
    assert cells[2][4] == "1500.0 s"
    text = " ".join(_texts(content))
    assert "단변량 기준 추천" in text
    assert "last 30 of 150" in text and "20 steps" in text
    assert "Prophet: a note" in text


def test_model_selection_shows_the_multivariate_basis():
    text = " ".join(_texts(create_model_selection_content(_result(feature_cols=["a", "b", "c"]))))
    assert "다변량 기준 추천 (특징 3, 외생 0)" in text


# --- request flow ------------------------------------------------------------------------------------------------------


def _settings(prop_id, auto, algorithm="Sarima", calls=None):
    def recommend(*args):
        (calls if calls is not None else []).append(args)
        raise ValueError("no recommendation in this test")

    table = create_param_table({"order": {"default": "(4, 1, 2)"}}).to_plotly_json()
    return handle_train_settings(
        prop_id=prop_id,
        algorithm=algorithm,
        table=table,
        load_train_df=lambda: None,
        target_col="value",
        feature_cols=[],
        specs=None,
        values=None,
        recommend=recommend,
        auto=auto,
    )


@pytest.mark.parametrize("algorithm", ["Sarima", "AutoETS", None])
def test_auto_train_does_not_open_the_popup(algorithm):
    calls = []
    is_open, content, inputs, specs, error, request, table = _settings("forecasting-train-btn", True, algorithm, calls)
    assert calls == []  # the recommendation shown in the popup is not even computed
    assert is_open is False
    assert request["auto"] is True and request["overrides"] == {}
    assert table is dash.no_update


def test_manual_train_still_computes_the_popup():
    calls = []
    *_, request, table = _settings("forecasting-train-btn", False, calls=calls)
    assert len(calls) == 1
    assert "auto" not in request


def test_auto_request_is_not_estimated():
    def estimate(request):
        raise AssertionError("should not be estimated")

    request = {"time": 1.0, "overrides": {}, "auto": True}
    is_open, content, pending, trigger = handle_train_request("forecasting-train-request", request, None, estimate)
    assert is_open is False and pending is None
    assert trigger == request


def test_algorithm_dropdown_is_disabled_in_auto_mode():
    assert algorithm_dropdown_disabled(True) is True
    assert algorithm_dropdown_disabled(False) is False
    assert algorithm_dropdown_disabled(None) is False


# --- analysis type -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "features, exog, label",
    [
        (None, None, "단변량"),
        ([], [], "단변량"),
        (["a", "b"], [], "다변량 (특징 2, 외생 0)"),
        ([], ["x"], "다변량 (특징 0, 외생 1)"),
        (["a", "b"], ["x"], "다변량 (특징 2, 외생 1)"),
    ],
)
def test_auto_mode_label(features, exog, label):
    assert auto_mode_label(True, features, exog) == label


def test_auto_mode_label_is_empty_without_auto():
    assert auto_mode_label(False, ["a"], ["x"]) == ""
    assert auto_mode_label(None, [], []) == ""


def test_analysis_mode():
    assert analysis_mode(None, None) == {"kind": "univariate", "n_features": 0, "n_exog": 0}
    assert analysis_mode(["a"], ["x", "y"]) == {"kind": "multivariate", "n_features": 1, "n_exog": 2}


# --- model selection ---------------------------------------------------------------------------------------------------


def test_horizon_is_the_number_of_test_steps():
    assert auto_horizon(_hourly(150), _hourly(20, start="2023-01-07 06:00"), "value") == 20


def test_horizon_of_irregular_test_data_is_resampled():
    # 2 hours of test data sampled every 5 minutes are 3 steps at the hourly training granularity
    idx = pd.date_range("2023-01-07 06:00", "2023-01-07 08:00", freq="5min")
    test_df = pd.DataFrame({"value": np.arange(len(idx), dtype=float)}, index=idx)
    assert len(test_df) == 25
    assert auto_horizon(_hourly(150), test_df, "value") == 3


def test_univariate_selection_gets_only_the_target():
    recommend = _recorder()
    train_df, test_df = _hourly(150), _hourly(20, start="2023-01-07 06:00")
    run_auto_selection(train_df, test_df, "value", [], [], recommend=recommend)
    ((args, kwargs),) = recommend.calls
    assert args[1:] == ("value", 20)
    assert args[0] is train_df
    assert kwargs == {}


def test_multivariate_selection_gets_the_selected_variables():
    recommend = _recorder()
    columns = ("value", "a", "b", "x")
    train_df, test_df = _hourly(150, columns=columns), _hourly(20, start="2023-01-07 06:00", columns=columns)
    run_auto_selection(train_df, test_df, "value", ["a", "b"], ["x"], recommend=recommend)
    ((args, kwargs),) = recommend.calls
    assert args[1:] == ("value", 20)
    assert kwargs == {"feature_columns": ["a", "b"], "exog_columns": ["x"]}


def test_selection_result():
    result = _result()
    assert result["algorithm"] == "ExtraTreesForecaster"
    assert result["params"] == {"maxlags": 24, "max_forecast_steps": 20}
    assert result["horizon"] == 20 and result["n_valid"] == 30 and result["n_points"] == 150
    assert result["mode"] == {"kind": "univariate", "n_features": 0, "n_exog": 0}
    assert result["notes"] == ["Prophet: a note"]
    assert [c["algorithm"] for c in result["candidates"]] == ["ExtraTreesForecaster", "Arima", "ETS"]
    assert result["candidates"][2]["valid_mae"] is None  # infinity is not valid JSON
    assert result["candidates"][2]["status"] == "left out: slow"
    json.dumps(result, allow_nan=False)  # the result goes to a Store


def test_no_selected_algorithm_raises_with_the_reasons():
    failed = [
        FakeCandidate("ETS", {}, status="left out: estimated training time 900 s > 300 s"),
        FakeCandidate("Arima", {}, status="failed: LinAlgError: singular matrix"),
    ]
    recommend = _recorder(FakeRecommendation(None, {}, failed))
    with pytest.raises(RuntimeError) as e:
        run_auto_selection(_hourly(150), _hourly(20, start="2023-01-07 06:00"), "value", [], [], recommend=recommend)
    message = str(e.value)
    assert "ETS: left out: estimated training time 900 s > 300 s" in message
    assert "Arima: failed: LinAlgError: singular matrix" in message


# --- showing the result ------------------------------------------------------------------------------------------------


PARAM_INFO = {
    "ExtraTreesForecaster": {"maxlags": {"default": ""}, "max_forecast_steps": {"default": 100}, "n_estimators": {}},
    "Arima": {"order": {"default": (4, 1, 2)}, "max_forecast_steps": {"default": 100}},
}


def _table_values(table):
    return {row["Parameter"]: row["Value"] for row in table.to_plotly_json()["props"]["data"]}


def _param_info(algorithm):
    return {name: {"default": info.get("default", 10)} for name, info in PARAM_INFO[algorithm].items()}


def test_selected_algorithm_table_has_the_selected_params():
    result = {"algorithm": "Arima", "params": {"order": [2, 1, 1], "max_forecast_steps": 20}}
    values = _table_values(handle_select_algorithm("Arima", result, param_info=_param_info))
    assert values == {"order": "(2, 1, 1)", "max_forecast_steps": "20"}


def test_other_algorithm_table_has_the_defaults():
    result = {"algorithm": "Arima", "params": {"order": [2, 1, 1], "max_forecast_steps": 20}}
    values = _table_values(handle_select_algorithm("ExtraTreesForecaster", result, param_info=_param_info))
    assert values == {"maxlags": "", "max_forecast_steps": "100", "n_estimators": "10"}
    values = _table_values(handle_select_algorithm("Arima", None, param_info=_param_info))
    assert values == {"order": "(4, 1, 2)", "max_forecast_steps": "100"}


def test_auto_result_updates_the_dropdown_and_the_card():
    value, options, content = handle_auto_result(_result())
    assert value == "ExtraTreesForecaster"
    assert {"label": "ExtraTreesForecaster", "value": "ExtraTreesForecaster"} in options
    assert "ExtraTreesForecaster" in " ".join(_texts(content))


def test_manual_result_clears_the_card():
    value, options, content = handle_auto_result(None)
    assert value is dash.no_update and options is dash.no_update
    assert _texts(content) == _texts(create_model_selection_content(None))


def test_auto_mode_content_explains_the_analysis_type():
    """The line under the buttons names the analysis type and says which variables the selection uses."""
    texts = lambda children: " ".join(str(c.children) for c in children)
    multi = texts(auto_mode_content(True, ["a", "b", "c"], []))
    assert "Auto 모드: 다변량 (특징 3, 외생 0)" in multi and "특징 변수와 외생 변수를 함께" in multi
    uni = texts(auto_mode_content(True, [], None))
    assert "Auto 모드: 단변량" in uni and "대상 변수만으로" in uni
    assert auto_mode_content(False, ["a"], []) == []

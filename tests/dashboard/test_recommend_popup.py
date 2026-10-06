#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Tests for the Forecasting tab's confirmation popup of the recommended settings: validating the confirmed values,
the popup content and input fields, and the Train -> popup -> Confirm/Cancel logic (without a Dash server).
"""
import dash
import pytest
from dash import dcc

from merlion.dashboard.callbacks.forecast import _create_recommendation_content, handle_train_settings
from merlion.dashboard.models.recommend import ParamRecommendation, Recommendation, parse_confirmed_values
from merlion.dashboard.pages.forecast import create_confirm_inputs, create_param_confirm_modal
from merlion.dashboard.pages.utils import create_param_table


def _specs(*params):
    return [p.spec() for p in params]


INT = ParamRecommendation("maxlags", 24, "int", "", is_steps=True)
OPTIONAL_INT = ParamRecommendation("seasonal_periods", None, "int", "", optional=True, is_steps=True)
CHOICE = ParamRecommendation("trend", "add", "choice", "", choices=["add", "mul", "None"])
BOOL = ParamRecommendation("damped_trend", True, "choice", "", choices=["True", "False"])
TUPLE = ParamRecommendation("seasonal_order", [1, 1, 0, 24], "int_tuple", "", tuple_len=4)
STR = ParamRecommendation("granularity", "5min", "str", "")


def _rec(*params):
    return Recommendation("Sarima", list(params), n_points=960, granularity=3600, notes=["a note"])


def _table(**values):
    return create_param_table({name: {"default": value} for name, value in values.items()}).to_plotly_json()


# --- parse_confirmed_values -------------------------------------------------------------------------------------------


def test_parse_valid_values():
    overrides, errors = parse_confirmed_values(
        _specs(INT, OPTIONAL_INT, CHOICE, BOOL, TUPLE, STR), [24, None, "None", "False", "(1, 0, 1, 24)", " 1h "]
    )
    assert errors == []
    assert overrides == {
        "maxlags": 24,
        "seasonal_periods": None,
        "trend": None,
        "damped_trend": False,
        "seasonal_order": [1, 0, 1, 24],
        "granularity": "1h",
    }


@pytest.mark.parametrize("value", [0, -3, 2.5, None, "", "abc"])
def test_parse_rejects_invalid_int(value):
    overrides, errors = parse_confirmed_values(_specs(INT), [value])
    assert "maxlags" not in overrides
    assert len(errors) == 1 and errors[0].startswith("maxlags")


def test_parse_accepts_integral_float():
    assert parse_confirmed_values(_specs(INT), [24.0]) == ({"maxlags": 24}, [])


def test_parse_rejects_unknown_choice():
    _, errors = parse_confirmed_values(_specs(CHOICE), ["linear"])
    assert errors and "trend must be one of add, mul, None" in errors[0]


@pytest.mark.parametrize("value", ["(1, 0, 1)", "(1, 0, -1, 24)", "1, a, 1, 24", ""])
def test_parse_rejects_invalid_tuple(value):
    _, errors = parse_confirmed_values(_specs(TUPLE), [value])
    assert len(errors) == 1 and errors[0].startswith("seasonal_order")


def test_parse_accepts_list_syntax_for_tuples():
    assert parse_confirmed_values(_specs(TUPLE), ["[2,1,1,12]"]) == ({"seasonal_order": [2, 1, 1, 12]}, [])


# --- popup layout and content -----------------------------------------------------------------------------------------


def test_modal_layout_has_dynamic_inputs():
    layout = str(create_param_confirm_modal().to_plotly_json())
    assert "forecasting-confirm-inputs" in layout
    assert "forecasting-confirm-specs" in layout
    assert "forecasting-confirm-maxlags" not in layout


def test_inputs_match_parameter_kinds():
    inputs = create_confirm_inputs([INT, CHOICE, TUPLE, STR])
    components = [div.children[1] for div in inputs]
    assert [type(c) for c in components] == [dcc.Input, dcc.Dropdown, dcc.Input, dcc.Input]
    assert components[0].type == "number" and components[0].value == 24
    assert components[1].value == "add"
    assert components[2].type == "text" and components[2].value == "(1, 1, 0, 24)"
    assert components[0].id == {"type": "forecasting-confirm-param", "name": "maxlags"}


def test_content_shows_duration_only_for_steps():
    content = _create_recommendation_content(_rec(INT, TUPLE), _table(maxlags="None", seasonal_order="(2, 0, 1, 24)"))
    rows = content[1].children[1:]
    cells = [[td.children for td in row.children] for row in rows]
    assert cells[0][:4] == ["maxlags", "None", "24", "1 days 00:00:00"]
    assert cells[1][:4] == ["seasonal_order", "(2, 0, 1, 24)", "(1, 1, 0, 24)", ""]
    assert any(getattr(c, "children", None) == "a note" for c in content)


# --- Train -> popup -> Confirm / Cancel --------------------------------------------------------------------------------


def _handle(prop_id, algorithm="Sarima", specs=None, values=None, recommend=None, table=None):
    return handle_train_settings(
        prop_id=prop_id,
        algorithm=algorithm,
        table=table or _table(order="(4, 1, 2)", seasonal_order="(2, 0, 1, 24)"),
        load_train_df=lambda: None,
        target_col="value",
        feature_cols=[],
        specs=specs,
        values=values,
        recommend=recommend or (lambda *args: _rec(TUPLE)),
    )


def test_train_opens_popup_for_tuned_algorithm():
    is_open, content, inputs, specs, error, trigger, table = _handle("forecasting-train-btn")
    assert is_open is True
    assert specs == _specs(TUPLE)
    assert len(inputs) == 1
    assert trigger is dash.no_update


def test_train_starts_right_away_for_other_algorithms():
    def recommend(*args):
        raise AssertionError("should not be called")

    is_open, *_, trigger, table = _handle("forecasting-train-btn", algorithm="AutoETS", recommend=recommend)
    assert is_open is False
    assert trigger["overrides"] == {}


def test_train_starts_right_away_if_recommendation_fails():
    def recommend(*args):
        raise AssertionError("The training data is too short")

    is_open, *_, trigger, table = _handle("forecasting-train-btn", recommend=recommend)
    assert is_open is False
    assert trigger["overrides"] == {}


def test_confirm_trains_with_values_and_updates_table():
    is_open, *_, error, trigger, table = _handle(
        "forecasting-param-confirm-btn", specs=_specs(TUPLE), values=["(1, 1, 1, 24)"]
    )
    assert is_open is False and error == ""
    assert trigger["overrides"] == {"seasonal_order": [1, 1, 1, 24]}
    data = {row["Parameter"]: row["Value"] for row in table.to_plotly_json()["props"]["data"]}
    assert data["seasonal_order"] == "(1, 1, 1, 24)"
    assert data["order"] == "(4, 1, 2)"


def test_confirm_with_invalid_value_keeps_popup_open():
    is_open, *_, error, trigger, table = _handle(
        "forecasting-param-confirm-btn", specs=_specs(TUPLE), values=["(1, 0, 1)"]
    )
    assert is_open is True
    assert "seasonal_order" in error
    assert trigger is dash.no_update and table is dash.no_update


def test_cancel_does_not_train():
    is_open, *_, trigger, table = _handle("forecasting-param-cancel-btn")
    assert is_open is False
    assert trigger is dash.no_update and table is dash.no_update

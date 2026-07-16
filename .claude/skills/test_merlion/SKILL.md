---
name: test_merlion
description: Use when asked to run or report on the dashboard<->merlion core interface tests (tests/dashboard/conftest.py, test_anomaly_model.py, test_forecast_model.py). Runs pytest against tests/dashboard/ and reports a clear pass/fail/xfail breakdown with root causes for anything unexpected. For environment/dependency diagnostics use checkenv_merlion instead; for launching the dashboard use run_merlion.
---

# Test Merlion Dashboard Interface

Purpose: run the `tests/dashboard/` suite (written to verify the AnomalyModel/ForecastModel <-> merlion core interface: ModelFactory, TimeSeries, ModelBase/Config -- see analayze_codebase.md section 5-1) and report the results clearly. This is a read-only diagnostic — it doesn't install dependencies or fix code.

## Files under test

- `tests/dashboard/conftest.py` — fixtures: `file_manager` (isolated per-test `FileManager`), `train_test_df`/`forecast_train_test_df`, `set_progress`.
- `tests/dashboard/test_anomaly_model.py` — `AnomalyModel` happy path (train/save/load) + failure paths (unregistered algorithm, missing column, missing model directory).
- `tests/dashboard/test_forecast_model.py` — same coverage for `ForecastModel`.

## Steps

1. **Run the suite**:
   ```
   python -m pytest tests/dashboard/ -v
   ```
   If `pytest` isn't installed in the active interpreter, install it first (`pip install pytest`) — this suite only needs pytest itself plus whatever merlion/dashboard dependencies are already present.

2. **Read the summary line** (`N passed, M xfailed, K failed`, etc.) and the per-test PASSED/FAILED/XFAIL/XPASS markers.

3. **Interpret known-expected results** (don't report these as new problems, but do mention them so the user has full context):
   - On Python 3.14: `test_train_returns_model_and_metrics` and `test_save_and_load_model_round_trip` in both `test_anomaly_model.py` and `test_forecast_model.py` are marked `xfail` (conditional on `sys.version_info >= (3, 14)`) because of the documented Enum/`functools.partial` incompatibility (analayze_codebase.md section 1-3). Seeing these as XFAIL is expected and not a regression.
   - On Python <3.14 (e.g. the verified `.venv312`), these same tests should run for real and PASS.
   - If any test shows **XPASS** (unexpectedly passed) where `strict=True` is set, pytest turns that into a FAILED result — this is a signal worth reporting explicitly, since it usually means either (a) a documented bug got fixed and the xfail marker's condition/reason needs updating, or (b) the xfail condition is wrong for this interpreter. Don't silently treat XPASS-as-FAILED the same as a genuine new bug — check which case it is (e.g. by checking `sys.version_info` against the condition in the test file) before reporting.

4. **For any genuinely unexpected FAILED test** (not one of the known xfails above), show the actual pytest failure output (assertion or traceback) rather than just the test name, and identify:
   - Whether it looks like a real merlion/dashboard bug (report file:line and the error).
   - Whether it looks like a fixture/test-data problem introduced by a change to `tests/dashboard/` itself.

5. **Report a concise summary** to the user: total counts (passed/xfailed/failed), which Python interpreter was used, and — if anything is a genuine new failure — the root cause and suggested fix. If everything matches expectations for the current interpreter, say so plainly.

## Cross-checking against multiple interpreters (optional, only if asked)

If the user wants confirmation across both verified environments, run the suite under both and report both summaries:
```
python -m pytest tests/dashboard/ -v            # active interpreter (e.g. Python 3.14 -- expect 6 passed, 4 xfailed)
.venv312/Scripts/python.exe -m pytest tests/dashboard/ -v   # Python 3.12 (e.g. expect 10 passed, 0 xfailed)
```
(Adjust the `.venv312` path/activation command per-OS if it doesn't exist in this exact form.)

## Not this skill's job

- Don't fix failing tests or merlion source automatically -- report findings and ask before making changes.
- Don't install/upgrade dependencies -- if pytest or another import is missing, mention it, but broader dependency repair is `verify_merlion`'s job.
- Don't launch the dashboard server -- that's `run_merlion`.

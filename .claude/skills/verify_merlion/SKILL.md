---
name: verify_merlion
description: Use when asked to verify, check, or smoke-test the Merlion dashboard's dependencies and that it actually boots (Dash-based GUI at merlion/dashboard/). Handles dependency install pitfalls on this repo (numpy build issues on newer Python, dash version pinning, missing psutil) and runs the automated smoke-test script, then tears the server down. For just launching the dashboard for manual browser use, use run_merlion instead.
---

# Verify Merlion Dashboard

Purpose: check that the Merlion dashboard's dependencies are correctly installed and that the app actually boots — an automated check, not a manual/browser session. Ends with the server torn down, not left running.

## Known constraints (verified by actually running this)

- **dash must be `>=2.4,<3.0`.** `setup.py`'s `dash[diskcache]>=2.4` has no upper bound, but `merlion/dashboard/utils/file_manager.py` imports `from dash.long_callback import DiskcacheLongCallbackManager` — that module was renamed/removed starting in dash 3.0 (confirmed broken on dash 3.0.0 and 4.4.0; confirmed working on dash 2.18.2). Installing latest dash breaks the dashboard.
- **`psutil` is required but not auto-installed** by the `dash[diskcache]` extra even though `DiskcacheLongCallbackManager` needs it at runtime. Install `psutil` (and `multiprocess`) explicitly if missing.
- **On very new Python versions (e.g. 3.14), `pip install -e .[dashboard]` as-is may fail** — `numpy<2.0` (pinned in `install_requires`) has no prebuilt wheel and needs a C/C++ compiler to build from source, which may not be present. If that happens, install core deps without the strict pin (`pip install -e . --no-deps`, then install `numpy`, `pandas`, `dash`, etc. individually) rather than fighting the pin.

## Steps

1. **Check current install state** before reinstalling anything:
   ```
   python -c "import dash; print(dash.__version__)"
   ```
   If dash is missing or `>=3.0`, fix it (see step 2). If it's already in `[2.4, 3.0)` with `psutil` present, skip to step 3.

2. **Install/fix dependencies**:
   ```
   pip install -e .[dashboard]
   ```
   If this fails on `numpy` metadata generation (seen on Python 3.14 — no compiler available):
   ```
   pip install -e . --no-deps
   pip install "numpy>=2.0" pandas dash dash-bootstrap-components diskcache dill GitPython py4j tqdm packaging plotly matplotlib scipy scikit-learn statsmodels
   ```
   Then pin dash to a working 2.x version and add the missing runtime dep:
   ```
   pip install "dash[diskcache]==2.18.2" "dash-bootstrap-components==1.5.0" --no-deps
   pip install psutil multiprocess
   ```

3. **Verify dash version lands in the working range**:
   ```
   python -c "import dash.long_callback; print('OK', dash.__version__)"
   ```
   Should print `OK 2.x.x` with no `ModuleNotFoundError`.

4. **Run the automated smoke test**:
   ```
   python scripts/test_dashboard.py
   ```
   This launches `python -m merlion.dashboard` as a subprocess, polls until the port opens, `GET`s `/`, checks for HTTP 200 with "Merlion" in the body, then tears the server down itself. Optional flags: `--host`, `--port` (default 8050), `--timeout` (default 30s).
   If it fails, it dumps the server's stdout/stderr — read that output to diagnose (most failures at this repo state are the dash version / psutil issues above, not new bugs).

5. **Report pass/fail** to the user, plus any dependency fixes that were needed to get there.

## Files involved

- `scripts/test_dashboard.py` — the smoke-test script itself. Read/extend this file if more routes need checking (e.g. tab callbacks) rather than writing a new script.
- `merlion/dashboard/__main__.py` — entrypoint (`app.run_server(debug=False)`), binds to `127.0.0.1:8050` by default.
- `merlion/dashboard/server.py` — the Dash/Flask app definition; this is what fails to import if the dash version is wrong.
- `merlion/dashboard/utils/file_manager.py` — where the dash version dependency (`dash.long_callback`) and the `psutil` runtime requirement actually get exercised.

## Reporting back

If deviating from the documented `setup.py` constraint was needed to get it running (e.g. downgrading dash), say so explicitly — this is a real gap in `setup.py`'s version pins, not just an environment quirk, and is worth surfacing to the user (see `analayze_codebase.md` for the existing writeup of this issue).

## Not this skill's job

Do not leave the server running for manual browser testing — that's `run_merlion`'s job. This skill always tears the server down (via the smoke-test script) before finishing.

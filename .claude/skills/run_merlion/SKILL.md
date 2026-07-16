---
name: run_merlion
description: Use when asked to run, start, or launch the Merlion dashboard web app (Dash-based GUI at merlion/dashboard/) for manual/interactive browser use. Purpose is only to launch the dashboard and hand the user a working URL to test in their browser, then keep it running until the user is done. Does not check/fix dependencies — for dependency verification and automated smoke-testing, use verify_merlion instead.
---

# Run Merlion Dashboard

Purpose: launch the Merlion dashboard (`merlion/dashboard/`, a Dash/Flask web app) for manual/interactive use and print the URL the user can open to test it themselves. This skill assumes dependencies already work — if launching fails, point the user (or yourself) at `verify_merlion` to diagnose and fix dependency issues, rather than fixing them inline here.

## Steps

1. **Launch the dashboard server**:
   ```
   python -m merlion.dashboard
   ```
   Run this with `run_in_background: true` via the Bash tool — it must stay running for the user to test it. Default bind address is `127.0.0.1:8050` (see `merlion/dashboard/__main__.py` / `app.run_server(debug=False)`).

2. **Poll readiness**:
   ```
   curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8050/
   ```
   If it doesn't return 200 within a few seconds, check the background task's output. A `ModuleNotFoundError` on `dash.long_callback` or a missing `psutil` means dependencies aren't in a working state — stop and run `verify_merlion` instead of debugging install issues here.

3. **Print the URL** to the user once the port responds with 200:
   ```
   http://127.0.0.1:8050
   ```
   This is the required output of this skill — always surface it explicitly, not just "server started."

4. **Leave it running** until the user says they're done testing, then stop the background task (TaskStop on its task id). Do not tear it down before the user has had a chance to check it, and do not leave it running indefinitely after they confirm they've finished.

## Files involved

- `merlion/dashboard/__main__.py` — entrypoint (`app.run_server(debug=False)`), binds to `127.0.0.1:8050` by default.
- `merlion/dashboard/server.py` — the Dash/Flask app definition.

## Related skill

`verify_merlion` — checks/fixes dependencies (dash version pinning, psutil, numpy build issues) and runs the automated smoke test (`scripts/test_dashboard.py`), tearing the server down when finished. Use that one first if you're unsure the app currently boots, or if this skill's launch fails.

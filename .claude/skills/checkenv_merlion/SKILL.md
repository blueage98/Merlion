---
name: checkenv_merlion
description: Use when asked to check, verify, or diagnose whether the current Python environment satisfies Merlion's runtime constraints (Python/numpy/pandas/dash version issues documented in analayze_codebase.md). Runs scripts/check_environment.py and reports pass/warn/fail findings plus concrete fixes. For actually launching the dashboard, use run_merlion; for a full dependency install + smoke test, use verify_merlion.
---

# Check Merlion Environment

Purpose: run `scripts/check_environment.py` against the active Python interpreter and report which of Merlion's known runtime constraints are satisfied, which aren't, and what to do about each failure. This is a read-only diagnostic — it doesn't install or change anything.

## Background

`analayze_codebase.md` (sections 1, 1-2, 1-3) documents several environment constraints discovered by actually running Merlion, not by reading `setup.py` alone:

- **Python must not be 3.14** — `merlion/utils/resample.py::AggregationPolicy` uses Enum members whose values are `functools.partial` objects; Python 3.14 no longer recognizes these as members, which breaks `TimeSeries.align()` and thus all real `model.train()` calls. Confirmed broken on 3.14, confirmed fine on 3.12. Python 3.13 is untested.
- **numpy** should stay within `setup.py`'s declared `>=1.21,<2.0` — outside Python 3.14 (no prebuilt wheel, needs a C/C++ compiler), this range installs cleanly.
- **pandas `[-1]` positional-indexing bug** — `sarima.py`/`ets.py` used to do a bare `series[-1]` (label lookup) on a `DatetimeIndex`-based Series, which silently fell back to positional access on old pandas but raises `KeyError: -1` on pandas 3.x. Already fixed via `.iloc[-1]`; the script also acts as a regression guard against this being reintroduced.
- **dash must be `>=2.4,<3.0`** — `dash 3.0` removed/renamed the `dash.long_callback` module that `merlion/dashboard/utils/file_manager.py` imports.
- **psutil** is required by `DiskcacheLongCallbackManager` at runtime but isn't pulled in automatically by the `dash[diskcache]` extra.

`scripts/check_environment.py` checks all of these **functionally** (it reproduces the actual bug behavior, e.g. building a real `Enum` with a `functools.partial` value and checking `__members__`) rather than just comparing version strings, since version ranges alone were shown to be an unreliable proxy.

## Steps

1. **Run the check**:
   ```
   python scripts/check_environment.py
   ```
   Add `--no-dashboard` if you only care about core (non-dashboard) constraints and want to skip the dash/psutil/diskcache checks.

2. **Read the exit code**: `0` means no FAILs (WARNs are still worth mentioning to the user but don't block). Non-zero means at least one FAIL.

3. **Report results to the user** in a short table or list: which checks PASSed, WARNed, or FAILed, with the one-line detail the script prints for each. Don't just say "some checks failed" — name them.

4. **For each FAIL, give the concrete fix**, not just "something's wrong":
   - Python Enum/functools.partial FAIL → the interpreter itself is the problem (e.g. Python 3.14). Recommend switching to Python 3.12 (verified working) — e.g. via a venv (`py -3.12 -m venv .venv312` on Windows, or `python3.12 -m venv .venv312` elsewhere), not by editing merlion source.
   - numpy WARN → `pip install "numpy>=1.21,<2.0"` (or investigate why a newer numpy got pulled in, e.g. no compatible wheel for the current Python version).
   - pandas `[-1]` regression FAIL → someone reintroduced a bare `series[-1]` in `sarima.py`/`ets.py` (the script's error message names the exact file:line) — change it back to `.iloc[-1]`.
   - dash FAIL → `pip install "dash[diskcache]>=2.4,<3.0"` (e.g. `dash==2.18.2`, the version verified in analayze_codebase.md).
   - psutil FAIL → `pip install psutil`.
   - DiskcacheLongCallbackManager FAIL → read the actual exception text the script prints; this is a catch-all functional check, so the fix depends on what broke (could be a permissions issue, a diskcache/dash incompatibility, etc.).

5. **If everything passes**, say so plainly and don't invent caveats — this environment matches the verified-working combination (Python 3.12 + numpy<2.0 + dash 2.18.2 + psutil, per analayze_codebase.md).

## Not this skill's job

- Don't install or fix anything automatically — this is a diagnostic. If the user wants fixes applied, that's a follow-up action they should explicitly ask for (or use `verify_merlion` for the install-and-fix workflow).
- Don't launch the dashboard server — that's `run_merlion`.

#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Evaluates whether the current Python environment satisfies the runtime
constraints for Merlion (core library + dashboard) discovered and verified
in analayze_codebase.md sections 1, 1-2, and 1-3.

Each check is FUNCTIONAL where possible (it reproduces the actual bug
behavior) rather than just comparing version strings, since version ranges
alone were shown to be an unreliable proxy (e.g. the Enum/functools.partial
bug is Python-3.14-specific and not simply "any Python 3.13+").

Usage:
    python scripts/check_environment.py            # core + dashboard checks
    python scripts/check_environment.py --no-dashboard   # skip dash/psutil checks
"""
import argparse
import importlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_SYMBOL = {PASS: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}


class Result:
    def __init__(self, name, status, detail):
        self.name = name
        self.status = status
        self.detail = detail


def check_python_version():
    """merlion/utils/resample.py::AggregationPolicy uses Enum members whose values
    are functools.partial objects. Python 3.14 no longer recognizes these as members
    (see analayze_codebase.md section 1-3), which breaks TimeSeries.align() and thus
    all real model training. Verified broken on 3.14, verified fine on 3.12; 3.13 is
    untested, so it's flagged as a warning rather than asserted safe."""
    from enum import Enum
    from functools import partial

    class _Probe(Enum):
        A = partial(lambda x: x)

    if not _Probe.__members__:
        return Result(
            "Python Enum/functools.partial behavior",
            FAIL,
            f"Python {sys.version.split()[0]}: Enum members with functools.partial values are not "
            "recognized (reproduces the bug that breaks merlion.utils.resample.AggregationPolicy, "
            "which breaks TimeSeries.align() and all real model.train() calls). "
            "Known-good: Python 3.12. Known-bad: Python 3.14.",
        )
    if sys.version_info >= (3, 14):
        # Shouldn't happen given the check above, but keep as a belt-and-suspenders warning.
        return Result(
            "Python Enum/functools.partial behavior",
            WARN,
            f"Python {sys.version.split()[0]}: functional probe passed, but this Python version "
            "wasn't the one verified in analayze_codebase.md -- re-verify if issues appear.",
        )
    if sys.version_info[:2] == (3, 13):
        return Result(
            "Python Enum/functools.partial behavior",
            WARN,
            f"Python {sys.version.split()[0]}: functional probe passed. Python 3.13 itself was never "
            "directly tested in analayze_codebase.md (only 3.12 and 3.14) -- treat as provisionally OK.",
        )
    return Result(
        "Python Enum/functools.partial behavior",
        PASS,
        f"Python {sys.version.split()[0]}: Enum members with functools.partial values work correctly.",
    )


def check_numpy():
    """setup.py pins numpy>=1.21,<2.0. On Python 3.14 this range has no prebuilt wheel
    and fails to build without a C/C++ compiler (analayze_codebase.md section 1-2)."""
    try:
        import numpy
    except ImportError:
        return Result("numpy", FAIL, "numpy is not installed.")
    version = tuple(int(p) for p in re.match(r"(\d+)\.(\d+)", numpy.__version__).groups())
    if version < (1, 21) or version >= (2, 0):
        return Result(
            "numpy",
            WARN,
            f"numpy {numpy.__version__} is outside setup.py's declared range (>=1.21,<2.0). "
            "merlion's compatibility with numpy>=2.0 has not been fully verified.",
        )
    return Result("numpy", PASS, f"numpy {numpy.__version__} (within setup.py's >=1.21,<2.0)")


def check_pandas_positional_indexing_fix():
    """sarima.py:112,131 and ets.py:145 used to do a bare `series[-1]` (label lookup)
    on a DatetimeIndex-based Series, intending positional "last value" access. Older
    pandas silently fell back to positional indexing for this; pandas 3.x does not,
    raising KeyError: -1 (analayze_codebase.md section 1-3, item 8). Fixed in source
    via .iloc[-1]. This check is a regression guard against that fix being reverted,
    plus a live functional probe of the actual pandas behavior difference."""
    import pandas as pd
    import numpy as np

    idx = pd.date_range("2023-01-01", periods=3, freq="h")
    series = pd.Series(np.arange(3), index=idx)
    try:
        series[-1]
        pandas_allows_bare_negative_index = True
    except KeyError:
        pandas_allows_bare_negative_index = False

    offending = []
    for relative_path in ["merlion/models/forecast/sarima.py", "merlion/models/forecast/ets.py"]:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for lineno, line in enumerate(source.splitlines(), start=1):
            if re.search(r"(?<!i)(?<!iloc)\[[^\]]*-1\]", line) and "iloc" not in line and "=" in line:
                # Heuristic: a bare `name[-1]` (not `.iloc[-1]`) assigned to/from a Series-like name.
                if re.search(r"\b\w+\[-1\]", line) and ".iloc[-1]" not in line:
                    offending.append(f"{relative_path}:{lineno}: {line.strip()}")

    if offending:
        return Result(
            "pandas [-1] positional indexing regression",
            FAIL,
            "Found bare `series[-1]` usage that should be `.iloc[-1]` (KeyError: -1 risk on modern "
            "pandas):\n    " + "\n    ".join(offending),
        )
    if not pandas_allows_bare_negative_index:
        detail = (
            f"pandas {pd.__version__}: bare `series[-1]` on a DatetimeIndex raises KeyError as expected "
            "on modern pandas -- the .iloc[-1] fix in sarima.py/ets.py is required and present."
        )
    else:
        detail = (
            f"pandas {pd.__version__}: bare `series[-1]` still silently falls back to positional "
            "indexing on this pandas version, so the underlying bug wouldn't reproduce here -- but "
            "the .iloc[-1] fix is present regardless (forward-compatible)."
        )
    return Result("pandas [-1] positional indexing regression", PASS, detail)


def check_dash(required):
    """setup.py declares dash[diskcache]>=2.4 with no upper bound, but
    merlion/dashboard/utils/file_manager.py imports dash.long_callback, which was
    removed/renamed to dash.background_callback starting in dash 3.0
    (analayze_codebase.md section 1-2). Required range: >=2.4,<3.0."""
    try:
        import dash
    except ImportError:
        return Result("dash", FAIL if required else WARN, "dash is not installed.")
    version = tuple(int(p) for p in re.match(r"(\d+)\.(\d+)", dash.__version__).groups())
    if version >= (3, 0):
        return Result(
            "dash",
            FAIL,
            f"dash {dash.__version__}: >=3.0 removed the `dash.long_callback` module that "
            "merlion/dashboard/utils/file_manager.py imports. Required: dash>=2.4,<3.0.",
        )
    if version < (2, 4):
        return Result("dash", FAIL, f"dash {dash.__version__}: below setup.py's declared >=2.4 minimum.")
    try:
        importlib.import_module("dash.long_callback")
        long_callback_ok = True
    except ModuleNotFoundError:
        long_callback_ok = False
    if not long_callback_ok:
        return Result(
            "dash",
            FAIL,
            f"dash {dash.__version__} is in the expected version range, but `dash.long_callback` "
            "still isn't importable -- investigate before trusting the dashboard to boot.",
        )
    return Result("dash", PASS, f"dash {dash.__version__} (within required >=2.4,<3.0), dash.long_callback importable")


def check_psutil(required):
    """DiskcacheLongCallbackManager (used by merlion/dashboard/utils/file_manager.py)
    requires psutil at runtime, but it isn't pulled in by the dash[diskcache] extra
    (analayze_codebase.md section 1-2)."""
    try:
        import psutil  # noqa: F401
    except ImportError:
        return Result("psutil", FAIL if required else WARN, "psutil is not installed (required by dash's DiskcacheLongCallbackManager).")
    return Result("psutil", PASS, f"psutil {psutil.__version__} importable")


def check_diskcache_long_callback_manager(required):
    """End-to-end functional check: actually construct the object the dashboard
    constructs at import time (merlion/dashboard/utils/file_manager.py:37), in a
    throwaway temp directory, to catch anything the version/import checks above miss."""
    import shutil
    import tempfile

    try:
        import diskcache
        from dash.long_callback import DiskcacheLongCallbackManager
    except ImportError as e:
        return Result(
            "DiskcacheLongCallbackManager (functional)",
            FAIL if required else WARN,
            f"Could not import diskcache/dash.long_callback: {e}",
        )
    tmp = tempfile.mkdtemp()
    cache = None
    try:
        cache = diskcache.Cache(tmp)
        DiskcacheLongCallbackManager(cache)
    except Exception as e:
        return Result(
            "DiskcacheLongCallbackManager (functional)",
            FAIL,
            f"Constructing DiskcacheLongCallbackManager raised {type(e).__name__}: {e}",
        )
    finally:
        # diskcache.Cache holds an open sqlite handle; close it before cleanup so
        # Windows doesn't raise PermissionError when we remove the temp dir.
        if cache is not None:
            cache.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return Result("DiskcacheLongCallbackManager (functional)", PASS, "Constructed successfully in a temp cache dir.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-dashboard", action="store_true", help="Skip dash/psutil/diskcache checks.")
    args = parser.parse_args()

    checks = [check_python_version, check_numpy, check_pandas_positional_indexing_fix]
    if not args.no_dashboard:
        checks += [
            lambda: check_dash(required=True),
            lambda: check_psutil(required=True),
            lambda: check_diskcache_long_callback_manager(required=True),
        ]

    results = [check() for check in checks]

    print(f"Merlion environment check -- Python {sys.version.split()[0]} at {sys.executable}\n")
    for r in results:
        print(f"{_SYMBOL[r.status]} {r.name}")
        for line in r.detail.splitlines():
            print(f"       {line}")
        print()

    n_fail = sum(r.status == FAIL for r in results)
    n_warn = sum(r.status == WARN for r in results)
    n_pass = sum(r.status == PASS for r in results)
    print(f"Summary: {n_pass} passed, {n_warn} warnings, {n_fail} failed")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

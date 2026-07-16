## Context

The Data page (`merlion/dashboard/pages/data.py`) currently offers one path to load data: `dcc.Upload` → `upload_file` callback (`merlion/dashboard/callbacks/data.py:36-50`) saves the file into `FileManager.data_directory` (a runtime directory, `~/merlion/data`, see `merlion/dashboard/utils/file_manager.py:24-30`) and repopulates the `select-file` dropdown. `click_run` (`callbacks/data.py:53-90`) then joins `file_manager.data_directory` with the selected filename, calls `DataAnalyzer().load_data(file_path)`, and renders stats/table/figure. `DataMixin.load_data` (`merlion/dashboard/models/utils.py:17-25`) is a plain `pd.read_csv(file_path)` + datetime-index setup — it only needs a valid path, so it can serve either upload directory or the repo's sample data unchanged.

The repo root also ships a `./data` folder with ready-to-use sample datasets (top-level `example.csv` plus subfolders like `walmart/`, `smap/`, etc., used by `ts_datasets`). Nothing in the dashboard currently reads from this folder.

## Goals / Non-Goals

**Goals:**
- Let a user pick a sample CSV from repo-root `./data` (recursively, including subfolders) via a dedicated dropdown, and plot it through the existing `click_run` pipeline.
- Keep the existing upload/`select-file` flow fully intact and unaffected.

**Non-Goals:**
- No changes to how uploaded files are parsed, stored, or plotted.
- No support for non-CSV sample files (e.g. `.pkl`) in the new dropdown.
- No new visualization or stats logic — 100% reuse of `DataAnalyzer`/`DataMixin`.

## Decisions

**1. Separate "Sample Data" dropdown, not merged into `select-file`.**
Keeps upload-sourced files and repo sample files visually and semantically distinct, and avoids collisions if a sample filename happens to match an uploaded filename. Matches the user's explicit choice.

**2. Recursive scan of `./data`, filtered to `*.csv`.**
`./data` contains non-CSV artifacts (e.g. `test_transform.pkl`) and dataset-specific subfolder structures (`ts_datasets` raw data) that are not directly loadable via `pd.read_csv` in a generic way. Scanning is filtered to `.csv` extension only, so entries always resolve cleanly through the existing `load_data`. Options are labeled with their path relative to `./data` (e.g. `walmart/train.csv`) so the folder of origin is visible.

**3. Scan happens on-demand (dropdown click), following the existing `update_select_file_dropdown` pattern** (`callbacks/anomaly.py:21-40`), not on app startup.
Consistent with how the existing dropdown already refreshes its options, and picks up newly added sample files without an app restart.

**4. An explicit "Data Source" radio button (Uploaded File / Sample Data) determines which dropdown is active, rather than inferring exclusivity from the dropdowns' own values.**
Two earlier approaches were tried and rejected during implementation:
- *Silent precedence at Run time* (Sample Data wins if both have values): confusing — the user has no visible indication of which source will actually be used.
- *Disable-the-other-dropdown-based-on-value* (e.g. selecting `select-file` disables and clears `select-sample-file`, and vice versa): this coupled disabling to a dropdown's *value*, but a dropdown's value legitimately persists after a successful Load (so the user can Load the same file again, or see what's currently loaded). That meant after loading an uploaded file once, `select-file` permanently held a non-empty value, permanently disabling Sample Data with no way back except manually clearing the dropdown — a dead end for switching sources.

The adopted design decouples "which source is active" from "what value happens to be sitting in a dropdown": a `dcc.RadioItems(id="data-source")` with options `upload`/`sample` (default `upload`) is the single source of truth for which dropdown is enabled. A callback (`Input("data-source", "value")` → `Output("select-file", "disabled")`, `Output("select-sample-file", "disabled")`) toggles both dropdowns directly off the radio value. `click_run` reads `State("data-source", "value")` and uses *only* the corresponding dropdown's value to build `file_path` — the other dropdown's (possibly stale, disabled) value is never consulted:
- `data-source == "sample"` → resolve against repo-root `./data` using `select-sample-file`'s value
- otherwise (`"upload"`) → resolve against `file_manager.data_directory` using `select-file`'s value (existing behavior)

This is fully reversible at any time (switching the radio back and forth always re-enables the other dropdown) and makes the active source explicit in the UI rather than implicit.

**5. Directory listing helper lives alongside `FileManager`, not inside it.**
The recursive scan targets the repo-root `./data` path (a fixed, source-controlled location), which is conceptually different from `FileManager`'s runtime upload directory. A small standalone helper (e.g. `list_sample_files()` in `file_manager.py` or a new `sample_data.py` util) keeps that distinction clear rather than overloading `FileManager` with a second, unrelated directory concept.

## Risks / Trade-offs

- **[Risk]** Repo-root `./data` could grow large (it already holds multi-MB datasets), making a full recursive scan slow on every dropdown click. → **Mitigation**: scan is a simple `os.walk`/`glob` over a local filesystem; acceptable for expected dataset sizes. Revisit with caching only if it becomes a measured problem.
- **[Risk]** A sample CSV might not be in the shape `DataMixin.load_data` expects (e.g. no usable datetime-like first column, since some `./data` subfolders are `ts_datasets`-specific formats). → **Mitigation**: this is identical to the existing behavior for uploaded files — `load_data` already assumes a compatible CSV shape, and errors surface the same way for both paths. No new handling needed.
- **[Risk]** Disabling a dropdown via a callback keyed on the radio button's value means the disabled state won't update until Dash processes that callback (a brief round-trip). → **Mitigation**: negligible in practice for a local UI interaction; no separate mitigation needed.
- **[Risk]** A disabled dropdown can still hold a stale value in the Dash component tree (its `value` prop isn't forcibly cleared when disabled). → **Mitigation**: `click_run` never reads the inactive dropdown's value at all — it branches entirely on `data-source` — so a stale value in the disabled dropdown has no effect on what gets loaded.

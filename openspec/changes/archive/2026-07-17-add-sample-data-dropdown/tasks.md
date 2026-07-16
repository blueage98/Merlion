## 1. Sample file listing helper

- [x] 1.1 Add a helper function (e.g. `list_sample_files()`) that recursively scans repo-root `./data` for `*.csv` files and returns paths relative to `./data`
- [x] 1.2 Resolve the repo-root `./data` path robustly (independent of current working directory), e.g. relative to the package/repo location

## 2. UI: Sample Data dropdown

- [x] 2.1 Add a new `dcc.Dropdown(id="select-sample-file")` (or similar id) to `merlion/dashboard/pages/data.py`, alongside the existing `select-file` dropdown
- [x] 2.2 Add a callback (mirroring `update_select_file_dropdown` in `callbacks/anomaly.py:21-40`) that populates the new dropdown's `options` from `list_sample_files()` on click/open
- [x] 2.3 Add a `dcc.RadioItems(id="data-source")` ("Uploaded File" / "Sample Data", default "Uploaded File") and a callback keyed on its value that disables/enables `select-file` and `select-sample-file` accordingly (superseded the earlier value-based disable approach, which broke after a successful Load left a stale dropdown value in place)

## 3. Wire into existing Run pipeline

- [x] 3.1 Add `State("data-source", "value")` and `State("select-sample-file", "value")` to the `click_run` callback in `merlion/dashboard/callbacks/data.py`
- [x] 3.2 Update the file-path assembly logic to branch on `data-source`: `"sample"` resolves `file_path` as `<repo>/data/<select-sample-file value>`; otherwise resolves via existing `file_manager.data_directory` + `select-file` value behavior — the inactive dropdown's value is never read
- [x] 3.3 Confirm no changes are needed in `DataAnalyzer`/`DataMixin.load_data` — the resolved `file_path` is passed through unchanged

## 4. Verification

- [x] 4.1 Manually launch the dashboard and confirm the Sample Data dropdown lists nested sample files with relative paths (e.g. `walmart/train.csv`, `example.csv`) and excludes non-CSV files
- [x] 4.2 Confirm selecting a sample file and clicking Run loads and plots it correctly
- [x] 4.3 Confirm the existing upload → `select-file` → Run flow still works unchanged
- [x] 4.4 Confirm switching the Data Source radio button correctly enables/disables the corresponding dropdown, and that switching sources after a successful Load still works (no permanent lock-out)

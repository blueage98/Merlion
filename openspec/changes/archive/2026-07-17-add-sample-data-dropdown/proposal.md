## Why

Testing the dashboard currently requires the user to manually upload a CSV file every time, even though representative sample datasets already exist in the repo's `./data` folder. This is a repetitive, unnecessary friction point during development and demos. Letting the user pick a sample file directly in the UI removes that friction.

## What Changes

- Add a new "Sample Data" dropdown to the Data page, separate from the existing upload/`select-file` dropdown.
- The Sample Data dropdown lists CSV files found by recursively scanning the repo-root `./data` folder, showing each as a relative path that includes its subfolder (e.g. `walmart/train.csv`).
- Selecting a Sample Data entry and clicking Run loads and plots that file using the existing `click_run` callback pipeline — no new parsing/plotting code.
- The existing upload flow (`dcc.Upload`, `select-file` dropdown, `FileManager.data_directory`) is unchanged.
- `click_run`'s file-path assembly is extended to branch on which dropdown supplied the selected value: `select-file` resolves against `file_manager.data_directory` (existing behavior); the new Sample Data dropdown resolves against the repo-root `./data` directory.
- A "Data Source" radio button (Uploaded File / Sample Data, default Uploaded File) makes the active source explicit and controls which of the two dropdowns is enabled; `click_run` reads only the active source's dropdown value, so a value left over in the inactive dropdown (e.g. after a prior successful Load) never affects what gets loaded.

## Capabilities

### New Capabilities
- `sample-data-selection`: Lets a user browse and pick a pre-existing sample CSV from the repo's `./data` folder in the dashboard UI, and plot it via the existing data-loading pipeline, without needing to upload a file first.

### Modified Capabilities
(none — no existing OpenSpec specs currently document the dashboard's upload/select/run behavior, so nothing is being modified under contract; this change only adds new capability)

## Impact

- `merlion/dashboard/pages/data.py` — add a new `dcc.Dropdown(id="select-sample-file")` (or similar) alongside the existing upload/select controls.
- `merlion/dashboard/callbacks/data.py` — add a callback to populate the new dropdown's options by recursively scanning `./data`; extend `click_run` to branch base directory by dropdown source.
- Possibly `merlion/dashboard/utils/file_manager.py` or a new small utility — helper to recursively list CSV files under repo-root `./data` with relative paths.
- No changes to `DataMixin.load_data`, `DataAnalyzer`, or plotting utilities — fully reused as-is.

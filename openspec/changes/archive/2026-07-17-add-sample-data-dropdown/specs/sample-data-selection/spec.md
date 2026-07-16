## ADDED Requirements

### Requirement: Sample Data dropdown lists repo sample CSV files
The Data page SHALL display a "Sample Data" dropdown, separate from the existing upload/`select-file` dropdown, whose options are populated by recursively scanning the repo-root `./data` folder for files with a `.csv` extension.

#### Scenario: Dropdown populated with nested sample files
- **WHEN** the user opens/clicks the Sample Data dropdown
- **THEN** it lists every `.csv` file found anywhere under `./data`, each shown as its path relative to `./data` (e.g. `walmart/train.csv`, `example.csv`)

#### Scenario: Non-CSV files excluded
- **WHEN** the Sample Data dropdown options are computed
- **THEN** files under `./data` that do not have a `.csv` extension (e.g. `test_transform.pkl`) SHALL NOT appear as options

### Requirement: Selecting a sample file plots it via the existing pipeline
Selecting an entry from the Sample Data dropdown (with the "Sample Data" data source active) and running the existing Run action SHALL load and plot that file using the same `DataAnalyzer`/`DataMixin` pipeline already used for uploaded files, with no new parsing or plotting logic.

#### Scenario: Run with a Sample Data selection
- **WHEN** the Data Source is set to "Sample Data", the user selects a file in the Sample Data dropdown (e.g. `walmart/train.csv`), and clicks Run
- **THEN** the system resolves the full path as `./data/walmart/train.csv`, loads it via the existing `load_data` function, and renders the same stats/table/figure output as for an uploaded file

### Requirement: Existing upload flow is unaffected
The existing `dcc.Upload` control, `select-file` dropdown, and their backing `FileManager.data_directory` behavior SHALL continue to function exactly as before this change when the "Uploaded File" data source is active.

#### Scenario: Run with an uploaded file selection
- **WHEN** the Data Source is set to "Uploaded File", the user selects a file in the `select-file` dropdown, and clicks Run
- **THEN** the system resolves the path against `file_manager.data_directory` exactly as it did before this change

### Requirement: A Data Source selector determines which dropdown is active
The Data page SHALL display a "Data Source" control (radio buttons: "Uploaded File" / "Sample Data", defaulting to "Uploaded File") that determines exactly one of `select-file` or the Sample Data dropdown is enabled at a time; the other SHALL be disabled. This determination SHALL depend only on the Data Source selection, not on whether either dropdown currently holds a value.

#### Scenario: Selecting "Sample Data" as the source disables the Uploaded File dropdown
- **WHEN** the user sets Data Source to "Sample Data"
- **THEN** the Sample Data dropdown becomes enabled and the `select-file` dropdown becomes disabled

#### Scenario: Selecting "Uploaded File" as the source disables the Sample Data dropdown
- **WHEN** the user sets Data Source to "Uploaded File"
- **THEN** the `select-file` dropdown becomes enabled and the Sample Data dropdown becomes disabled

#### Scenario: Switching sources remains possible after a successful Load
- **WHEN** the user has successfully loaded data from one source (leaving that source's dropdown holding a value) and then switches the Data Source radio to the other source
- **THEN** the other source's dropdown becomes enabled and selectable, regardless of any value still held by the previously active dropdown

#### Scenario: Run always uses the active Data Source, not a stale dropdown value
- **WHEN** the user clicks Run
- **THEN** the system uses only the dropdown value belonging to the currently selected Data Source to resolve the file path, ignoring any value present in the other (inactive) dropdown

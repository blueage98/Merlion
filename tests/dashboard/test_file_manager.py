#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""Tests for FileManager's sample data handling (File Manager tab -> Sample Data -> Load)."""
import os

import pytest


def test_copy_sample_file_into_data_folder(file_manager):
    relpath = "synthetic_anomaly/horizontal.csv"
    assert relpath in file_manager.list_sample_files()

    name = file_manager.copy_sample_file(relpath)

    assert name == "horizontal.csv"
    assert name in file_manager.uploaded_files()
    with open(os.path.join(file_manager.sample_data_directory, relpath), "rb") as f:
        expected = f.read()
    with open(os.path.join(file_manager.data_directory, name), "rb") as f:
        assert f.read() == expected


def test_copy_sample_file_overwrites_existing_copy(file_manager):
    relpath = "example.csv"
    with open(os.path.join(file_manager.data_directory, "example.csv"), "w") as f:
        f.write("stale")

    file_manager.copy_sample_file(relpath)

    with open(os.path.join(file_manager.data_directory, "example.csv")) as f:
        assert f.read() != "stale"


def test_copy_sample_file_missing_raises(file_manager):
    with pytest.raises(AssertionError, match="does not exist"):
        file_manager.copy_sample_file("no_such_file.csv")


def test_copy_sample_file_rejects_path_outside_sample_dir(file_manager):
    with pytest.raises(AssertionError, match="Invalid sample file path"):
        file_manager.copy_sample_file("../setup.py")
    assert file_manager.uploaded_files() == []

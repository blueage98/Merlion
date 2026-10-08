#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import os
import requests
from tqdm import tqdm
import pandas as pd

from ts_datasets.base import BaseDataset

logger = logging.getLogger(__name__)


def _freq_aliases():
    """
    The pandas offset alias of each M4 subset. pandas 2.2 renamed the hourly and period-end aliases ("H" -> "h",
    "M" -> "ME", "Q" -> "QE", "Y" -> "YE"), and pandas 3 no longer accepts the old ones.
    """
    try:
        pd.tseries.frequencies.to_offset("ME")
        return dict(Hourly="h", Daily="D", Weekly="W", Monthly="ME", Quarterly="QE", Yearly="YE")
    except ValueError:
        return dict(Hourly="H", Daily="D", Weekly="W", Monthly="M", Quarterly="Q", Yearly="Y")


FREQ_ALIASES = _freq_aliases()


class M4(BaseDataset):
    """
    The M4 Competition data is an extended and diverse set of time series to
    identify the most accurate forecasting method(s) for different types
    of domains, including Business, financial and economic forecasting,
    and different type of granularity, including Yearly (23,000 sequences),
    Quarterly (24,000 sequences), Monthly (48,000 sequences),
    Weekly(359 sequences), Daily (4,227 sequences) and Hourly (414 sequences)
    data.

    - source: https://github.com/Mcompetitions/M4-methods/tree/master/Dataset
    - timeseries sequences: 100,000
    """

    valid_subsets = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
    url = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/{}.csv"

    def __init__(self, subset="Hourly", rootdir=None):
        super().__init__()
        self.subset = subset
        assert subset in self.valid_subsets, f"subset should be in {self.valid_subsets}, but got {subset}"

        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "M4")

        # download dataset if it is not found in root dir
        if not os.path.isdir(rootdir):
            logger.info(
                f"M4 {subset} dataset cannot be found from {rootdir}.\n"
                f"M4 {subset} dataset will be downloaded from {self.url}.\n"
            )
            download(rootdir, self.url, "M4-info")

        # extract starting date from meta-information of dataset
        self.freq = FREQ_ALIASES[subset]
        self.info_dataset = pd.read_csv(os.path.join(rootdir, "M4-info.csv"), parse_dates=True).set_index("M4id")

        train_csv = os.path.join(rootdir, f"train/{subset}-train.csv")
        if not os.path.isfile(train_csv):
            download(os.path.join(rootdir, "train"), self.url, f"{subset}-train", "Train")
        test_csv = os.path.join(rootdir, f"test/{subset}-test.csv")
        if not os.path.isfile(test_csv):
            download(os.path.join(rootdir, "test"), self.url, f"{subset}-test", "Test")

        self.train_set = pd.read_csv(train_csv).set_index("V1")
        self.test_set = pd.read_csv(test_csv).set_index("V1")

    def __getitem__(self, i):
        id = self.train_set.index[i]
        train, test = self.train_set.loc[id].dropna(), self.test_set.loc[id].dropna()
        ts = pd.concat((train, test)).to_frame()
        # raw data do not follow consistent timestamp format
        t0 = self.info_dataset.loc[id, "StartingDate"]
        try:
            ts.index = pd.date_range(start=t0, periods=len(ts), freq=self.freq)
        except Exception as e:
            if self.subset == "Yearly":
                logger.warning(f"Time series {i} too long for yearly granularity. Using quarterly instead.")
                ts.index = pd.date_range(start=t0, periods=len(ts), freq=FREQ_ALIASES["Quarterly"])
            else:
                raise e
        md = pd.DataFrame({"trainval": ts.index < ts.index[len(train)]}, index=ts.index)
        return ts, md

    def __len__(self):
        return len(self.train_set)


def download(datapath, url, name, split=None):
    os.makedirs(datapath, exist_ok=True)
    if split is not None:
        namesplit = split + "/" + name
    else:
        namesplit = name
    url = url.format(namesplit)
    file_path = os.path.join(datapath, name) + ".csv"
    if os.path.isfile(file_path):
        logger.info(name + " already exists")
        return
    logger.info("Downloading " + url)
    r = requests.get(url, stream=True)
    with open(file_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=16 * 1024**2):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

# Copyright 2022 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


""" Useful functions for handling the experiment data. """

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Optional

import pandas as pd


def dict_product(dicts):
    """
    Return the combination of all elements in dicts
    """

    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def combine_experiment_data(
    results_dir: str, filename: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetches all csv files in ``results_dir`` and concatenates them into a big pandas data frame.
    This data frame is then returned and saved as a csv under ``filename`` if given.
    """
    results_files = glob(f"{results_dir}/*.csv")
    if len(results_files) > 0:
        loaded_dfs = [pd.read_csv(file) for file in results_files]
        all_results = pd.concat(loaded_dfs, axis=0)
        if filename is not None:
            all_results.to_csv(filename, index=False)
            return None
        else:
            return all_results
    else:
        return None


def make_experiment_dir(path: str):
    """Create all directories for storing data, results and log from experiments."""
    Path(path).mkdir(exist_ok=True, parents=True)

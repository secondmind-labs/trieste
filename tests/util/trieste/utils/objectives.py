# Copyright 2021 The Trieste Contributors
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
"""
Simple dataset generators for testing.
"""

from __future__ import annotations

from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.objectives import branin, hartmann_6
from trieste.objectives.utils import mk_observer
from trieste.space import Box


def hartmann_6_dataset(num_query_points: int) -> Dataset:
    """
    Generate example dataset based on Hartmann 6 objective function.
    :param num_query_points: A number of samples from the objective function.
    :return: A dataset.
    """
    search_space = Box([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1])
    query_points = search_space.sample(num_query_points)

    observer = mk_observer(hartmann_6, OBJECTIVE)
    data = observer(query_points)

    return data[OBJECTIVE]


def branin_dataset(num_query_points: int) -> Dataset:
    """
    Generate example dataset based on Hartmann 6 objective function.
    :param num_query_points: A number of samples from the objective function.
    :return: A dataset.
    """
    search_space = Box([0, 0], [1, 1])
    query_points = search_space.sample(num_query_points)

    observer = mk_observer(branin, OBJECTIVE)
    data = observer(query_points)

    return data[OBJECTIVE]

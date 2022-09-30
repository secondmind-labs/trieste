# Copyright 2020 The Trieste Contributors
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
"""This module contains functionality for computing the non-dominated set
given a set of data points."""
from __future__ import annotations

import numpy as np

from ...types import TensorType


def non_dominated(observations: TensorType) -> tuple[TensorType, TensorType]:
    """
    Computes the non-dominated set for a set of data points. Loosely based on:
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    If there are duplicate point(s) in the non-dominated set, this function will return
    as it is without removing the duplicate.

    Note that reordering the observations by standard score beforehand can speed up the search.

    :param observations: set of points with shape [N,D]
    :return: tf.Tensor of the non-dominated set [P,D] and a non-dominated point mask [N],
        P is the number of points in pareto front, the mask specifies whether each data point
        is non-dominated or not.
    """
    nondominated_point_mask = np.full(observations.shape[0], True)

    next_point_index = -1
    while next_point_index + 1 < len(observations):
        next_point_index += 1 + int(np.argmax(nondominated_point_mask[next_point_index + 1 :]))
        nondominated_point_mask &= np.any(
            observations < observations[next_point_index], axis=1
        ) | np.all(observations == observations[next_point_index], axis=1)

    return observations[nondominated_point_mask], nondominated_point_mask

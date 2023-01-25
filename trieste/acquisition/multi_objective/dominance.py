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

import tensorflow as tf

from ...types import TensorType


def non_dominated(observations: TensorType) -> tuple[TensorType, TensorType]:
    """
    Computes the non-dominated set for a set of data points. Based on:
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    If there are duplicate point(s) in the non-dominated set, this function will return
    as it is without removing the duplicate.

    :param observations: set of points with shape [N,D]
    :return: tf.Tensor of the non-dominated set [P,D] and a non-dominated point mask [N],
        P is the number of points in pareto front, the mask specifies whether each data point
        is non-dominated or not.
    """
    num_points = tf.shape(observations)[0]

    # Reordering the observations beforehand speeds up the search:
    mean = tf.reduce_mean(observations, axis=0)
    std = tf.math.reduce_std(observations, axis=0)
    weights = tf.reduce_sum(((observations - mean) / (std + 1e-7)), axis=1)
    sorting_indices = tf.argsort(weights)

    def cond(i: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
        return i < len(indices)

    def body(i: tf.Tensor, indices: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        obs = tf.gather(observations, indices)
        nondominated = tf.reduce_any(obs < obs[i], axis=1) | tf.reduce_all(obs == obs[i], axis=1)
        i = tf.reduce_sum(tf.cast(nondominated[:i], tf.int32)) + 1
        indices = indices[nondominated]
        return i, indices

    _, indices = tf.while_loop(
        cond,
        body,
        loop_vars=(
            0,  # i
            tf.gather(tf.range(num_points), sorting_indices),  # indices
        ),
        shape_invariants=(
            tf.TensorShape([]),  # i
            tf.TensorShape([None]),  # indices
        ),
    )

    nondominated_observations = tf.gather(observations, indices)
    trues = tf.ones(tf.shape(indices), tf.bool)
    is_nondominated = tf.scatter_nd(indices[:, None], trues, [num_points])
    return nondominated_observations, is_nondominated

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
""" This module contains functions and classes for Pareto based multi-objective optimization. """

from typing import Tuple

import tensorflow as tf
from typing_extensions import Final

from .type import TensorType


def non_dominated(observations: TensorType) -> Tuple[TensorType, TensorType]:
    """
    Computes the non-dominated set for a set of data points.
    if there are duplicate point(s) in the non-dominated set, this function will return
    as it is without removing the duplicate.

    :param observations: set of points with shape [N,D]
    :return: tf.Tensor of the non-dominated set [P,D] and the degree of dominance [N],
        P is the number of points in pareto front
        dominances gives the number of dominating points for each data point


    """
    extended = tf.tile(observations[None], [len(observations), 1, 1])
    swapped_ext = tf.transpose(extended, [1, 0, 2])
    dominance = tf.math.count_nonzero(
        tf.logical_and(
            tf.reduce_all(extended <= swapped_ext, axis=2),
            tf.reduce_any(extended < swapped_ext, axis=2),
        ),
        axis=1,
    )

    return tf.boolean_mask(observations, dominance == 0), dominance


class BoundedVolumes:
    """
    A :class:`BoundedVolumes store the index of the Pareto front to form lower and upper
    bounds of the pseudo cells decomposition.
    """

    def __init__(self, lower_idx: tf.Tensor, upper_idx: tf.Tensor):
        """
        Construct bounded volumes.

        :param lower_idx: the lowerbounds index of the volumes
        :param upper_idx: the upperbounds index of the volumes
        """

        tf.debugging.assert_shapes([(lower_idx, ["N","D"]), (upper_idx, ["N","D"])])
        self.lower_idx = lower_idx
        self.upper_idx = upper_idx


class Pareto:
    """
    A :class:`Pareto` Construct a Pareto set.
    Stores a Pareto set and calculates the cell bounds covering the non-dominated region.
    The latter is needed for certain multiobjective acquisition functions.
    """

    def __init__(self, observations: TensorType):
        """
        :param observations: The observations for all objectives, with shape [N, 2].
        :raise ValueError (or InvalidArgumentError): If ``observations`` has an invalid shape.
        """
        tf.debugging.assert_shapes([(observations, [None, 2])])

        pf, _ = non_dominated(observations)
        self.front: Final[TensorType] = tf.gather_nd(pf, tf.argsort(pf[:, :1], axis=0))
        self.bounds: Final[BoundedVolumes] = self._bounds_2d(self.front)

    @staticmethod
    def _bounds_2d(front: TensorType) -> BoundedVolumes:
        """
        :param front: pareto front of current observations
        """
        # this assumes the Pareto set has been sorted in ascending order on the first
        # objective, which implies the second objective is sorted in descending order
        len_front, number_of_objectives = front.shape

        pf_ext_idx = tf.concat(
            [
                tf.zeros([1, number_of_objectives], dtype=tf.int32),
                tf.argsort(front, axis=0) + 1,
                tf.ones([1, number_of_objectives], dtype=tf.int32) * len_front + 1,
            ],
            axis=0,
        )

        range_ = tf.range(len_front + 1)[:, None]
        lower = tf.concat([range_, tf.zeros_like(range_)], axis=-1)
        upper = tf.concat([range_ + 1, pf_ext_idx[::-1, 1:][: pf_ext_idx[-1, 0]]], axis=-1)

        return BoundedVolumes(lower, upper)

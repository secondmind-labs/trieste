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
from __future__ import annotations

import tensorflow as tf
from typing import Callable
from typing_extensions import Final

from ..type import TensorType
from .misc import DEFAULTS
from trieste.utils.partition import divided_and_conqure, _BoundedVolumes


def non_dominated(observations: TensorType) -> tuple[TensorType, TensorType]:
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


class Pareto:
    """
    A :class:`Pareto` Construct a Pareto set.
    Stores a Pareto set and calculates the cell bounds covering the non-dominated region.
    The latter is needed for certain multiobjective acquisition functions.

    For hypervolume-based multiobjective optimisation with n>2 objectives, this class
    defaultly use branch and bound procedure algorithm. a divide and conquer method introduced
    in :cite:`Couckuyt2012`.
    """

    def __init__(self, observations: TensorType, *, partition: [str, Callable] = 'default',
                 jitter: float = DEFAULTS.JITTER):
        """
        :param observations: The observations for all objectives, with shape [N, D].
        :raise ValueError (or InvalidArgumentError): If ``observations`` has an invalid shape.
        """
        tf.debugging.assert_rank(observations, 2)
        tf.debugging.assert_greater_equal(tf.shape(observations)[-1], 2)
        self._partition = partition

        pfront, _ = non_dominated(observations)
        self.front: Final[TensorType] = tf.gather_nd(pfront, tf.argsort(pfront[:, :1], axis=0))
        self._bounds = self._get_bounds(self.front, jitter)

    def _get_bounds(self, front: TensorType, jitter: float) -> _BoundedVolumes:
        if self._partition == 'default':
            if front.shape[-1] > 2:
                return divided_and_conqure(front, jitter)
            else:
                return self._bounds_2d(front)
        elif isinstance(self._partition, Callable):
            return self._partition(front)
        else:
            raise TypeError (f' Specified partition method : {self._partition} not understood')

    @staticmethod
    def _bounds_2d(front: TensorType) -> _BoundedVolumes:
        # Compute the cells covering the non-dominated region for 2 dimension case
        # this assumes the Pareto set has been sorted in ascending order on the first
        # objective, which implies the second objective is sorted in descending order
        len_front, number_of_objectives = front.shape

        pseudo_front_idx = tf.concat(
            [
                tf.zeros([1, number_of_objectives], dtype=tf.int32),
                tf.argsort(front, axis=0) + 1,
                tf.ones([1, number_of_objectives], dtype=tf.int32) * len_front + 1,
            ],
            axis=0,
        )

        range_ = tf.range(len_front + 1)[:, None]
        lower_result = tf.concat([range_, tf.zeros_like(range_)], axis=-1)
        upper_result = tf.concat(
            [range_ + 1, pseudo_front_idx[::-1, 1:][: pseudo_front_idx[-1, 0]]], axis=-1
        )

        return _BoundedVolumes(lower_result, upper_result)

    def hypervolume_indicator(self, reference: TensorType) -> TensorType:
        """
        Calculate the hypervolume indicator
        The hypervolume indicator is the volume of the dominated region.

        :param reference: a reference point to use, with shape [D].
            Defines the upper bound of the hypervolume.
            Should be equal or bigger than the anti-ideal point of the Pareto set.
            For comparing results across runs, the same reference point must be used.
        :return: hypervolume indicator
        :raise ValueError (or `tf.errors.InvalidArgumentError`): If ``reference`` has an invalid
            shape.
        :raise `tf.errors.InvalidArgumentError`: If ``reference`` is less than the anti-ideal point
            in any dimension.
        """
        tf.debugging.assert_greater_equal(reference, self.front)

        tf.debugging.assert_shapes(
            [
                (self._bounds.lower_idx, ["N", "D"]),
                (self._bounds.upper_idx, ["N", "D"]),
                (self.front, ["M", "D"]),
                (reference, ["D"]),
            ]
        )

        min_front = tf.reduce_min(self.front, 0, keepdims=True)
        pseudo_front = tf.concat((min_front, self.front, reference[None]), 0)
        N, D = tf.shape(self._bounds.upper_idx)

        idx = tf.tile(tf.expand_dims(tf.range(D), -1), [1, N])
        upper_idx = tf.reshape(
            tf.stack([tf.transpose(self._bounds.upper_idx), idx], axis=2), [N * D, 2]
        )
        lower_idx = tf.reshape(
            tf.stack([tf.transpose(self._bounds.lower_idx), idx], axis=2), [N * D, 2]
        )
        upper = tf.reshape(tf.gather_nd(pseudo_front, upper_idx), [D, N])
        lower = tf.reshape(tf.gather_nd(pseudo_front, lower_idx), [D, N])
        hypervolume = tf.reduce_sum(tf.reduce_prod(upper - lower, 0))

        return tf.reduce_prod(reference[None] - min_front) - hypervolume

    def hypercell_bounds(
        self, anti_reference: TensorType, reference: TensorType
    ) -> tuple[TensorType, TensorType]:
        """
        Get the partitioned hypercell's lower and upper bounds.

        :param anti_reference: a worst point to use with shape [D].
            Defines the lower bound of the hypercell
        :param reference: a reference point to use, with shape [D].
            Defines the upper bound of the hypervolume.
            Should be equal to or bigger than the anti-ideal point of the Pareto set.
            For comparing results across runs, the same reference point must be used.
        :return: lower, upper bounds of the partitioned cell
        :raise ValueError (or `tf.errors.InvalidArgumentError`): If ``reference`` has an invalid
            shape.
        """
        tf.debugging.assert_greater_equal(reference, self.front)
        tf.debugging.assert_greater_equal(self.front, anti_reference)
        tf.debugging.assert_type(anti_reference, self.front.dtype)
        tf.debugging.assert_type(reference, self.front.dtype)

        tf.debugging.assert_shapes(
            [
                (self._bounds.lower_idx, ["N", "D"]),
                (self._bounds.upper_idx, ["N", "D"]),
                (self.front, ["M", "D"]),
                (reference, ["D"]),
                (anti_reference, ["D"]),
            ]
        )

        pseudo_pfront = tf.concat((anti_reference[None], self.front, reference[None]), axis=0)
        N = tf.shape(self._bounds.upper_idx)[0]
        D = tf.shape(self._bounds.upper_idx)[1]
        idx = tf.tile(tf.range(D), (N,))

        lower_idx = tf.stack((tf.reshape(self._bounds.lower_idx, [-1]), idx), axis=1)
        upper_idx = tf.stack((tf.reshape(self._bounds.upper_idx, [-1]), idx), axis=1)

        lower = tf.reshape(tf.gather_nd(pseudo_pfront, lower_idx), [N, D])
        upper = tf.reshape(tf.gather_nd(pseudo_pfront, upper_idx), [N, D])

        return lower, upper


def get_reference_point(front: TensorType) -> TensorType:
    """
    reference point calculation method
    """
    f = tf.math.reduce_max(front, axis=0) - tf.math.reduce_min(front, axis=0)
    return tf.math.reduce_max(front, axis=0) + 2 * f / front.shape[0]

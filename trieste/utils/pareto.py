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

from abc import ABC, abstractmethod

import tensorflow as tf
from typing import Callable

from ..type import TensorType
from trieste.utils.partition import _BoundedVolumes, ExactHvPartition2d, DividedAndConqure


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


def pareto_frontier(observations: TensorType) -> TensorType:
    """
    Get the pareto frontier from observations

    :param observations: set of points with shape [N,D]
    :return: tf.Tensor of the non-dominated set [P,D] and the degree of dominance [N],
        P is the number of points in pareto front
        dominances gives the number of dominating points for each data point

    """
    pfront, _ = non_dominated(observations)
    return tf.gather_nd(pfront, tf.argsort(pfront[:, :1], axis=0))


class _Pareto(ABC):
    """
    A :class:`_Pareto` prepare the necessary functionality for calculation of Pareto.
    """

    @abstractmethod
    def hypervolume_indicator(
            self, reference: TensorType) -> TensorType:
        """
        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :return: An acquisition function.
        """

    @abstractmethod
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


class Pareto(_Pareto):
    """
    A :class:`Pareto` Construct a Pareto set.
    Stores a Pareto set and calculates the cell bounds covering the non-dominated region.
    The latter is needed for certain multiobjective acquisition functions.

    For hypervolume-based multiobjective optimisation with n>2 objectives, this class
    defaultly use branch and bound procedure algorithm. a divide and conquer method introduced
    in :cite:`Couckuyt2012`.
    """

    def __init__(self, observations: TensorType, *, partition: [str, Callable] = 'default',
                 reference_point: [TensorType, None] = None):
        """
        :param observations: The observations for all objectives, with shape [N, D].
        :param partition: method of partitioning based on the (screened) pareto frontier
        :param reference_point: The reference point used to screen out not interested frontier in
          observations.

        :raise ValueError (or InvalidArgumentError): If ``observations`` has an invalid shape.
        """
        tf.debugging.assert_rank(observations, 2)
        tf.debugging.assert_greater_equal(tf.shape(observations)[-1], 2)
        self._partition = partition

        # get screened front according to sort of concentration: the 1st step
        if reference_point is None:
            pfront, _ = non_dominated(observations)
        else:  # screen possible not interested points
            screen_mask = tf.reduce_any(observations <= reference_point, -1)
            pfront, _ = non_dominated(observations[screen_mask])

        # get front from partition: in case any approximation has been made
        orded_pfront = tf.gather_nd(pfront, tf.argsort(pfront[:, :1], axis=0))
        self._bounds, self.front = self._get_partitioned_bounds(orded_pfront)

    def _get_partitioned_bounds(self, front: TensorType) -> _BoundedVolumes:
        if self._partition == 'default':
            if front.shape[-1] > 2:
                return DividedAndConqure()(front)
            else:
                return ExactHvPartition2d()(front)
        elif isinstance(self._partition, Callable):
            return self._partition(front)
        else:
            raise TypeError(f' Specified partition method : {self._partition} not understood')

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

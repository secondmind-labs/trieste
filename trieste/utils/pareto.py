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

from trieste.utils.mo_utils.partition import (
    DividedAndConquerNonDominated,
    DominatedPartition,
    ExactPartition2dNonDominated,
    NonDominatedPartition,
    Partition,
)

from ..type import TensorType
from .mo_utils.dominance import non_dominated


class Pareto:
    """
    A :class:`Pareto` Construct a Pareto set.
    Stores a Pareto set and calculates the cell bounds covering the non-dominated region.
    The latter is needed for certain multiobjective acquisition functions.

    For hypervolume-based multiobjective optimisation with n>2 objectives, this class
    defaultly use branch and bound procedure algorithm. a divide and conquer method introduced
    in :cite:`Couckuyt2012`.
    """

    def __init__(
        self,
        observations: TensorType,
        *,
        partition: [str, Partition] = "default",
        screen_concentration_point: [TensorType, None] = None,
    ):
        """
        :param observations: The observations for all objectives, with shape [N, D].
        :param partition: method of partitioning based on the (screened) pareto frontier
        :param screen_concentration_point: The reference point used to screen out not interested frontier in
          observations.

        :raise ValueError (or InvalidArgumentError): If ``observations`` has an invalid shape.
        """
        tf.debugging.assert_rank(observations, 2)
        tf.debugging.assert_greater_equal(tf.shape(observations)[-1], 2)

        # get screened front according to sort of concentration:
        if screen_concentration_point is None:
            screened_front, _ = non_dominated(observations)
        else:  # screen possible not interested points
            screen_mask = tf.reduce_all(observations <= screen_concentration_point, -1)
            screened_front, _ = non_dominated(observations[screen_mask])
        self.front = screened_front

        self._prepare_partition(partition)

    def _prepare_partition(self, partition_method: [str, Partition] = "default"):
        if partition_method == "default":
            if self.front.shape[-1] > 2:
                self._partition = DividedAndConquerNonDominated(self.front)
            elif self.front.shape[-1] == 2:
                self._partition = ExactPartition2dNonDominated(self.front)
        else:
            assert isinstance(partition_method, Partition), ValueError(
                "specified partition method must inherit from Partition abstract class but found"
            )
            self._partition = partition_method

    def hypervolume_indicator(self, reference: TensorType) -> TensorType:
        """
        Calculate the hypervolume indicator based on self.front and a reference point
        The hypervolume indicator is the volume of the dominated region.

        :param reference: a reference point to use, with shape [D].
            Defines the upper bound of the hypervolume.
            Should be equal or bigger than the anti-ideal point of the Pareto set.
            For comparing results across runs, the same reference point must be used.
        :return: hypervolume indicator, if reference point is less than all of the front in any dimension,
            the hypervolume indicator will be zero.
        :raise ValueError (or `tf.errors.InvalidArgumentError`): If ``reference`` has an invalid
            shape.
        :raise ValueError (or `tf.errors.InvalidArgumentError`): If ``self.front`` is empty (which can happen
        if the concentration point is too strict so no frontier exists after the screening)
        """
        if tf.equal(tf.size(self.front), 0):
            raise ValueError('empty front cannot be used to calculate hypervolume indicator')

        # FIXME: Figure out why dummy_anti_reference cnnot use original reduce min
        dummy_anti_reference = tf.reduce_min(self.front, axis=0) - tf.ones(
            shape=1, dtype=self.front.dtype
        )
        if isinstance(self._partition, DominatedPartition):
            lower, upper = self._partition.partition_bounds(dummy_anti_reference, reference)
            hypervolume_indicator = tf.reduce_sum(tf.reduce_prod(upper - lower, 1))
        elif isinstance(self._partition, NonDominatedPartition):
            lower, upper = self._partition.partition_bounds(dummy_anti_reference, reference)
            hypervolume = tf.reduce_sum(tf.reduce_prod(upper - lower, 1))
            hypervolume_indicator = tf.reduce_prod(reference - dummy_anti_reference) - hypervolume
        else:
            raise ValueError('partition strategy need to be either DominatedPartition or NonDominatedPartition')
        return hypervolume_indicator

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
        tf.debugging.assert_greater_equal(reference, anti_reference)
        tf.debugging.assert_type(anti_reference, reference.dtype)

        tf.debugging.assert_shapes(
            [
                (reference, ["D"]),
                (anti_reference, ["D"]),
            ]
        )

        if tf.equal(tf.size(self.front), 0):
            return anti_reference[None], reference[None]
        else:
            assert isinstance(self._partition, NonDominatedPartition)
            return self._partition.partition_bounds(anti_reference, reference)


# FIXME: ensure front is not empty
def get_reference_point(front: TensorType) -> TensorType:
    """
    reference point calculation method

    :raise ValueError : If ``front`` is empty
    """
    f = tf.math.reduce_max(front, axis=0) - tf.math.reduce_min(front, axis=0)
    return tf.math.reduce_max(front, axis=0) + 2 * f / front.shape[0]

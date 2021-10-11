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

from ...types import TensorType
from .dominance import non_dominated
from .partition import prepare_default_non_dominated_partition_bounds


class Pareto:
    """
    A :class:`Pareto` constructs a Pareto set.
    Stores a Pareto set and calculates hypervolume of the Pareto set given a
    specified reference point
    """

    def __init__(
        self,
        observations: TensorType,
    ):
        """
        :param observations: The observations for all objectives, with shape [N, D].
        :raise ValueError (or InvalidArgumentError): If ``observations`` has an invalid shape.
        """
        tf.debugging.assert_rank(observations, 2)
        tf.debugging.assert_greater_equal(tf.shape(observations)[-1], 2)

        self.front = non_dominated(observations)[0]

    def hypervolume_indicator(self, reference: TensorType) -> TensorType:
        """
        Calculate the hypervolume indicator based on self.front and a reference point
        The hypervolume indicator is the volume of the dominated region.

        :param reference: a reference point to use, with shape [D].
            Defines the upper bound of the hypervolume.
            Should be equal or bigger than the anti-ideal point of the Pareto set.
            For comparing results across runs, the same reference point must be used.
        :return: hypervolume indicator, if reference point is less than all of the front
            in any dimension, the hypervolume indicator will be zero.
        :raise ValueError (or `tf.errors.InvalidArgumentError`): If ``reference`` has an invalid
            shape.
        :raise ValueError (or `tf.errors.InvalidArgumentError`): If ``self.front`` is empty
            (which can happen if the concentration point is too strict so no frontier
            exists after the screening)
        """
        if tf.equal(tf.size(self.front), 0):
            raise ValueError("empty front cannot be used to calculate hypervolume indicator")

        helper_anti_reference = tf.reduce_min(self.front, axis=0) - tf.ones(
            shape=1, dtype=self.front.dtype
        )
        lower, upper = prepare_default_non_dominated_partition_bounds(
            reference, self.front, helper_anti_reference
        )
        non_dominated_hypervolume = tf.reduce_sum(tf.reduce_prod(upper - lower, 1))
        hypervolume_indicator = (
            tf.reduce_prod(reference - helper_anti_reference) - non_dominated_hypervolume
        )
        return hypervolume_indicator


def get_reference_point(front: TensorType) -> TensorType:
    """
    reference point calculation method. Note if the front only contains one point, this reference
    point calculation method will result a reference point the same as the input.

    :param front: Pareto front referred to calculate the reference point, with shape [..., N, D]
    :return: a reference point to use, with shape [..., D].
    :raise ValueError: If ``front`` is empty
    """
    if tf.equal(tf.size(front), 0):
        raise ValueError("empty front cannot be used to calculate reference point")

    f = tf.math.reduce_max(front, axis=-2) - tf.math.reduce_min(front, axis=-2)
    return tf.math.reduce_max(front, axis=-2) + 2 * f / tf.cast(tf.shape(front)[-2], f.dtype)

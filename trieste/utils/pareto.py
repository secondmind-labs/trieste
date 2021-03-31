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

import gpflow
import tensorflow as tf
from typing_extensions import Final

from ..type import TensorType


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
    A :class:`BoundedVolumes` store the index of the Pareto front to form lower and upper
    bounds of the pseudo cells decomposition.
    """

    def __init__(self, lower_idx: tf.Tensor, upper_idx: tf.Tensor):
        """
        Construct bounded volumes.

        :param lower_idx: the lowerbounds index of the volumes
        :param upper_idx: the upperbounds index of the volumes
        """

        tf.debugging.assert_shapes([(lower_idx, ["N", "D"]), (upper_idx, ["N", "D"])])
        self.lower_idx: Final[TensorType] = lower_idx
        self.upper_idx: Final[TensorType] = upper_idx


class Pareto:
    """
    A :class:`Pareto` Construct a Pareto set.
    Stores a Pareto set and calculates the cell bounds covering the non-dominated region.
    The latter is needed for certain multiobjective acquisition functions.
    """

    def __init__(self, observations: TensorType):
        """
        :param observations: The observations for all objectives, with shape [N, D].
        :raise ValueError (or InvalidArgumentError): If ``observations`` has an invalid shape.
        """
        tf.debugging.assert_rank(observations, 2)

        pfront, _ = non_dominated(observations)
        self.front: Final[TensorType] = tf.gather_nd(pfront, tf.argsort(pfront[:, :1], axis=0))
        self.bounds = self._get_bounds(self.front)

    def _get_bounds(self, front: TensorType) -> BoundedVolumes:
        if front.shape[-1] > 2:
            return self._divide_conquer_nd(front)
        else:
            return self._bounds_2d(front)

    @staticmethod
    def _bounds_2d(front: TensorType) -> BoundedVolumes:

        # This assumes the Pareto set has been sorted in ascending order on the first
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

        return BoundedVolumes(lower_result, upper_result)

    @staticmethod
    def _is_test_required(smaller: TensorType) -> TensorType:
        idx_dom_augm = tf.reduce_any(smaller, axis=1)
        is_dom_augm = tf.reduce_all(idx_dom_augm)

        return is_dom_augm

    def _divide_conquer_nd(
        self, front: TensorType, threshold: TensorType | float = 0
    ) -> TensorType:
        # Divide and conquer strategy to compute the cells covering the non-dominated region.
        # Generic version: works for an arbitrary number of objectives.

        len_front, number_of_objectives = front.shape
        jitter = gpflow.config.default_jitter()
        lower_result = tf.zeros([0, number_of_objectives], dtype=tf.int32)
        upper_result = tf.zeros([0, number_of_objectives], dtype=tf.int32)

        min_front = tf.reduce_min(front, axis=0, keepdims=True) - 1
        max_front = tf.reduce_max(front, axis=0, keepdims=True) + 1
        pseudo_front = tf.concat([min_front, front, max_front], axis=0)

        pseudo_front_idx = tf.concat(
            [
                tf.zeros([1, number_of_objectives], dtype=tf.int32),
                tf.argsort(front, axis=0) + 1,  # +1 as index zero is reserved for the ideal point
                tf.ones([1, number_of_objectives], dtype=tf.int32) * len_front + 1,
            ],
            axis=0,
        )

        divide_conquer_cells = tf.stack(
            [
                tf.zeros(number_of_objectives, dtype=tf.int32),
                (int(pseudo_front_idx.shape[0]) - 1)
                * tf.ones(number_of_objectives, dtype=tf.int32),
            ],
            axis=0,
        )[None]

        total_size = tf.reduce_prod(max_front - min_front)

        def while_body(
            divide_conquer_cells: TensorType,
            lower_result: TensorType,
            upper_result: TensorType,
        ) -> Tuple[TensorType, TensorType, TensorType]:
            divide_conquer_cells = tf.unstack(divide_conquer_cells, axis=0)
            cell = divide_conquer_cells[-1]
            divide_conquer_cells = tf.cond(
                tf.not_equal(tf.size(divide_conquer_cells[:-1]), 0),
                lambda: tf.stack(divide_conquer_cells[:-1]),
                lambda: tf.zeros([0, 2, number_of_objectives], dtype=tf.int32),
            )

            arr = tf.range(number_of_objectives)
            lower_idx = tf.gather_nd(pseudo_front_idx, tf.stack((cell[0], arr), -1))
            upper_idx = tf.gather_nd(pseudo_front_idx, tf.stack((cell[1], arr), -1))
            lower = tf.gather_nd(pseudo_front, tf.stack((lower_idx, arr), -1))
            upper = tf.gather_nd(pseudo_front, tf.stack((upper_idx, arr), -1))

            test_accepted = self._is_test_required((upper - jitter) < front)
            lower_result, upper_result = tf.cond(
                test_accepted,
                lambda: self._accepted_test_body(lower_result, upper_result, lower_idx, upper_idx),
                lambda: (lower_result, upper_result),
            )

            test_rejected = self._is_test_required((lower + jitter) < front)
            divide_conquer_cells = tf.cond(
                tf.logical_and(test_rejected, tf.logical_not(test_accepted)),
                lambda: self._rejected_test_body(
                    cell, lower, upper, divide_conquer_cells, total_size, threshold
                ),
                lambda: divide_conquer_cells,
            )

            return divide_conquer_cells, lower_result, upper_result

        _, lower_result, upper_result = tf.while_loop(
            lambda divide_conquer_cells, lower_result, upper_result: len(divide_conquer_cells) > 0,
            while_body,
            loop_vars=[divide_conquer_cells, lower_result, upper_result],
            shape_invariants=[
                tf.TensorShape([None, 2, number_of_objectives]),
                tf.TensorShape([None, number_of_objectives]),
                tf.TensorShape([None, number_of_objectives]),
            ],
        )
        return BoundedVolumes(lower_result, upper_result)

    def _accepted_test_body(
        self,
        lower_result: TensorType,
        upper_result: TensorType,
        lower_idx: TensorType,
        upper_idx: TensorType,
    ) -> Tuple[TensorType, TensorType]:
        lower_result = tf.concat([lower_result, lower_idx[None]], axis=0)
        upper_result = tf.concat([upper_result, upper_idx[None]], axis=0)
        return lower_result, upper_result

    def _rejected_test_body(
        self,
        cell: TensorType,
        lower: TensorType,
        upper: TensorType,
        divide_conquer_cells: TensorType,
        total_size: TensorType,
        threshold: TensorType,
    ) -> TensorType:

        divide_conquer_cells_dist = cell[1] - cell[0]
        hc_size = tf.math.reduce_prod(upper - lower, axis=0, keepdims=True)

        not_unit_cell = tf.reduce_any(divide_conquer_cells_dist > 1)
        vol_above_thresh = tf.reduce_all((hc_size[0] / total_size) > threshold)
        divide_conquer_cells = tf.cond(
            tf.logical_and(not_unit_cell, vol_above_thresh),
            lambda: self._divide_body(divide_conquer_cells, divide_conquer_cells_dist, cell),
            lambda: tf.identity(divide_conquer_cells),
        )
        return divide_conquer_cells

    def _divide_body(
        self,
        divide_conquer_cells: TensorType,
        divide_conquer_cells_dist: TensorType,
        cell: TensorType,
    ) -> TensorType:
        edge_size = tf.reduce_max(divide_conquer_cells_dist)
        idx = tf.argmax(divide_conquer_cells_dist)
        edge_size1 = int(tf.round(tf.cast(edge_size, dtype=tf.float32) / 2.0))
        edge_size2 = edge_size - edge_size1

        upper = tf.unstack(tf.identity(cell[1]))
        upper[idx] = upper[idx] - edge_size1
        upper = tf.stack(upper)
        divide_conquer_cells = tf.concat(
            [divide_conquer_cells, tf.stack([tf.identity(cell[0]), upper], axis=0)[None]], axis=0
        )
        lower = tf.unstack(tf.identity(cell[0]))
        lower[idx] = lower[idx] + edge_size2
        lower = tf.stack(lower)
        divide_conquer_cells = tf.concat(
            [divide_conquer_cells, tf.stack([lower, tf.identity(cell[1])], axis=0)[None]], axis=0
        )
        return divide_conquer_cells

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
                (self.bounds.lower_idx, ["N", "D"]),
                (self.bounds.upper_idx, ["N", "D"]),
                (self.front, ["M", "D"]),
                (reference, ["D"]),
            ]
        )

        min_front = tf.reduce_min(self.front, 0, keepdims=True)
        pseudo_front = tf.concat((min_front, self.front, reference[None]), 0)
        N, D = tf.shape(self.bounds.upper_idx)

        idx = tf.tile(tf.expand_dims(tf.range(D), -1), [1, N])
        upper_idx = tf.reshape(
            tf.stack([tf.transpose(self.bounds.upper_idx), idx], axis=2), [N * D, 2]
        )
        lower_idx = tf.reshape(
            tf.stack([tf.transpose(self.bounds.lower_idx), idx], axis=2), [N * D, 2]
        )
        upper = tf.reshape(tf.gather_nd(pseudo_front, upper_idx), [D, N])
        lower = tf.reshape(tf.gather_nd(pseudo_front, lower_idx), [D, N])
        hypervolume = tf.reduce_sum(tf.reduce_prod(upper - lower, 0))

        return tf.reduce_prod(reference[None] - min_front) - hypervolume

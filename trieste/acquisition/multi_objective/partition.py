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
"""This module contains functions of different methods for
partitioning the dominated/non-dominated region in multi-objective optimization problems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import tensorflow as tf

from ...types import TensorType
from ...utils.misc import DEFAULTS
from .dominance import non_dominated


def prepare_default_non_dominated_partition_bounds(
    reference: TensorType,
    observations: Optional[TensorType] = None,
    anti_reference: Optional[TensorType] = None,
) -> tuple[TensorType, TensorType]:
    """
    Prepare the default non-dominated partition boundary for acquisition function usage.
    This functionality will trigger different partition according to objective numbers, if
    objective number is 2, an `ExactPartition2dNonDominated` will be used. If the objective
    number is larger than 2, a `DividedAndConquerNonDominated` will be used.

    :param observations: The observations for all objectives, with shape [N, D], if not specified
        or is an empty Tensor, a single non-dominated partition bounds constructed by reference
        and anti_reference point will be returned.
    :param anti_reference: a worst point to use with shape [D].
        Defines the lower bound of the hypercell. If not specified, will use a default value:
        -[1e10] * D.
    :param reference: a reference point to use, with shape [D].
        Defines the upper bound of the hypervolume.
        Should be equal to or bigger than the anti-ideal point of the Pareto set.
        For comparing results across runs, the same reference point must be used.
    :return: lower, upper bounds of the partitioned cell, each with shape [N, D]
    :raise ValueError (or `tf.errors.InvalidArgumentError`): If ``reference`` has an invalid
        shape.
    :raise ValueError (or `tf.errors.InvalidArgumentError`): If ``anti_reference`` has an invalid
        shape.
    """

    def is_empty_obs(obs: Optional[TensorType]) -> bool:
        return obs is None or tf.equal(tf.size(observations), 0)

    def specify_default_anti_reference_point(
        ref: TensorType, obs: Optional[TensorType]
    ) -> TensorType:
        anti_ref = -1e10 * tf.ones(shape=(tf.shape(reference)), dtype=reference.dtype)
        tf.debugging.assert_greater_equal(
            ref,
            anti_ref,
            message=f"reference point: {ref} containing at least one value below default "
            "anti-reference point ([-1e10, ..., -1e10]), try specify a lower "
            "anti-reference point.",
        )
        if not is_empty_obs(obs):  # make sure given (valid) observations are larger than -1e10
            tf.debugging.assert_greater_equal(
                obs,
                anti_ref,
                message=f"observations: {obs} containing at least one value below default "
                "anti-reference point ([-1e10, ..., -1e10]), try specify a lower "
                "anti-reference point.",
            )
        return anti_ref

    tf.debugging.assert_shapes([(reference, ["D"])])
    if anti_reference is None:
        # if anti_reference point is not specified, use a -1e10 as default (act as -inf)
        anti_reference = specify_default_anti_reference_point(reference, observations)
    else:
        # anti_reference point is specified
        tf.debugging.assert_shapes([(anti_reference, ["D"])])

    if is_empty_obs(observations):  # if no valid observations
        assert tf.reduce_all(tf.less_equal(anti_reference, reference)), ValueError(
            f"anti_reference point: {anti_reference} contains at least one value larger "
            f"than reference point: {reference}"
        )
        return tf.expand_dims(anti_reference, 0), tf.expand_dims(reference, 0)
    elif tf.shape(observations)[-1] > 2:
        return DividedAndConquerNonDominated(observations).partition_bounds(
            anti_reference, reference
        )
    else:
        return ExactPartition2dNonDominated(observations).partition_bounds(
            anti_reference, reference
        )


@dataclass(frozen=True)
class _BoundedVolumes:
    # stores the index of the Pareto front to form lower and upper
    # bounds of the pseudo cells decomposition.

    # the lowerbounds index of the volumes
    lower_idx: TensorType

    # the upperbounds index of the volumes
    upper_idx: TensorType

    def __post_init__(self) -> None:
        tf.debugging.assert_shapes([(self.lower_idx, ["N", "D"]), (self.upper_idx, ["N", "D"])])


class _BoundIndexPartition:
    """
    A collection of partition strategies that are based on storing the index of pareto fronts
    & other auxiliary points
    """

    front: TensorType
    _bounds: _BoundedVolumes

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls is _BoundIndexPartition:
            raise TypeError("BoundIndexPartition may not be instantiated directly")
        return object.__new__(cls)

    def partition_bounds(
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
        :return: lower, upper bounds of the partitioned cell, each with shape [N, D]
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

        # concatenate the pseudo front to have the same corresponding of bound index
        pseudo_pfront = tf.concat((anti_reference[None], self.front, reference[None]), axis=0)
        N = tf.shape(self._bounds.upper_idx)[0]
        D = tf.shape(self._bounds.upper_idx)[1]
        idx = tf.tile(tf.range(D), (N,))

        lower_idx = tf.stack((tf.reshape(self._bounds.lower_idx, [-1]), idx), axis=1)
        upper_idx = tf.stack((tf.reshape(self._bounds.upper_idx, [-1]), idx), axis=1)

        lower = tf.reshape(tf.gather_nd(pseudo_pfront, lower_idx), [N, D])
        upper = tf.reshape(tf.gather_nd(pseudo_pfront, upper_idx), [N, D])

        return lower, upper


class ExactPartition2dNonDominated(_BoundIndexPartition):
    """
    Exact partition of non-dominated space, used as a default option when the
    objective number equals 2.
    """

    def __init__(self, front: TensorType):
        """
        :param front: non-dominated pareto front.
        """
        tf.debugging.assert_equal(
            tf.reduce_all(non_dominated(front)[1]),
            True,
            message=f"\ninput {front} contains dominated points",
        )
        self.front = tf.gather_nd(front, tf.argsort(front[:, :1], axis=0))  # sort input front
        self._bounds = self._get_bound_index()

    def _get_bound_index(self) -> _BoundedVolumes:
        # Compute the cells covering the non-dominated region for 2 dimension case
        # this assumes the Pareto set has been sorted in ascending order on the first
        # objective, which implies the second objective is sorted in descending order
        len_front, number_of_objectives = self.front.shape

        pseudo_front_idx = tf.concat(
            [
                tf.zeros([1, number_of_objectives], dtype=tf.int32),
                tf.argsort(self.front, axis=0) + 1,
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


class DividedAndConquerNonDominated(_BoundIndexPartition):
    """
    branch and bound procedure algorithm. a divide and conquer method introduced
    in :cite:`Couckuyt2012`.
    """

    def __init__(self, front: TensorType, threshold: TensorType | float = 0):
        """
        :param front: non-dominated pareto front.
        :param threshold: a threshold used to screen out cells in partition : when its volume is
            below this threshold, its rejected directly in order to be more computationally
            efficient, if setting above 0, this partition strategy tends to return an
            approximated partition.
        """
        tf.debugging.assert_equal(
            tf.reduce_all(non_dominated(front)[1]),
            True,
            message=f"\ninput {front} contains dominated points",
        )
        self.front = tf.gather_nd(front, tf.argsort(front[:, :1], axis=0))  # sort
        self.front = front
        self._bounds = self._get_bound_index(threshold)

    def _get_bound_index(self, threshold: TensorType | float = 0) -> _BoundedVolumes:
        len_front, number_of_objectives = self.front.shape
        lower_result = tf.zeros([0, number_of_objectives], dtype=tf.int32)
        upper_result = tf.zeros([0, number_of_objectives], dtype=tf.int32)

        min_front = tf.reduce_min(self.front, axis=0, keepdims=True) - 1
        max_front = tf.reduce_max(self.front, axis=0, keepdims=True) + 1
        pseudo_front = tf.concat([min_front, self.front, max_front], axis=0)

        pseudo_front_idx = tf.concat(
            [
                tf.zeros([1, number_of_objectives], dtype=tf.int32),
                tf.argsort(self.front, axis=0)
                + 1,  # +1 as index zero is reserved for the ideal point
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
        ) -> tuple[TensorType, TensorType, TensorType]:
            divide_conquer_cells_unstacked = tf.unstack(divide_conquer_cells, axis=0)
            cell = divide_conquer_cells_unstacked[-1]
            divide_conquer_cells_new = tf.cond(
                tf.not_equal(tf.size(divide_conquer_cells_unstacked[:-1]), 0),
                lambda: tf.stack(divide_conquer_cells_unstacked[:-1]),
                lambda: tf.zeros([0, 2, number_of_objectives], dtype=tf.int32),
            )

            arr = tf.range(number_of_objectives)
            lower_idx = tf.gather_nd(pseudo_front_idx, tf.stack((cell[0], arr), -1))
            upper_idx = tf.gather_nd(pseudo_front_idx, tf.stack((cell[1], arr), -1))
            lower = tf.gather_nd(pseudo_front, tf.stack((lower_idx, arr), -1))
            upper = tf.gather_nd(pseudo_front, tf.stack((upper_idx, arr), -1))

            test_accepted = self._is_test_required((upper - DEFAULTS.JITTER) < self.front)
            lower_result_final, upper_result_final = tf.cond(
                test_accepted,
                lambda: self._accepted_test_body(lower_result, upper_result, lower_idx, upper_idx),
                lambda: (lower_result, upper_result),
            )

            test_rejected = self._is_test_required((lower + DEFAULTS.JITTER) < self.front)
            divide_conquer_cells_final = tf.cond(
                tf.logical_and(test_rejected, tf.logical_not(test_accepted)),
                lambda: self._rejected_test_body(
                    cell, lower, upper, divide_conquer_cells_new, total_size, threshold
                ),
                lambda: divide_conquer_cells_new,
            )

            return divide_conquer_cells_final, lower_result_final, upper_result_final

        _, lower_result_final, upper_result_final = tf.while_loop(
            lambda divide_conquer_cells, lower_result, upper_result: len(divide_conquer_cells) > 0,
            while_body,
            loop_vars=[divide_conquer_cells, lower_result, upper_result],
            shape_invariants=[
                tf.TensorShape([None, 2, number_of_objectives]),
                tf.TensorShape([None, number_of_objectives]),
                tf.TensorShape([None, number_of_objectives]),
            ],
        )
        return _BoundedVolumes(lower_result_final, upper_result_final)

    @staticmethod
    def _is_test_required(smaller: TensorType) -> TensorType:
        idx_dom_augm = tf.reduce_any(smaller, axis=1)
        is_dom_augm = tf.reduce_all(idx_dom_augm)

        return is_dom_augm

    @staticmethod
    def _accepted_test_body(
        lower_result: TensorType,
        upper_result: TensorType,
        lower_idx: TensorType,
        upper_idx: TensorType,
    ) -> tuple[TensorType, TensorType]:
        lower_result_accepted = tf.concat([lower_result, lower_idx[None]], axis=0)
        upper_result_accepted = tf.concat([upper_result, upper_idx[None]], axis=0)
        return lower_result_accepted, upper_result_accepted

    @classmethod
    def _rejected_test_body(
        cls,
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
        divide_conquer_cells_rejected = tf.cond(
            tf.logical_and(not_unit_cell, vol_above_thresh),
            lambda: cls._divide_body(divide_conquer_cells, divide_conquer_cells_dist, cell),
            lambda: tf.identity(divide_conquer_cells),
        )
        return divide_conquer_cells_rejected

    @staticmethod
    def _divide_body(
        divide_conquer_cells: TensorType,
        divide_conquer_cells_dist: TensorType,
        cell: TensorType,
    ) -> TensorType:
        edge_size = tf.reduce_max(divide_conquer_cells_dist)
        idx = tf.argmax(divide_conquer_cells_dist)
        edge_size1 = int(tf.round(tf.cast(edge_size, dtype=tf.float32) / 2.0))
        edge_size2 = int(edge_size - edge_size1)

        sparse_edge_size1 = tf.concat(
            [tf.zeros([idx]), edge_size1 * tf.ones([1]), tf.zeros([len(cell[1]) - idx - 1])], axis=0
        )
        upper = tf.identity(cell[1]) - tf.cast(sparse_edge_size1, dtype=tf.int32)

        divide_conquer_cells_new = tf.concat(
            [divide_conquer_cells, tf.stack([tf.identity(cell[0]), upper], axis=0)[None]], axis=0
        )

        sparse_edge_size2 = tf.concat(
            [tf.zeros([idx]), edge_size2 * tf.ones([1]), tf.zeros([len(cell[1]) - idx - 1])], axis=0
        )
        lower = tf.identity(cell[0]) + tf.cast(sparse_edge_size2, dtype=tf.int32)

        divide_conquer_cells_final = tf.concat(
            [divide_conquer_cells_new, tf.stack([lower, tf.identity(cell[1])], axis=0)[None]],
            axis=0,
        )

        return divide_conquer_cells_final

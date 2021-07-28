"""This module contains functions of different methods for partitioning the dominated/non-dominated
 region in multi-objective optimization, assuming a front is given upfront """
from __future__ import annotations

import tensorflow as tf
from ..type import TensorType
from .misc import DEFAULTS
from typing_extensions import Final
from abc import ABC
import numpy as np


class _BoundedVolumes:
    """
    A :class:`_BoundedVolumes` store the index of the Pareto front to form lower and upper
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


class _Partition(ABC):
    """
    Base class of partition
    """


class _NonDominatedPartition(_Partition):
    """
    Partition methods focusing on partitioning non-dominated regions
    """


class _DominatedPartition(_Partition):
    """
    Partition methods focusing on partitioning dominated-regions
    """


class ExactHvPartition2d(_Partition):
    def __call__(self, front: TensorType):
        return exact_hv_partition_2d(front), front


def exact_hv_partition_2d(front: TensorType):

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


class DividedAndConquerNonDominated(_Partition):
    def __call__(self, front: TensorType, jitter: float = DEFAULTS.JITTER,
                 threshold: TensorType | float = 0
    ):
        return divided_and_conquer_non_dominated(front, jitter, threshold), front


# TODO: Add reference point
def divided_and_conquer_non_dominated(front: TensorType, jitter: float = DEFAULTS.JITTER, threshold: TensorType | float = 0
                                      ) -> _BoundedVolumes:
    """
    Branch and bound procedure to compute the cells covering the non-dominated region.
    Generic version: works for an arbitrary number of objectives.

    :param front
    :param jitter
    :param threshold
    """

    def _is_test_required(smaller: TensorType) -> TensorType:
        idx_dom_augm = tf.reduce_any(smaller, axis=1)
        is_dom_augm = tf.reduce_all(idx_dom_augm)

        return is_dom_augm

    def _accepted_test_body(
        lower_result: TensorType,
        upper_result: TensorType,
        lower_idx: TensorType,
        upper_idx: TensorType,
    ) -> tuple[TensorType, TensorType]:
        lower_result_accepted = tf.concat([lower_result, lower_idx[None]], axis=0)
        upper_result_accepted = tf.concat([upper_result, upper_idx[None]], axis=0)
        return lower_result_accepted, upper_result_accepted

    def _rejected_test_body(
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
            lambda: _divide_body(divide_conquer_cells, divide_conquer_cells_dist, cell),
            lambda: tf.identity(divide_conquer_cells),
        )
        return divide_conquer_cells_rejected

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

    len_front, number_of_objectives = front.shape
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

        test_accepted = _is_test_required((upper - jitter) < front)
        lower_result_final, upper_result_final = tf.cond(
            test_accepted,
            lambda: _accepted_test_body(lower_result, upper_result, lower_idx, upper_idx),
            lambda: (lower_result, upper_result),
        )

        test_rejected = _is_test_required((lower + jitter) < front)
        divide_conquer_cells_final = tf.cond(
            tf.logical_and(test_rejected, tf.logical_not(test_accepted)),
            lambda: _rejected_test_body(
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


class HypervolumeBoxDecompositionNonIncremental(_DominatedPartition):
    """
    A method of partitioning the dominated region.

    The main idea is of using a sort of auxiliary points assosiating to existing Pareto points to
    describe the Pareto frontier, (which is referred to as local upper bounds in the original context)
    then, we could use an alternative partition as an replacement of original partition.

    Main reference: Section 2.2.2 of lacour2017box
    Assumptions
    One of the assumption made here is no any points have the same value in any dimension

    An implementation of
    Also noted as LKF17
    """


class CoperatePartition(_NonDominatedPartition):
    """
    Main refer Algorithm 2, Algorithm 3 of yang2019efficient
    as well as https://github.com/pytorch/botorch/blob/ddc97bb4dc5de8ea1cf0f0d11519b826f6a56868/botorch/utils/multi_objective/box_decompositions/non_dominated.py#L371
    """


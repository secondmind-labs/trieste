from __future__ import annotations

import tensorflow as tf
from ..type import TensorType
from typing_extensions import Final
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


# TODO: Add reference point
def divided_and_conqure(front: TensorType, jitter: float, threshold: TensorType | float = 0
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


def divided_and_conqure_numpy_ver(front: TensorType, jitter: float, threshold: TensorType | float = 0
    ) -> _BoundedVolumes:
    """
    Divide and conquer strategy to compute the cells covering the non-dominated region.
    Generic version: works for an arbitrary number of objectives.
    """
    outdim = front.shape[1]
    # The divide and conquer algorithm operates on a pseudo Pareto set
    # that is a mapping of the real Pareto set to discrete values
    pseudo_pf = np.argsort(front, axis=0) + 1  # +1 as index zero is reserved for the ideal point
    # Extend front with the ideal and anti-ideal point
    min_pf = np.min(front, axis=0) - 1
    max_pf = np.max(front, axis=0) + 1
    pf_ext = np.vstack((min_pf, front, max_pf))  # Needed for early stopping check (threshold)
    pf_ext_idx = np.vstack((np.zeros(outdim, dtype=np_int_type),
                            pseudo_pf,
                            np.ones(outdim, dtype=np_int_type) * front.shape[0] + 1))
    # Start with one cell covering the whole front
    dc = [(np.zeros(outdim, dtype=np_int_type),
           (int(pf_ext_idx.shape[0]) - 1) * np.ones(outdim, dtype=np_int_type))]
    total_size = np.prod(max_pf - min_pf)
    # Start divide and conquer until we processed all cells
    while dc:
        # Process test cell
        cell = dc.pop()
        arr = np.arange(outdim)
        lb = pf_ext[pf_ext_idx[cell[0], arr], arr]
        ub = pf_ext[pf_ext_idx[cell[1], arr], arr]
        # Acceptance test:
        if self._is_test_required((ub - jitter) < front):
            # Cell is a valid integral bound: store
            self.bounds.append(pf_ext_idx[cell[0], np.arange(outdim)],
                               pf_ext_idx[cell[1], np.arange(outdim)])
        # Reject test:
        elif self._is_test_required((lb + jitter) < front):
            # Cell can not be discarded: calculate the size of the cell
            dc_dist = cell[1] - cell[0]
            hc = _BoundedVolumes(pf_ext[pf_ext_idx[cell[0], np.arange(outdim)], np.arange(outdim)],
                                pf_ext[pf_ext_idx[cell[1], np.arange(outdim)], np.arange(outdim)])
            # Only divide when it is not an unit cell and the volume is above the approx. threshold
            if np.any(dc_dist > 1) and np.all((hc.size()[0] / total_size) > self.threshold):
                # Divide the test cell over its largest dimension
                edge_size, idx = np.max(dc_dist), np.argmax(dc_dist)
                edge_size1 = int(np.round(edge_size / 2.0))
                edge_size2 = edge_size - edge_size1
                # Store divided cells
                ub = np.copy(cell[1])
                ub[idx] -= edge_size1
                dc.append((np.copy(cell[0]), ub))
                lb = np.copy(cell[0])
                lb[idx] += edge_size2
                dc.append((lb, np.copy(cell[1])))
        # else: cell can be discarded


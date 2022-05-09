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
            tf.cast(tf.reduce_sum(tf.abs(non_dominated(front)[1])), dtype=front.dtype),
            tf.zeros(shape=1, dtype=front.dtype),
            message=f"\ninput {front} " f"contains dominated points",
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
            tf.cast(tf.reduce_sum(tf.abs(non_dominated(front)[1])), dtype=front.dtype),
            tf.zeros(shape=1, dtype=front.dtype),
            message=f"\ninput {front} " f"contains dominated points",
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


class HypervolumeBoxDecompositionIncrementalDominated:
    """
    A Hypervolume Box Decomposition Algorithm (incremental version) (HBDA in short).
    The main idea is of using a set auxiliary points (which is referred to as local
    upper bound set in the paper) associating to existing Pareto points to describe the
    dominated region and to decompose it to disjoint hyper rectangles. Main reference:
    Section 2.2.2 of :cite:`lacour2017box`.
    """

    def __init__(
        self,
        observations: TensorType,
        reference_point: TensorType,
        dummy_anti_ref_value: float = -1e10,
    ):
        """
        :param observations: the objective observations, preferably this can be a
            non-dominated set, but any set is acceptable here.
        :param reference_point: a reference point to use, with shape [D]
            (same as p in the paper). Defines the upper bound of the hypervolume.
        :param dummy_anti_ref_value: float, a value used to setup the dummy anti-reference point:
            _dummy_anti_reference_point = [dummy_anti_ref_value, ..., dummy_anti_ref_value].
            This anti-reference point will not affect the partitioned bounds, but it is required
            to be smaller than any (potential) observations
        """
        tf.debugging.assert_type(reference_point, observations.dtype)
        tf.debugging.assert_shapes([(reference_point, ["D"])])

        tf.debugging.assert_greater_equal(reference_point, observations)
        tf.debugging.assert_greater_equal(
            reference_point,
            observations,
            message=f"reference_point: {reference_point} containing points below "
            f"observations:\n {observations} ",
        )

        self._dummy_anti_reference_point = dummy_anti_ref_value * tf.ones(
            (1, observations.shape[-1]), dtype=observations.dtype
        )
        self._reference_point = reference_point

        # making sure objective space has been lower bounded by [-1e10, ..., -1e10]
        tf.debugging.assert_greater_equal(
            observations,
            self._dummy_anti_reference_point,
            f"observations: {observations} contains value smaller than dummy "
            f"anti-reference point: {self._dummy_anti_reference_point}",
        )
        # initialize local upper bound set using reference point (step 1 of Alg 1)
        self.U_set = reference_point[tf.newaxis]

        self.Z_set = (  # initialize local upper bound's defining points _Z to be the dummy
            # points \hat{z} (defined in Sec 2.1). Note: 1. the original defined objective space
            # [0, reference_point] has been replaced by [-1e10, reference_point].  2. the dummy
            # anti reference point value will not affect lower/upper bounds of this dominated
            # partition method
            dummy_anti_ref_value
            * tf.ones((1, observations.shape[-1], observations.shape[-1]), dtype=observations.dtype)
            - dummy_anti_ref_value
            * tf.eye(
                observations.shape[-1],
                observations.shape[-1],
                batch_shape=[1],
                dtype=observations.dtype,
            )
            + tf.linalg.diag(reference_point)[tf.newaxis, ...]
        )  # [1, D, D]

        (
            self.U_set,
            self.Z_set,
        ) = _update_local_upper_bounds_incremental(  # incrementally update local upper
            new_observations=observations,  # bounds and defining points for each new observation
            u_set=self.U_set,
            z_set=self.Z_set,
        )

    def update(self, new_obs: TensorType) -> None:
        """
        Update with new observations, this can be computed with the incremental method

        :param new_obs with shape [N, D]
        """
        tf.debugging.assert_greater_equal(self._reference_point, new_obs)
        tf.debugging.assert_greater_equal(new_obs, self._dummy_anti_reference_point)
        (
            self.U_set,
            self.Z_set,
        ) = _update_local_upper_bounds_incremental(  # incrementally update local upper
            new_observations=new_obs,  # bounds and defining points for each new Pareto point
            u_set=self.U_set,
            z_set=self.Z_set,
        )

    def partition_bounds(self) -> tuple[TensorType, TensorType]:
        return _get_partition_bounds_hbda(self.Z_set, self.U_set, self._reference_point)


def _update_local_upper_bounds_incremental(
    new_observations: TensorType, u_set: TensorType, z_set: TensorType
) -> tuple[TensorType, TensorType]:
    r"""Update the current local upper bound with the new pareto points. (note:
    this does not require the input: new_front_points must be non-dominated points)

    :param new_observations: with shape [N, D].
    :param u_set: with shape [N', D], the set containing all the existing local upper bounds.
    :param z_set: with shape [N', D, D] contain the existing local upper bounds defining points,
        note the meaning of the two D is different: first D denotes for any element
        in u_set, it has D defining points; the second D denotes each defining point is
        D dimensional.
    :return: a new [N'', D] new local upper bounds set.
             a new [N'', D, D] contain the new local upper bounds defining points
    """

    tf.debugging.assert_shapes([(new_observations, ["N", "D"])])
    for new_obs in new_observations:  # incrementally update local upper bounds
        u_set, z_set = _compute_new_local_upper_bounds(u_set, z_set, z_bar=new_obs)
    return u_set, z_set


def _compute_new_local_upper_bounds(
    u_set: TensorType, z_set: TensorType, z_bar: TensorType
) -> tuple[TensorType, TensorType]:
    r"""Given new observation z_bar, compute new local upper bounds (and their defining points).
    This uses the incremental algorithm (Alg. 1 and Theorem 2.2) from :cite:`lacour2017box`:
    Given a new observation z_bar, if z_bar dominates any of the element in existing
    local upper bounds set: u_set, we need to:
    1. calculating the new local upper bounds set introduced by z_bar, and its corresponding
    defining set z_set
    2. remove the old local upper bounds set from u_set that has been dominated by z_bar and
    their corresponding defining points from z_set
    3. concatenate u_set, z_set with the new local upper bounds set and its corresponding
    defining set
    otherwise the u_set and z_set keep unchanged.

    :param u_set: (U in the paper) with shape [N, D] dim tensor containing the local upper bounds.
    :param z_set: (Z in the paper) with shape [N, D, D] dim tensor containing the local
        upper bounds.
    :param z_bar: with shape [D] denoting the new point (same notation in the paper)
    :return: new u_set with shape [N', D], new defining z_set with shape [N', D, D].
    """
    tf.debugging.assert_shapes([(z_bar, ["D"])])
    tf.debugging.assert_type(z_bar, u_set.dtype)

    num_outcomes = u_set.shape[-1]

    # condition check in Theorem 2.2: if not need to update (z_bar doesn't strict dominate anything)
    z_bar_dominates_u_set_mask = tf.reduce_all(z_bar < u_set, -1)
    if not tf.reduce_any(z_bar_dominates_u_set_mask):  # z_bar does not dominate any point in set U
        return u_set, z_set

    # elements in u that has been dominated by zbar and needs to be updated (step 5 of Alg. 2)
    u_set_need_update = u_set[z_bar_dominates_u_set_mask]  # [M, D], same as A in the paper
    z_set_need_update = z_set[z_bar_dominates_u_set_mask]  # [M, D, D]

    # Container of new local upper bound (lub) points and its defining set
    updated_u_set = tf.zeros(shape=(0, num_outcomes), dtype=u_set.dtype)
    updated_z_set = tf.zeros(shape=(0, num_outcomes, num_outcomes), dtype=u_set.dtype)

    # update local upper bound and its corresponding defining points
    for j in tf.range(num_outcomes):  # check update per dimension
        # for jth output dimension, check if if zbar_j can form a new lub
        # (if zbar_j ≥ max_{k≠j}{z_j^k(u)} get all lub's defining point and check:
        indices = tf.constant([dim for dim in range(num_outcomes) if dim != j], dtype=tf.int32)
        mask_j_dim = tf.constant(
            [0 if dim != j else 1 for dim in range(num_outcomes)], dtype=z_bar.dtype
        )
        mask_not_j_dim = tf.constant(
            [1 if dim != j else 0 for dim in range(num_outcomes)], dtype=z_bar.dtype
        )
        # get except jth defining point's jth dim
        z_uj_k = tf.gather(z_set_need_update, indices, axis=-2, batch_dims=0)[
            ..., j
        ]  # [M, D, D] -> [M, D-1]
        z_uj_max = tf.reduce_max(z_uj_k, -1)  # [M, D-1] -> [M]
        u_mask_to_be_replaced_by_zbar_j = z_bar[j] >= z_uj_max  # [M]
        # any of original local upper bounds (in A) can be updated
        if tf.reduce_any(u_mask_to_be_replaced_by_zbar_j):
            # update u with new lub: (zbar_j, u_{-j})
            u_need_update_j_dim = u_set_need_update[u_mask_to_be_replaced_by_zbar_j]  # [M', D]
            # tensorflow tricky to replace u_j's j dimension with z_bar[j]
            new_u_updated_j_dim = u_need_update_j_dim * mask_not_j_dim + tf.repeat(
                (mask_j_dim * z_bar[j])[tf.newaxis], u_need_update_j_dim.shape[0], axis=0
            )
            # add the new local upper bound point: u_j
            updated_u_set = tf.concat([updated_u_set, new_u_updated_j_dim], 0)

            # update u's defining point z
            z_need_update = z_set_need_update[
                u_mask_to_be_replaced_by_zbar_j
            ]  # get its original defining point
            # replace jth : [M', D - 1, D] & [1, D] -> [M', D, D]
            z_uj_new = (
                z_need_update * mask_not_j_dim[..., tf.newaxis]
                + z_bar * mask_j_dim[..., tf.newaxis]
            )
            # add its (updated) defining point
            updated_z_set = tf.concat([updated_z_set, z_uj_new], 0)

    # filter out elements of U (and it defining points) that are in A
    z_not_dominates_u_set_mask = ~z_bar_dominates_u_set_mask
    no_need_update_u_set = u_set[z_not_dominates_u_set_mask]
    no_need_update_z_set = z_set[z_not_dominates_u_set_mask]

    # combine remained lub points with new lub points (as well as their corresponding
    # defining points)
    if tf.shape(updated_u_set)[0] > 0:
        # add points from lub_new and lub_new_z
        u_set = tf.concat([no_need_update_u_set, updated_u_set], axis=0)
        z_set = tf.concat([no_need_update_z_set, updated_z_set], axis=0)
    return u_set, z_set


def _get_partition_bounds_hbda(
    z_set: TensorType, u_set: TensorType, reference_point: TensorType
) -> tuple[TensorType, TensorType]:
    r"""Get the hyper cell bounds through given the local upper bounds and the
    defining points. Main referred from Equation 2 of :cite:`lacour2017box`.

    :param u_set: with shape [N, D], local upper bounds set, N is the
        sets number, D denotes the objective numbers.
    :param z_set: with shape [N, D, D].
    :param reference_point: z^r in the paper, with shape [D].
    :return: lower, upper bounds of the partitioned cell, each with shape [N, D].
    """
    tf.debugging.assert_shapes([(reference_point, ["D"])])
    l_bounds = tf.zeros(shape=[0, tf.shape(u_set)[-1]], dtype=u_set.dtype)
    u_bounds = tf.zeros(shape=[0, tf.shape(u_set)[-1]], dtype=u_set.dtype)

    for u_idx in range(tf.shape(u_set)[0]):  # get partition through each of the lub point
        l_bound_new = tf.zeros(shape=0, dtype=u_set.dtype)
        u_bound_new = tf.zeros(shape=0, dtype=u_set.dtype)
        # for each (partitioned) hyper-cell, get its bounds through each objective dimension
        # get bounds on 1st dim: [z_1^1(u), z_1^r]
        l_bound_new = tf.concat([l_bound_new, z_set[u_idx, 0, 0][tf.newaxis]], 0)
        u_bound_new = tf.concat([u_bound_new, reference_point[0][tf.newaxis]], 0)

        for j in range(1, u_set.shape[-1]):  # get bounds on rest dim
            l_bound_new = tf.concat(
                [l_bound_new, tf.reduce_max(z_set[u_idx, :j, j])[tf.newaxis]], 0
            )
            u_bound_new = tf.concat([u_bound_new, u_set[u_idx, j][tf.newaxis]], 0)
        l_bounds = tf.concat([l_bounds, l_bound_new[tf.newaxis]], 0)
        u_bounds = tf.concat([u_bounds, u_bound_new[tf.newaxis]], 0)

    # remove empty partitions (i.e., if lb_bounds == ub_bounds on certain obj dimension)
    empty = tf.reduce_any(u_bounds <= l_bounds, axis=-1)
    return l_bounds[~empty], u_bounds[~empty]

"""This module contains functions of different methods for partitioning the dominated/non-dominated
 region in multi-objective optimization, assuming a front is given upfront """
from __future__ import annotations

from abc import ABC

import tensorflow as tf
from typing_extensions import Final

from trieste.type import TensorType
from trieste.utils.misc import DEFAULTS
from trieste.utils.mo_utils.dominance import non_dominated


def prepare_default_non_dominated_partition(observations):
    if observations.shape[-1] > 2:
        return DividedAndConquerNonDominated(observations)
    elif observations.shape[-1] == 2:
        return ExactPartition2dNonDominated(observations)
    else:
        raise ValueError


class Partition(ABC):
    """
    Base class of partition
    """

    front: TensorType

    def partition_bounds(
        self, anti_reference: TensorType, reference: TensorType
    ) -> tuple[TensorType, TensorType]:
        """
        Get partition bounds according to the refernece point, anti_reference point
        as well as the self.front
        """


class NonDominatedPartition(Partition):
    """
    Partition methods focusing on partitioning non-dominated regions
    """

    def partition_bounds(
        self, anti_reference: TensorType, reference: TensorType
    ) -> tuple[TensorType, TensorType]:
        """
        Get partition bounds according to the refernece point, anti_reference point
        as well as the self.front, note the returned lower and upper bounds is a partition
        of the non-dominated region.
        """


class DominatedPartition(Partition):
    """
    Partition methods focusing on partitioning dominated-regions
    """

    def partition_bounds(
        self, anti_reference: TensorType, reference: TensorType
    ) -> tuple[TensorType, TensorType]:
        """
        Get partition bounds according to the refernece point, anti_reference point
        as well as the self.front, note the returned lower and upper bounds is a partition
        of the non-dominated region.
        """


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


class BoundIndexPartition(NonDominatedPartition):
    """
    A collection of partition strategy that is based on storing the index of pareto fronts & other auxilary points
    """
    _bounds: _BoundedVolumes

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


class ExactPartition2dNonDominated(BoundIndexPartition):
    def __init__(self, front: TensorType):
        """
        :param front
        """
        tf.debugging.assert_equal(
            tf.cast(tf.reduce_sum(tf.abs(non_dominated(front)[1])), dtype=front.dtype),
            tf.zeros(shape=1, dtype=front.dtype),
            message=f"\ninput {front} " f"contains dominated points",
        )
        self.front = tf.gather_nd(front, tf.argsort(front[:, :1], axis=0))  # sort
        self._bounds = self._get_bound_index()

    def _get_bound_index(self):
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


class DividedAndConquerNonDominated(BoundIndexPartition):
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

    def __init__(
        self, front: TensorType, jitter: float = DEFAULTS.JITTER, threshold: TensorType | float = 0
    ):
        """
        :param front
        :param jitter
        :param threshold
        """
        tf.debugging.assert_equal(
            tf.cast(tf.reduce_sum(tf.abs(non_dominated(front)[1])), dtype=front.dtype),
            tf.zeros(shape=1, dtype=front.dtype),
            message=f"\ninput {front} " f"contains dominated points",
        )
        self.front = tf.gather_nd(front, tf.argsort(front[:, :1], axis=0))  # sort
        self.front = front
        self._bounds = self._get_bound_index(jitter, threshold)

    def _get_bound_index(self, jitter: float = DEFAULTS.JITTER, threshold: TensorType | float = 0):
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

            test_accepted = self._is_test_required((upper - jitter) < self.front)
            lower_result_final, upper_result_final = tf.cond(
                test_accepted,
                lambda: self._accepted_test_body(lower_result, upper_result, lower_idx, upper_idx),
                lambda: (lower_result, upper_result),
            )

            test_rejected = self._is_test_required((lower + jitter) < self.front)
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


class HypervolumeBoxDecompositionIncrementalDominated(DominatedPartition):
    """
    A method of partitioning the dominated region.

    The main idea is of using a sort of auxiliary points (which is referred to as local
    upper bounds in the original context) assosiating to existing Pareto points to
    describe the Pareto frontier, then, one could use an alternative partition as
    an replacement of original partition.

    Main reference: Section 2.2.2 of :cite:`lacour2017box`
    Assumptions
    One of the assumption made here is no any points have the same value in any dimension
    """

    def __init__(self, front: TensorType, reference_point: TensorType):
        tf.debugging.assert_shapes([(reference_point, ["D"])])
        tf.debugging.assert_greater_equal(reference_point, front)

        self._reference_point = reference_point
        self.U_set = reference_point[
            tf.newaxis
        ]  # initialize local upper bounds with reference point

        self.Z_set = (  # initialize defining points _Z to be the dummy points \hat{z} that are
            # defined in Sec 2.1
            -1e10 * tf.ones((1, front.shape[-1], front.shape[-1]), dtype=front.dtype)
            + 1e10 * tf.eye(front.shape[-1], front.shape[-1], batch_shape=[1], dtype=front.dtype)
            + tf.linalg.diag(reference_point)[tf.newaxis, ...]
        )  # note we replace the original defined objective space [0, reference_point] by [-1e10, reference_point]

        (
            self.U_set,
            self.Z_set,
        ) = _update_local_upper_bounds_incremental(  # incrementally update local upper
            new_front_points=front,  # bounds and defining points for each new Pareto point
            u_set=self.U_set,
            z_set=self.Z_set,
        )

    def update(self, new_front: TensorType):
        """
        Update with new front, this can be computed with the incremental method
        """
        (
            self.U_set,
            self.Z_set,
        ) = _update_local_upper_bounds_incremental(  # incrementally update local upper
            new_front_points=new_front,  # bounds and defining points for each new Pareto point
            u_set=self.U_set,
            z_set=self.Z_set,
        )

    def partition_bounds(
        self, anti_reference: TensorType, reference: TensorType
    ) -> tuple[TensorType, TensorType]:
        return _get_partition_bounds_hbda(self.Z_set, self.U_set, self._reference_point)


def _update_local_upper_bounds_incremental(
    new_front_points: TensorType, u_set: TensorType, z_set: TensorType
) -> tuple[TensorType, TensorType]:
    r"""Update the current local upper with the new pareto points. (note: this does not
    require input: new_front_points be non-dominated points)

    :param new_front_points: with shape [n, p], the new Pareto frontier points.
    :param u_set: with shape [n', p], the set containing all the existing local upper bounds.
    :param z_set: with shape [n', p, p] contain the existing local upper bounds defining points,
        note the meaning of the two p is different: first p denotes for any element
        in u_set, it has p defining points,  the second p denotes each defining points is p dimensional.

    Returns: A new [n'', p] new local upper bounds set.
             A [n'', p, p]  contain the new local upper bounds defining points
    """

    # from matplotlib import pyplot as plt
    tf.debugging.assert_shapes([(new_front_points, ["N", "D"])])
    # front_evalued = []
    for new_front_pt in new_front_points:  # incrementally update local upper bounds
        # plt.scatter(1.2, 1.2, label='ref points')
        # front_evalued.append(new_front_pt)
        # for front in front_evalued:
        #     plt.scatter(front[0], front[1], label='PF points', color='b')
        u_set, z_set = _compute_new_local_upper_bounds(u_set, z_set, z_bar=new_front_pt)
        # plt.scatter(u_set[:, 0], u_set[:, 1], label='LUB points', color='r')
        # plt.close()
    return u_set, z_set


def _compute_new_local_upper_bounds(
    u_set: TensorType, z_set: TensorType, z_bar: TensorType
) -> tuple[TensorType, TensorType]:
    r"""Compute new local upper bounds.
    This uses the incremental algorithm (Alg. 1 and Theorem 2.2) from :cite:`lacour2017box`: Given a new point z_bar,
    if z_bar (new point) dominates any of the element in existing local upper bounds set: u_set, we need to:
    1. calculating the new local upper bounds set introduced by z_bar, and its corresponding defining set z_set
    2. remove the old local upper bounds set from u_set that has been dominated by z_bar and their corresponding
        defining points from z_set
    3. concatenate u_set, z_set with the new local upper bounds set and its corresponding defining set

    :param u_set: (U in the paper) with shape `[n,  p] dim tensor containing the local upper bounds.
    :param z_set: (Z in the paper) with shape `[n, p, p] dim tensor containing the local upper bounds.
    :param z_bar: with shape [p] denoting the new point
    :return: new u_set with shape [n, p], new defining z_set with shape [n', p, p].
    """
    tf.debugging.assert_shapes([(z_bar, ["D"])])
    tf.debugging.assert_type(z_bar, u_set.dtype)

    num_outcomes = u_set.shape[-1]
    # condition check in Theorem 2.2: if not need to update (z_bar doesn't strict dominate anything)
    z_bar_dominates_u_set_mask = tf.reduce_all(z_bar < u_set, -1)
    if not tf.reduce_any(z_bar_dominates_u_set_mask):  # z_bar does not dominate any point in set U
        return u_set, z_set

    # A: elements in u that has been dominated by zbar, which needs to be replaced
    capital_a = u_set[z_bar_dominates_u_set_mask]  # [m, p]
    capital_a_z = z_set[z_bar_dominates_u_set_mask]  # [m, p, p]

    # Container of new local upper bound (lub) points and its defining set
    lub_new = []
    lub_new_z = []

    for j in range(num_outcomes):  # update per each dimension
        # calculate for jth output dimension, if zbar_j ≥ max_{k≠j}{z_j^k(u)}
        indices = tf.constant([dim for dim in range(num_outcomes) if dim != j], dtype=tf.int32)
        mask_j = tf.constant(
            [0 if dim != j else 1 for dim in range(num_outcomes)], dtype=z_bar.dtype
        )
        mask_not_j = tf.constant(
            [1 if dim != j else 0 for dim in range(num_outcomes)], dtype=z_bar.dtype
        )
        z_uj_k = tf.gather(capital_a_z, indices, axis=1, batch_dims=0)[
            :, :, j
        ]  # [m, p, p] -> [m, p-1]
        z_uj_max = tf.reduce_max(z_uj_k, -1)  # [m, p-1] -> [m]
        u_mask_to_be_replaced_by_zbar_j = z_bar[j] >= z_uj_max  # [m, 1]
        # for jth dimension, any one of m local upper bounds can be replaced
        if tf.reduce_any(u_mask_to_be_replaced_by_zbar_j):
            # add new lub: (zbar_j, u_{-j})
            a_filtered = capital_a[u_mask_to_be_replaced_by_zbar_j]  # [m', p]
            # tensorflow tricky to replace u_j's j dimension with z_bar[j]
            u_j = a_filtered * mask_not_j + tf.repeat(
                (mask_j * z_bar[j])[tf.newaxis], a_filtered.shape[0], axis=0
            )
            lub_new.append(u_j)  # add the new local upper bound point: u_j

            # add its defining point
            a_z_filtered = capital_a_z[
                u_mask_to_be_replaced_by_zbar_j
            ]  # get its original defining point

            z_uj_new = a_z_filtered * mask_not_j[..., tf.newaxis] + z_bar * mask_j[..., tf.newaxis]
            lub_new_z.append(z_uj_new)

    # filter out elements of U that are in A
    z_not_dominates_u_set_mask = ~z_bar_dominates_u_set_mask
    u_set = u_set[z_not_dominates_u_set_mask]
    # remaining indices
    z_set = z_set[z_not_dominates_u_set_mask]

    # combine original untouched lub points with new lub points and their corresponding defining points
    if len(lub_new) > 0:
        # add points from lub_new and lub_new_z
        u_set = tf.concat([u_set, *lub_new], axis=0)
        z_set = tf.concat([z_set, *lub_new_z], axis=0)
    return u_set, z_set


def _get_partition_bounds_hbda(
    z: TensorType, u: TensorType, reference_point: TensorType
) -> tuple(TensorType, TensorType):
    r"""Get the cell bounds given the local upper bounds and the defining points.
    Main referred from Equation 2 in Hypervolume Box Decomposition Algorithm (HBDA) paper
    of :cite:`lacour2017box`.

    :param u: with shape [|U(N)|, p], local upper bounds set, |U(N)| is the sets number,
       p denotes the objective dimensionality
    :param z: with shape [|U(N)|, p, p]
    :param reference_point: z^r in the paper

    :return: l_bounds, u_bounds
    """
    tf.debugging.assert_shapes([(reference_point, ["D"])])
    l_bounds = tf.zeros(shape=[0, u.shape[-1]], dtype=u.dtype)
    u_bounds = tf.zeros(shape=[0, u.shape[-1]], dtype=u.dtype)

    for u_idx in range(u.shape[0]):
        l_bound_new = tf.zeros(shape=0, dtype=u.dtype)
        u_bound_new = tf.zeros(shape=0, dtype=u.dtype)
        # get bounds on 1st dim: [z_1^1(u), z_1^r]
        l_bound_new = tf.concat([l_bound_new, z[u_idx, 0, 0][tf.newaxis]], 0)
        u_bound_new = tf.concat([u_bound_new, reference_point[0][tf.newaxis]], 0)

        for j in range(1, u.shape[-1]):  # get bounds on rest dim
            l_bound_new = tf.concat([l_bound_new, tf.reduce_max(z[u_idx, :j, j])[tf.newaxis]], 0)
            u_bound_new = tf.concat([u_bound_new, u[u_idx, j][tf.newaxis]], 0)
        l_bounds = tf.concat([l_bounds, l_bound_new[tf.newaxis]], 0)
        u_bounds = tf.concat([u_bounds, u_bound_new[tf.newaxis]], 0)

    # remove empty partitions
    # Note: the equality will evaluate as True if the lower and upper bound
    # are both (-inf), which could happen if the reference point is -inf.
    empty = tf.reduce_any(u_bounds <= l_bounds, axis=-1)
    return l_bounds[~empty], u_bounds[~empty]


class FlipTrickNonDominatedPartition(NonDominatedPartition):
    """
    Main refer Algorithm 2, Algorithm 3 of :cite:yang2019efficient, a slight alter of
    method is utilized as we are performing minimization.

    The idea behind the proposed algorithm is transforming the problem of
    partitioning a non-dominated space into the problem of partitioning the
    dominated space.

    For instance, consider minimization problem, we could use lacour2017box's methods to locate the
    local upper bound point set (by partitioning the dominated region), given this set,
    we can combine with a fake reference point (e.g., [-inf, ..., -inf]) and flip the problem
    as maximization, in this case, we are able to use lacour2017box's method again to partition
    the 'dominated' region, which will then provide us with the partition of the non-dominated region
    """

    def __init__(self, front: TensorType, reference_point: TensorType):
        lub_sets = HypervolumeBoxDecompositionIncrementalDominated(front, reference_point).U_set
        flipped_partition = HypervolumeBoxDecompositionIncrementalDominated(
            -lub_sets, 3 * tf.ones(shape=front.shape[-1], dtype=front.dtype)
        )
        flipped_lb_pts, flipped_ub_pts = flipped_partition.partition_bounds()
        self.lb_pts = -flipped_ub_pts
        self.ub_pts = -flipped_lb_pts

    def partition_bounds(
        self, anti_reference: TensorType, reference: TensorType
    ) -> tuple[TensorType, TensorType]:
        return self.lb_pts, self.ub_pts


if __name__ == "__main__":
    # z = tf.constant([[[0.5, 1.0], [1, 0.5]]])
    # u = tf.constant([[0.5, 0.5]])
    # get_partition_bounds_hbda(z, u, tf.constant([1.0, 1.0]))
    from trieste.utils.multi_objectives import DTLZ1, VLMOP2

    # TODO: Implement non incremental partitioning
    # TODO: Visual Check
    # TODO: Profile on jupyter notebook
    # TODO: Write Test
    # plot check
    # 2d case
    # from trieste.utils.multi_objectives import VLMOP2
    # from matplotlib import pyplot as plt
    # pf = VLMOP2().gen_pareto_optimal_points(10)
    # lb, ub = partition = FlipTrickNonDominatedPartition(pf, 2 * tf.ones(2)).partition_bounds()
    # # plt.scatter(1.2, 1.2, label='ref points')
    # # plt.scatter(pf[:, 0], pf[:, 1], label='PF points')
    # plt.scatter(lb[:, 0], lb[:, 1], label='Lower bound points')
    # plt.scatter(ub[:, 0], ub[:, 1], label='Upper bound points')
    # plt.legend()
    # plt.show()
    # partition = HypervolumeBoxDecompositionIncremental(pf, 1.2 * tf.ones(2))
    # from matplotlib import pyplot as plt
    # plt.scatter(1.2, 1.2, label='ref points')
    # plt.scatter(pf[:, 0], pf[:, 1], label='PF points')
    # plt.scatter(partition.U_set[:, 0], partition.U_set[:, 1], label='U set points')
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # lb_pts, ub_pts = partition.partition_bounds()
    # plt.scatter(1.2, 1.2, label='ref points')
    # plt.scatter(lb_pts[:, 0], lb_pts[:, 1], label='lb points')
    # plt.scatter(ub_pts[:, 0], ub_pts[:, 1], label='ub points')
    # plt.scatter(pf[:, 0], pf[:, 1], label='PF points')
    # plt.legend()
    # plt.show()
    # 3D case
    outdim = 3
    pf = DTLZ1(10, outdim).gen_pareto_optimal_points(100)
    partition = FlipTrickNonDominatedPartition(pf, 12 * tf.ones(outdim))
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )

    # from botorch.test_functions.multi_objective import DTLZ1
    # import torch
    # _ = FastNondominatedPartitioning(-1.2 * torch.ones(2), -torch.Tensor(pf))

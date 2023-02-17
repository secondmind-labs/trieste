# %% [markdown]
# # Multi-Objective Optimization (MOO) using \{PF\}$^2$ES

# %% [markdown]
# In this tutorial, we demonstrate how to utilize our proposed new information-theoretic acquisition function: \{PF\}$^2$ES (https://arxiv.org/abs/2204.05411) for multi-objective optimization (MOO).
#
# \{PF\}$^2$ES is suitable for:
# - Observation Noise Free MOO problem by sequential sampling / parallel sampling
# - Observation Noise Free C(onstraint)MOO problem by sequential sampling / parallel sampling
#
# Notes:
# - that the running of the whole notebook can take up to 40 minutes.
# - it is recommended to run cells sequentially instead of `run all` at once, as figures might not show completely for the latter scenarios.

# %%
import numpy as np
import tensorflow as tf

np.random.seed(1793)
tf.random.set_seed(1793)

# %%
try:
    import pymoo
    from pymoo.core.result import Result as PyMOOResult
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
except ImportError:
    raise ImportError("PF2ES requires pymoo, " "which can be installed via `pip install pymoo`")

from functools import partial
from typing import Callable, List, Mapping, Optional, Tuple, Union

import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from trieste.acquisition.interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    AcquisitionFunctionClass,
)
from trieste.acquisition.multi_objective.dominance import non_dominated
from trieste.acquisition.multi_objective.partition import (
    ExactPartition2dNonDominated,
)
from trieste.data import Dataset
from trieste.models.interfaces import (
    HasReparamSampler,
    HasReparamSamplerModelStack,
    HasTrajectorySampler,
    ModelStack,
    ProbabilisticModelType,
    ReparametrizationSampler,
    SupportsPredictJoint,
    TrainableModelStack,
    TrainablePredictJointModelStack,
    TrajectoryFunction,
)
from trieste.observer import OBJECTIVE
from trieste.space import Box
from trieste.types import TensorType, Tag
from trieste.utils.misc import DEFAULTS


# %% [markdown]
# ## Additioanal Supporting Functionalities
#

# %% [markdown]
# Besides the functionalities provided by Trieste, {PF}$^2$ES builds on following additional supporting functionality.

# %%
def prepare_default_non_dominated_partition_bounds(
    reference: TensorType,
    observations: Optional[TensorType] = None,
    anti_reference: Optional[TensorType] = None,
) -> Tuple[TensorType, TensorType]:
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

    if tf.shape(observations)[-1] > 2:
        return FlipTrickPartitionNonDominated(
            observations, anti_reference, reference
        ).partition_bounds()
    else:
        return ExactPartition2dNonDominated(observations).partition_bounds(
            anti_reference, reference
        )


class HypervolumeBoxDecompositionIncrementalDominated:
    """
    A Hypervolume Box Decomposition Algorithm (incremental version) (HBDA in short) which
    is used to partition the dominated space into disjoint hyperrectangles. Given the
    dominated region D(N_nd, Z^r) constructed by non dominated set N_nd extracted from
    ``observations`` set N, and a corresponding ``reference_point`` Z^r. HBDA use a
    set U of auxiliary points (which is referred to as local upper bound set in the paper
    and each element in U is defined by D number of points in N_nd) associated with the
    non dominated set N_nd to describe the dominated region D(U, Z^r) and to decompose it
    to disjoint hyper rectangles. Main reference: Section 2.2.2 of :cite:`lacour2017box`.
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

        tf.debugging.assert_greater_equal(
            reference_point,
            observations,
            message=f"observations: {observations} contains at least one value larger "
            f"than reference point: {reference_point}",
        )

        self._dummy_anti_reference_point = dummy_anti_ref_value * tf.ones(
            (1, observations.shape[-1]), dtype=observations.dtype
        )
        self._reference_point = reference_point

        tf.debugging.assert_greater_equal(  # making sure objective space has been
            # lower bounded by [dummy_anti_ref_value, ..., dummy_anti_ref_value]
            observations,
            self._dummy_anti_reference_point,
            f"observations: {observations} contains value smaller than dummy "
            f"anti-reference point: {self._dummy_anti_reference_point}, try "
            f"using a smaller dummy anti ref value.",
        )

        # initialize local upper bound set U using reference point Z^r (step 1 of Alg 1)
        self.U_set = reference_point[tf.newaxis]

        self.Z_set = (  # initialize local upper bound set element's defining points Z, the
            # defining points of Z^r is the dummy points \hat{z}^1, ..., \hat{z}^D defined
            # in Sec 2.1. Note: 1. the original defined objective space [0, reference_point] has
            # been replaced by [dummy anti-reference point, reference_point].
            # 2. the dummy anti-reference point value will not affect lower/upper bounds of this
            # dominated partition method
            dummy_anti_ref_value
            * tf.ones(
                (1, observations.shape[-1], observations.shape[-1]),
                dtype=observations.dtype,
            )
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

    def partition_bounds(self) -> Tuple[TensorType, TensorType]:
        return _get_partition_bounds_hbda(self.Z_set, self.U_set, self._reference_point)


def _update_local_upper_bounds_incremental(
    new_observations: TensorType, u_set: TensorType, z_set: TensorType
) -> Tuple[TensorType, TensorType]:
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
) -> Tuple[TensorType, TensorType]:
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

    # elements in U that has been dominated by z_bar and needs to be updated (step 5 of Alg. 2)
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
            [0 if dim != j else 1 for dim in range(num_outcomes)],
            dtype=z_bar.dtype,
        )
        mask_not_j_dim = tf.constant(
            [1 if dim != j else 0 for dim in range(num_outcomes)],
            dtype=z_bar.dtype,
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
                (mask_j_dim * z_bar[j])[tf.newaxis],
                u_need_update_j_dim.shape[0],
                axis=0,
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

    # filter out elements of U (and its defining points) that are in A
    z_not_dominates_u_set_mask = ~z_bar_dominates_u_set_mask
    no_need_update_u_set = u_set[z_not_dominates_u_set_mask]
    no_need_update_z_set = z_set[z_not_dominates_u_set_mask]

    # combine remaining lub points with new lub points (as well as their
    # corresponding defining points)
    if tf.shape(updated_u_set)[0] > 0:
        # add points from lub_new and lub_new_z
        u_set = tf.concat([no_need_update_u_set, updated_u_set], axis=0)
        z_set = tf.concat([no_need_update_z_set, updated_z_set], axis=0)
    return u_set, z_set


def _get_partition_bounds_hbda(
    z_set: TensorType, u_set: TensorType, reference_point: TensorType
) -> Tuple[TensorType, TensorType]:
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


class FlipTrickPartitionNonDominated:
    """
    Main refer Algorithm 3 of :cite:yang2019efficient, a slight alter of
    method is utilized as we are performing minimization.
    The idea behind the proposed algorithm is transforming the problem of
    partitioning a non-dominated space into the problem of partitioning the
    dominated space.
    For instance, consider minimization problem, we could use lacour2017box's methods to
    locate the local upper bound point set (by partitioning the dominated region), by
    treating these local upper bound as fake Pareto front, we can combine with a fake
    reference point (e.g., [-inf, ..., -inf]) and flip the problem as maximization, in
    this case, we are able to use :cite:`lacour2017box`s method again to partition the
    'dominated' region, which will then provide us with the partition of the non-dominated
    region
    """

    def __init__(
        self,
        observations: TensorType,
        anti_reference_point: TensorType,
        reference_point: TensorType,
    ):
        """
        :param observations: the objective observations, preferably this can be a
            non-dominated set, but any set is acceptable here.
        :param anti_reference_point: a worst point to use with shape [D].
            Defines the lower bound of the hypercell. If not specified, will use a default value:
            -[1e10] * D.
        :param reference_point: a reference point to use, with shape [D]
            (same as p in the paper). Defines the upper bound of the hypervolume.
        """
        lub_sets = HypervolumeBoxDecompositionIncrementalDominated(
            observations, reference_point
        ).U_set
        flipped_partition = HypervolumeBoxDecompositionIncrementalDominated(
            -lub_sets,
            -anti_reference_point,
            dummy_anti_ref_value=tf.reduce_min(-lub_sets - 1.0, axis=-2),
        )
        flipped_lb_pts, flipped_ub_pts = flipped_partition.partition_bounds()
        self.lb_pts = -flipped_ub_pts
        self.ub_pts = -flipped_lb_pts

    def partition_bounds(self) -> Tuple[TensorType, TensorType]:
        return self.lb_pts, self.ub_pts


# %%
class HasTrajectorySamplerModelStack(ModelStack[HasTrajectorySampler]):
    """
    A Model Stack that each sub model is an instantiation os `HasTrajectorySampler`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for model in self._models:
            if not isinstance(model, HasTrajectorySampler):
                raise ValueError(
                    f"HasTrajectorySamplerModelStack only supports models with a trajectory_sampler "
                    f"method; received {model.__repr__()}"
                )

        self._trajectory_samplers = [model.trajectory_sampler() for model in self._models]
        self._trajectory_samples: list = []
        self.gen_initial_trajectories()

    def gen_initial_trajectories(self):
        assert len(self._trajectory_samples) == 0
        for trajectory_sampler in self._trajectory_samplers:
            self._trajectory_samples.append(trajectory_sampler.get_trajectory())

    def get_trajectories(self, regenerate: bool = False) -> List[TrajectoryFunction]:
        """
        Get sampled trajectories, if no trajectories have been generated, it will run
        `gen_initial_trajectories` first to generate trajectories first, otherwise, it will
        just return the existing trajectory

        :param regenerate once trajectories have been genearted they are fixed in batch dim
          (i.e. the trajectory samples are fixed), if wanna use different trajectory samples, set regenerate to True
        Note that if one would like evaluate on trajectory with a different (joint) batch size, `gen_initial_trajectories`
        must be called
        """
        if regenerate is True:
            if len(self._trajectory_samples) == 0:
                print("No existing trajectories have been found, generate new trajectories")
                self.gen_initial_trajectories()
            else:  # regenerate trajectories
                self._trajectory_samples = []
                for trajectory_sampler in self._trajectory_samplers:
                    self._trajectory_samples.append(trajectory_sampler.get_trajectory())
        return self._trajectory_samples

    def update_trajectories(self):
        """
        Update trajectories
        """
        assert len(self._trajectory_samples) != 0
        for trajectory, idx in zip(self._trajectory_samples, range(len(self._trajectory_samples))):
            self._trajectory_samples[idx] = self._trajectory_samplers[idx].update_trajectory(
                trajectory
            )

    def resample_trajectories(self):
        """
        Resample trajectories
        """
        assert len(self._trajectory_samples) != 0
        for trajectory, idx in zip(self._trajectory_samples, range(len(self._trajectory_samples))):
            self._trajectory_samples[idx] = self._trajectory_samplers[idx].update_trajectory(
                trajectory
            )

    def eval_on_trajectory(self, query_points: TensorType) -> TensorType:
        r"""
        :param query_points: The points at which to sample, with shape [N, B, D].
        :return: The samples from all the wrapped models, For wrapped models with predictive
            distributions with event shapes 1, this has
            shape [..., N, B, :math:`\sum_i 1`].
        """
        if len(self._trajectory_samples) == 0:
            self.gen_initial_trajectories()
        return tf.concat(
            [trajectory_sample(query_points) for trajectory_sample in self._trajectory_samples],
            axis=-1,
        )


class TrainableHasTrajectorySamplerModelStack(HasTrajectorySamplerModelStack, TrainableModelStack):
    """
    A stack of models that are both trainable and support trajectory sample
    """

    pass


class TrainableHasTrajectoryAndPredictJointReparamModelStack(
    HasTrajectorySamplerModelStack,
    TrainablePredictJointModelStack,
    HasReparamSamplerModelStack,
):
    """
    A stack of models that are both trainable and support predict_joint
    """

    def get_internal_data(self):
        return Dataset(
            self._models[0].get_internal_data().query_points,
            tf.concat(
                [model.get_internal_data().observations for model in self._models],
                axis=-1,
            ),
        )


def extract_pf_from_data(
    dataset: Mapping[Tag, Dataset],
    objective_tag: Tag = OBJECTIVE,
    constraint_tag: Optional[Tag] = None,
) -> Tuple[TensorType, ...]:
    """
    Extract (feasible) Pareto Frontier Input and Output from Given dataset

    This assumes the objective and constraints are at the same location!
    return PF_X, PF_Y
    """
    obj_obs = dataset[objective_tag].observations
    pf_obs, pf_boolean_mask = non_dominated(obj_obs)
    if constraint_tag is not None:  # extract feasible pf data
        assert tf.reduce_all(
            tf.equal(
                dataset[objective_tag].query_points,
                dataset[constraint_tag].query_points,
            )
        )

        feasible_mask = tf.reduce_all(
            dataset[constraint_tag].observations <= tf.zeros(shape=1, dtype=obj_obs.dtype),
            axis=-1,
        )
        _, un_constraint_dominance_rank = non_dominated(dataset[OBJECTIVE].observations)
        un_constraint_dominance_mask = tf.squeeze(un_constraint_dominance_rank == True)
        feasible_pf_obs = dataset[OBJECTIVE].observations[
            tf.logical_and(un_constraint_dominance_mask, feasible_mask)
        ]
        feasible_pf_x = dataset[OBJECTIVE].query_points[
            tf.logical_and(un_constraint_dominance_mask, feasible_mask)
        ]
        return feasible_pf_x, feasible_pf_obs
    else:
        return (
            tf.boolean_mask(dataset[objective_tag].query_points, pf_boolean_mask == True),
            pf_obs,
        )


class MOOResult:
    """
    Wrapper for pymoo result, the main difference the constraint, if have, is <0 is feasible
    """

    def __init__(self, result: PyMOOResult):
        self._res = result

    @property
    def inputs(self):
        return self._res.X

    @property
    def fronts(self):
        """
        return the (feasible) Pareto Front from Pymoo, if no constraint has been satisfied
        """
        return self._res.F

    @property
    def constraint(
        self,
    ):  # Note in Pymoo, <0 is feasible, so now we need to inverse
        return self._res.G

    def _initialize_existing_result_as_empty(
        self,
        inputs: TensorType,
        observations: TensorType,
        constraints: TensorType = None,
    ):
        self._res.X = tf.zeros(shape=(0, inputs.shape[-1]), dtype=inputs.dtype)
        self._res.F = tf.zeros(shape=(0, observations.shape[-1]), dtype=observations.dtype)
        if constraints is not None:
            self._res.G = tf.zeros(shape=(0, constraints.shape[-1]), dtype=constraints.dtype)

    def _check_if_existing_result_is_empty(self):
        if self._res.X is None:
            return True

    def concatenate_with(
        self,
        inputs: TensorType,
        observations: TensorType,
        constraints: TensorType = None,
    ):
        """
        Add result with some other input & observations that possibly is also Pareto optimal
        """

        if self._check_if_existing_result_is_empty():
            self._initialize_existing_result_as_empty(inputs, observations, constraints)
        aug_inputs = tf.concat([self._res.X, inputs], axis=0)
        aug_observations = tf.concat([self._res.F, observations], axis=0)
        if constraints is not None:
            aug_constraints = tf.concat([self.constraint, constraints], axis=0)
            feasible_mask = tf.reduce_all(aug_constraints <= 0, axis=-1)
        else:  # no constrain, all feasible
            aug_constraints = None
            feasible_mask = tf.ones(aug_observations.shape[0], dtype=tf.bool)
        _, dominance_mask_on_feasible_candidate = non_dominated(aug_observations[feasible_mask])
        self._res.X = tf.boolean_mask(
            aug_inputs[feasible_mask],
            dominance_mask_on_feasible_candidate == True,
        )
        self._res.F = tf.boolean_mask(
            aug_observations[feasible_mask],
            dominance_mask_on_feasible_candidate == True,
        )
        if constraints is not None:
            assert aug_constraints is not None
            self._res.G = tf.boolean_mask(
                aug_constraints[feasible_mask],
                dominance_mask_on_feasible_candidate == True,
            )


def moo_nsga2_pymoo(
    f: Callable[[TensorType], TensorType],
    input_dim: int,
    obj_num: int,
    bounds: tuple,
    popsize: int,
    num_generation: int = 1000,
    cons: Optional[Callable] = None,
    cons_num: int = 0,
    initial_candidates: Optional[TensorType] = None,
    verbose: bool = False,
) -> MOOResult:
    """
    Multi-Objective Optimizer using NSGA2 algorithm by pymoo

    When there is no optimal result, the return
    :param f
    :param obj_num
    :param input_dim
    :param bounds: [[lb_0, lb_1， ..., lb_D], [ub_0, ub_1， ..., ub_D]]
    :param popsize: population size for NSGA2
    :param num_generation: number of generations used for NSGA2
    :param cons: Callable function representing the constraints of the problem
    :param cons_num: constraint number
    :return if no feasible pareto frontier has been located, return None or [None, None]
    """

    if cons is not None:
        assert cons_num > 0

    def func(x):
        "wrapper objective function for Pymoo written in numpy"
        return f(tf.convert_to_tensor(x)).numpy()

    def cfunc(x):
        "wrapper constraint function for Pymoo written in numpy"
        return cons(tf.convert_to_tensor(x)).numpy()

    class MyProblem(Problem):
        def __init__(self, n_var: int, n_obj: int, n_constr: int = cons_num):
            """
            :param n_var input variables
            :param n_obj number of objective functions
            :param n_constr number of constraint numbers
            """
            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                n_constr=n_constr,
                xl=bounds[0].numpy(),
                xu=bounds[1].numpy(),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = func(x)
            if cons_num > 0:
                out["G"] = cfunc(x)

    problem = MyProblem(n_var=input_dim, n_obj=obj_num)
    if initial_candidates is None:  # random start sampling
        pop = Box(bounds[0], bounds[1]).sample_halton(popsize).numpy()
    else:
        # https://pymoo.org/customization/initialization.html
        if initial_candidates.shape[0] >= popsize:  #
            pop = initial_candidates[:popsize].numpy()  # we only use the first pop size
        else:  # we need to fill a bit more
            pop = tf.concat(
                [
                    initial_candidates,
                    Box(bounds[0], bounds[1]).sample_halton(popsize - initial_candidates.shape[0]),
                ],
                axis=0,
            ).numpy()

    algorithm = NSGA2(
        pop_size=popsize,
        n_offsprings=10,
        sampling=pop,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", num_generation),
        save_history=False,
        verbose=verbose,
    )

    return MOOResult(res)


def sample_pareto_fronts_from_parametric_gp_posterior(
    objective_models: HasTrajectorySamplerModelStack,
    obj_num: int,
    sample_pf_num: int,
    search_space: Box,
    cons_num: int = 0,
    constraint_models: Optional[Union[HasTrajectorySamplerModelStack, ModelStack]] = None,
    moo_solver="nsga2",
    return_pf_input: bool = False,
    return_pf_constraints: Optional[bool] = False,
    popsize: int = 50,
    num_moo_iter: int = 500,
    reference_pf_inputs: Optional[TensorType] = None,
) -> Union[List, Tuple]:
    """
    Sample (feasible) Pareto frontier from Gaussian Process posteriors

    approximate the Pareto frontier by using multi-objective based evolutionary algorithm (MOEA)
    based on a parametric GP posterior, this is the most common approach in existing literatures.

    Note which parametric trajectory sampler to used (i.e., RFF or decoupled sampler) is specified within model
    level

    :param objective_models
    :param obj_num: objective number
    :param sample_pf_num:
    :param search_space:
    :param constraint_models
    :param cons_num
    :param popsize: MOEA pop size
    :param num_moo_iter: MOEA number of iterations
    :cons_num: constraint number
    :param moo_solver: [nsga2, monte_carlo]
    :param return_pf_input return Pareto front corresponding input
    :param return_pf_constraints return constraint of Pareto Frontier
    :param reference_pf_inputs the reference pareto frontier used to
        extract potential candidate point and sampled PF

    """
    con_func = None
    pf_samples: list = []
    pf_samples_x: list = []
    pf_sample_cons: list = []
    # pre-processing

    if moo_solver == "nsga2":
        assert isinstance(objective_models, HasTrajectorySamplerModelStack)
        if constraint_models is not None:
            assert isinstance(constraint_models, HasTrajectorySamplerModelStack)
        for trajectory_idx in range(sample_pf_num):  # for each sample of trajectories, calculate pf
            # construct objective and constraint function
            objective_models.resample_trajectories()
            obj_func = lambda x: tf.squeeze(
                objective_models.eval_on_trajectory(tf.expand_dims(x, -2)),
                axis=-2,
            )

            if constraint_models is not None:
                constraint_models.resample_trajectories()
                con_func = lambda x: tf.squeeze(
                    constraint_models.eval_on_trajectory(tf.expand_dims(x, -2)),
                    axis=-2,
                )
            moo_res = moo_nsga2_pymoo(
                obj_func,
                obj_num=obj_num,
                input_dim=len(search_space.lower),
                bounds=(search_space.lower, search_space.upper),
                popsize=popsize,
                num_generation=num_moo_iter,
                cons=con_func,
                cons_num=cons_num
                # here assume each model only have 1 output
            )
            if reference_pf_inputs is not None and tf.size(reference_pf_inputs) != 0:
                reference_obj = obj_func(reference_pf_inputs)
                if constraint_models is not None:
                    assert con_func is not None
                    reference_con = con_func(reference_pf_inputs)
                else:
                    reference_con = None
                moo_res.concatenate_with(reference_pf_inputs, reference_obj, reference_con)
            pf_samples.append(moo_res.fronts)
            if return_pf_input:
                pf_samples_x.append(moo_res.inputs)
            if return_pf_constraints:
                pf_sample_cons.append(moo_res.constraint)
    else:
        raise NotImplementedError(
            f"moo_solver: {moo_solver} do not supported yet! " f"only support [nsga2] at the moment"
        )
    if return_pf_input and return_pf_constraints:
        return pf_samples, pf_samples_x, pf_sample_cons
    elif return_pf_input and not return_pf_constraints:
        return pf_samples, pf_samples_x
    elif not return_pf_input and return_pf_constraints:
        return pf_samples, pf_sample_cons
    return pf_samples


def inference_pareto_fronts_from_gp_mean(
    models: TrainableHasTrajectoryAndPredictJointReparamModelStack,
    search_space: Box,
    popsize: int = 20,
    num_moo_iter: int = 500,
    cons_models: Optional[TrainableHasTrajectoryAndPredictJointReparamModelStack] = None,
    min_feasibility_probability=0.5,
    constraint_enforce_percentage: float = 0.0,
    use_model_data_as_initialization: bool = True,
) -> Tuple[TensorType, ...]:
    """
    Get the (feasible) pareto frontier from GP posterior mean optionally subject to
    a probability of being feasible constraint
    """
    assert isinstance(models, ModelStack)

    def obj_post_mean(at):
        return models.predict(at)[0]

    def con_prob_feasible(at, enforcement=0.0):
        """
        Calculate the probability of being feasible
        """
        mean, var = cons_models.predict(at)
        prob_fea = tfd.Normal(mean, tf.sqrt(var)).cdf(0.0 - enforcement)
        return min_feasibility_probability - tf.reduce_prod(prob_fea, axis=-1, keepdims=True)

    if cons_models is not None and constraint_enforce_percentage != 0.0:
        assert constraint_enforce_percentage >= 0
        stacked_constraint_obs = tf.concat(
            [_model.get_internal_data().observations for _model in cons_models._models],
            1,
        )
        constraint_range = tf.reduce_max(stacked_constraint_obs, -2) - tf.reduce_min(
            stacked_constraint_obs, -2
        )
        constraint_enforcement = constraint_range * constraint_enforce_percentage
    else:
        constraint_enforcement = 0.0

    if use_model_data_as_initialization is True:
        if cons_models is not None:
            constraint_tag: Optional[Tag] = "CONSTRAINT"
            _dataset = {
                OBJECTIVE: models.get_internal_data(),
                constraint_tag: cons_models.get_internal_data(),
            }
        else:
            constraint_tag = None
            _dataset = {OBJECTIVE: models.get_internal_data()}
        initial_candidates, _ = extract_pf_from_data(
            _dataset, objective_tag=OBJECTIVE, constraint_tag=constraint_tag
        )
    else:
        initial_candidates = None

    moo_res = moo_nsga2_pymoo(
        obj_post_mean,
        input_dim=len(search_space.lower),
        obj_num=len(models._models),
        bounds=(search_space.lower, search_space.upper),
        popsize=popsize,
        num_generation=num_moo_iter,
        cons=partial(con_prob_feasible, enforcement=constraint_enforcement)
        if cons_models is not None
        else None,
        cons_num=len(cons_models._models) if cons_models is not None else 0,
        initial_candidates=initial_candidates,
    )
    return moo_res.fronts, moo_res.inputs


# %%
import math


class QuasiMonteCarloNormalSampler:
    """
    Sample from a univariate normal:
    N~(0, I_d), where d is the dimensionality
    """

    def __init__(self, dimensionality: int):
        self.dimensionality = dimensionality
        self._box_muller_req_dim = tf.cast(
            2 * tf.math.ceil(dimensionality / 2), dtype=tf.int64
        )  # making sure this dim is even number
        self.dimensionality = dimensionality
        self._uniform_engine = Box(
            tf.zeros(shape=self._box_muller_req_dim),
            tf.ones(shape=self._box_muller_req_dim),
        )

    def sample(self, sample_size: int, dtype=None, seed: Optional[int] = None):
        """
        main reference:
        """
        uniform_samples = self._uniform_engine.sample_halton(
            sample_size, seed=seed
        )  # [sample_size, ceil_even(dimensionality)]

        even_indices = tf.range(0, uniform_samples.shape[-1], 2)
        Rs = tf.sqrt(
            -2 * tf.math.log(tf.gather(uniform_samples, even_indices, axis=-1))
        )  # [sample_size, ceil_even(dimensionality)/2]
        thetas = (
            2 * math.pi * tf.gather(uniform_samples, 1 + even_indices, axis=-1)
        )  # [sample_size, ceil_even(dimensionality)/2]
        cos = tf.cos(thetas)
        sin = tf.sin(thetas)
        samples_tf = tf.reshape(
            tf.stack([Rs * cos, Rs * sin], -1),
            shape=(sample_size, self._box_muller_req_dim),
        )
        # make sure we only return the number of dimension requested
        samples_tf = samples_tf[:, : self.dimensionality]
        if dtype is None:
            return samples_tf
        else:
            return tf.cast(samples_tf, dtype=dtype)


class QuasiMonteCarloMultivariateNormalSampler:
    """
    Sample from a multivariate d dimensional normal:
    N~(μ, σ)
    """

    def __init__(self, mean: TensorType, cov: TensorType):
        """
        :param mean
        :param cov full covariance matrices
        """

        self._mean = mean
        self._cov = cov
        self._chol_covariance = tf.linalg.cholesky(cov)
        self.base_sampler = QuasiMonteCarloNormalSampler(tf.shape(mean)[-1])

    def sample(self, sample_size: int):
        return self._mean + tf.squeeze(
            tf.matmul(
                self._chol_covariance,
                tf.cast(
                    self.base_sampler.sample(sample_size)[..., tf.newaxis],
                    dtype=self._mean.dtype,
                ),
            ),
            -1,
        )


# %%
from trieste.bayesian_optimizer import EfficientGlobalOptimization


class BuilderAccesableEfficientGlobalOptimization(EfficientGlobalOptimization):
    @property
    def builder(self):
        return self._builder


class PF2ES(AcquisitionFunctionBuilder[HasReparamSampler]):
    """
    Implementation of Parallel Feasible Pareto Frontier Entropy Search
    """

    def __init__(
        self,
        search_space: Box,
        *,
        objective_tag: Tag = OBJECTIVE,
        constraint_tag: Optional[str] = None,
        sample_pf_num: int = 5,
        moo_solver: str = "nsga2",
        moo_iter_for_approx_pf_searching: int = 500,
        population_size_for_approx_pf_searching: int = 50,
        discretize_input_sample_size: Optional[int] = 5000,
        parallel_sampling: bool = False,
        extreme_cons_ref_value: Optional[TensorType] = None,
        batch_mc_sample_size: int = 64,
        temperature_tau=1e-3,
        pareto_epsilon: float = 0.04,
        enable_qmc: bool = True,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_tag: The tag for the constraint data and model.
        :param sample_pf_num: Pareto frontier MC sample number to approximate acq function
        :param extreme_cons_ref_value: in case no feasible Pareto frontier exists (in constraint case),
            use this value as a reference value
        :param batch_mc_sample_size: Monte Carlo sample size for joint batch acquisition function calculation,
            only used when doing batch optimization
        :param parallel_sampling whether to use Batch acquisition function
        :param pareto_epsilon, heuristically used to make Pareto frontier bit better, this can enforce exploration of
            the Pareto Frontier itself. By default we use 0.04
        :raise ValueError (or InvalidArgumentError): If ``min_feasibility_probability`` is not a
            scalar in the unit interval :math:`[0, 1]`.
        """
        self._search_space = search_space
        self._objective_tag = objective_tag
        self._constraint_tag = constraint_tag
        self._num_pf_samples = sample_pf_num
        self._extreme_cons_ref_value = extreme_cons_ref_value
        self._moo_solver = moo_solver
        self._pf_samples: List = []  # sampled pareto frontier
        self._pf_samples_inputs: Optional[
            list
        ] = None  # sampled pareto frontier corresponding input
        self._partitioned_bounds: List = []
        self._pop_size = population_size_for_approx_pf_searching
        self._moo_iter = moo_iter_for_approx_pf_searching
        self._discretize_input_sample_size = discretize_input_sample_size
        self._parametric_obj_sampler = None
        self._parametric_con_sampler = None
        self._q_mc_sample_size = batch_mc_sample_size
        self._tau = temperature_tau
        assert 0.0 <= pareto_epsilon <= 1.0, ValueError(
            f"Pareto Epsilon is a percentage value must between [0, 1] but received: {pareto_epsilon}"
        )
        self._percentage_pareto_epsilon = pareto_epsilon
        self._pareto_epsilon: Optional[TensorType] = None
        self._parallel_sampling = parallel_sampling
        self._qMC = enable_qmc

    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param models
        :param datasets
        :param pending_points If `pending_points` is not None: perform a greedy batch acquisition function optimization,
        otherwise perform a joint batch acquisition function optimization
        """
        if self._constraint_tag is None:  # prepare unconstrained acquisition function
            return self.prepare_unconstrained_acquisition_function(models, datasets)
        else:  # prepare constraint acquisition function
            return self.prepare_constrained_acquisition_function(models, datasets)

    def estimate_pareto_frontier_ranges(
        self, obj_num: int, dtype, sample_wise_maximum: bool = True
    ) -> TensorType:
        """
        Estimate Pareto Frontier ranges based on sampled Pareto Frontier
        :param obj_num: number of objective functions
        :param sample_wise_maximum: whether to use pf sample wise maximum, if enabled to True, the ranges will be
            calculated per pf sample, if set to False, only the maximum range w.r.t all the pf samples will be used,
            this will make a even more conservative estimation of the feasible non-dominated region
        """

        if sample_wise_maximum is False:
            obj_wise_max = tf.zeros(obj_num, dtype=dtype)
            obj_wise_min = tf.zeros(obj_num, dtype=dtype)
            for pf in self._pf_samples:
                if pf is not None:  # handle strong constraint scenario
                    obj_wise_max = tf.maximum(
                        tf.cast(obj_wise_max, pf.dtype),
                        tf.reduce_max(pf, axis=-2),
                    )
                    obj_wise_min = tf.minimum(
                        tf.cast(obj_wise_min, pf.dtype),
                        tf.reduce_min(pf, axis=-2),
                    )
            return tf.stack([[obj_wise_max - obj_wise_min] * len(self._pf_samples)], axis=0)
        else:
            pareto_frontier_ranges = []
            for pf in self._pf_samples:
                if pf is not None:  # handle strong constraint scenario
                    pareto_frontier_ranges.append(
                        tf.reduce_max(pf, axis=-2) - tf.reduce_min(pf, axis=-2)
                    )
                else:
                    pareto_frontier_ranges.append(tf.zeros(shape=obj_num, dtype=dtype))
            return tf.stack(pareto_frontier_ranges, axis=0)

    def calculate_maximum_discrepancy_objective_vise(self, obj_num: int) -> TensorType:
        """
        Calculate Maximum Discrepancy for each sub Pareto Frontier
        """

        maximum_discrepancy_obj_wise = []
        for pf in self._pf_samples:
            max_discrepancy_obj_wise_per_pf = tf.zeros(obj_num, dtype=pf.dtype)
            # handle strong constraint scenario, if pf size is smaller than 2, there is no need to do clustering and
            # we assume the discrepancy in this case is 0

            # none clustering version
            if pf is not None and pf.shape[0] > 2:
                sorted_sub_pf = tf.sort(pf, axis=0)
                sub_maximum_discrepancy = tf.reduce_max(
                    sorted_sub_pf[1:] - sorted_sub_pf[:-1], axis=0
                )
                max_discrepancy_obj_wise_per_pf = tf.maximum(
                    tf.cast(sub_maximum_discrepancy, pf.dtype),
                    max_discrepancy_obj_wise_per_pf,
                )
            maximum_discrepancy_obj_wise.append(max_discrepancy_obj_wise_per_pf)
        return tf.stack(maximum_discrepancy_obj_wise, axis=0)

    def prepare_unconstrained_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        prepare parallel pareto frontier entropy search acquisition function
        :param datasets
        :param models
        """
        assert datasets is not None, ValueError("Dataset must be populated.")
        tf.debugging.assert_positive(len(datasets), message="Dataset must be populated.")
        obj_model = models[self._objective_tag]
        assert isinstance(obj_model, TrainableHasTrajectoryAndPredictJointReparamModelStack)
        obj_number = len(obj_model._models)  # assume each obj has only
        current_pf_x, _ = extract_pf_from_data(datasets)

        (
            self._pf_samples,
            self._pf_samples_inputs,
        ) = sample_pareto_fronts_from_parametric_gp_posterior(  # sample pareto frontier
            obj_model,
            obj_number,
            self._num_pf_samples,
            self._search_space,
            popsize=self._pop_size,
            num_moo_iter=self._moo_iter,
            moo_solver=self._moo_solver,
            reference_pf_inputs=current_pf_x,
            return_pf_input=True,
        )
        self._pareto_epsilon = self.estimate_pareto_frontier_ranges(
            obj_num=obj_number, dtype=current_pf_x.dtype
        ) * tf.convert_to_tensor(self._percentage_pareto_epsilon, dtype=current_pf_x.dtype)

        # get partition bounds
        self._partitioned_bounds = [
            prepare_default_non_dominated_partition_bounds(
                tf.constant([1e20] * obj_number, dtype=_pf.dtype),
                non_dominated(_pf - _pf_epsilon)[0],
            )
            for _pf, _pf_epsilon in zip(self._pf_samples, self._pareto_epsilon)
        ]

        if not isinstance(obj_model, HasReparamSampler):
            raise ValueError(
                f"The PF2ES function only supports "
                f"models that implement a reparam_sampler method; received {obj_model.__repr__()}"
            )

        sampler = obj_model.reparam_sampler(self._q_mc_sample_size)

        if self._parallel_sampling:
            return parallel_pareto_frontier_entropy_search(
                models=obj_model,
                partition_bounds=self._partitioned_bounds,
                pf_samples=self._pf_samples,
                sampler=sampler,
                tau=self._tau,
            )
        else:
            return sequential_pareto_frontier_entropy_search(
                models=obj_model,
                partition_bounds=self._partitioned_bounds,
                pf_samples=self._pf_samples,
                sampler=sampler,
                tau=self._tau,
            )

    def prepare_constrained_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        prepare parallel feasible pareto frontier entropy search acquisition function
        :param datasets
        :param models
        """
        obj_model = models[self._objective_tag]
        cons_model = models[self._constraint_tag]
        assert datasets is not None, ValueError("Dataset must be populated.")
        assert isinstance(obj_model, TrainableHasTrajectoryAndPredictJointReparamModelStack)
        assert isinstance(cons_model, TrainableHasTrajectoryAndPredictJointReparamModelStack)
        obj_number = len(obj_model._models)
        cons_number = len(cons_model._models)
        _constraint_threshold = tf.zeros(
            shape=cons_number,
            dtype=datasets[self._objective_tag].query_points.dtype,
        )
        current_pf_x, current_pf_y = extract_pf_from_data(
            datasets,
            objective_tag=self._objective_tag,
            constraint_tag=self._constraint_tag,
        )
        # Note: constrain threshold is not used in sample Feasible PF
        (
            self._pf_samples,
            self._pf_samples_inputs,
        ) = sample_pareto_fronts_from_parametric_gp_posterior(  # sample pareto frontier
            obj_model,
            obj_number,
            constraint_models=cons_model,
            cons_num=cons_number,
            sample_pf_num=self._num_pf_samples,
            search_space=self._search_space,
            popsize=self._pop_size,
            num_moo_iter=self._moo_iter,
            moo_solver=self._moo_solver,
            reference_pf_inputs=current_pf_x,
            return_pf_input=True,
        )

        self._pareto_epsilon = self.calculate_maximum_discrepancy_objective_vise(obj_num=obj_number)

        for _fea_pf, _id in zip(self._pf_samples, range(len(self._pf_samples))):
            if _fea_pf is None or tf.size(_fea_pf) == 0:
                print(f"no feasible obs in this {_id}th PF sample ")
        # get partition bounds
        self._partitioned_bounds = [
            prepare_default_non_dominated_partition_bounds(
                tf.constant(
                    [1e20] * obj_number,
                    dtype=datasets[self._objective_tag].query_points.dtype,
                ),
                non_dominated(_fea_pf - _pf_epsilon)[0] if _fea_pf is not None else None,
            )
            for _fea_pf, _pf_epsilon in zip(self._pf_samples, self._pareto_epsilon)
        ]

        if not isinstance(obj_model, HasReparamSampler):
            raise ValueError(
                f"The batch Monte-Carlo expected hyper-volume improvement function only supports "
                f"models that implement a reparam_sampler method; received {obj_model.__repr__()}"
            )
        if not isinstance(cons_model, HasReparamSampler):
            raise ValueError(
                f"The batch Monte-Carlo expected hyper-volume improvement function only supports "
                f"models that implement a reparam_sampler method; received {cons_model.__repr__()}"
            )

        obj_sampler = obj_model.reparam_sampler(self._q_mc_sample_size)
        cons_sampler = cons_model.reparam_sampler(self._q_mc_sample_size)
        if self._parallel_sampling:
            return parallel_feasible_pareto_frontier_entropy_search(
                objective_models=obj_model,
                constraint_models=cons_model,
                partition_bounds=self._partitioned_bounds,
                constraint_threshold=_constraint_threshold,
                obj_sampler=obj_sampler,
                con_sampler=cons_sampler,
                tau=self._tau,
                enable_qmc=self._qMC,
            )
        else:
            return sequential_feasible_pareto_frontier_entropy_search(
                objective_models=obj_model,
                constraint_models=cons_model,
                partition_bounds=self._partitioned_bounds,
                constraint_threshold=_constraint_threshold,
                obj_sampler=obj_sampler,
                con_sampler=cons_sampler,
                tau=self._tau,
            )

    @property
    def get_pf_samples(self):
        return self._pf_samples


class sequential_pareto_frontier_entropy_search(AcquisitionFunctionClass):
    def __init__(
        self,
        models: ProbabilisticModelType,
        partition_bounds: List[Tuple[TensorType, TensorType]],
        pf_samples: List[TensorType],
        sampler: Optional[ReparametrizationSampler] = None,
        tau: float = 1e-2,
        enable_qmc: bool = True,
    ):
        """
        :param partition_bounds
        :param models
        :param sampler reparameterization sampler for obj model
        :param tau temperature parameter, used to soft handle 0-1 event
        :param enable_qmc whether to use quasi-MC sampling for
        """
        assert len(partition_bounds) == len(pf_samples)
        self._model = models
        self._partition_bounds = partition_bounds
        self._sampler = sampler
        self._tau = tau
        self._pf_samples = pf_samples
        self._qMC = enable_qmc

    @tf.function
    def __call__(self, x: TensorType):
        prob_improve = tf.zeros(shape=(tf.shape(x)[0], 1), dtype=x.dtype)  # [N, 1]
        for (lb_points, ub_points), pareto_frontier in zip(
            self._partition_bounds, self._pf_samples
        ):
            # partition the dominated region
            # The upper bound is also a placeholder: as idealy it is inf
            lb_points = tf.maximum(lb_points, -1e100)  # increase numerical stability
            prob_iprv = analytic_non_dominated_prob(
                self._model,
                x,
                lb_points,
                ub_points,
                clip_to_enable_numerical_stability=True,
            )
            prob_improve = tf.concat([prob_improve, prob_iprv], axis=-1)  # [..., N, pf_mc_size + 1]

        # [N, 1 + pf_mc_size] -> [N, 1]
        return tf.reduce_mean(prob_improve[..., 1:], axis=-1, keepdims=True)


class parallel_pareto_frontier_entropy_search(sequential_pareto_frontier_entropy_search):
    """q-PF2ES for MOO problem"""

    def __call__(self, x: TensorType):
        prob_improve = tf.zeros(shape=(tf.shape(x)[0], 1), dtype=x.dtype)  # [N, 1]
        for (lb_points, ub_points), pareto_frontier in zip(
            self._partition_bounds, self._pf_samples
        ):
            # partition the dominated region
            # The upper bound is also a placeholder: as idealy it is inf
            lb_points = tf.maximum(lb_points, -1e100)  # increase numerical stability
            prob_iprv = monte_carlo_non_dominated_prob(
                self._sampler,
                x,
                lb_points,
                ub_points,
                self._tau,
                enable_qmc=self._qMC,
            )
            prob_improve = tf.concat([prob_improve, prob_iprv], axis=-1)  # [..., N, pf_mc_size + 1]

        # [N, 1 + pf_mc_size] -> [N, 1]
        return tf.reduce_mean(-tf.math.log(1 - prob_improve[..., 1:]), axis=-1, keepdims=True)


class sequential_feasible_pareto_frontier_entropy_search(AcquisitionFunctionClass):
    def __init__(
        self,
        objective_models: ProbabilisticModelType,
        constraint_models: ProbabilisticModelType,
        partition_bounds: List[Tuple[TensorType, TensorType]],
        constraint_threshold: TensorType,
        obj_sampler: Optional[ReparametrizationSampler] = None,
        con_sampler: Optional[ReparametrizationSampler] = None,
        tau: float = 1e-2,
        enable_qmc: bool = True,
    ):
        """
        :param objective_models
        :param constraint_models
        :param partition_bounds
        :param obj_sampler
        :param con_sampler
        :param tau
        """
        self._obj_model = objective_models
        self._con_model = constraint_models
        self._partition_bounds = partition_bounds
        self._obj_sampler = obj_sampler
        self._con_sampler = con_sampler
        self._tau = tau
        self._constraint_threshold = constraint_threshold
        self._qMC = enable_qmc

    @tf.function
    def __call__(self, x: TensorType):
        prob_improve = tf.zeros(shape=(x.shape[0], 1), dtype=x.dtype)  # [Batch_dim, 1]

        # pareto class is not yet supported batch, we have to hence rely on a loop
        for lb_points, ub_points in self._partition_bounds:
            # partition the dominated region
            lb_points = tf.maximum(lb_points, -1e100)
            _analytic_pof = analytic_pof(self._con_model, x, self._constraint_threshold)
            prob_iprv = (
                analytic_non_dominated_prob(
                    self._obj_model,
                    x,
                    lb_points,
                    ub_points,
                    clip_to_enable_numerical_stability=True,
                )
                * _analytic_pof
            )

            # improve stability
            prob_iprv = prob_iprv * tf.cast(
                tf.greater_equal(_analytic_pof, 1e-5), dtype=prob_iprv.dtype
            )
            tf.debugging.assert_all_finite(prob_iprv, f"prob_iprv: {prob_iprv} has NaN!")
            prob_improve = tf.concat(
                [prob_improve, prob_iprv], axis=-1
            )  # [pending_mc_size, N, pf_mc_size]
        return tf.reduce_mean(-tf.math.log(1 - prob_improve[..., 1:]), axis=-1, keepdims=True)


class parallel_feasible_pareto_frontier_entropy_search(
    sequential_feasible_pareto_frontier_entropy_search
):
    def __call__(self, x: TensorType):
        prob_improve = tf.zeros(shape=(x.shape[0], 1), dtype=x.dtype)  # [Batch_dim, 1]

        # pareto class is not yet supported batch, we have to hence rely on a loop
        for lb_points, ub_points in self._partition_bounds:
            # partition the dominated region
            lb_points = tf.maximum(lb_points, -1e100)
            prob_iprv = monte_carlo_non_dominated_feasible_prob(
                self._obj_sampler,
                self._con_sampler,
                x,
                lb_points,
                ub_points,
                self._constraint_threshold,
                self._tau,
                enable_qmc=self._qMC,
            )
            # tf.debugging.assert_all_finite(prob_iprv, f"prob_iprv: {prob_iprv} has NaN!")
            prob_improve = tf.concat(
                [prob_improve, prob_iprv], axis=-1
            )  # [pending_mc_size, N, pf_mc_size]
        return tf.reduce_mean(-tf.math.log(1 - prob_improve[..., 1:]), axis=-1, keepdims=True)


def prob_being_in_triangle_region(
    model: ProbabilisticModelType,
    input: TensorType,
    pareto_frontier: TensorType,
):
    def analytical_calculation_of_the_probability(mu_x, mu_y, sigma_x, sigma_y, l1, l2):
        """
        Test of being in the triangular region
        """
        __b = -l2 * sigma_x / (l1 * sigma_y)
        __a = (l1 * l2 - l1 * mu_y - l2 * mu_x) / (l1 * sigma_y)
        rho = -__b / (1 + __b ** 2) ** 0.5
        rv = multivariate_normal(
            [
                0.0,
                0.0,
            ],
            [[1.0, rho], [rho, 1.0]],
        )
        cdf_diff = rv.cdf([__a / ((1 + __b ** 2) ** 0.5), (l1 - mu_x) / sigma_x]) - rv.cdf(
            [__a / ((1 + __b ** 2) ** 0.5), (-mu_x) / sigma_x]
        )
        sub_part = norm().cdf(-mu_y / sigma_y) * (
            norm().cdf((l1 - mu_x) / sigma_x) - norm().cdf((-mu_x) / sigma_x)
        )
        return cdf_diff - sub_part

    from scipy.stats import multivariate_normal, norm

    means, vars = model.predict(input)
    means = tf.squeeze(means, -2)
    vars = tf.squeeze(vars, -2)
    stds = tf.sqrt(vars)
    # sort input front
    sorted_pareto_frontier = tf.gather_nd(
        pareto_frontier, tf.argsort(pareto_frontier[:, :1], axis=0)
    )
    element_res = []
    for mean, std in zip(means, stds):
        element_prob = 0.0
        for pf_point_a, pf_point_b in zip(sorted_pareto_frontier, sorted_pareto_frontier[1:]):
            _l1 = (pf_point_b - pf_point_a)[0]
            _l2 = (pf_point_a - pf_point_b)[1]
            # since we derive the expresion in 1st quadrant， we transform our problem there
            moved_mean = -(mean - tf.convert_to_tensor([pf_point_b[0], pf_point_a[-1]]))
            element_prob += analytical_calculation_of_the_probability(
                moved_mean[0], moved_mean[1], std[0], std[1], _l1, _l2
            )
        element_res.append(element_prob)

    res = tf.convert_to_tensor(element_res)[..., tf.newaxis]
    print(f"max prob in triangle region is: {tf.reduce_max(res)}")
    return res


def analytic_non_dominated_prob(
    model: ProbabilisticModelType,
    inputs: TensorType,
    lower_bounds: TensorType,
    upper_bounds: TensorType,
    clip_to_enable_numerical_stability: bool = True,
) -> TensorType:
    """
    Calculate the probability of non-dominance given the mean and std, this is the
    same as hyper-volume probability of improvemet unless remove_triangle_part is set to True
    :param model
    :param inputs: [N, D]
    :param: lower_bounds: [N, M]
    :param: upper_bounds: [N, M]
    :param clip_to_enable_numerical_stability: if set to True, clip small
        amount ensure numerical stability under logarithm
    :param pareto_frontier
    return
    """
    tf.debugging.assert_shapes(
        [(inputs, ["N", 1, None])],
        message="This acquisition function only supports batch sizes of one.",
    )
    standard_normal = tfp.distributions.Normal(tf.cast(0, inputs.dtype), tf.cast(1, inputs.dtype))
    fmean, fvar = model.predict(tf.squeeze(inputs, -2))
    fvar = tf.clip_by_value(fvar, 1e-100, 1e100)  # clip below to improve numerical stability

    mean = tf.expand_dims(fmean, -2)
    std = tf.expand_dims(tf.sqrt(fvar), -2)
    alpha_u = (upper_bounds - mean) / std  # [..., N_data, N, L]
    alpha_l = (lower_bounds - mean) / std  # [..., N_data, N, L]
    alpha_u = tf.clip_by_value(
        alpha_u, alpha_u.dtype.min, alpha_u.dtype.max
    )  # clip to improve numerical stability
    alpha_l = tf.clip_by_value(
        alpha_l, alpha_l.dtype.min, alpha_l.dtype.max
    )  # clip to improve numerical stability
    z_ml = standard_normal.cdf(alpha_u) - standard_normal.cdf(alpha_l)  # [..., N_Data, N, M+C]
    z_m = tf.reduce_prod(z_ml, axis=-1)  # [..., N_Data, N, M+C] -> [..., N_Data, N]
    z = tf.reduce_sum(z_m, axis=-1, keepdims=True)  # [..., N_Data, 1]
    if clip_to_enable_numerical_stability:
        z = tf.maximum(z, 1e-10)  # clip to improve numerical stability
        z = tf.minimum(z, 1 - 1e-10)  # clip to improve numerical stability

    return z


def monte_carlo_non_dominated_prob(
    sampler,
    inputs: TensorType,
    lower_bound: TensorType,
    upper_bound: TensorType,
    epsilon,
    enable_qmc: bool = True,
) -> TensorType:
    """
    In order to enable batch, we need to change this a bit
    Calculate the probability of non-dominance given the mean and std, this is the
    same as hyper-volume probability of improvemet
    :param sampler
    :param inputs
    :param lower_bound: [N_bd, M]
    :param upper_bound: [N_bd, M]
    :param epsilon
    :param enable_qmc
    return
    """
    observations = sampler.sample(inputs, qMC=enable_qmc)  # [N, mc_num, q, M]
    expand_obs = tf.expand_dims(observations, -3)  # [N, mc_num, 1, q, M]
    # calculate the probability that it is non-dominated
    expand_lower_bound = tf.expand_dims(lower_bound, -2)  # [N_bd, 1, M]
    expand_upper_bound = tf.expand_dims(upper_bound, -2)  # [N_bd, 1, M]
    soft_above_lower_bound = tf.sigmoid(
        (expand_obs - expand_lower_bound) / epsilon
    )  # [N, mc_num, 1, q, M] - [N_bd, 1, M] -> [N, mc_num, N_bd, q, M]
    soft_below_upper_bound = tf.sigmoid(
        (expand_upper_bound - expand_obs) / epsilon
    )  # [N, mc_num, N_bd, q, M]
    soft_any_of_q_in_cell = tf.reduce_prod(
        soft_above_lower_bound * soft_below_upper_bound, axis=-1, keepdims=True
    )  # [N, mc_num, N_bd, q, 1]
    soft_any_of_q_in_cell = tf.reduce_max(soft_any_of_q_in_cell, axis=-2)  # [N, mc_num, N_bd, 1]
    prob_any_non_dominated = tf.reduce_max(soft_any_of_q_in_cell, -2)  # [N, mc_num, 1]
    prob_any_non_dominated = tf.reduce_mean(prob_any_non_dominated, -2)  # [N, 1]
    return prob_any_non_dominated


def monte_carlo_non_dominated_feasible_prob(
    obj_sampler,
    cons_sampler,
    input: TensorType,
    lower_bound: TensorType,
    upper_bound: TensorType,
    constraint_threshold: TensorType,
    tau: float,
    enable_qmc: bool = True,
) -> TensorType:
    """
    Note, this represent the probability that
    probability that at least one of q is feasible and satisfy the constraint
    """
    observations_smp = obj_sampler.sample(input, qMC=enable_qmc)  # [N, mc_num, q, M]
    constraints_smp = cons_sampler.sample(input, qMC=enable_qmc)  # [N, mc_num, q, C]
    tf.debugging.assert_all_finite(observations_smp, f"observations_smp samples: has NaN")
    tf.debugging.assert_all_finite(constraints_smp, f"constraints_smp samples: has NaN")
    # aug_samp = tf.concat([observations_smp, constraints_smp], axis=-1) # [N, mc_num, q, M+C]
    expand_obs = tf.expand_dims(observations_smp, -3)  # [N, mc_num, 1, q, M]
    expand_cons = tf.expand_dims(constraints_smp, -3)  # [N, mc_num, 1, q, C]
    # calculate the probability that it is 1. non-dominated, 2. feasible
    expand_lower_bound = tf.expand_dims(lower_bound, -2)  # [N_bd, 1, M+C]
    expand_upper_bound = tf.expand_dims(upper_bound, -2)  # [N_bd, 1, M+C]

    soft_above_lower_bound = tf.sigmoid(
        (expand_obs - expand_lower_bound) / tau
    )  # [N, mc_num, N_bd, q, M]
    soft_below_upper_bound = tf.sigmoid(
        (expand_upper_bound - expand_obs) / tau
    )  # [N, mc_num, N_bd, q, M]
    # calc feasibility
    soft_satisfy_constraint = tf.reduce_prod(
        tf.sigmoid((constraint_threshold - expand_cons) / tau),
        -1,
        keepdims=True,
    )  # [N, mc_num, 1, q, 1]
    soft_of_any_cand_in_cell_and_feasible = (
        tf.reduce_prod(
            soft_above_lower_bound * soft_below_upper_bound,
            axis=-1,
            keepdims=True,
        )
        * soft_satisfy_constraint
    )  # [N, mc_num, N_bd, q, 1]

    soft_of_any_cand_in_cell_and_feasible = tf.reduce_max(
        soft_of_any_cand_in_cell_and_feasible, axis=-2
    )  # [N, mc_num, N_bd, 1]
    soft_of_any_cand_in_any_cell = tf.reduce_max(
        soft_of_any_cand_in_cell_and_feasible, axis=-2
    )  # [N, mc_num, 1]
    prob_any_non_dominated = tf.reduce_mean(soft_of_any_cand_in_any_cell, axis=-2)  # [N, 1]
    prob_any_non_dominated = tf.minimum(
        prob_any_non_dominated, 1 - 1e-10
    )  # clip to improve numerical stability
    return prob_any_non_dominated


def analytic_pof(
    constraint_predictor: ProbabilisticModelType,
    input: TensorType,
    constraint_threshold,
) -> TensorType:
    """
    assume c(x) < constraint_threshold is feasible
    """
    cmean, cvar = constraint_predictor.predict(tf.squeeze(input, -2))

    cvar = tf.clip_by_value(cvar, 1e-100, 1e100)  # clip below to improve numerical stability
    pof = tf.reduce_prod(
        tfp.distributions.Normal(cmean, tf.sqrt(cvar)).cdf(constraint_threshold),
        axis=-1,
        keepdims=True,
    )  # [MC_size, batch_size, 1]
    return pof


# %% [markdown]
# For a fully demonstration of \{PF\}$^2$ES's utility, besides using the ` BayesianOptimizer` as a defualt choice, we recommend using an `Ask-Tell` interface, since it provides the functionality of visualizing the intermediate Pareto frontiers samples generated acquisition function, which is helpful for us to understand the uncertainty of the Pareto frontier. We provide a modified `AskTellOptimizer` below to return the intermediate data that we would like to inspect.
#
# from typing import Tuple

# %%
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.types import TensorType


class AskTellOptimizer_with_PF_inference(AskTellOptimizer):
    """
    Here, we construct a modified AskTell Interface so that  in each (Batch) iteration, we are able to see
    how the current GP inferred Pareto Frontier looks like, this can give us a hint that how the uncertainty interms
    of Pareto frontier is!
    """

    def ask(self) -> Tuple[TensorType, TensorType]:
        """Suggests a point (or points in batch mode) to observe by optimizing the acquisition
        function. If the acquisition is stateful, its state is saved.

        :return: A :class:`TensorType` instance representing suggested point(s).
        """
        # This trick deserves a comment to explain what's going on
        # acquisition_rule.acquire can return different things:
        # - when acquisition has no state attached, it returns just points
        # - when acquisition has state, it returns a Callable
        #   which, when called, returns state and points
        # so code below is needed to cater for both cases

        with Timer() as query_point_generation_timer:
            points_or_stateful = self._acquisition_rule.acquire(
                self._search_space, self._models, datasets=self._datasets
            )

        if callable(points_or_stateful):
            self._acquisition_state, query_points = points_or_stateful(self._acquisition_state)
        else:
            query_points = points_or_stateful

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                if tf.rank(query_points) == 2:
                    for i in tf.range(tf.shape(query_points)[1]):
                        if len(query_points) == 1:
                            logging.scalar(f"query_points/[{i}]", float(query_points[0, i]))
                        else:
                            logging.histogram(f"query_points/[{i}]", query_points[:, i])
                    logging.histogram(
                        "query_points/euclidean_distances",
                        lambda: pdist(query_points),
                    )
                logging.scalar(
                    "wallclock/query_point_generation",
                    query_point_generation_timer.time,
                )
        assert isinstance(
            ask_tell._acquisition_rule,
            BuilderAccesableEfficientGlobalOptimization,
        )
        assert hasattr(ask_tell._acquisition_rule.builder, "get_pf_samples")
        return query_points, ask_tell._acquisition_rule.builder.get_pf_samples


# %% [markdown]
# ## Demonstration on MOO problem


# %%
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from trieste.experimental.plotting.plotting import create_grid
from trieste.experimental.plotting.inequality_constraints import (
    Simulation,
    plot_2obj_cst_query_points,
)


def plot_mobo_points_in_obj_space(
    obs_values: TensorType,
    num_init: Optional[int] = None,
    mask_fail: Optional[TensorType] = None,
    figsize: Optional[Tuple[float]] = None,
    xlabel: str = "Obj 1",
    ylabel: str = "Obj 2",
    zlabel: str = "Obj 3",
    title: Optional[str] = None,
    m_init: str = "x",
    m_add: str = "o",
    c_pass: str = "tab:green",
    c_fail: str = "tab:red",
    c_pareto: str = "tab:purple",
    only_plot_pareto: bool = False,
    inverse_plot: bool = False,
    ax4plot=None,
    return_path_collections: bool = False,
) -> Union[Axes, Tuple]:
    """
    A modified `plot_mobo_points_in_obj_space` supporting plot on specified axis.

    Adds scatter points in objective space, used for multi-objective optimization (2 or 3
    objectives only). Markers and colors are chosen according to BO factors.

    :param obs_values: TF Tensor or numpy array of objective values, shape (N, 2) or (N, 3).
    :param num_init: initial number of BO points
    :param mask_fail: Bool vector, True if the corresponding observation violates the constraint(s)
    :param figsize: Size of the figure.
    :param xlabel: Label of the X axis.
    :param ylabel: Label of the Y axis.
    :param zlabel: Label of the Z axis (in 3d case).
    :param title: Title of the plot.
    :param m_init: Marker for initial points.
    :param m_add: Marker for the points observed during the BO loop.
    :param c_pass: color for the regular BO points
    :param c_fail: color for the failed BO points
    :param c_pareto: color for the Pareto front points
    :param only_plot_pareto: if set to `True`, only plot the pareto points. Default is `False`.
    :return: figure and axes
    """
    obj_num = obs_values.shape[-1]
    tf.debugging.assert_shapes([])
    assert obj_num == 2 or obj_num == 3, NotImplementedError(
        f"Only support 2/3-objective functions but found: {obj_num}"
    )

    _, dom = non_dominated(obs_values)
    idx_pareto = np.where(dom) if mask_fail is None else np.where(np.logical_and(dom, ~mask_fail))

    pts = obs_values.numpy() if tf.is_tensor(obs_values) else obs_values
    num_pts = pts.shape[0]

    col_pts, mark_pts = format_point_markers(
        num_pts,
        num_init,
        idx_pareto,
        mask_fail,
        m_init,
        m_add,
        c_pass,
        c_fail,
        c_pareto,
    )
    if only_plot_pareto:
        col_pts = col_pts[idx_pareto]
        mark_pts = mark_pts[idx_pareto]
        pts = pts[idx_pareto]

    if ax4plot is None:
        if obj_num == 2:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
    else:
        ax = ax4plot
    path_collections = []
    for i in range(pts.shape[0]):
        if not inverse_plot:
            path_collections.append(ax.scatter(*pts[i], c=col_pts[i], marker=mark_pts[i]))
        else:
            path_collections.append(ax.scatter(*-pts[i], c=col_pts[i], marker=mark_pts[i]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if obj_num == 3:
        ax.set_zlabel(zlabel)
    if title is not None:
        ax.set_title(title)
    if ax4plot is None:
        if return_path_collections is False:
            return fig, ax
        else:
            return fig, ax, path_collections
    else:
        if return_path_collections is False:
            return ax
        else:
            return ax, path_collections


# %% [markdown]
# ### MOO Problem Definition

# %% [markdown]
# We consider the VLMOP2 function --- a synthetic benchmark problem with two objectives. We start by defining the problem parameters.

# %%
import timeit

import gpflow
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist
from tensorflow import linalg

# %%
import trieste
from trieste import logging
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.data import Dataset
from trieste.experimental.plotting import plot_bo_points, plot_function_2d
from trieste.experimental.plotting.plotting import format_point_markers
from trieste.models.gpflow import build_gpr
from trieste.objectives.multi_objectives import VLMOP2
from trieste.observer import OBJECTIVE
from trieste.space import Box, SearchSpace
from trieste.utils import Timer

vlmop2 = VLMOP2(input_dim=2).objective
observer = trieste.objectives.utils.mk_observer(vlmop2)

# %%
mins = [-2, -2]
maxs = [2, 2]
vlmop2_search_space = Box(mins, maxs)
vlmop2_num_objective = 2

# %% [markdown]
# Let's randomly sample some initial data from the observer ...

# %%
num_initial_points = 5  # We use 2d+1 as initial doe number
initial_query_points = vlmop2_search_space.sample(num_initial_points)
vlmop2_initial_data = observer(initial_query_points)

# %% [markdown]
# ... and visualise the data across the design space: each figure contains the contour lines of each objective function.

# %%
_, ax = plot_function_2d(
    vlmop2,
    mins,
    maxs,
    grid_density=100,
    contour=True,
    title=["Obj 1", "Obj 2"],
    figsize=(12, 6),
    colorbar=True,
    xlabel="$X_1$",
    ylabel="$X_2$",
)
plot_bo_points(initial_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(initial_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()


# %%
def _cholesky(matrix):
    """Return a Cholesky factor and boolean success."""
    try:
        chol = tf.linalg.cholesky(matrix)
        ok = tf.reduce_all(tf.math.is_finite(chol))
        return chol, ok
    except tf.errors.InvalidArgumentError:
        return matrix, False


def safer_cholesky(matrix, max_attempts: int = 10, jitter: float = 1e-6):
    def update_diag(matrix, jitter):
        diag = tf.linalg.diag_part(matrix)
        diag_add = tf.ones_like(diag) * jitter
        new_diag = diag_add + diag
        new_matrix = tf.linalg.set_diag(matrix, new_diag)
        return new_matrix

    def cond(state):
        return state[0]

    def body(state):

        _, matrix, jitter, _ = state
        res, ok = _cholesky(matrix)
        new_matrix = tf.cond(ok, lambda: matrix, lambda: update_diag(matrix, jitter))
        break_flag = tf.logical_not(ok)
        return [(break_flag, new_matrix, jitter * 10, res)]

    jitter = tf.cast(jitter, matrix.dtype)
    init_state = (True, update_diag(matrix, jitter), jitter, matrix)
    result = tf.while_loop(cond, body, [init_state], maximum_iterations=max_attempts)

    return result[-1][-1]


class BatchReparametrizationSampler(ReparametrizationSampler[SupportsPredictJoint]):
    r"""
    This sampler employs the *reparameterization trick* to approximate batches of samples from a
    :class:`ProbabilisticModel`\ 's predictive joint distribution as

    .. math:: x \mapsto \mu(x) + \epsilon L(x)

    where :math:`L` is the Cholesky factor s.t. :math:`LL^T` is the covariance, and
    :math:`\epsilon \sim \mathcal N (0, 1)` is constant for a given sampler, thus ensuring samples
    form a continuous curve.
    """

    def __init__(self, sample_size: int, model: SupportsPredictJoint):
        """
        :param sample_size: The number of samples for each batch of points. Must be positive.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        super().__init__(sample_size, model)
        if not isinstance(model, SupportsPredictJoint):
            raise NotImplementedError(
                f"BatchReparametrizationSampler only works with models that support "
                f"predict_joint; received {model.__repr__()}"
            )

        # _eps is essentially a lazy constant. It is declared and assigned an empty tensor here, and
        # populated on the first call to sample
        self._eps = tf.Variable(
            tf.ones([0, 0, sample_size], dtype=tf.float64),
            shape=[None, None, sample_size],
        )  # [0, 0, S]

    def sample(
        self,
        at: TensorType,
        *,
        jitter: float = DEFAULTS.JITTER,
        max_trial: int = 10,
        qMC: bool = False,
        seed: Optional[int] = None,
    ) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`BatchReparametrizationSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`BatchReparametrizationSampler` instances will produce different samples.

        If a Cholesky Decomposition Error happens, we try to remove point that is close together to possibly raise the
        Cholesky issue

        :param at: Batches of query points at which to sample the predictive distribution, with
            shape `[..., B, D]`, for batches of size `B` of points of dimension `D`. Must have a
            consistent batch size across all calls to :meth:`sample` for any given
            :class:`BatchReparametrizationSampler`.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :param max_trial: if Cholesky decomposition has failed, rerun the Cholesky decomposition with a larger Jitter
            till success if the max_trial number for Cholesky decomposition is not reached
        :param qMC: whether to enable quasi-Monte Carlo Sampling for the batch sampler
        :return: The samples, of shape `[..., S, B, L]`, where `S` is the `sample_size`, `B` the
            number of points per batch, and `L` the dimension of the model's predictive
            distribution.
        :raise ValueError (or InvalidArgumentError): If any of the following are true:
            - ``at`` is a scalar.
            - The batch size `B` of ``at`` is not positive.
            - The batch size `B` of ``at`` differs from that of previous calls.
            - ``jitter`` is negative.
        """
        tf.debugging.assert_rank_at_least(at, 2)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        batch_size = at.shape[-2]

        tf.debugging.assert_positive(batch_size)

        if self._initialized:
            if qMC is False:
                tf.debugging.assert_equal(
                    batch_size,
                    tf.shape(self._eps)[-2],
                    f"{type(self).__name__} requires a fixed batch size. Got batch size {batch_size}"
                    f" but previous batch size was {tf.shape(self._eps)[-2]}.",
                )
            else:
                pass

        mean, cov = self._model.predict_joint(at)  # [..., B, L], [..., L, B, B]

        if qMC is False:
            if not self._initialized:
                self._eps.assign(
                    tf.random.normal(
                        [tf.shape(mean)[-1], batch_size, self._sample_size],
                        dtype=tf.float64,
                        seed=seed,
                    )  # [L, B, S]
                )
                self._initialized.assign(True)

            identity = tf.eye(batch_size, dtype=cov.dtype)  # [B, B]

            cov_cholesky = safer_cholesky(cov + jitter * identity)  # [..., L, B, B]

            variance_contribution = cov_cholesky @ tf.cast(self._eps, cov.dtype)  # [..., L, B, S]
            tf.debugging.assert_all_finite(
                variance_contribution,
                message=f"variance_contribution {variance_contribution} have NaN",
            )
            leading_indices = tf.range(tf.rank(variance_contribution) - 3)
            absolute_trailing_indices = [-1, -2, -3] + tf.rank(variance_contribution)
            new_order = tf.concat([leading_indices, absolute_trailing_indices], axis=0)

            return mean[..., None, :, :] + tf.transpose(variance_contribution, new_order)
        else:  # enable qMC sampler
            if not self._initialized:
                # The cholesky matrix for decomposition is BL * BL
                # note the 1st dim is just used for align with the specification of eps shape, its not used in reality
                self._eps.assign(  # NOTE!!! This is not exactly the same shape as the eps used in naive Monte Carlo!
                    tf.transpose(
                        QuasiMonteCarloNormalSampler(
                            dimensionality=batch_size * tf.shape(mean)[-1]
                        ).sample(self._sample_size, dtype=tf.float64, seed=seed)
                    )[None, ...]
                )  # [1, BL, S])
                # self._eps.assign(tf.random.normal([1, batch_size * tf.shape(mean)[
                #     -1], self._sample_size], dtype = tf.float64, seed = seed))  # [1, BL, S])
                self._initialized.assign(True)
            # We need to sample q * M in a whole, first we construct the full covariance matrix [..., BL, BL],
            # which is a block matrix with block size B * B
            splitted_cov = tf.split(
                cov, axis=-3, num_or_size_splits=cov.shape[-3]
            )  # split along output dimension
            linop_blocks = [
                linalg.LinearOperatorFullMatrix(tf.squeeze(block, axis=-3))
                for block in splitted_cov
            ]
            aug_cov = linalg.LinearOperatorBlockDiag(linop_blocks).to_dense()  # [..., B * L, B * L]
            aug_identity = tf.eye(batch_size * tf.shape(mean)[-1], dtype=aug_cov.dtype)
            cov_cholesky = safer_cholesky(aug_cov + jitter * aug_identity)  # [..., BL, BL]
            cov_cholesky = tf.where(
                tf.math.is_nan(cov_cholesky),
                tf.zeros_like(cov_cholesky, dtype=cov_cholesky.dtype),
                cov_cholesky,
            )
            tf.debugging.assert_all_finite(
                cov_cholesky,
                message="qMC sampler covariance decomposition has NaN",
            )
            # [..., BL, BL] * [..., BL, S] -> [..., BL, S]
            variance_contribution = cov_cholesky @ tf.cast(self._eps[0], cov.dtype)
            # [..., B, L] -> [..., BL]
            aug_mean = tf.reshape(
                tf.transpose(
                    mean,
                    perm=tf.concat(
                        [tf.range(tf.rank(mean) - 2), [-1, -2] + tf.rank(mean)],
                        axis=0,
                    ),
                ),
                shape=tf.concat([mean.shape[:-2], [batch_size * tf.shape(mean)[-1]]], axis=0),
            )
            aug_sample = tf.expand_dims(aug_mean, axis=-1) + variance_contribution  # [..., BL, S]
            # [..., BL, S] -> [..., S, BL]
            leading_indices = tf.range(tf.rank(variance_contribution) - 2)
            absolute_trailing_indices = [-1, -2] + tf.rank(variance_contribution)
            new_order = tf.concat([leading_indices, absolute_trailing_indices], axis=0)
            aug_sample = tf.transpose(aug_sample, perm=new_order)
            # [..., S, BL] -> [..., S, B, L]
            new_shape = tf.concat(
                [tf.shape(aug_sample)[:-1], [tf.shape(mean)[-1]], [batch_size]],
                axis=0,
            )
            # return tf.reshape(aug_sample, shape=new_shape)
            return tf.transpose(
                tf.reshape(aug_sample, shape=new_shape),
                perm=tf.concat(
                    [
                        tf.range(tf.shape(new_shape) - 2),
                        [-1, -2] + tf.shape(new_shape),
                    ],
                    axis=0,
                ),
            )


from trieste.models.gpflow.models import (
    GaussianProcessRegression,
    GPflowPredictor,
)


class GaussianProcessRegression_with_qMC_sampler(GaussianProcessRegression):
    """
    GaussianProcessRegression with qMC reparam sampler
    """

    def reparam_sampler(self, num_samples: int) -> ReparametrizationSampler[GPflowPredictor]:
        """
        Return a reparametrization sampler providing `num_samples` samples.

        :return: The reparametrization sampler.
        """
        return BatchReparametrizationSampler(num_samples, self)


def build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
    data: Dataset, num_output: int, search_space: SearchSpace
) -> TrainableHasTrajectoryAndPredictJointReparamModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        gpr = build_gpr(
            single_obj_data,
            search_space,
            likelihood_variance=1e-7,
            trainable_likelihood=False,
        )
        gprs.append(
            (
                GaussianProcessRegression_with_qMC_sampler(gpr, use_decoupled_sampler=False),
                1,
            )
        )

    return TrainableHasTrajectoryAndPredictJointReparamModelStack(*gprs)


# %%
vlmop2_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
    vlmop2_initial_data, vlmop2_num_objective, vlmop2_search_space
)
ask_tell = AskTellOptimizer_with_PF_inference(
    vlmop2_search_space,
    vlmop2_initial_data,
    vlmop2_model,
    acquisition_rule=BuilderAccesableEfficientGlobalOptimization(PF2ES(vlmop2_search_space)),
)

# %% [markdown]
# ### Sequential MOO by \{PF\}$^2$ES (~10min)

# %% [markdown]
# We now conduct multi-objective optimization on VLMOP2 based on our sequential \{PF\}$^2$ES, the whole below process may takes around 6-10 minutes.
#
# We note since we demonstrate the plot investigation of intermediate results within each BO iter here, this consumes larger time than just performing BO for \{PF\}$^2$ES, which's acquiring time has been printed below and only takes around 4 mins.

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from IPython.display import HTML

n_steps = 20

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axs[0].set_xlabel("Objective 1")
axs[0].set_ylabel("Objective 2")
axs[0].set_title("GP Inferred Pareto Frontiers $\\tilde{\mathcal{F}}$")

axs[1].set_title("BO Samples in Objective Space")

ims = []  # for plot usage
for step in range(n_steps):
    start = timeit.default_timer()
    new_point, gp_inferred_pfs = ask_tell.ask()
    stop = timeit.default_timer()

    print(f"Acq Func Sample Time at step {step + 1}: {stop - start} sec")

    pred_mean, pred_var = ask_tell._models[OBJECTIVE].predict(new_point)

    new_data = observer(new_point)
    ask_tell.tell(new_data)

    # plot inferred Pareto frontier
    _ims = [
        axs[0].scatter(gp_inferred_pf[:, 0], gp_inferred_pf[:, 1], s=5)
        for gp_inferred_pf in gp_inferred_pfs
    ]

    im1 = axs[0].scatter(
        *tf.split(pred_mean, 2, axis=-1),
        label="Predicted BO Sample Data",
        color="orange",
    )
    ellipse = Ellipse(
        (pred_mean[0, 0], pred_mean[0, 1]),
        2 * tf.sqrt(pred_var[0, 0]),
        2 * tf.sqrt(pred_var[0, 1]),
        angle=0,
        alpha=0.2,
        edgecolor="k",
    )
    im2 = axs[0].add_artist(ellipse)

    if step == 0:
        im3 = axs[0].legend()
    axs[0].add_artist(im3)  # https://github.com/matplotlib/matplotlib/issues/12833
    # plot actual BO samples
    axs[1], ploted_path_lists = plot_mobo_points_in_obj_space(
        vlmop2_model.get_internal_data().observations,
        ax4plot=axs[1],
        return_path_collections=True,
    )

    ims.append(_ims + [im1, im2, im3] + ploted_path_lists)

ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=False)
plt.close()


HTML(ani.to_jshtml())

# %%
# %matplotlib notebook
data_query_points = ask_tell.datasets[OBJECTIVE].query_points
data_observations = ask_tell.datasets[OBJECTIVE].observations

_, ax = plot_function_2d(
    vlmop2,
    mins,
    maxs,
    grid_density=100,
    contour=True,
    figsize=(12, 6),
    title=["Obj 1", "Obj 2"],
    xlabel="$X_1$",
    ylabel="$X_2$",
    colorbar=True,
)
plot_bo_points(data_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(data_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()

# %% [markdown]
# As we are able to see from the 1st subfigure: the uncertainty of Pareto frontier samples getting decrease w.r.t the BO samples.
#
# For the optimal recommendation, we perform an *Out-of-sample* recommendation on the lastest GP model we have, here we also plot a "reference Pareto frontier" investigating how good the Out-of-sample recommendation is

# %%
rec_pf, rec_pf_inputs = inference_pareto_fronts_from_gp_mean(
    vlmop2_model, vlmop2_search_space, popsize=50
)

real_pf = vlmop2(rec_pf_inputs)  # 'expensive' evaluation

fig, ax = plt.subplots()
plot_mobo_points_in_obj_space(real_pf, ax4plot=ax)
ax.scatter(
    *tf.split(VLMOP2(input_dim=2).gen_pareto_optimal_points(50), 2, -1),
    label="reference Pareto Frontier",
    s=5,
)
plt.legend()
plt.title("Comparison of Out-of-sample recommendation vs reference Pareto Frontier")
plt.show()

# %% [markdown]
# ### Batched MOO by q- \{PF\}$^2$ES (~15-20min)

# %% [markdown]
# We now conduct show that \{PF\}$^2$ES can also conduct batch multi-objective optimization (we referred to as q-\{PF\}$^2$ES), given parallel computation resources, this allow taking smaller BO iterations while achiving similar performance result on optimal Pareto frontier recommendation.
#
# We note the only change to enable q- \{PF\}$^2$ES is by specifying `parallel_sampling=True` to let the acquisition function know its performed in a batch setting, we can also specify the Monte Carlo sample size to 128 by using `batch_mc_sample_size=128` (here for speed we only use 64), eventually, we set `num_query_points=2`as we would like to sample 2 points in a batch.

# %%
vlmop2_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
    vlmop2_initial_data, vlmop2_num_objective, vlmop2_search_space
)
ask_tell = AskTellOptimizer_with_PF_inference(
    vlmop2_search_space,
    vlmop2_initial_data,
    vlmop2_model,
    acquisition_rule=BuilderAccesableEfficientGlobalOptimization(
        PF2ES(vlmop2_search_space, parallel_sampling=True, batch_mc_sample_size=64),
        num_query_points=2,
    ),
)

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from IPython.display import HTML

n_steps = 10

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axs[0].set_xlabel("Objective 1")
axs[0].set_ylabel("Objective 2")
axs[0].set_title("GP Inferred Pareto Frontiers $\\tilde{\mathcal{F}}$")
axs[0].legend()
axs[1].legend()
axs[1].set_title("BO Samples in Objective Space")

ims = []  # for plot usage
for step in range(n_steps):
    start = timeit.default_timer()
    new_point, gp_inferred_pfs = ask_tell.ask()
    stop = timeit.default_timer()

    print(f"Acq Func Sample Time at step {step + 1}: {stop - start} sec")

    pred_means, pred_vars = ask_tell._models[OBJECTIVE].predict(new_point)

    new_data = observer(new_point)
    ask_tell.tell(new_data)
    # plot inferred Pareto frontier
    _ims = [
        axs[0].scatter(gp_inferred_pf[:, 0], gp_inferred_pf[:, 1], s=5)
        for gp_inferred_pf in gp_inferred_pfs
    ]

    im12s = []
    for pred_mean, pred_var in zip(pred_means, pred_vars):
        im12s.append(
            axs[0].scatter(
                *tf.split(pred_mean, 2, axis=-1),
                color="r",
                label="Predicted BO Sample Data",
            )
        )
        ellipse = Ellipse(
            (pred_mean[0], pred_mean[1]),
            2 * tf.sqrt(pred_var[0]),
            2 * tf.sqrt(pred_var[1]),
            angle=0,
            alpha=0.2,
            edgecolor="k",
        )
        im12s.append(axs[0].add_artist(ellipse))
    if step == 0:
        im3 = axs[0].legend()
    axs[0].add_artist(im3)  # https://github.com/matplotlib/matplotlib/issues/12833
    im12s.append(im3)
    # plot actual BO samples
    axs[1], ploted_path_lists = plot_mobo_points_in_obj_space(
        vlmop2_model.get_internal_data().observations,
        ax4plot=axs[1],
        return_path_collections=True,
    )

    ims.append(_ims + im12s + ploted_path_lists)

ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=False)
plt.close()


HTML(ani.to_jshtml())

# %%
# %matplotlib notebook
data_query_points = ask_tell.datasets[OBJECTIVE].query_points
data_observations = ask_tell.datasets[OBJECTIVE].observations
plt.figure()
_, ax = plot_function_2d(
    vlmop2,
    mins,
    maxs,
    grid_density=100,
    contour=True,
    figsize=(12, 6),
    title=["Obj 1", "Obj 2"],
    xlabel="$X_1$",
    ylabel="$X_2$",
    colorbar=True,
)
plot_bo_points(data_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(data_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()

# %% [markdown]
# As we are able to see from the 1st subfigure: the uncertainty of Pareto frontier samples getting decrease w.r.t the BO samples.
#
# For the optimal recommendation, we can perform an *Out-of-sample* recommendation on the lastest GP model we have, here we also plot a "reference Pareto frontier" investigating how good the Out-of-sample recommendation is

# %%
rec_pf, rec_pf_inputs = inference_pareto_fronts_from_gp_mean(
    vlmop2_model, vlmop2_search_space, popsize=50
)

real_pf = vlmop2(rec_pf_inputs)  # 'expensive' evaluation

fig, ax = plt.subplots()
plot_mobo_points_in_obj_space(real_pf, ax4plot=ax)
ax.scatter(
    *tf.split(VLMOP2(input_dim=2).gen_pareto_optimal_points(50), 2, -1),
    label="reference Pareto Frontier",
    s=5,
)
plt.legend()
plt.title("Comparison of Out-of-sample recommendation vs reference Pareto Frontier")
plt.show()

# %% [markdown]
# -----------------

# %% [markdown]
# ## Demonstration on  CMOO problem

# %% [markdown]
# We now demonstrate that \{PF\}$^2$ES and its parallel version are also able to perform Constraint MOO (CMOO) problem.

# %% [markdown]
# ### CMOO Problem Definition

# %% [markdown]
# As we are able to see from the 1st subfigure: the uncertainty of Pareto frontier samples getting decrease w.r.t the BO samples.
#
# For the optimal recommendation, we can perform an *Out-of-sample* recommendation on the lastest GP model we have, here we also plot a "reference Pareto frontier" investigating how good the Out-of-sample recommendation is

# %%
CONSTRAINT = "CONSTRAINT"

# %%
num_initial_points = 5
cvlmop2_num_objective = 2
cvlmop2_num_constraints = 1


class Sim:
    threshold = 0.0

    @staticmethod
    def objective(input_data):
        return vlmop2(input_data)

    @staticmethod
    def constraint(input_data):
        x, y = input_data[:, -2], input_data[:, -1]
        z = tf.cos(x) * tf.cos(y) - tf.sin(x) * tf.sin(y)
        return z[:, None] - 0.75


def observer_cst(query_points):
    return {
        OBJECTIVE: Dataset(query_points, Sim.objective(query_points)),
        CONSTRAINT: Dataset(query_points, Sim.constraint(query_points)),
    }


cvlmop2_initial_query_points = vlmop2_search_space.sample(num_initial_points)
cvlmop2_initial_data_with_cst = observer_cst(cvlmop2_initial_query_points)
cvlmop2_search_space = vlmop2_search_space

# %%
plot_2obj_cst_query_points(
    cvlmop2_search_space,
    Sim,
    cvlmop2_initial_data_with_cst[OBJECTIVE].astuple(),
    cvlmop2_initial_data_with_cst[CONSTRAINT].astuple(),
)
plt.show()

mask_fail = cvlmop2_initial_data_with_cst[CONSTRAINT].observations.numpy() > 0
plot_mobo_points_in_obj_space(
    cvlmop2_initial_data_with_cst[OBJECTIVE].observations,
    mask_fail=mask_fail[:, 0],
)
plt.show()

# %% [markdown]
# ### Sequential MOO by \{PF\}$^2$ES (~20min)

# %%
import copy  # for not changing cvlmop2_initial_data_with_cst, which is reused in Batch CMOO

cvlmop2_model = {
    OBJECTIVE: build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        copy.deepcopy(cvlmop2_initial_data_with_cst[OBJECTIVE]),
        num_output=cvlmop2_num_objective,
        search_space=cvlmop2_search_space,
    ),
    CONSTRAINT: build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        copy.deepcopy(cvlmop2_initial_data_with_cst[CONSTRAINT]),
        num_output=cvlmop2_num_constraints,
        search_space=cvlmop2_search_space,
    ),
}

ask_tell = AskTellOptimizer_with_PF_inference(
    cvlmop2_search_space,
    copy.deepcopy(cvlmop2_initial_data_with_cst),
    cvlmop2_model,
    acquisition_rule=BuilderAccesableEfficientGlobalOptimization(
        PF2ES(cvlmop2_search_space, constraint_tag=CONSTRAINT)
    ),
)

# %% [markdown]
# We now conduct CMOO on constraint-VLMOP2 based on our sequential \{PF\}$^2$ES, the whole below process may takes around 6-10 minutes. Note the experiments conducted below may take 20 minutes

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from IPython.display import HTML

n_steps = 30

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axs[0].set_xlabel("Objective 1")
axs[0].set_ylabel("Objective 2")
axs[0].set_title("GP Inferred Pareto Frontiers $\\tilde{\mathcal{F}}$")

axs[1].set_title("BO Samples in Objective Space")

ims = []  # for plot usage
for step in range(n_steps):
    start = timeit.default_timer()
    new_point, gp_inferred_pfs = ask_tell.ask()
    stop = timeit.default_timer()

    print(f"Acq Func Sample Time at step {step + 1}: {stop - start} sec")

    pred_mean, pred_var = ask_tell._models[OBJECTIVE].predict(new_point)

    new_data = observer_cst(new_point)
    ask_tell.tell(new_data)

    # plot inferred Pareto frontier
    _ims = [
        axs[0].scatter(gp_inferred_pf[:, 0], gp_inferred_pf[:, 1], s=5)
        for gp_inferred_pf in gp_inferred_pfs
    ]

    im1 = axs[0].scatter(
        *tf.split(pred_mean, 2, axis=-1),
        label="Predicted BO Sample Data",
        color="orange",
    )
    ellipse = Ellipse(
        (pred_mean[0, 0], pred_mean[0, 1]),
        2 * tf.sqrt(pred_var[0, 0]),
        2 * tf.sqrt(pred_var[0, 1]),
        angle=0,
        alpha=0.2,
        edgecolor="k",
    )
    im2 = axs[0].add_artist(ellipse)
    if step == 0:
        im3 = axs[0].legend()
    axs[0].add_artist(im3)  # https://github.com/matplotlib/matplotlib/issues/12833
    # plot actual BO
    mask_fail = ask_tell._models[CONSTRAINT].get_internal_data().observations.numpy() > 0
    axs[1], ploted_path_lists = plot_mobo_points_in_obj_space(
        cvlmop2_model[OBJECTIVE].get_internal_data().observations,
        ax4plot=axs[1],
        return_path_collections=True,
        mask_fail=mask_fail[:, 0],
    )

    ims.append(_ims + [im1, im2, im3] + ploted_path_lists)

ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=False)
plt.close()


HTML(ani.to_jshtml())

# %%
# %matplotlib notebook
objective_dataset = ask_tell.datasets[OBJECTIVE]
constraint_dataset = ask_tell.datasets[CONSTRAINT]

plot_2obj_cst_query_points(
    cvlmop2_search_space,
    Sim,
    objective_dataset.astuple(),
    constraint_dataset.astuple(),
)
plt.show()

# %% [markdown]
# As we are able to see from the 1st subfigure: the uncertainty of Pareto frontier samples getting decrease w.r.t the BO samples.
#
# For the optimal recommendation, we can perform an *Out-of-sample* recommendation on the lastest GP model we have, here we also plot a "reference Pareto frontier" investigating how good the Out-of-sample recommendation is

# %%
# Out-of-sample recommendation
rec_pf, rec_pf_inputs = inference_pareto_fronts_from_gp_mean(
    cvlmop2_model[OBJECTIVE],
    cvlmop2_search_space,
    popsize=50,
    cons_models=cvlmop2_model[CONSTRAINT],
    min_feasibility_probability=0.95,
    constraint_enforce_percentage=5e-3,
)


# we generate the reference feasible Pareto frontier based on the real problem
reference_pf = moo_nsga2_pymoo(
    Sim().objective,
    input_dim=2,
    obj_num=2,
    bounds=tf.convert_to_tensor(VLMOP2(input_dim=2).bounds),
    popsize=50,
    cons=Sim().constraint,
    cons_num=1,
)

# %%
rec_actual_obs_datasets = observer_cst(rec_pf_inputs)  # 'expensive' evaluation

fig, ax = plt.subplots()
mask_fail = rec_actual_obs_datasets[CONSTRAINT].observations.numpy() > 0
plot_mobo_points_in_obj_space(
    rec_actual_obs_datasets[OBJECTIVE].observations,
    mask_fail=mask_fail[:, 0],
    ax4plot=ax,
)
ax.scatter(
    *tf.split(reference_pf.fronts, 2, -1),
    label="reference Pareto Frontier",
    s=5,
)
plt.legend()
plt.title("Comparison of Out-of-sample recommendation vs reference Pareto Frontier")
plt.show()


# %% [markdown]
# ### Batch CMOO by q- \{PF\}$^2$ES (~20min)

# %% [markdown]
# Eventually, we use q-\{PF\}$^2$ES) for CMOO. Note the whole experiments may take around 20 min.

# %%
import copy

cvlmop2_model = {
    OBJECTIVE: build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        copy.deepcopy(cvlmop2_initial_data_with_cst[OBJECTIVE]),
        num_output=cvlmop2_num_objective,
        search_space=cvlmop2_search_space,
    ),
    CONSTRAINT: build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        copy.deepcopy(cvlmop2_initial_data_with_cst[CONSTRAINT]),
        num_output=cvlmop2_num_constraints,
        search_space=cvlmop2_search_space,
    ),
}

ask_tell = AskTellOptimizer_with_PF_inference(
    cvlmop2_search_space,
    copy.deepcopy(cvlmop2_initial_data_with_cst),
    cvlmop2_model,
    acquisition_rule=BuilderAccesableEfficientGlobalOptimization(
        PF2ES(
            cvlmop2_search_space,
            constraint_tag=CONSTRAINT,
            parallel_sampling=True,
            batch_mc_sample_size=64,
        ),
        num_query_points=2,
    ),
)

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from IPython.display import HTML

n_steps = 15

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axs[0].set_xlabel("Objective 1")
axs[0].set_ylabel("Objective 2")
axs[0].set_title("GP Inferred Pareto Frontiers $\\tilde{\mathcal{F}}$")

axs[1].set_title("BO Samples in Objective Space")

ims = []  # for plot usage
for step in range(n_steps):
    start = timeit.default_timer()
    new_point, gp_inferred_pfs = ask_tell.ask()
    stop = timeit.default_timer()

    print(f"Acq Func Sample Time at step {step + 1}: {stop - start} sec")

    pred_means, pred_vars = ask_tell._models[OBJECTIVE].predict(new_point)

    new_data = observer_cst(new_point)
    ask_tell.tell(new_data)

    # plot inferred Pareto frontier
    _ims = [
        axs[0].scatter(gp_inferred_pf[:, 0], gp_inferred_pf[:, 1], s=5)
        for gp_inferred_pf in gp_inferred_pfs
    ]

    im12s = []
    for pred_mean, pred_var in zip(pred_means, pred_vars):
        im12s.append(
            axs[0].scatter(
                *tf.split(pred_mean, 2, axis=-1),
                color="r",
                label="Predicted BO Sample Data",
            )
        )
        ellipse = Ellipse(
            (pred_mean[0], pred_mean[1]),
            2 * tf.sqrt(pred_var[0]),
            2 * tf.sqrt(pred_var[1]),
            angle=0,
            alpha=0.2,
            edgecolor="k",
        )
        im12s.append(axs[0].add_artist(ellipse))
    if step == 0:
        im3 = axs[0].legend()
    axs[0].add_artist(im3)  # https://github.com/matplotlib/matplotlib/issues/12833
    # plot actual BO
    mask_fail = ask_tell._models[CONSTRAINT].get_internal_data().observations.numpy() > 0
    axs[1], ploted_path_lists = plot_mobo_points_in_obj_space(
        cvlmop2_model[OBJECTIVE].get_internal_data().observations,
        ax4plot=axs[1],
        return_path_collections=True,
        mask_fail=mask_fail[:, 0],
    )

    ims.append(_ims + im12s + ploted_path_lists)

ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=False)
plt.close()


HTML(ani.to_jshtml())

# %%
# %matplotlib notebook
objective_dataset = ask_tell.datasets[OBJECTIVE]
constraint_dataset = ask_tell.datasets[CONSTRAINT]
data_query_points = objective_dataset.query_points
data_observations = objective_dataset.observations

plot_2obj_cst_query_points(
    cvlmop2_search_space,
    Sim,
    objective_dataset.astuple(),
    constraint_dataset.astuple(),
)
plt.show()

# %% [markdown]
# As we are able to see from the 1st subfigure: the uncertainty of Pareto frontier samples getting decrease w.r.t the BO samples.
#
# For the optimal recommendation, we can perform an *Out-of-sample* recommendation on the lastest GP model we have, here we also plot a "reference Pareto frontier" investigating how good the Out-of-sample recommendation is

# %%
# Out-of-sample recommendation
rec_pf, rec_pf_inputs = inference_pareto_fronts_from_gp_mean(
    cvlmop2_model[OBJECTIVE],
    cvlmop2_search_space,
    popsize=50,
    cons_models=cvlmop2_model[CONSTRAINT],
    min_feasibility_probability=0.95,
    constraint_enforce_percentage=5e-3,
)


# we generate the reference feasible Pareto frontier based on the real problem
reference_pf = moo_nsga2_pymoo(
    Sim().objective,
    input_dim=2,
    obj_num=2,
    bounds=tf.convert_to_tensor(VLMOP2(input_dim=2).bounds),
    popsize=50,
    cons=Sim().constraint,
    cons_num=1,
)

# %%
rec_actual_obs_datasets = observer_cst(rec_pf_inputs)  # 'expensive' evaluation

fig, ax = plt.subplots()
mask_fail = rec_actual_obs_datasets[CONSTRAINT].observations.numpy() > 0
plot_mobo_points_in_obj_space(
    rec_actual_obs_datasets[OBJECTIVE].observations,
    mask_fail=mask_fail[:, 0],
    ax4plot=ax,
)
ax.scatter(
    *tf.split(reference_pf.fronts, 2, -1),
    label="reference Pareto Frontier",
    s=5,
)
plt.legend()
plt.title("Comparison of Out-of-sample recommendation vs reference Pareto Frontier")
plt.show()

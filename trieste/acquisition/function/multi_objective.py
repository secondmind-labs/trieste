# Copyright 2021 The Trieste Contributors
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
"""
This module contains multi-objective acquisition function builders.
"""
from __future__ import annotations

import math
from itertools import combinations, product
from typing import Callable, Mapping, Optional, Sequence, cast

import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...models import ProbabilisticModel, ReparametrizationSampler
from ...models.interfaces import HasReparamSampler
from ...observer import OBJECTIVE
from ...types import Tag, TensorType
from ...utils import DEFAULTS
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    AcquisitionFunctionClass,
    GreedyAcquisitionFunctionBuilder,
    PenalizationFunction,
    ProbabilisticModelType,
    SingleModelAcquisitionBuilder,
)
from ..multi_objective.pareto import (
    Pareto,
    get_reference_point,
    prepare_default_non_dominated_partition_bounds,
)
from .function import ExpectedConstrainedImprovement


class ExpectedHypervolumeImprovement(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    """
    Builder for the expected hypervolume improvement acquisition function.
    The implementation of the acquisition function largely
    follows :cite:`yang2019efficient`
    """

    def __init__(
        self,
        reference_point_spec: (
            Sequence[float] | TensorType | Callable[..., TensorType]
        ) = get_reference_point,
    ):
        """
        :param reference_point_spec: this method is used to determine how the reference point is
            calculated. If a Callable function specified, it is expected to take existing
            posterior mean-based observations (to screen out the observation noise) and return
            a reference point with shape [D] (D represents number of objectives). If the Pareto
            front location is known, this arg can be used to specify a fixed reference point
            in each bo iteration. A dynamic reference point updating strategy is used by
            default to set a reference point according to the datasets.
        """
        if callable(reference_point_spec):
            self._ref_point_spec: tf.Tensor | Callable[..., TensorType] = reference_point_spec
        else:
            self._ref_point_spec = tf.convert_to_tensor(reference_point_spec)
        self._ref_point = None

    def __repr__(self) -> str:
        """"""
        if callable(self._ref_point_spec):
            return f"ExpectedHypervolumeImprovement({self._ref_point_spec.__name__})"
        else:
            return f"ExpectedHypervolumeImprovement({self._ref_point_spec!r})"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :return: The expected hypervolume improvement acquisition function.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)

        if callable(self._ref_point_spec):
            self._ref_point = tf.cast(self._ref_point_spec(mean), dtype=mean.dtype)
        else:
            self._ref_point = tf.cast(self._ref_point_spec, dtype=mean.dtype)

        _pf = Pareto(mean)
        screened_front = _pf.front[tf.reduce_all(_pf.front <= self._ref_point, -1)]
        # prepare the partitioned bounds of non-dominated region for calculating of the
        # hypervolume improvement in this area
        _partition_bounds = prepare_default_non_dominated_partition_bounds(
            self._ref_point, screened_front
        )
        return expected_hv_improvement(model, _partition_bounds)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, expected_hv_improvement), [tf.constant([])])
        mean, _ = model.predict(dataset.query_points)

        if callable(self._ref_point_spec):
            self._ref_point = self._ref_point_spec(mean)
        else:
            assert isinstance(self._ref_point_spec, tf.Tensor)  # specified a fixed ref point
            self._ref_point = tf.cast(self._ref_point_spec, dtype=mean.dtype)

        _pf = Pareto(mean)
        screened_front = _pf.front[tf.reduce_all(_pf.front <= self._ref_point, -1)]
        _partition_bounds = prepare_default_non_dominated_partition_bounds(
            self._ref_point, screened_front
        )
        function.update(_partition_bounds)  # type: ignore
        return function


class expected_hv_improvement(AcquisitionFunctionClass):
    def __init__(self, model: ProbabilisticModel, partition_bounds: tuple[TensorType, TensorType]):
        r"""
        expected Hyper-volume (HV) calculating using Eq. 44 of :cite:`yang2019efficient` paper.
        The expected hypervolume improvement calculation in the non-dominated region
        can be decomposed into sub-calculations based on each partitioned cell.
        For easier calculation, this sub-calculation can be reformulated as a combination
        of two generalized expected improvements, corresponding to Psi (Eq. 44) and Nu (Eq. 45)
        function calculations, respectively.

        Note:
        1. Since in Trieste we do not assume the use of a certain non-dominated region partition
        algorithm, we do not assume the last dimension of the partitioned cell has only one
        (lower) bound (i.e., minus infinity, which is used in the :cite:`yang2019efficient` paper).
        This is not as efficient as the original paper, but is applicable to different non-dominated
        partition algorithm.
        2. As the Psi and nu function in the original paper are defined for maximization problems,
        we inverse our minimisation problem (to also be a maximisation), allowing use of the
        original notation and equations.

        :param model: The model of the objective function.
        :param partition_bounds: with shape ([N, D], [N, D]), partitioned non-dominated hypercell
            bounds for hypervolume improvement calculation
        :return: The expected_hv_improvement acquisition function modified for objective
            minimisation. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        """
        self._model = model
        self._lb_points = tf.Variable(
            partition_bounds[0], trainable=False, shape=[None, partition_bounds[0].shape[-1]]
        )
        self._ub_points = tf.Variable(
            partition_bounds[1], trainable=False, shape=[None, partition_bounds[1].shape[-1]]
        )
        self._cross_index = tf.constant(
            list(product(*[[0, 1]] * self._lb_points.shape[-1]))
        )  # [2^d, indices_at_dim]

    def update(self, partition_bounds: tuple[TensorType, TensorType]) -> None:
        """Update the acquisition function with new partition bounds."""
        self._lb_points.assign(partition_bounds[0])
        self._ub_points.assign(partition_bounds[1])

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        normal = tfp.distributions.Normal(
            loc=tf.zeros(shape=1, dtype=x.dtype), scale=tf.ones(shape=1, dtype=x.dtype)
        )

        def Psi(a: TensorType, b: TensorType, mean: TensorType, std: TensorType) -> TensorType:
            return std * normal.prob((b - mean) / std) + (mean - a) * (
                1 - normal.cdf((b - mean) / std)
            )

        def nu(lb: TensorType, ub: TensorType, mean: TensorType, std: TensorType) -> TensorType:
            return (ub - lb) * (1 - normal.cdf((ub - mean) / std))

        def ehvi_based_on_partitioned_cell(
            neg_pred_mean: TensorType, pred_std: TensorType
        ) -> TensorType:
            r"""
            Calculate the ehvi based on cell i.
            """

            neg_lb_points, neg_ub_points = -self._ub_points, -self._lb_points

            neg_ub_points = tf.minimum(neg_ub_points, 1e10)  # clip to improve numerical stability

            psi_ub = Psi(
                neg_lb_points, neg_ub_points, neg_pred_mean, pred_std
            )  # [..., num_cells, out_dim]
            psi_lb = Psi(
                neg_lb_points, neg_lb_points, neg_pred_mean, pred_std
            )  # [..., num_cells, out_dim]

            psi_lb2ub = tf.maximum(psi_lb - psi_ub, 0.0)  # [..., num_cells, out_dim]
            nu_contrib = nu(neg_lb_points, neg_ub_points, neg_pred_mean, pred_std)

            stacked_factors = tf.concat(
                [tf.expand_dims(psi_lb2ub, -2), tf.expand_dims(nu_contrib, -2)], axis=-2
            )  # Take the cross product of psi_diff and nu across all outcomes
            # [..., num_cells, 2(operation_num, refer Eq. 45), num_obj]

            factor_combinations = tf.linalg.diag_part(
                tf.gather(stacked_factors, self._cross_index, axis=-2)
            )  # [..., num_cells, 2^d, 2(operation_num), num_obj]

            return tf.reduce_sum(tf.reduce_prod(factor_combinations, axis=-1), axis=-1)

        candidate_mean, candidate_var = self._model.predict(tf.squeeze(x, -2))
        candidate_std = tf.sqrt(candidate_var)

        neg_candidate_mean = -tf.expand_dims(candidate_mean, 1)  # [..., 1, out_dim]
        candidate_std = tf.expand_dims(candidate_std, 1)  # [..., 1, out_dim]

        ehvi_cells_based = ehvi_based_on_partitioned_cell(neg_candidate_mean, candidate_std)

        return tf.reduce_sum(
            ehvi_cells_based,
            axis=-1,
            keepdims=True,
        )


class BatchMonteCarloExpectedHypervolumeImprovement(
    SingleModelAcquisitionBuilder[HasReparamSampler]
):
    """
    Builder for the batch expected hypervolume improvement acquisition function.
    The implementation of the acquisition function largely
    follows :cite:`daulton2020differentiable`
    """

    def __init__(
        self,
        sample_size: int,
        reference_point_spec: (
            Sequence[float] | TensorType | Callable[..., TensorType]
        ) = get_reference_point,
        *,
        jitter: float = DEFAULTS.JITTER,
    ):
        """
        :param sample_size: The number of samples from model predicted distribution for
            each batch of points.
        :param reference_point_spec: this method is used to determine how the reference point is
            calculated. If a Callable function specified, it is expected to take existing
            posterior mean-based observations (to screen out the observation noise) and return
            a reference point with shape [D] (D represents number of objectives). If the Pareto
            front location is known, this arg can be used to specify a fixed reference point
            in each bo iteration. A dynamic reference point updating strategy is used by
            default to set a reference point according to the datasets.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or
            ``jitter`` is negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        self._sample_size = sample_size
        self._jitter = jitter
        if callable(reference_point_spec):
            self._ref_point_spec: tf.Tensor | Callable[..., TensorType] = reference_point_spec
        else:
            self._ref_point_spec = tf.convert_to_tensor(reference_point_spec)
        self._ref_point = None

    def __repr__(self) -> str:
        """"""
        if callable(self._ref_point_spec):
            return (
                f"BatchMonteCarloExpectedHypervolumeImprovement({self._sample_size!r},"
                f" {self._ref_point_spec.__name__},"
                f" jitter={self._jitter!r})"
            )
        else:
            return (
                f"BatchMonteCarloExpectedHypervolumeImprovement({self._sample_size!r},"
                f" {self._ref_point_spec!r}"
                f" jitter={self._jitter!r})"
            )

    def prepare_acquisition_function(
        self,
        model: HasReparamSampler,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model. Must have event shape [1].
        :param dataset: The data from the observer. Must be populated.
        :return: The batch expected hypervolume improvement acquisition function.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)

        if callable(self._ref_point_spec):
            self._ref_point = tf.cast(self._ref_point_spec(mean), dtype=mean.dtype)
        else:
            self._ref_point = tf.cast(self._ref_point_spec, dtype=mean.dtype)

        _pf = Pareto(mean)
        screened_front = _pf.front[tf.reduce_all(_pf.front <= self._ref_point, -1)]
        # prepare the partitioned bounds of non-dominated region for calculating of the
        # hypervolume improvement in this area
        _partition_bounds = prepare_default_non_dominated_partition_bounds(
            self._ref_point, screened_front
        )

        if not isinstance(model, HasReparamSampler):
            raise ValueError(
                f"The batch Monte-Carlo expected hyper-volume improvement function only supports "
                f"models that implement a reparam_sampler method; received {model!r}"
            )

        sampler = model.reparam_sampler(self._sample_size)
        return batch_ehvi(sampler, self._jitter, _partition_bounds)


def batch_ehvi(
    sampler: ReparametrizationSampler[HasReparamSampler],
    sampler_jitter: float,
    partition_bounds: tuple[TensorType, TensorType],
) -> AcquisitionFunction:
    """
    :param sampler: The posterior sampler, which given query points `at`, is able to sample
        the possible observations at 'at'.
    :param sampler_jitter: The size of the jitter to use in sampler when stabilising the Cholesky
        decomposition of the covariance matrix.
    :param partition_bounds: with shape ([N, D], [N, D]), partitioned non-dominated hypercell
        bounds for hypervolume improvement calculation
    :return: The batch expected hypervolume improvement acquisition
        function for objective minimisation.
    """

    def acquisition(at: TensorType) -> TensorType:
        _batch_size = at.shape[-2]  # B

        def gen_q_subset_indices(q: int) -> tf.RaggedTensor:
            # generate all subsets of [1, ..., q] as indices
            indices = list(range(q))
            return tf.ragged.constant([list(combinations(indices, i)) for i in range(1, q + 1)])

        samples = sampler.sample(at, jitter=sampler_jitter)  # [..., S, B, num_obj]

        q_subset_indices = gen_q_subset_indices(_batch_size)

        hv_contrib = tf.zeros(tf.shape(samples)[:-2], dtype=samples.dtype)
        lb_points, ub_points = partition_bounds

        def hv_contrib_on_samples(
            obj_samples: TensorType,
        ) -> TensorType:  # calculate samples overlapped area's hvi for obj_samples
            # [..., S, Cq_j, j, num_obj] -> [..., S, Cq_j, num_obj]
            overlap_vertices = tf.reduce_max(obj_samples, axis=-2)

            overlap_vertices = tf.maximum(  # compare overlap vertices and lower bound of each cell:
                tf.expand_dims(overlap_vertices, -3),  # expand a cell dimension
                lb_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :],
            )  # [..., S, K, Cq_j, num_obj]

            lengths_j = tf.maximum(  # get hvi length per obj within each cell
                (ub_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :] - overlap_vertices), 0.0
            )  # [..., S, K, Cq_j, num_obj]

            areas_j = tf.reduce_sum(  # sum over all subsets Cq_j -> [..., S, K]
                tf.reduce_prod(lengths_j, axis=-1), axis=-1  # calc hvi within each K
            )

            return tf.reduce_sum(areas_j, axis=-1)  # sum over cells -> [..., S]

        for j in tf.range(1, _batch_size + 1):  # Inclusion-Exclusion loop
            q_choose_j = tf.gather(q_subset_indices, j - 1).to_tensor()
            # gather all combinations having j points from q batch points (Cq_j)
            j_sub_samples = tf.gather(samples, q_choose_j, axis=-2)  # [..., S, Cq_j, j, num_obj]
            hv_contrib += tf.cast((-1) ** (j + 1), dtype=samples.dtype) * hv_contrib_on_samples(
                j_sub_samples
            )

        return tf.reduce_mean(hv_contrib, axis=-1, keepdims=True)  # average through MC

    return acquisition


class ExpectedConstrainedHypervolumeImprovement(
    ExpectedConstrainedImprovement[ProbabilisticModelType]
):
    """
    Builder for the constrained expected hypervolume improvement acquisition function.
    This function essentially combines ExpectedConstrainedImprovement and
    ExpectedHypervolumeImprovement.
    """

    def __init__(
        self,
        objective_tag: Tag,
        constraint_builder: AcquisitionFunctionBuilder[ProbabilisticModelType],
        min_feasibility_probability: float | TensorType = 0.5,
        reference_point_spec: (
            Sequence[float] | TensorType | Callable[..., TensorType]
        ) = get_reference_point,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_builder: The builder for the constraint function.
        :param min_feasibility_probability: The minimum probability of feasibility for a
            "best point" to be considered feasible.
        :param reference_point_spec: this method is used to determine how the reference point is
            calculated. If a Callable function specified, it is expected to take existing posterior
            mean-based feasible observations (to screen out the observation noise) and return a
            reference point with shape [D] (D represents number of objectives). If the feasible
            Pareto front location is known, this arg can be used to specify a fixed reference
            point in each bo iteration. A dynamic reference point updating strategy is used by
            default to set a reference point according to the datasets.
        """
        super().__init__(objective_tag, constraint_builder, min_feasibility_probability)
        if callable(reference_point_spec):
            self._ref_point_spec: tf.Tensor | Callable[..., TensorType] = reference_point_spec
        else:
            self._ref_point_spec = tf.convert_to_tensor(reference_point_spec)
        self._ref_point = None

    def __repr__(self) -> str:
        """"""
        if callable(self._ref_point_spec):
            return (
                f"ExpectedConstrainedHypervolumeImprovement({self._objective_tag!r},"
                f" {self._constraint_builder!r}, {self._min_feasibility_probability!r},"
                f" {self._ref_point_spec.__name__})"
            )
        else:
            return (
                f"ExpectedConstrainedHypervolumeImprovement({self._objective_tag!r}, "
                f" {self._constraint_builder!r}, {self._min_feasibility_probability!r},"
                f" ref_point_specification={repr(self._ref_point_spec)!r}"
            )

    def _update_expected_improvement_fn(
        self, objective_model: ProbabilisticModelType, feasible_mean: TensorType
    ) -> None:
        """
        Set or update the unconstrained expected improvement function.

        :param objective_model: The objective model.
        :param feasible_mean: The mean of the feasible query points.
        """
        if callable(self._ref_point_spec):
            self._ref_point = tf.cast(
                self._ref_point_spec(feasible_mean),
                dtype=feasible_mean.dtype,
            )
        else:
            self._ref_point = tf.cast(self._ref_point_spec, dtype=feasible_mean.dtype)

        _pf = Pareto(feasible_mean)
        screened_front = _pf.front[tf.reduce_all(_pf.front <= self._ref_point, -1)]
        # prepare the partitioned bounds of non-dominated region for calculating of the
        # hypervolume improvement in this area
        _partition_bounds = prepare_default_non_dominated_partition_bounds(
            self._ref_point,
            screened_front,
        )

        self._expected_improvement_fn: Optional[AcquisitionFunction]
        if self._expected_improvement_fn is None:
            self._expected_improvement_fn = expected_hv_improvement(
                objective_model, _partition_bounds
            )
        else:
            tf.debugging.Assert(
                isinstance(self._expected_improvement_fn, expected_hv_improvement), []
            )
            self._expected_improvement_fn.update(_partition_bounds)  # type: ignore


class HIPPO(GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]):
    r"""
    HIPPO: HIghly Parallelizable Pareto Optimization

    Builder of the acquisition function for greedily collecting batches by HIPPO
    penalization in multi-objective optimization by penalizing batch points
    by their distance in the objective space. The resulting acquistion function
    takes in a set of pending points and returns a base multi-objective acquisition function
    penalized around those points.

    Penalization is applied to the acquisition function multiplicatively. However, to
    improve numerical stability, we perform additive penalization in a log space.
    """

    def __init__(
        self,
        objective_tag: Tag = OBJECTIVE,
        base_acquisition_function_builder: (
            AcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelAcquisitionBuilder[ProbabilisticModelType]
            | None
        ) = None,
    ):
        """
        Initializes the HIPPO acquisition function builder.

        :param objective_tag: The tag for the objective data and model.
        :param base_acquisition_function_builder: Base acquisition function to be
            penalized. Defaults to Expected Hypervolume Improvement, also supports
            its constrained version.
        """
        self._objective_tag = objective_tag
        if base_acquisition_function_builder is None:
            self._base_builder: AcquisitionFunctionBuilder[ProbabilisticModelType] = (
                ExpectedHypervolumeImprovement().using(self._objective_tag)
            )
        else:
            if isinstance(base_acquisition_function_builder, SingleModelAcquisitionBuilder):
                self._base_builder = base_acquisition_function_builder.using(self._objective_tag)
            else:
                self._base_builder = base_acquisition_function_builder

        self._base_acquisition_function: Optional[AcquisitionFunction] = None
        self._penalization: Optional[PenalizationFunction] = None
        self._penalized_acquisition: Optional[AcquisitionFunction] = None

    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        Creates a new instance of the acquisition function.

        :param models: The models.
        :param datasets: The data from the observer. Must be populated.
        :param pending_points: The points we penalize with respect to.
        :return: The HIPPO acquisition function.
        :raise tf.errors.InvalidArgumentError: If the ``dataset`` is empty.
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        datasets = cast(Mapping[Tag, Dataset], datasets)
        tf.debugging.Assert(datasets[self._objective_tag] is not None, [tf.constant([])])
        tf.debugging.assert_positive(
            len(datasets[self._objective_tag]),
            message=f"{self._objective_tag} dataset must be populated.",
        )

        acq = self._update_base_acquisition_function(models, datasets)
        if pending_points is not None and len(pending_points) != 0:
            acq = self._update_penalization(acq, models[self._objective_tag], pending_points)

        return acq

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
        Updates the acquisition function.

        :param function: The acquisition function to update.
        :param models: The models.
        :param datasets: The data from the observer. Must be populated.
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, of to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: The updated acquisition function.
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        datasets = cast(Mapping[Tag, Dataset], datasets)
        tf.debugging.Assert(datasets[self._objective_tag] is not None, [tf.constant([])])
        tf.debugging.assert_positive(
            len(datasets[self._objective_tag]),
            message=f"{self._objective_tag} dataset must be populated.",
        )
        tf.debugging.Assert(self._base_acquisition_function is not None, [tf.constant([])])

        if new_optimization_step:
            self._update_base_acquisition_function(models, datasets)

        if pending_points is None or len(pending_points) == 0:
            # no penalization required if no pending_points
            return cast(AcquisitionFunction, self._base_acquisition_function)

        return self._update_penalization(function, models[self._objective_tag], pending_points)

    def _update_penalization(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        tf.debugging.assert_rank(pending_points, 2)

        if self._penalized_acquisition is not None and isinstance(
            self._penalization, hippo_penalizer
        ):
            # if possible, just update the penalization function variables
            # (the type ignore is due to mypy getting confused by tf.function)
            self._penalization.update(pending_points)  # type: ignore[unreachable]
            return self._penalized_acquisition
        else:
            # otherwise construct a new penalized acquisition function
            self._penalization = hippo_penalizer(model, pending_points)

        @tf.function
        def penalized_acquisition(x: TensorType) -> TensorType:
            log_acq = tf.math.log(
                cast(AcquisitionFunction, self._base_acquisition_function)(x)
            ) + tf.math.log(cast(PenalizationFunction, self._penalization)(x))
            return tf.math.exp(log_acq)

        self._penalized_acquisition = penalized_acquisition
        return penalized_acquisition

    def _update_base_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        if self._base_acquisition_function is None:
            self._base_acquisition_function = self._base_builder.prepare_acquisition_function(
                models, datasets
            )
        else:
            self._base_acquisition_function = self._base_builder.update_acquisition_function(
                self._base_acquisition_function, models, datasets
            )
        return self._base_acquisition_function


class hippo_penalizer:
    r"""
    Returns the penalization function used for multi-objective greedy batch Bayesian
    optimization.

    A candidate point :math:`x` is penalized based on the Mahalanobis distance to a
    given pending point :math:`p_i`. Since we assume objectives to be independent,
    the Mahalanobis distance between these points becomes a Eucledian distance
    normalized by standard deviation. Penalties for multiple pending points are multiplied,
    and the resulting quantity is warped with the arctan function to :math:`[0, 1]` interval.

    :param model: The model over the specified ``dataset``.
    :param pending_points: The points we penalize with respect to.
    :return: The penalization function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    """

    def __init__(self, model: ProbabilisticModel, pending_points: TensorType):
        """Initialize the MO penalizer.

        :param model: The model.
        :param pending_points: The points we penalize with respect to.
        :raise ValueError: If pending points are empty or None.
        :return: The penalization function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one."""
        tf.debugging.Assert(
            pending_points is not None and len(pending_points) != 0, [tf.constant([])]
        )

        self._model = model
        self._pending_points = tf.Variable(pending_points, shape=[None, *pending_points.shape[1:]])
        pending_means, pending_vars = self._model.predict(self._pending_points)
        self._pending_means = tf.Variable(pending_means, shape=[None, *pending_means.shape[1:]])
        self._pending_vars = tf.Variable(pending_vars, shape=[None, *pending_vars.shape[1:]])

    def update(self, pending_points: TensorType) -> None:
        """Update the penalizer with new pending points."""
        tf.debugging.Assert(
            pending_points is not None and len(pending_points) != 0, [tf.constant([])]
        )

        self._pending_points.assign(pending_points)
        pending_means, pending_vars = self._model.predict(self._pending_points)
        self._pending_means.assign(pending_means)
        self._pending_vars.assign(pending_vars)

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This penalization function cannot be calculated for batches of points.",
        )

        # x is [N, 1, D]
        x = tf.squeeze(x, axis=1)  # x is now [N, D]

        x_means, x_vars = self._model.predict(x)
        # x_means is [N, K], x_vars is [N, K]
        # where K is the number of models/objectives

        # self._pending_points is [B, D] where B is the size of the batch collected so far
        tf.debugging.assert_shapes(
            [
                (x, ["N", "D"]),
                (self._pending_points, ["B", "D"]),
                (self._pending_means, ["B", "K"]),
                (self._pending_vars, ["B", "K"]),
                (x_means, ["N", "K"]),
                (x_vars, ["N", "K"]),
            ],
            message="""Encountered unexpected shapes while calculating mean and variance
                       of given point x and pending points""",
        )

        x_means_expanded = x_means[:, None, :]
        pending_means_expanded = self._pending_means[None, :, :]
        pending_vars_expanded = self._pending_vars[None, :, :]
        pending_stddevs_expanded = tf.sqrt(pending_vars_expanded)

        # this computes Mahalanobis distance between x and pending points
        # since we assume objectives to be independent
        # it reduces to regular Eucledian distance normalized by standard deviation
        standardize_mean_diff = (
            tf.abs(x_means_expanded - pending_means_expanded) / pending_stddevs_expanded
        )  # [N, B, K]
        d = tf.norm(standardize_mean_diff, axis=-1)  # [N, B]

        # warp the distance so that resulting value is from 0 to (nearly) 1
        warped_d = (2.0 / math.pi) * tf.math.atan(d)
        penalty = tf.reduce_prod(warped_d, axis=-1)  # [N,]

        return tf.reshape(penalty, (-1, 1))

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
"""
This module contains acquisition function builders, which build and define our acquisition
functions --- functions that estimate the utility of evaluating sets of candidate points.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from itertools import product
from math import inf
from typing import Callable, Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp

from ..data import Dataset
from ..models import ProbabilisticModel
from ..space import SearchSpace
from ..type import TensorType
from ..utils import DEFAULTS
from ..utils.pareto import Pareto, get_reference_point
from .sampler import BatchReparametrizationSampler, GumbelSampler

AcquisitionFunction = Callable[[TensorType], TensorType]
"""
Type alias for acquisition functions.

An :const:`AcquisitionFunction` maps a set of `B` query points (each of dimension `D`) to a single
value that describes how useful it would be evaluate all these points together (to our goal of
optimizing the objective function). Thus, with leading dimensions, an :const:`AcquisitionFunction`
takes input shape `[..., B, D]` and returns shape `[..., 1]`.

Note that :const:`AcquisitionFunction`s which do not support batch optimization still expect inputs
with a batch dimension, i.e. an input of shape `[..., 1, D]`.
"""


class AcquisitionFunctionBuilder(ABC):
    """ An :class:`AcquisitionFunctionBuilder` builds an acquisition function. """

    @abstractmethod
    def prepare_acquisition_function(
        self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        """
        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :return: An acquisition function.
        """


class SingleModelAcquisitionBuilder(ABC):
    """
    Convenience acquisition function builder for an acquisition function (or component of a
    composite acquisition function) that requires only one model, dataset pair.
    """

    def using(self, tag: str) -> AcquisitionFunctionBuilder:
        """
        :param tag: The tag for the model, dataset pair to use to build this acquisition function.
        :return: An acquisition function builder that selects the model and dataset specified by
            ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """
        single_builder = self

        class _Anon(AcquisitionFunctionBuilder):
            def prepare_acquisition_function(
                self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
            ) -> AcquisitionFunction:
                return single_builder.prepare_acquisition_function(datasets[tag], models[tag])

            def __repr__(self) -> str:
                return f"{single_builder!r} using tag {tag!r}"

        return _Anon()

    @abstractmethod
    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: The data to use to build the acquisition function.
        :param model: The model over the specified ``dataset``.
        :return: An acquisition function.
        """


class ExpectedImprovement(SingleModelAcquisitionBuilder):
    """
    Builder for the expected improvement function where the "best" value is taken to be the minimum
    of the posterior mean at observed points.
    """

    def __repr__(self) -> str:
        """"""
        return "ExpectedImprovement()"

    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: The data from the observer. Must be populated.
        :param model: The model over the specified ``dataset``.
        :return: The expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise ValueError: If ``dataset`` is empty.
        """
        if len(dataset.query_points) == 0:
            raise ValueError("Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        return expected_improvement(model, eta)


def expected_improvement(model: ProbabilisticModel, eta: TensorType) -> AcquisitionFunction:
    r"""
    Return the Expected Improvement (EI) acquisition function for single-objective global
    optimization. Improvement is with respect to the current "best" observation ``eta``, where an
    improvement moves towards the objective function's minimum, and the expectation is calculated
    with respect to the ``model`` posterior. For model posterior :math:`f`, this is

    .. math:: x \mapsto \mathbb E \left[ \max (\eta - f(x), 0) \right]

    This function was introduced by Mockus et al, 1975. See :cite:`Jones:1998` for details.

    :param model: The model of the objective function.
    :param eta: The "best" observation.
    :return: The expected improvement function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    """

    def acquisition(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        mean, variance = model.predict(tf.squeeze(x, -2))
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        return (eta - mean) * normal.cdf(eta) + variance * normal.prob(eta)

    return acquisition


class MinValueEntropySearch(SingleModelAcquisitionBuilder):
    r"""
    Builder for the max-value entropy search acquisition function modified for objective
    minimisation. :class:`MinValueEntropySearch` estimates the information in the distribution
    of the objective minimum that would be gained by evaluating the objective at a given point.

    This implementation largely follows :cite:`wang2017max` and samples the objective minimum
    :math:`y^*` via a Gumbel sampler.
    """

    def __init__(self, search_space: SearchSpace, num_samples: int = 10, grid_size: int = 5000):
        """
        :param search_space: The global search space over which the optimisation is defined.
        :param num_samples: Number of samples to draw from the distribution over the minimum of the
            objective function.
        :param grid_size: Size of the grid with which to fit the Gumbel distribution. We recommend
            scaling this with search space dimension.
        """
        self._search_space = search_space

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self._num_samples = num_samples

        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        self._grid_size = grid_size

    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: The data from the observer.
        :param model: The model over the specified ``dataset``.
        :return: The max-value entropy search acquisition function modified for objective
            minimisation. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        """
        if len(dataset.query_points) == 0:
            raise ValueError("Dataset must be populated.")

        gumbel_sampler = GumbelSampler(self._num_samples, model)

        query_points = self._search_space.sample(num_samples=self._grid_size)
        tf.debugging.assert_same_float_dtype([dataset.query_points, query_points])
        query_points = tf.concat([dataset.query_points, query_points], 0)
        gumbel_samples = gumbel_sampler.sample(query_points)

        return min_value_entropy_search(model, gumbel_samples)


def min_value_entropy_search(model: ProbabilisticModel, samples: TensorType) -> AcquisitionFunction:
    r"""
    Return the max-value entropy search acquisition function (adapted from :cite:`wang2017max`),
    modified for objective minimisation. This function calculates the information gain (or change in
    entropy) in the distribution over the objective minimum :math:`y^*`, if we were to evaluate the
    objective at a given point.

    :param model: The model of the objective function.
    :param samples: Samples from the distribution over :math:`y^*`.
    :return: The max-value entropy search acquisition function modified for objective
        minimisation. This function will raise :exc:`ValueError` or
        :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
    """
    tf.debugging.assert_rank(samples, 2)

    if len(samples) == 0:
        raise ValueError("Gumbel samples must be populated.")

    def acquisition(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        fmean, fvar = model.predict(tf.squeeze(x, -2))
        fsd = tf.math.sqrt(fvar)
        fsd = tf.clip_by_value(
            fsd, 1.0e-8, fmean.dtype.max
        )  # clip below to improve numerical stability

        normal = tfp.distributions.Normal(tf.cast(0, fmean.dtype), tf.cast(1, fmean.dtype))
        gamma = (tf.squeeze(samples) - fmean) / fsd

        minus_cdf = 1 - normal.cdf(gamma)
        minus_cdf = tf.clip_by_value(
            minus_cdf, 1.0e-8, 1
        )  # clip below to improve numerical stability
        f_acqu_x = -gamma * normal.prob(gamma) / (2 * minus_cdf) - tf.math.log(minus_cdf)

        return tf.math.reduce_mean(f_acqu_x, axis=1, keepdims=True)

    return acquisition


class NegativeLowerConfidenceBound(SingleModelAcquisitionBuilder):
    """
    Builder for the negative of the lower confidence bound. The lower confidence bound is typically
    minimised, so the negative is suitable for maximisation.
    """

    def __init__(self, beta: float = 1.96):
        """
        :param beta: Weighting given to the variance contribution to the lower confidence bound.
            Must not be negative.
        """
        self._beta = beta

    def __repr__(self) -> str:
        """"""
        return f"NegativeLowerConfidenceBound({self._beta!r})"

    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: Unused.
        :param model: The model over the specified ``dataset``.
        :return: The negative lower confidence bound function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise ValueError: If ``beta`` is negative.
        """
        lcb = lower_confidence_bound(model, self._beta)
        return lambda at: -lcb(at)


class NegativePredictiveMean(NegativeLowerConfidenceBound):
    """
    Builder for the negative of the predictive mean. The predictive mean is minimised on minimising
    the objective function. The negative predictive mean is therefore maximised.
    """

    def __init__(self):
        super().__init__(beta=0.0)

    def __repr__(self) -> str:
        """"""
        return "NegativePredictiveMean()"


def lower_confidence_bound(model: ProbabilisticModel, beta: float) -> AcquisitionFunction:
    r"""
    The lower confidence bound (LCB) acquisition function for single-objective global optimization.

    .. math:: x^* \mapsto \mathbb{E} [f(x^*)|x, y] - \beta \sqrt{ \mathrm{Var}[f(x^*)|x, y] }

    See :cite:`Srinivas:2010` for details.

    :param model: The model of the objective function.
    :param beta: The weight to give to the standard deviation contribution of the LCB. Must not be
        negative.
    :return: The lower confidence bound function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    :raise ValueError: If ``beta`` is negative.
    """
    if beta < 0:
        raise ValueError(
            f"Standard deviation scaling parameter beta must not be negative, got {beta}"
        )

    def acquisition(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        mean, variance = model.predict(tf.squeeze(x, -2))
        return mean - beta * tf.sqrt(variance)

    return acquisition


class ProbabilityOfFeasibility(SingleModelAcquisitionBuilder):
    r"""
    Builder for the :func:`probability_of_feasibility` acquisition function, defined in
    :cite:`gardner14` as

    .. math::

        \int_{-\infty}^{\tau} p(c(\mathbf{x}) | \mathbf{x}, \mathcal{D}) \mathrm{d} c(\mathbf{x})
        \qquad ,

    where :math:`\tau` is a threshold. Values below the threshold are considered feasible by the
    constraint function. See also :cite:`schonlau1998global` for details.
    """

    def __init__(self, threshold: float | TensorType):
        """
        :param threshold: The (scalar) probability of feasibility threshold.
        :raise ValueError (or InvalidArgumentError): If ``threshold`` is not a scalar.
        """
        tf.debugging.assert_scalar(threshold)

        super().__init__()

        self._threshold = threshold

    def __repr__(self) -> str:
        """"""
        return f"ProbabilityOfFeasibility({self._threshold!r})"

    @property
    def threshold(self) -> float | TensorType:
        """ The probability of feasibility threshold. """
        return self._threshold

    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: Unused.
        :param model: The model over the specified ``dataset``.
        :return: The probability of feasibility function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """
        return probability_of_feasibility(model, self.threshold)


def probability_of_feasibility(
    model: ProbabilisticModel, threshold: float | TensorType
) -> AcquisitionFunction:
    r"""
    The probability of feasibility acquisition function defined in :cite:`gardner14` as

    .. math::

        \int_{-\infty}^{\tau} p(c(\mathbf{x}) | \mathbf{x}, \mathcal{D}) \mathrm{d} c(\mathbf{x})
        \qquad ,

    where :math:`\tau` is a threshold. Values below the threshold are considered feasible by the
    constraint function.

    :param model: The model of the objective function.
    :param threshold: The (scalar) probability of feasibility threshold.
    :return: The probability of feasibility function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    :raise ValueError: If ``threshold`` is not a scalar.
    """
    tf.debugging.assert_scalar(threshold)

    def acquisition(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        mean, var = model.predict(tf.squeeze(x, -2))
        distr = tfp.distributions.Normal(mean, tf.sqrt(var))
        return distr.cdf(tf.cast(threshold, x.dtype))

    return acquisition


class ExpectedConstrainedImprovement(AcquisitionFunctionBuilder):
    """
    Builder for the *expected constrained improvement* acquisition function defined in
    :cite:`gardner14`. The acquisition function computes the expected improvement from the best
    feasible point, where feasible points are those that (probably) satisfy some constraint. Where
    there are no feasible points, this builder simply builds the constraint function.
    """

    def __init__(
        self,
        objective_tag: str,
        constraint_builder: AcquisitionFunctionBuilder,
        min_feasibility_probability: float | TensorType = 0.5,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_builder: The builder for the constraint function.
        :param min_feasibility_probability: The minimum probability of feasibility for a
            "best point" to be considered feasible.
        :raise ValueError (or InvalidArgumentError): If ``min_feasibility_probability`` is not a
            scalar in the unit interval :math:`[0, 1]`.
        """
        tf.debugging.assert_scalar(min_feasibility_probability)

        if not 0 <= min_feasibility_probability <= 1:
            raise ValueError(
                f"Minimum feasibility probability must be between 0 and 1 inclusive,"
                f" got {min_feasibility_probability}"
            )

        self._objective_tag = objective_tag
        self._constraint_builder = constraint_builder
        self._min_feasibility_probability = min_feasibility_probability

    def __repr__(self) -> str:
        """"""
        return (
            f"ExpectedConstrainedImprovement({self._objective_tag!r}, {self._constraint_builder!r},"
            f" {self._min_feasibility_probability!r})"
        )

    def prepare_acquisition_function(
        self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        """
        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :return: The expected constrained improvement acquisition function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise KeyError: If `objective_tag` is not found in ``datasets`` and ``models``.
        :raise ValueError: If the objective data is empty.
        """
        objective_model = models[self._objective_tag]
        objective_dataset = datasets[self._objective_tag]

        if len(objective_dataset) == 0:
            raise ValueError(
                "Expected improvement is defined with respect to existing points in the objective"
                " data, but the objective data is empty."
            )

        constraint_fn = self._constraint_builder.prepare_acquisition_function(datasets, models)
        pof = constraint_fn(objective_dataset.query_points[:, None, ...])
        is_feasible = pof >= self._min_feasibility_probability

        if not tf.reduce_any(is_feasible):
            return constraint_fn

        mean, _ = objective_model.predict(objective_dataset.query_points)
        eta = tf.reduce_min(tf.boolean_mask(mean, is_feasible), axis=0)

        return lambda at: expected_improvement(objective_model, eta)(at) * constraint_fn(at)


class ExpectedHypervolumeImprovement(SingleModelAcquisitionBuilder):
    """
    Builder for the expected hypervolume improvement acquisition function.
    The implementation of the acquisition function largely
    follows :cite:`yang2019efficient`
    """

    def __repr__(self) -> str:
        """"""
        return "ExpectedHypervolumeImprovement()"

    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: The data from the observer. Must be populated.
        :param model: The model over the specified ``dataset``.
        :return: The expected hypervolume improvement acquisition function.
        """
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)

        _pf = Pareto(mean)
        _reference_pt = get_reference_point(_pf.front)
        return expected_hv_improvement(model, _pf, _reference_pt)


def expected_hv_improvement(
    model: ProbabilisticModel,
    pareto: Pareto,
    reference_point: TensorType,
) -> AcquisitionFunction:
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
    :param pareto: Pareto class
    :param reference_point: The reference point for calculating hypervolume
    :return: The expected_hv_improvement acquisition function modified for objective
        minimisation. This function will raise :exc:`ValueError` or
        :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
    """

    def acquisition(x: TensorType) -> TensorType:
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

            lb_points, ub_points = pareto.hypercell_bounds(
                tf.constant([-inf] * neg_pred_mean.shape[-1], dtype=x.dtype), reference_point
            )

            neg_lb_points, neg_ub_points = -ub_points, -lb_points

            neg_ub_points = tf.minimum(neg_ub_points, 1e10)  # clip to improve numerical stability

            psi_ub = Psi(
                neg_lb_points, neg_ub_points, neg_pred_mean, pred_std
            )  # [..., num_cells, out_dim]
            psi_lb = Psi(
                neg_lb_points, neg_lb_points, neg_pred_mean, pred_std
            )  # [..., num_cells, out_dim]

            psi_lb2ub = tf.maximum(psi_lb - psi_ub, 0.0)  # [..., num_cells, out_dim]
            nu_contrib = nu(neg_lb_points, neg_ub_points, neg_pred_mean, pred_std)

            cross_index = tf.constant(
                list(product(*[[0, 1]] * reference_point.shape[-1]))
            )  # [2^d, indices_at_dim]

            stacked_factors = tf.concat(
                [tf.expand_dims(psi_lb2ub, -2), tf.expand_dims(nu_contrib, -2)], axis=-2
            )  # Take the cross product of psi_diff and nu across all outcomes
            # [..., num_cells, 2(operation_num, refer Eq. 45), num_obj]

            factor_combinations = tf.linalg.diag_part(
                tf.gather(stacked_factors, cross_index, axis=-2)
            )  # [..., num_cells, 2^d, 2(operation_num), num_obj]

            return tf.reduce_sum(tf.reduce_prod(factor_combinations, axis=-1), axis=-1)

        candidate_mean, candidate_var = model.predict(tf.squeeze(x, -2))
        candidate_std = tf.sqrt(candidate_var)

        neg_candidate_mean = -tf.expand_dims(candidate_mean, 1)  # [..., 1, out_dim]
        candidate_std = tf.expand_dims(candidate_std, 1)  # [..., 1, out_dim]

        ehvi_cells_based = ehvi_based_on_partitioned_cell(neg_candidate_mean, candidate_std)

        return tf.reduce_sum(
            ehvi_cells_based,
            axis=-1,
            keepdims=True,
        )

    return acquisition


class BatchMonteCarloExpectedImprovement(SingleModelAcquisitionBuilder):
    """
    Expected improvement for batches of points (or :math:`q`-EI), approximated using Monte Carlo
    estimation with the reparametrization trick. See :cite:`Ginsbourger2010` for details.

    Improvement is measured with respect to the minimum predictive mean at observed query points.
    This is calculated in :class:`BatchMonteCarloExpectedImprovement` by assuming observations
    at new points are independent from those at known query points. This is faster, but is an
    approximation for noisy observers.
    """

    def __init__(self, sample_size: int, *, jitter: float = DEFAULTS.JITTER):
        """
        :param sample_size: The number of samples for each batch of points.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or
            ``jitter`` is negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        super().__init__()

        self._sample_size = sample_size
        self._jitter = jitter

    def __repr__(self) -> str:
        """"""
        return f"BatchMonteCarloExpectedImprovement({self._sample_size!r}, jitter={self._jitter!r})"

    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: The data from the observer. Must be populated.
        :param model: The model over the specified ``dataset``. Must have event shape [1].
        :return: The batch *expected improvement* acquisition function.
        :raise ValueError (or InvalidArgumentError): If ``dataset`` is not populated, or ``model``
            does not have an event shape of [1].
        """
        tf.debugging.assert_positive(len(dataset))

        mean, _ = model.predict(dataset.query_points)

        tf.debugging.assert_shapes(
            [(mean, ["_", 1])], message="Expected model with event shape [1]."
        )

        eta = tf.reduce_min(mean, axis=0)
        sampler = BatchReparametrizationSampler(self._sample_size, model)

        def batch_ei(at: TensorType) -> TensorType:
            samples = tf.squeeze(sampler.sample(at, jitter=self._jitter), axis=-1)  # [..., S, B]
            min_sample_per_batch = tf.reduce_min(samples, axis=-1)  # [..., S]
            batch_improvement = tf.maximum(eta - min_sample_per_batch, 0.0)  # [..., S]
            return tf.reduce_mean(batch_improvement, axis=-1, keepdims=True)  # [..., 1]

        return batch_ei


class GreedyAcquisitionFunctionBuilder(ABC):
    """
    A :class:`GreedyAcquisitionFunctionBuilder` builds an acquisition function
    suitable for greedily building batches for batch Bayesian
    Optimization. :class:`GreedyAcquisitionFunctionBuilder` differs
    from :class:`AcquisitionFunctionBuilder` by requiring that a set
    of pending points is passed to the builder. Note that this acquisition function
    is typically called `B` times each Bayesian optimization step, when building batches
    of size `B`.
    """

    @abstractmethod
    def prepare_acquisition_function(
        self,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ProbabilisticModel],
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :return: An acquisition function.
        """


class SingleModelGreedyAcquisitionBuilder(ABC):
    """
    Convenience acquisition function builder for a greedy acquisition function (or component of a
    composite greedy acquisition function) that requires only one model, dataset pair.
    """

    def using(self, tag: str) -> GreedyAcquisitionFunctionBuilder:
        """
        :param tag: The tag for the model, dataset pair to use to build this acquisition function.
        :return: An acquisition function builder that selects the model and dataset specified by
            ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """
        single_builder = self

        class _Anon(GreedyAcquisitionFunctionBuilder):
            def prepare_acquisition_function(
                self,
                datasets: Mapping[str, Dataset],
                models: Mapping[str, ProbabilisticModel],
                pending_points: Optional[TensorType] = None,
            ) -> AcquisitionFunction:
                return single_builder.prepare_acquisition_function(
                    datasets[tag], models[tag], pending_points=pending_points
                )

            def __repr__(self) -> str:
                return f"{single_builder!r} using tag {tag!r}"

        return _Anon()

    @abstractmethod
    def prepare_acquisition_function(
        self,
        dataset: Dataset,
        model: ProbabilisticModel,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        :param dataset: The data to use to build the acquisition function.
        :param model: The model over the specified ``dataset``.
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :return: An acquisition function.
        """


class LocalPenalizationAcquisitionFunction(SingleModelGreedyAcquisitionBuilder):
    r"""
    Builder of the acquisition function maker for greedily collecting batches by local
    penalization.  The resulting :const:`AcquisitionFunctionMaker` takes in a set of pending
    points and returns a base acquisition function penalized around those points.
    An estimate of the objective function's Lipschitz constant is used to control the size
    of penalization.

    Local penalization allows us to perform batch Bayesian optimization with a standard (non-batch)
    acqusition function. All that we require is that the acquisition function takes strictly
    positive values. By iteratively building a batch of points though sequentially maximizing
    this acquisition function but down-weighted around locations close to the already
    chosen (pending) points, local penalization provides diverse batches of candidate points.

    Local penalization is applied to the acquisition function multiplicatively. However, to
    improve numerical stability, we perfom additive penalization in a log space.

    The Lipschitz constant and additional penalization parameters are estimated once
    when first preparing the acquisition function with no pending points. These estimates
    are reused for all subsequent function calls.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        num_samples: int = 500,
        penalizer: Callable[..., PenalizationFunction] = None,
        base_acquisition_function_builder: Optional[
            Union[ExpectedImprovement, MinValueEntropySearch]
        ] = None,
    ):
        """
        :param search_space: The global search space over which the optimisation is defined.
        :param num_samples: Size of the random sample over which the Lipschitz constant
            is estimated. We recommend scaling this with search space dimension.
        :param penalizer: The chosen penalization method (defaults to soft penalization).
        :param base_acquisition_function_builder: Base acquisition function to be
            penalized (defaults to expected improvement). Local penalization only supports
            strictly positive acquisition functions.

        """
        self._search_space = search_space
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self._num_samples = num_samples

        self._lipschitz_penalizer = soft_local_penalizer if penalizer is None else penalizer

        if base_acquisition_function_builder is None:
            self._base_builder: SingleModelAcquisitionBuilder = ExpectedImprovement()
        elif isinstance(
            base_acquisition_function_builder, (ExpectedImprovement, MinValueEntropySearch)
        ):
            self._base_builder = base_acquisition_function_builder
        else:
            raise ValueError(
                f"""
                Local penalization can only be applied to strictly positive acquisition functions,
                we got {base_acquisition_function_builder}.
                """
            )

        self._lipschitz_constant = None
        self._eta = None
        self._base_acquisition_function: Optional[AcquisitionFunction] = None

    def prepare_acquisition_function(
        self,
        dataset: Dataset,
        model: ProbabilisticModel,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        :param dataset: The data from the observer.
        :param model: The model over the specified ``dataset``.
        :param pending_points: The points we penalize with respect to.
        :return: The (log) expected improvement penalized with respect to the pending points.
        :raise ValueError: if the first call does not have pending_points=None.
        """
        if len(dataset.query_points) == 0:
            raise ValueError("Dataset must be populated.")

        if (
            pending_points is None
        ):  # compute penalization params and base acquisition once per optimization step
            samples = self._search_space.sample(num_samples=self._num_samples)
            samples = tf.concat([dataset.query_points, samples], 0)

            def get_lipschitz_estimate(
                sampled_points,
            ) -> tf.Tensor:  # use max norm of posterior mean gradients
                with tf.GradientTape() as g:
                    g.watch(sampled_points)
                    mean, _ = model.predict(sampled_points)
                grads = g.gradient(mean, sampled_points)
                grads_norm = tf.norm(grads, axis=1)
                max_grads_norm = tf.reduce_max(grads_norm)
                eta = tf.reduce_min(mean, axis=0)
                return max_grads_norm, eta

            lipschitz_constant, eta = get_lipschitz_estimate(samples)
            if (
                lipschitz_constant < 1e-5
            ):  # threshold to improve numerical stability for 'flat' models
                lipschitz_constant = 10

            self._lipschitz_constant = lipschitz_constant
            self._eta = eta

            if isinstance(self._base_builder, ExpectedImprovement):  # reuse eta estimate
                self._base_acquisition_function = expected_improvement(model, self._eta)
            else:
                self._base_acquisition_function = self._base_builder.prepare_acquisition_function(
                    dataset, model
                )

        if (self._lipschitz_constant is None) or (self._base_acquisition_function is None):
            raise ValueError("Local penalization must be first called with no pending_points.")

        if pending_points is None:
            return self._base_acquisition_function  # no penalization required if no pending_points.

        tf.debugging.assert_shapes(
            [(pending_points, ["N", len(self._search_space.upper)])],
            message="pending_points must be of shape [N,D]",
        )

        penalization = self._lipschitz_penalizer(
            model, pending_points, self._lipschitz_constant, self._eta
        )

        def penalized_acquisition(x: TensorType) -> TensorType:
            log_acq = tf.math.log(self._base_acquisition_function(x)) + tf.math.log(penalization(x))
            return tf.math.exp(log_acq)

        return penalized_acquisition


PenalizationFunction = Callable[[TensorType], TensorType]
"""
An :const:`PenalizationFunction` maps a query point (of dimension `D`) to a single
value that described how heavily it should be penalized (a positive quantity).
As penalization is applied multiplicatively to acquisition functions, small
penalization outputs correspond to a stronger penalization effect. Thus, with
leading dimensions, an :const:`PenalizationFunction` takes input
shape `[..., 1, D]` and returns shape `[..., 1]`.
"""


def soft_local_penalizer(
    model: ProbabilisticModel,
    pending_points: TensorType,
    lipschitz_constant: TensorType,
    eta: TensorType,
) -> PenalizationFunction:
    r"""
    Return the soft local penalization function used for single-objective greedy batch Bayesian
    optimization in :cite:`Gonzalez:2016`.

    Soft penalization returns the probability that a candidate point does not belong
    in the exclusion zones of the pending points. For model posterior mean :math:`\mu`, model
    posterior variance :math:`\sigma^2`, current "best" function value :math:`\eta`, and an
    estimated Lipschitz constant :math:`L`,the penalization from a set of pending point :math:`x'`
    on a candidate point :math:`x` is given by
    .. math:: \phi(x, x') = \frac{1}{2}\textrm{erfc}(-z)
    where :math:`z = \frac{1}{\sqrt{2\sigma^2(x')}}(L||x'-x|| + \eta - \mu(x'))`.

    The penalization from a set of pending points is just product of the individual penalizations.
    See :cite:`Gonzalez:2016` for a full derivation.

    :param model: The model over the specified ``dataset``.
    :param pending_points: The points we penalize with respect to.
    :param lipschitz_constant: The estimated Lipschitz constant of the objective function.
    :param eta: The estimated global minima.
    :return: The local penalization function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    """

    mean_pending, variance_pending = model.predict(pending_points)
    radius = tf.transpose((mean_pending - eta) / lipschitz_constant)
    scale = tf.transpose(tf.sqrt(variance_pending) / lipschitz_constant)

    def penalization_function(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This penalization function cannot be calculated for batches of points.",
        )

        pairwise_distances = tf.norm(
            tf.expand_dims(x, 1) - tf.expand_dims(pending_points, 0), axis=-1
        )
        standardised_distances = (pairwise_distances - radius) / scale

        normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
        penalization = normal.cdf(standardised_distances)
        return tf.reduce_prod(penalization, axis=-1)

    return penalization_function


def hard_local_penalizer(
    model: ProbabilisticModel,
    pending_points: TensorType,
    lipschitz_constant: TensorType,
    eta: TensorType,
) -> PenalizationFunction:
    r"""
    Return the hard local penalization function used for single-objective greedy batch Bayesian
    optimization in :cite:`Alvi:2019`.

    Hard penalization is a stronger penalizer than soft penalization and is sometimes more effective
    See :cite:`Alvi:2019` for details. Our implementation follows theirs, with the penalization from
    a set of pending points being the product of the individual penalizations.

    :param model: The model over the specified ``dataset``.
    :param pending_points: The points we penalize with respect to.
    :param lipschitz_constant: The estimated Lipschitz constant of the objective function.
    :param eta: The estimated global minima.
    :return: The local penalization function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    """

    mean_pending, variance_pending = model.predict(pending_points)
    radius = tf.transpose((mean_pending - eta) / lipschitz_constant)
    scale = tf.transpose(tf.sqrt(variance_pending) / lipschitz_constant)

    def penalization_function(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This penalization function cannot be calculated for batches of points.",
        )

        pairwise_distances = tf.norm(
            tf.expand_dims(x, 1) - tf.expand_dims(pending_points, 0), axis=-1
        )

        p = -5  # following experiments of :cite:`Alvi:2019`.
        penalization = ((pairwise_distances / (radius + scale)) ** p + 1) ** (1 / p)

        return tf.reduce_prod(penalization, axis=-1)

    return penalization_function

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

from .empiric import SingleModelEmpiric, Empiric
from ..data import Dataset
from ..models import ProbabilisticModel
from ..space import SearchSpace
from ..type import TensorType
from ..utils import DEFAULTS
from ..utils.pareto import Pareto, get_reference_point
from .sampler import (
    BatchReparametrizationSampler,
    ExactThompsonSampler,
    GumbelSampler,
    RandomFourierFeatureThompsonSampler,
    ThompsonSampler,
)

CLAMP_LB = 1e-8

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


class ExpectedImprovement(SingleModelEmpiric[AcquisitionFunction]):
    """
    Builder for the expected improvement function where the "best" value is taken to be the minimum
    of the posterior mean at observed points.
    """

    def __repr__(self) -> str:
        """"""
        return "ExpectedImprovement()"

    def acquire(self, dataset: Dataset, model: ProbabilisticModel) -> AcquisitionFunction:
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


class AugmentedExpectedImprovement(SingleModelEmpiric[AcquisitionFunction]):
    """
    Builder for the augmented expected improvement function for optimization single-objective
    optimization problems with high levels of observation noise.
    """

    def __repr__(self) -> str:
        """"""
        return "AugmentedExpectedImprovement()"

    def acquire(self, dataset: Dataset, model: ProbabilisticModel) -> AcquisitionFunction:
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
        return augmented_expected_improvement(model, eta)


def augmented_expected_improvement(
    model: ProbabilisticModel, eta: TensorType
) -> AcquisitionFunction:
    r"""
    Return the Augmented Expected Improvement (AEI) acquisition function for single-objective global
    optimization under homoscedastic observation noise.
    Improvement is with respect to the current "best" observation ``eta``, where an
    improvement moves towards the objective function's minimum, and the expectation is calculated
    with respect to the ``model`` posterior. In contrast to standard EI, AEI has an additional
    multiplicative factor that penalizes evaluations made in areas of the space with very small
    posterior predictive variance. Thus, when applying standard EI to noisy optimisation
    problems, AEI avoids getting trapped and repeatedly querying the same point.
    For model posterior :math:`f`, this is
    .. math:: x \mapsto EI(x) * \left(1 - frac{\tau^2}{\sqrt{s^2(x)+\tau^2}}\right),
    where :math:`s^2(x)` is the predictive variance and :math:`\tau` is observation noise.
    This function was introduced by Huang et al, 2006. See :cite:`Huang:2006` for details.

    :param model: The model of the objective function.
    :param eta: The "best" observation.
    :return: The expected improvement function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    """

    try:
        noise_variance = model.get_observation_noise()
    except NotImplementedError:
        raise ValueError(
            """
            Augmented expected improvement only currently supports homoscedastic gpflow models
            with a likelihood.variance attribute.
            """
        )

    def acquisition(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        mean, variance = model.predict(tf.squeeze(x, -2))
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        expected_improvement = (eta - mean) * normal.cdf(eta) + variance * normal.prob(eta)

        augmentation = 1 - (tf.math.sqrt(noise_variance)) / (
            tf.math.sqrt(noise_variance + variance)
        )

        return expected_improvement * augmentation

    return acquisition


class MinValueEntropySearch(SingleModelEmpiric[AcquisitionFunction]):
    r"""
    Builder for the max-value entropy search acquisition function modified for objective
    minimisation. :class:`MinValueEntropySearch` estimates the information in the distribution
    of the objective minimum that would be gained by evaluating the objective at a given point.

    This implementation largely follows :cite:`wang2017max` and samples the objective's minimum
    :math:`y^*` across a large set of sampled locations via either a Gumbel sampler, an exact
    Thompson sampler or an approximate random Fourier feature-based Thompson sampler, with the
    Gumbel sampler being the cheapest but least accurate. Default behavior is to use the
    exact Thompson sampler.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        num_samples: int = 5,
        grid_size: int = 1000,
        use_thompson: bool = True,
        num_fourier_features: Optional[int] = None,
    ):
        """
        :param search_space: The global search space over which the optimisation is defined.
        :param num_samples: Number of samples to draw from the distribution over the minimum of the
            objective function.
        :param grid_size: Size of the grid from which to sample the min-values. We recommend
            scaling this with search space dimension.
        :param use_thompson: If True then use Thompson sampling to sample the objective's
            minimum, else use Gumbel sampling.
        :param num_fourier_features: Number of Fourier features used for approximate Thompson
            sampling. If None, then do exact Thompson sampling.
        """
        self._search_space = search_space

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self._num_samples = num_samples

        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        self._grid_size = grid_size

        if num_fourier_features is not None:
            if not use_thompson:
                raise ValueError(
                    f"""
                    Fourier features approximation can only be applied to Thompson sampling
                    however `use_thompson` is {use_thompson}.
                    """
                )
            if num_fourier_features <= 0:
                raise ValueError(
                    f"num_fourier_features must be positive, got {num_fourier_features}"
                )
        self._use_thompson = use_thompson
        self._num_fourier_features = num_fourier_features

    def acquire(self, dataset: Dataset, model: ProbabilisticModel) -> AcquisitionFunction:
        """
        :param dataset: The data from the observer.
        :param model: The model over the specified ``dataset``.
        :return: The max-value entropy search acquisition function modified for objective
            minimisation. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        """
        if len(dataset.query_points) == 0:
            raise ValueError("Dataset must be populated.")

        if not self._use_thompson:  # use Gumbel sampler
            sampler: ThompsonSampler = GumbelSampler(self._num_samples, model)
        elif self._num_fourier_features is not None:  # use approximate Thompson sampler
            sampler = RandomFourierFeatureThompsonSampler(
                self._num_samples,
                model,
                dataset,
                sample_min_value=True,
                num_features=self._num_fourier_features,
            )
        else:  # use exact Thompson sampler
            sampler = ExactThompsonSampler(self._num_samples, model, sample_min_value=True)

        query_points = self._search_space.sample(num_samples=self._grid_size)
        tf.debugging.assert_same_float_dtype([dataset.query_points, query_points])
        query_points = tf.concat([dataset.query_points, query_points], 0)
        min_value_samples = sampler.sample(query_points)

        return min_value_entropy_search(model, min_value_samples)


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
        raise ValueError("Min value samples must be populated.")

    def acquisition(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        fmean, fvar = model.predict(tf.squeeze(x, -2))
        fsd = tf.math.sqrt(fvar)
        fsd = tf.clip_by_value(
            fsd, CLAMP_LB, fmean.dtype.max
        )  # clip below to improve numerical stability

        normal = tfp.distributions.Normal(tf.cast(0, fmean.dtype), tf.cast(1, fmean.dtype))
        gamma = (tf.squeeze(samples) - fmean) / fsd

        minus_cdf = 1 - normal.cdf(gamma)
        minus_cdf = tf.clip_by_value(
            minus_cdf, CLAMP_LB, 1
        )  # clip below to improve numerical stability
        f_acqu_x = -gamma * normal.prob(gamma) / (2 * minus_cdf) - tf.math.log(minus_cdf)

        return tf.math.reduce_mean(f_acqu_x, axis=1, keepdims=True)

    return acquisition


class NegativeLowerConfidenceBound(SingleModelEmpiric[AcquisitionFunction]):
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

    def acquire(self, dataset: Dataset, model: ProbabilisticModel) -> AcquisitionFunction:
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


class ProbabilityOfFeasibility(SingleModelEmpiric[AcquisitionFunction]):
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
        """The probability of feasibility threshold."""
        return self._threshold

    def acquire(self, dataset: Dataset, model: ProbabilisticModel) -> AcquisitionFunction:
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


class ExpectedConstrainedImprovement(Empiric[AcquisitionFunction]):
    """
    Builder for the *expected constrained improvement* acquisition function defined in
    :cite:`gardner14`. The acquisition function computes the expected improvement from the best
    feasible point, where feasible points are those that (probably) satisfy some constraint. Where
    there are no feasible points, this builder simply builds the constraint function.
    """

    def __init__(
        self,
        objective_tag: str,
        constraint_builder: Empiric[AcquisitionFunction],
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

    def acquire(
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

        constraint_fn = self._constraint_builder.acquire(datasets, models)
        pof = constraint_fn(objective_dataset.query_points[:, None, ...])
        is_feasible = tf.squeeze(pof >= self._min_feasibility_probability, axis=-1)

        if not tf.reduce_any(is_feasible):
            return constraint_fn

        feasible_query_points = tf.boolean_mask(objective_dataset.query_points, is_feasible)
        feasible_mean, _ = objective_model.predict(feasible_query_points)
        eta = tf.reduce_min(feasible_mean, axis=0)

        return lambda at: expected_improvement(objective_model, eta)(at) * constraint_fn(at)


class ExpectedHypervolumeImprovement(SingleModelEmpiric[AcquisitionFunction]):
    """
    Builder for the expected hypervolume improvement acquisition function.
    The implementation of the acquisition function largely
    follows :cite:`yang2019efficient`
    """

    def __repr__(self) -> str:
        """"""
        return "ExpectedHypervolumeImprovement()"

    def acquire(self, dataset: Dataset, model: ProbabilisticModel) -> AcquisitionFunction:
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


class BatchMonteCarloExpectedImprovement(SingleModelEmpiric[AcquisitionFunction]):
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

    def acquire(self, dataset: Dataset, model: ProbabilisticModel) -> AcquisitionFunction:
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


# todo this doesn't work atm
class GIBBON(SingleModelEmpiric[AcquisitionFunction]):
    r"""

    The General-purpose Information-Based Bayesian Optimisation (GIBBON) acquisition function
    of :cite:`Moss:2021`. :class:`GIBBON` provides a computationally cheap approximation of the
    information gained about (i.e the change in entropy of) the objective function's minimum by
    evaluating a batch of candidate points. Batches are built in a greedy manner.

    This implementation follows :cite:`Moss:2021` but is modified for function
    minimisation (rather than maximisation). We sample the objective's minimum
    :math:`y^*` across a large set of sampled locations via either a Gumbel sampler, an exact
    Thompson sampler or an approximate random Fourier feature-based Thompson sampler, with the
    Gumbel sampler being the cheapest but least accurate. Default behavior is to use the
    exact Thompson sampler.

    When performing BO with large batches (i.e. much larger than 10), we recommend the big batch
    formulation of GIBBON, as accessed through setting `big_batch` to be True.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        num_samples: int = 5,
        grid_size: int = 1000,
        use_thompson: bool = True,
        num_fourier_features: Optional[int] = None,
        big_batch: bool = False,
    ):
        """
        :param search_space: The global search space over which the optimisation is defined.
        :param num_samples: Number of samples to draw from the distribution over the minimum of
            the objective function.
        :param grid_size: Size of the grid from which to sample the min-values. We recommend
            scaling this with search space dimension.
        :param use_thompson: If True then use Thompson sampling to sample the objective's
            minimum, else use Gumbel sampling.
        :param num_fourier_features: Number of Fourier features used for approximate Thompson
            sampling. If None, then do exact Thompson sampling.
        :param big_batch: If True, then use the GIBBON formulation for BO with large
            batches (i.e. larger than 10).
        """
        self._search_space = search_space

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self._num_samples = num_samples

        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        self._grid_size = grid_size

        if num_fourier_features is not None:
            if not use_thompson:
                raise ValueError(
                    f"""
                    Fourier features approximation can only be applied to Thompson sampling
                    however `use_thompson` is {use_thompson}.
                    """
                )
            if num_fourier_features <= 0:
                raise ValueError(
                    f"num_fourier_features must be positive, got {num_fourier_features}"
                )
        self._use_thompson = use_thompson
        self._num_fourier_features = num_fourier_features
        self._big_batch = big_batch

        self._min_value_samples = None

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
        :return: The GIBBON acquisition function modified for objective minimisation.
        :raise ValueError: if the first call does not have pending_points=None.
        """

        if len(dataset.query_points) == 0:
            raise ValueError("Dataset must be populated.")

        if pending_points is None:  # only collect min-value samples once per optimization step

            if not self._use_thompson:  # use Gumbel sampler
                sampler: ThompsonSampler = GumbelSampler(self._num_samples, model)
            elif self._num_fourier_features is not None:  # use approximate Thompson sampler
                sampler = RandomFourierFeatureThompsonSampler(
                    self._num_samples,
                    model,
                    dataset,
                    sample_min_value=True,
                    num_features=self._num_fourier_features,
                )
            else:  # use exact Thompson sampler
                sampler = ExactThompsonSampler(self._num_samples, model, sample_min_value=True)

            query_points = self._search_space.sample(num_samples=self._grid_size)
            tf.debugging.assert_same_float_dtype([dataset.query_points, query_points])
            query_points = tf.concat([dataset.query_points, query_points], 0)
            self._min_value_samples = sampler.sample(query_points)

        if self._min_value_samples is None:
            raise ValueError("GIBBON must be first called with no pending_points.")

        return gibbon(model, self._min_value_samples, pending_points, self._big_batch)


def gibbon(
    model: ProbabilisticModel,
    samples: TensorType,
    pending_points: Optional[TensorType] = None,
    big_batch: bool = False,
) -> AcquisitionFunction:
    r"""
    Return the General-purpose Information-Based Bayesian Optimization (GIBBON) acquisition function
    of :cite:`Moss:2021`.The GIBBON acquisition function consists of two terms --- a quality term
    and a diversity term. The quality term measures the amount of information that each individual
    batch element provides about the objective function's minimal value :math:`y^*` (ensuring that
    evaluations are targeted in promising areas of the space), whereas the repulsion term encourages
    diversity within the batch (achieving high values for points with low predictive correlation).

    GIBBON's repulsion term :math:`r=\log |C|`  is given by the log determinant of the predictive
    correlation matrix :math:`C` between the `m` pending points and the current candidate.
    The predictive covariance :math:`V` can be expressed as :math:V = [[v, A], [A, B]]` for a
    tensor :math:`B` with shape [`m`,`m`] and so we can efficiently calculate :math:`|V|` using the
    formula for the determinant of block matrices, i.e :math:`|V| = (v - A^T * B^{-1} * A) * |B|`.
    Note that when using GIBBON for purely sequential optimization, the repulsion term is
    not required.

    As GIBBON's batches are built in a greedy manner, i.e sequentially adding points to build a set
    of `m` pending points, we need only ever calculate the entropy reduction provided by adding the
    current candidate point to the current pending points, not the full information gain provided by
    evaluating all the pending points. This allows for a modest computational saving.

    When performing BO with large batches (i.e. much larger than 10), GIBBON's approximations become
    less accurate and its repulsion term dominates. Therefore, for large batch BO, we follow the
    arguments of :cite:`Moss:2021` and divide GIBBON's repulsion term by :math:`B^{log B}`. This
    behavior is accessed though setting `big_batch` to True.

    :param model: The model of the objective function. GIBBON requires a model with
        a :method:covariance_between_points method and so GIBBON only
        supports :class:`GaussianProcessRegression` models.
    :param samples: Samples from the distribution over :math:`y^*`.
    :param pending_points: The points already chosen in the current batch.
    :param big_batch: If True, then use the GIBBON formulation for BO with large batches.
    :return: The GIBBON acquisition function. This function will raise :exc:`ValueError` or
        :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
    """
    tf.debugging.assert_rank(samples, 2)
    if len(samples) == 0:
        raise ValueError("Min-value samples must be populated.")

    try:
        noise_variance = model.get_observation_noise()
    except NotImplementedError:
        raise ValueError(
            """
            GIBBON only currently supports homoscedastic gpflow models
            with a likelihood.variance attribute.
            """
        )

    if not hasattr(model, "covariance_between_points"):
        raise AttributeError(
            """
            GIBBON only supports models with a covariance_between_points method.
            """
        )

    if pending_points is not None:
        tf.debugging.assert_rank(pending_points, 2)

    def acquisition(x: TensorType) -> TensorType:  # [N, D] -> [N, 1]

        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batches through greedy batch design.",
        )

        fmean, fvar = model.predict(tf.squeeze(x, -2))
        yvar = fvar + noise_variance  # need predictive variance of observations

        rho_squared = fvar / yvar  # squared correlation between observations and latent function
        fsd = tf.clip_by_value(
            tf.math.sqrt(fvar), CLAMP_LB, fmean.dtype.max
        )  # clip below to improve numerical stability
        gamma = (tf.squeeze(samples) - fmean) / fsd

        def quality_term(
            rho_squared: TensorType, gamma: TensorType
        ) -> TensorType:  # calculate GIBBON's quality term
            normal = tfp.distributions.Normal(tf.cast(0, fmean.dtype), tf.cast(1, fmean.dtype))
            minus_cdf = 1 - normal.cdf(gamma)
            minus_cdf = tf.clip_by_value(
                minus_cdf, CLAMP_LB, 1
            )  # clip below to improve numerical stability
            ratio = normal.prob(gamma) / minus_cdf
            inner_log = 1 + rho_squared * ratio * (gamma - ratio)
            acq = -0.5 * tf.math.reduce_mean(tf.math.log(inner_log), axis=1, keepdims=True)

            return acq  # [N, 1]

        def repulsion_term(
            x: TensorType, pending_points: TensorType, yvar: TensorType
        ) -> TensorType:  # calculate GIBBON's repulsion term
            _, B = model.predict_joint(pending_points)  # [1, m, m]
            L = tf.linalg.cholesky(
                B + noise_variance * tf.eye(len(pending_points), dtype=B.dtype)
            )  # need predictive variance of observations
            A = tf.expand_dims(
                model.covariance_between_points(tf.squeeze(x, -2), pending_points),  # type: ignore
                -1,
            )  # [N, m, 1]
            L_inv_A = tf.linalg.triangular_solve(L, A)
            V_det = yvar - tf.squeeze(
                tf.matmul(L_inv_A, L_inv_A, transpose_a=True), -1
            )  # equation for determinant of block matrices
            repulsion = 0.5 * (tf.math.log(V_det) - tf.math.log(yvar))

            return repulsion  # [N, 1]

        if pending_points is None:  # no repulsion term required if no pending_points
            return quality_term(rho_squared, gamma)  # [..., 1]
        else:
            if big_batch:
                batch_size, search_space_dim = tf.cast(tf.shape(pending_points), dtype=fmean.dtype)
                repulsion_weight = (1 / batch_size) ** (tf.math.log(search_space_dim))
            else:
                repulsion_weight = 1.0

            return quality_term(rho_squared, gamma) + repulsion_weight * repulsion_term(
                x, pending_points, yvar
            )

    return acquisition

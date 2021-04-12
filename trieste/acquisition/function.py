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
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import bisect

from ..data import Dataset
from ..models import ProbabilisticModel
from ..space import SearchSpace
from ..type import TensorType
from ..utils import DEFAULTS

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
    :math:`y^*` via the empirical cdf :math:`\operatorname{Pr}(y^*<y)`. The cdf is approximated
    by a Gumbel distribution

    .. math:: \mathcal G(y; a, b) = 1 - e^{-e^\frac{y - a}{b}}

    where :math:`a, b \in \mathbb R` are chosen such that the quartiles of the Gumbel and cdf match.
    Samples are obtained via the Gumbel distribution by sampling :math:`r` uniformly from
    :math:`[0, 1]` and applying the inverse probability integral transform
    :math:`y = \mathcal G^{-1}(r; a, b)`.
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

        query_points = self._search_space.sample(num_samples=self._grid_size)
        query_points = tf.concat([dataset.query_points, query_points], 0)
        fmean, fvar = model.predict(query_points)
        fsd = tf.math.sqrt(fvar)

        def probf(y: tf.Tensor) -> tf.Tensor:  # Build empirical CDF for Pr(y*^hat<y)
            unit_normal = tfp.distributions.Normal(tf.cast(0, fmean.dtype), tf.cast(1, fmean.dtype))
            log_cdf = unit_normal.log_cdf(-(y - fmean) / fsd)
            return 1 - tf.exp(tf.reduce_sum(log_cdf, axis=0))

        left = tf.reduce_min(fmean - 5 * fsd)
        right = tf.reduce_max(fmean + 5 * fsd)

        def binary_search(val: float) -> float:  # Find empirical interquartile range
            return bisect(lambda y: probf(y) - val, left, right, maxiter=10000)

        q1, q2 = map(binary_search, [0.25, 0.75])

        log = tf.math.log
        l1 = log(log(4.0 / 3.0))
        l2 = log(log(4.0))
        b = (q1 - q2) / (l1 - l2)
        a = (q2 * l1 - q1 * l2) / (l1 - l2)

        uniform_samples = tf.random.uniform([self._num_samples], dtype=fmean.dtype)
        gumbel_samples = log(-log(1 - uniform_samples)) * tf.cast(b, fmean.dtype) + tf.cast(
            a, fmean.dtype
        )

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
    tf.debugging.assert_rank(samples, 1)

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
        gamma = (samples - fmean) / fsd

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


class IndependentReparametrizationSampler:
    r"""
    This sampler employs the *reparameterization trick* to approximate samples from a
    :class:`ProbabilisticModel`\ 's predictive distribution as

    .. math:: x \mapsto \mu(x) + \epsilon \sigma(x)

    where :math:`\epsilon \sim \mathcal N (0, 1)` is constant for a given sampler, thus ensuring
    samples form a continuous curve.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel):
        """
        :param sample_size: The number of samples to take at each point. Must be positive.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)

        self._sample_size = sample_size

        # _eps is essentially a lazy constant. It is declared and assigned an empty tensor here, and
        # populated on the first call to sample
        self._eps = tf.Variable(
            tf.ones([sample_size, 0], dtype=tf.float64), shape=[sample_size, None]
        )  # [S, 0]
        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"IndependentReparametrizationSampler({self._sample_size!r}, {self._model!r})"

    def sample(self, at: TensorType) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`IndependentReparametrizationSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`IndependentReparametrizationSampler` instances will produce different samples.

        :param at: Where to sample the predictive distribution, with shape `[..., 1, D]`, for points
            of dimension `D`.
        :return: The samples, of shape `[..., S, 1, L]`, where `S` is the `sample_size` and `L` is
            the number of latent model dimensions.
        :raise ValueError (or InvalidArgumentError): If ``at`` has an invalid shape.
        """
        tf.debugging.assert_shapes([(at, [..., 1, None])])
        mean, var = self._model.predict(at[..., None, :, :])  # [..., 1, 1, L], [..., 1, 1, L]

        if tf.size(self._eps) == 0:
            self._eps.assign(
                tf.random.normal([self._sample_size, mean.shape[-1]], dtype=tf.float64)
            )  # [S, L]

        return mean + tf.sqrt(var) * tf.cast(self._eps[:, None, :], var.dtype)  # [..., S, 1, L]


class BatchReparametrizationSampler:
    r"""
    This sampler employs the *reparameterization trick* to approximate batches of samples from a
    :class:`ProbabilisticModel`\ 's predictive joint distribution as

    .. math:: x \mapsto \mu(x) + \epsilon L(x)

    where :math:`L` is the Cholesky factor s.t. :math:`LL^T` is the covariance, and
    :math:`\epsilon \sim \mathcal N (0, 1)` is constant for a given sampler, thus ensuring samples
    form a continuous curve.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel):
        """
        :param sample_size: The number of samples for each batch of points. Must be positive.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)

        self._sample_size = sample_size

        # _eps is essentially a lazy constant. It is declared and assigned an empty tensor here, and
        # populated on the first call to sample
        self._eps = tf.Variable(
            tf.ones([0, 0, sample_size], dtype=tf.float64), shape=[None, None, sample_size]
        )  # [0, 0, S]
        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"BatchReparametrizationSampler({self._sample_size!r}, {self._model!r})"

    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`BatchReparametrizationSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`BatchReparametrizationSampler` instances will produce different samples.

        :param at: Batches of query points at which to sample the predictive distribution, with
            shape `[..., B, D]`, for batches of size `B` of points of dimension `D`. Must have a
            consistent batch size across all calls to :meth:`sample` for any given
            :class:`BatchReparametrizationSampler`.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
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

        eps_is_populated = tf.size(self._eps) != 0

        if eps_is_populated:
            tf.debugging.assert_equal(
                batch_size,
                tf.shape(self._eps)[-2],
                f"{type(self).__name__} requires a fixed batch size. Got batch size {batch_size}"
                f" but previous batch size was {tf.shape(self._eps)[-2]}.",
            )

        mean, cov = self._model.predict_joint(at)  # [..., B, L], [..., L, B, B]

        if not eps_is_populated:
            self._eps.assign(
                tf.random.normal(
                    [mean.shape[-1], batch_size, self._sample_size], dtype=tf.float64
                )  # [L, B, S]
            )

        identity = tf.eye(batch_size, dtype=cov.dtype)  # [B, B]
        cov_cholesky = tf.linalg.cholesky(cov + jitter * identity)  # [..., L, B, B]

        variance_contribution = cov_cholesky @ tf.cast(self._eps, cov.dtype)  # [..., L, B, S]

        leading_indices = tf.range(tf.rank(variance_contribution) - 3)
        absolute_trailing_indices = [-1, -2, -3] + tf.rank(variance_contribution)
        new_order = tf.concat([leading_indices, absolute_trailing_indices], axis=0)

        return mean[..., None, :, :] + tf.transpose(variance_contribution, new_order)


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






class IndependentReparametrizationSampler:
    r"""
    This sampler employs the *reparameterization trick* to approximate samples from a
    :class:`ProbabilisticModel`\ 's predictive distribution as

    .. math:: x \mapsto \mu(x) + \epsilon \sigma(x)

    where :math:`\epsilon \sim \mathcal N (0, 1)` is constant for a given sampler, thus ensuring
    samples form a continuous curve.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel):
        """
        :param sample_size: The number of samples to take at each point. Must be positive.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)

        self._sample_size = sample_size

        # _eps is essentially a lazy constant. It is declared and assigned an empty tensor here, and
        # populated on the first call to sample
        self._eps = tf.Variable(
            tf.ones([sample_size, 0], dtype=tf.float64), shape=[sample_size, None]
        )  # [S, 0]
        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"IndependentReparametrizationSampler({self._sample_size!r}, {self._model!r})"

    def sample(self, at: TensorType) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`IndependentReparametrizationSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`IndependentReparametrizationSampler` instances will produce different samples.

        :param at: Where to sample the predictive distribution, with shape `[..., 1, D]`, for points
            of dimension `D`.
        :return: The samples, of shape `[..., S, 1, L]`, where `S` is the `sample_size` and `L` is
            the number of latent model dimensions.
        :raise ValueError (or InvalidArgumentError): If ``at`` has an invalid shape.
        """
        tf.debugging.assert_shapes([(at, [..., 1, None])])
        mean, var = self._model.predict(at[..., None, :, :])  # [..., 1, 1, L], [..., 1, 1, L]

        if tf.size(self._eps) == 0:
            self._eps.assign(
                tf.random.normal([self._sample_size, mean.shape[-1]], dtype=tf.float64)
            )  # [S, L]

        return mean + tf.sqrt(var) * tf.cast(self._eps[:, None, :], var.dtype)  # [..., S, 1, L]



PenalizationFunction = Callable[[TensorType], TensorType]
"""
Type alias for penalization functions.

A :const:`PenalizationFunction` maps a query point (of dimension `D`) to a single
value that measures how heavily that query point should be penalized (between 0 and 1) with 
respect to a set of `M` pending points (each of dimension `D`). Thus, with leading dimensions, an :const:`PenalizationFunction`
takes input shape `[..., 1, D]` and returns shape `[..., 1]`.
"""


class PenalizationFunctionBuilder(ABC):
    """ An :class:`PenalizationFunctionBuilder` builds a penalization function. """

    @abstractmethod
    def prepare_penalization_function(
        self, pending_points: TensorType, models: Mapping[str, ProbabilisticModel]
    ) -> PenalizationFunction:
        """
        :param pending_points: The set of pending points which we penalize with respect to.
        :param models: The models.
        :return: A penalization function.
        """

    @abstractmethod
    def estimate_penalization_parameters(self, search_space: SearchSpace, models: Mapping[str, ProbabilisticModel]):
        """
        :param search_space: The global search space over which the optimisation is defined.        
        :param models: The models.
        :return: A penalization function.
        """





class SingleModelPenalizationBuilder(ABC):
    """
    Convenience penalization function builder for an penalization function (or component of a
    composite penalization function) that requires only one model.
    """

    def using(self, tag: str) -> PenalizationFunctionBuilder:
        """
        :param tag: The tag for the model used to build this penalization function.
        :return: An penalization function builder that selects the model specified by
            ``tag``, as defined in :meth:`prepare_penalization_function`.
        """
        single_builder = self

        class _Anon(PenalizationFunctionBuilder):
            def prepare_penalization_function(
                self, pending_points: TensorType, models: Mapping[str, ProbabilisticModel]
            ) -> PenalizationFunction:
                return single_builder.prepare_penalization_function(pending_points, models[tag])

            def estimate_penalization_parameters(self, search_space: SearchSpace, models: Mapping[str, ProbabilisticModel]):
                return single_builder.estimate_penalization_parameters(search_space, models[tag])

            def __repr__(self) -> str:
                return f"{single_builder!r} using tag {tag!r}"

        return _Anon()

    @abstractmethod
    def prepare_penalization_function(
        self, pending_points: TensorType, model: ProbabilisticModel
    ) -> PenalizationFunction:
        """
        :param pending_points: The set of pending points which we penalize with respect to.
        :param model: The model over the specified ``dataset``.
        :return: A penalization function.
        """

    @abstractmethod
    def estimate_penalization_parameters(self, search_space: SearchSpace, model: ProbabilisticModel):
        """
        :param search_space: The global search space over which the optimisation is defined.        
        :param model: The model over the specified ``dataset``.
        :return: A penalization function.
        """


class LocalPenalization(SingleModelPenalizationBuilder):
    """
    Builder for the local penalization acqusiition function. See :cite:`Gonzalez:2016` for details.


    SAY WHAT THIS IS

    xpected improvement function where the "best" value is taken to be the minimum
    of the posterior mean at observed points.
    """


def __init__(self, grid_size: int = 5000):
        """
        :param grid_size: Size of the grid over which the Lipschitz constant is estimated. We recommend
            scaling this with search space dimension.
        """

        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        self._grid_size = grid_size

        self._lipschitz_constant = None
        self._eta = None

    def estimate_penalization_parameters(self, search_space: SearchSpace, model: ProbabilisticModel):
        r"""
        DESCRIBE MATHS FOR SAMPLES!


        :param search_space: The global search space over which the optimisation is defined.
        :param model: The model over the specified ``dataset``.
        """
        samples_points = search_space.sample(num_samples=self._grid_size)
        
        with tf.GradientTape() as g: # get gradients of posterior mean at samples
            g.watch(sample_points)
            mean, _ = model.predict(sample_points)
        grads = g.gradient(mean,sample_points)
        grads_norm =  tf.norm(grads, axis=1)
        max_grads_norm = tf.reduce_max(grads_norm)

        if max_grads_norm < 1e-5: # threshold to improve numerical stability for 'flat' models
                self._lipschitz_constant = 10
            else:
                self._lipschitz_constant = max_grads_norm

        self._eta = tf.reduce_min(mean, axis=0)


    def prepare_penalization_function(
        self, pending_points: TensorType, model: ProbabilisticModel
    ) -> PenalizationFunction:
        """
        :param pending_points: The set of pending points which we penalize with respect to.
        :param model: The model over the specified ``dataset``.
        :return: The local penalization function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise ValueError: If ``dataset`` is empty or if the penalization parameters have not yet been estimated.
        """

        if len(pending_points) == 0:
            raise ValueError("pending_points must be populated.")

        if (not self._lipschitz_constant) or (not self._eta):
            raise ValueError("penalization parameters must be estimated before penalization building a penalization function.")  

        return local_penalization(model, pending_points, self._eta, self._lipschitz_constant)




def local_penalization(model: ProbabilisticModel, pending_points: TensorType, eta: TensorType, lipschitz_constant: TensorType) -> PenalizationFunction:
    r"""
    Return the local penalizer as used for single-objective greedy batch Bayesian optimization by 
    :cite:`Gonzalez:2016`.


    DISCUSS MATHGS


    :param model: The model of the objective function.
    :param pending points: The set of pending points which we penalize with respect to.
    :param eta: The estimated global minimum value.
    :param lipschitz_constant: The estimated objective function's Lipschitz constant.
    :return: The local penalization function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one as it is to be used to build batches greedily 
    """

    mean, variance  = model.predict(self._X_pending)
    radius = tf.transpose((mean - eta) / lipschitz_constant)
    scale = tf.transpose(tf.sqrt(variance) / lipschitz_constant)

    def penalization_function(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This penalization function can only be used for greedy batch design.",
        )

        pairwise_distances = tf.norm(tf.expand_dims(x,1)-tf.expand_dims(pending_points,0),axis=-1)
        standardised_distances = (pairwise_distances - radius) / scale

        normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
        penalisation = normal.cdf(standardised_distances)
        return penalisation

    return penalization_function


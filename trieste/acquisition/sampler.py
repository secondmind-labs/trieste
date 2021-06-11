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
This module is the home of the sampling functionality required by Trieste's
acquisiiton functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import tensorflow as tf
import tensorflow_probability as tfp
from gpflux.layers.basis_functions import RandomFourierFeatures
from scipy.optimize import bisect

from ..data import Dataset
from ..models import ProbabilisticModel
from ..type import TensorType
from ..utils import DEFAULTS


class DiscreteSampler(ABC):
    r"""
    An :class:`DiscreteSampler` samples a specific quantity across a discrete set of points
    according to an underlying :class:`ProbabilisticModel`.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel):
        """
        :param sample_size: The desired number of samples.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)

        self._sample_size = sample_size
        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._sample_size!r}, {self._model!r})"

    @abstractmethod
    def sample(self, at: TensorType) -> TensorType:
        """
        :param at: Input points that define the sampler.
        :return: Samples.
        """


class DiscreteThompsonSampler(DiscreteSampler):
    r"""
    This sampler provides approximate Thompson samples of the objective function's
    maximiser :math:`x^*` over a discrete set of input locations.
    """

    def sample(self, at: TensorType) -> TensorType:
        """
        Return approximate samples from of the objective function's minimser. We return only
        unique samples.

        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :return: The samples, of shape `[S, D]`, where `S` is the `sample_size`.
        :raise ValueError (or InvalidArgumentError): If ``at`` has an invalid shape.
        """
        tf.debugging.assert_shapes([(at, ["N", None])])

        samples = self._model.sample(at, self._sample_size)  # [self._sample_size, len(at), 1]
        samples_2d = tf.squeeze(samples, -1)  # [self._sample_size, len(at)]
        indices = tf.math.argmin(samples_2d, axis=1)
        unique_indices = tf.unique(indices).y
        thompson_samples = tf.gather(at, unique_indices)
        return thompson_samples


class GumbelSampler(DiscreteSampler):
    r"""
    This sampler follows :cite:`wang2017max` and yields approximate samples of the objective
    minimum value :math:`y^*` via the empirical cdf :math:`\operatorname{Pr}(y^*<y)`. The cdf
    is approximated by a Gumbel distribution

    .. math:: \mathcal G(y; a, b) = 1 - e^{-e^\frac{y - a}{b}}

    where :math:`a, b \in \mathbb R` are chosen such that the quartiles of the Gumbel and cdf match.
    Samples are obtained via the Gumbel distribution by sampling :math:`r` uniformly from
    :math:`[0, 1]` and applying the inverse probability integral transform
    :math:`y = \mathcal G^{-1}(r; a, b)`.
    """

    def sample(self, at: TensorType) -> TensorType:
        """
        Return approximate samples from of the objective function's minimum value.

        :param at: Points at where to fit the Gumbel distribution, with shape `[N, D]`, for points
            of dimension `D`. We recommend scaling `N` with search space dimension.
        :return: The samples, of shape `[S, 1]`, where `S` is the `sample_size`.
        :raise ValueError (or InvalidArgumentError): If ``at`` has an invalid shape.
        """
        tf.debugging.assert_shapes([(at, ["N", None])])

        try:
            fmean, fvar = self._model.predict_y(at)
        except NotImplementedError:
            fmean, fvar = self._model.predict(at)

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

        uniform_samples = tf.random.uniform([self._sample_size], dtype=fmean.dtype)
        gumbel_samples = log(-log(1 - uniform_samples)) * tf.cast(b, fmean.dtype) + tf.cast(
            a, fmean.dtype
        )
        gumbel_samples = tf.expand_dims(gumbel_samples, axis=-1)  # [S, 1]
        return gumbel_samples


class IndependentReparametrizationSampler(DiscreteSampler):
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
        super().__init__(sample_size, model)

        # _eps is essentially a lazy constant. It is declared and assigned an empty tensor here, and
        # populated on the first call to sample
        self._eps = tf.Variable(
            tf.ones([sample_size, 0], dtype=tf.float64), shape=[sample_size, None]
        )  # [S, 0]

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


class BatchReparametrizationSampler(DiscreteSampler):
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
        super().__init__(sample_size, model)

        # _eps is essentially a lazy constant. It is declared and assigned an empty tensor here, and
        # populated on the first call to sample
        self._eps = tf.Variable(
            tf.ones([0, 0, sample_size], dtype=tf.float64), shape=[None, None, sample_size]
        )  # [0, 0, S]

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


class ContinuousSampler(ABC):
    r"""
    An :class:`ContinuousSampler` samples a specific quantity according
    to an underlying :class:`ProbabilisticModel`.
    Unlike our :class:`DiscreteSampler`, :class:`ContinuousSampler` returns a
    queryable function (a trajectory) that provides the sample's value at a query location.
    """

    def __init__(self, dataset: Dataset, model: ProbabilisticModel):
        """
        :param dataset: The data from the observer. Must be populated.
        :param model: The model to sample from.
        :raise ValueError: If ``dataset`` is empty.
        """

        if len(dataset.query_points) == 0:
            raise ValueError("Dataset must be populated.")

        self._dataset = dataset
        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._model!r}, {self._dataset!r})"

    @abstractmethod
    def get_trajectory(self) -> Callable[[TensorType], TensorType]:
        """
        Return a function that evaluates a particular sample at a set of `N` query
        points (each of dimension `D`) i.e. takes input of shape `[N, D]` and returns
        shape `[N, 1]`.

        :return: Queryable function representing a sample.
        """


class RandomFourierFeatureThompsonSampler(ContinuousSampler):
    r"""
    This class builds functions that approximate a trajectory sampled from an underlying Gaussian
    process model. For tractibility, the Gaussian process is approximated with a Bayesian
    Linear model across a set of features sampled from the Fourier feature decomposition of
    the model's kernel. See :cite:`hernandez2014predictive` for details.

    A key property of these trajectory functions is that the same sample draw is evaluated
    for all queries. This property is known as consistency. Achieving consistency for exact
    sample draws from a  GP is prohibitively costly because it scales cubically with the number
    of query points. However, finite feature representations can be evaluated with constant cost
    regardless of the required number of queries.

    In particular, we approximate the Gaussian processes' posterior samples as the finite feature
    approximation

    .. math:: \hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i

    where :math:`\phi_i` are m Fourier features and :math:`\theta_i` are
    feature weights sampled from a posterior distribution that depends on the feature values at the
    model's datapoints.

    Our implementation follows :cite:`hernandez2014predictive`, with our calculations
    differing slightly depending on properties of the problem. In particular,  we used different
    calculation strategies depending on the number of considered features m and the number
    of data points n.

    If :math:`m<n` then we follow Appendix A of :cite:`hernandez2014predictive` and calculate the
    posterior distribution for :math:`\theta` following their Bayesian linear regression motivation,
    i.e. the computation revolves around an O(m^3)  inversion of a design matrix.

    If :math:`n<m` then we use the kernel trick to recast computation to revolve around an O(n^3)
    inversion of a gram matrix. As well as being more efficient in early BO
    steps (where :math:`n<m`), this second computation method allows must larger choices
    of m (as required to approximate very flexible kernels).
    """

    def __init__(self, dataset: Dataset, model: ProbabilisticModel, num_features: int = 1000):
        """
        :param dataset: The data from the observer. Must be populated.
        :param model: The model to sample from.
        :param num_features: The number of features used to approximate the kernel.
        :raise ValueError: If ``dataset`` is empty.
        """
        super().__init__(dataset, model)

        tf.debugging.assert_positive(num_features)
        self._num_features = num_features  # m
        self._num_data = len(self._dataset.query_points)  # n

        try:
            self._noise_variance = model.get_observation_noise()
            self._kernel = model.get_kernel()
        except (NotImplementedError, AttributeError):
            raise ValueError(
                """
            Thompson sampling with random Fourier features only currently supports models
            with likelihood.variance and kernel attributes.
            """
            )

        self._pre_calc = False

    def __repr__(self) -> str:
        """"""
        return (
            f"{self.__class__.__name__}({self._model!r}, {self._dataset!r}, {self._num_features!r})"
        )

    def _prepare_theta_posterior_in_design_space(self) -> tfp.distributions.Distribution:
        r"""
        Calculate the posterior of theta (the feature weights) in the design space. This
        distribution is a Gaussian

        .. math:: \theta \sim N(D^{-1}\Phi^Ty,D^{-1}\sigma^2)

        where the [m,m] design matrix :math:`D=(\Phi^T\Phi + \sigma^2I_m)` is defined for
        the [n,m] matrix of feature evaluations across the training data :math:`\Phi`
        and observation noise variance :math:`\sigma^2`.

        :return: The posterior distribution for theta.
        """

        phi = self._feature_functions(self._dataset.query_points)  # [n, m]
        D = tf.matmul(phi, phi, transpose_a=True)  # [m, m]
        s = self._noise_variance * tf.eye(self._num_features, dtype=phi.dtype)
        L = tf.linalg.cholesky(D + s)
        D_inv = tf.linalg.cholesky_solve(L, tf.eye(self._num_features, dtype=phi.dtype))

        theta_posterior_mean = tf.matmul(
            D_inv, tf.matmul(phi, self._dataset.observations, transpose_a=True)
        )[
            :, 0
        ]  # [m,]
        theta_posterior_chol_covariance = tf.linalg.cholesky(D_inv * self._noise_variance)  # [m, m]

        return tfp.distributions.MultivariateNormalTriL(
            theta_posterior_mean, theta_posterior_chol_covariance
        )

    def _prepare_theta_posterior_in_gram_space(self) -> tfp.distributions.Distribution:
        r"""
        Calculate the posterior of theta (the feature weights) in the gram space.

         .. math:: \theta \sim N(\Phi^TG^{-1}y,I_m - \Phi^TG^{-1}\Phi)

        where the [n,n] gram matrix :math:`G=(\Phi\Phi^T + \sigma^2I_n)` is defined for the [n,m]
        matrix of feature evaluations across the training data :math:`\Phi` and
        observation noise variance :math:`\sigma^2`.

        :return: The posterior distribution for theta.
        """

        phi = self._feature_functions(self._dataset.query_points)  # [n, m]
        G = tf.matmul(phi, phi, transpose_b=True)  # [n, n]
        s = self._noise_variance * tf.eye(self._num_data, dtype=phi.dtype)
        L = tf.linalg.cholesky(G + s)
        L_inv_phi = tf.linalg.triangular_solve(L, phi)  # [n, m]
        L_inv_y = tf.linalg.triangular_solve(L, self._dataset.observations)  # [n, 1]

        theta_posterior_mean = tf.tensordot(tf.transpose(L_inv_phi), L_inv_y, [[-1], [-2]])[
            :, 0
        ]  # [m,]
        theta_posterior_covariance = tf.eye(self._num_features, dtype=phi.dtype) - tf.tensordot(
            tf.transpose(L_inv_phi), L_inv_phi, [[-1], [-2]]
        )  # [m, m]
        theta_posterior_chol_covariance = tf.linalg.cholesky(theta_posterior_covariance)  # [m, m]

        return tfp.distributions.MultivariateNormalTriL(
            theta_posterior_mean, theta_posterior_chol_covariance
        )

    def get_trajectory(self) -> Callable[[TensorType], TensorType]:
        """
        Generate an approximate function draw (trajectory) by sampling weights
        and evaluating the feature functions.

        If this is the first call, then we calculate the posterior distributions for
        the feautre weights.

        :return: A function representing an approximate trajectory from the Gaussian process,
            taking an input of shape `[N, D]` and returning shape `[N, 1]`
        """

        if not self._pre_calc:
            self._feature_functions = RandomFourierFeatures(
                self._kernel, self._num_features, dtype=self._dataset.query_points.dtype
            )

            if (
                self._num_features < self._num_data
            ):  # if m < n  then calculate posterior in design space (an m*m matrix inversion)
                self._theta_posterior = self._prepare_theta_posterior_in_design_space()
            else:  # if n < m  then calculate posterior in gram space (an n*n matrix inversion)
                self._theta_posterior = self._prepare_theta_posterior_in_gram_space()

            self._pre_calc = True

        theta_sample = self._theta_posterior.sample(1)  # [1, m]

        def trajectory(x: TensorType) -> TensorType:
            feature_evaluations = self._feature_functions(x)  # [N, m]
            return tf.matmul(feature_evaluations, theta_sample, transpose_b=True)  # [N,1]

        return trajectory

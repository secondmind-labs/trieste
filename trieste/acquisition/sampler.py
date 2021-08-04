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
from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp
from gpflux.layers.basis_functions import RandomFourierFeatures
from scipy.optimize import bisect

from ..data import Dataset
from ..models import ProbabilisticModel
from ..type import TensorType
from ..utils import DEFAULTS


class Sampler(ABC):
    r"""
    A :class:`Sampler` samples a specific quantity across a discrete set of points
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


class ThompsonSampler(Sampler, ABC):
    r"""
    A :class:`ThompsonSampler` samples either the minimum values or minimisers of a function
    modeled by an underlying :class:`ProbabilisticModel` across a  discrete set of points.

    """

    def __init__(self, sample_size: int, model: ProbabilisticModel, sample_min_value: bool = False):
        """
        :param sample_size: The desired number of samples.
        :param model: The model to sample from.
        :sample_min_value: If True then sample from the minimum value of the function,
            else sample the function's minimiser.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        super().__init__(sample_size, model)
        self._sample_min_value = sample_min_value

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}(
        {self._sample_size!r},
        {self._model!r},
        {self._sample_min_value})
        """


class ExactThompsonSampler(ThompsonSampler):
    r"""
    This sampler provides exact Thompson samples of the objective function's
    maximiser :math:`x^*` over a discrete set of input locations.

    Although exact Thompson sampling is costly (incuring with an :math:`O(N^3)` complexity to
    sample over a set of `N` locations), this method can be used for any probabilistic model
    with a sampling method.

    """

    def sample(self, at: TensorType) -> TensorType:
        """
        Return exact samples from either the objective function's minimser or its minimal value
        over the candidate set `at`.

        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :return: The samples, of shape `[S, D]` (where `S` is the `sample_size`) if sampling
            the function's minimser or shape `[S, 1]` if sampling the function's mimimal value.
        :raise ValueError (or InvalidArgumentError): If ``at`` has an invalid shape.
        """
        tf.debugging.assert_shapes([(at, ["N", None])])

        samples = self._model.sample(at, self._sample_size)  # [S, N, 1]

        if self._sample_min_value:
            thompson_samples = tf.reduce_min(samples, axis=1)  # [S, 1]
        else:
            samples_2d = tf.squeeze(samples, -1)  # [S, N]
            indices = tf.math.argmin(samples_2d, axis=1)
            thompson_samples = tf.gather(at, indices)  # [S, D]

        return thompson_samples


class GumbelSampler(ThompsonSampler):
    r"""
    This sampler follows :cite:`wang2017max` and yields approximate samples of the objective
    minimum value :math:`y^*` via the empirical cdf :math:`\operatorname{Pr}(y^*<y)`. The cdf
    is approximated by a Gumbel distribution

    .. math:: \mathcal G(y; a, b) = 1 - e^{-e^\frac{y - a}{b}}

    where :math:`a, b \in \mathbb R` are chosen such that the quartiles of the Gumbel and cdf match.
    Samples are obtained via the Gumbel distribution by sampling :math:`r` uniformly from
    :math:`[0, 1]` and applying the inverse probability integral transform
    :math:`y = \mathcal G^{-1}(r; a, b)`.

    Note that the :class:`GumbelSampler` can only sample a function's minimal value and not
    its minimiser.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel):
        """
        :param sample_size: The desired number of samples.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        super().__init__(sample_size, model, True)

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


class IndependentReparametrizationSampler(Sampler):
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


class BatchReparametrizationSampler(Sampler):
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


TrajectoryFunction = Callable[[TensorType], TensorType]
"""
Type alias for trajectory functions.

An :const:`TrajectoryFunction` evaluates a particular sample at a set of `N` query
points (each of dimension `D`) i.e. takes input of shape `[N, D]` and returns
shape `[N, 1]`.

A key property of these trajectory functions is that the same sample draw is evaluated
for all queries. This property is known as consistency.
"""


class RandomFourierFeatureThompsonSampler(ThompsonSampler):
    r"""
    This class builds functions that approximate a trajectory sampled from an underlying Gaussian
    process model. For tractibility, the Gaussian process is approximated with a Bayesian
    Linear model across a set of features sampled from the Fourier feature decomposition of
    the model's kernel. See :cite:`hernandez2014predictive` for details.

    Achieving consistency (ensuring that the same sample draw for all evalutions of a particular
    trajectory function) for exact sample draws from a GP is prohibitively costly because it scales
    cubically with the number of query points. However, finite feature representations can be
    evaluated with constant cost regardless of the required number of queries.

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

    def __init__(
        self,
        sample_size: int,
        model: ProbabilisticModel,
        dataset: Dataset,
        sample_min_value: bool = False,
        num_features: int = 1000,
    ):
        """
        :param sample_size: The desired number of samples.
        :param model: The model to sample from.
        :param dataset: The data from the observer. Must be populated.
        :sample_min_value: If True then sample from the minimum value of the function,
            else sample the function's minimiser.
        :param num_features: The number of features used to approximate the kernel. We use a default
            of 1000 as it typically perfoms well for a wide range of kernels. Note that very smooth
            kernels (e.g. RBF) can be well-approximated with fewer features.
        :raise ValueError: If ``dataset`` is empty.
        """

        super().__init__(sample_size, model, sample_min_value)

        if len(dataset.query_points) == 0:
            raise ValueError("Dataset must be populated.")

        self._dataset = dataset
        self._model = model

        tf.debugging.assert_positive(num_features)
        self._num_features = num_features  # m
        self._num_data = len(self._dataset.query_points)  # n

        try:
            self._noise_variance = model.get_observation_noise()
            self._kernel = model.get_kernel()  # type: ignore
        except (NotImplementedError, AttributeError):
            raise ValueError(
                """
            Thompson sampling with random Fourier features only currently supports models
            with a Gaussian likelihood and an accessible kernel attribute.
            """
            )

        self._feature_functions = RandomFourierFeatures(
            self._kernel, self._num_features, dtype=self._dataset.query_points.dtype
        )  # prep feature functions at data

        if (
            self._num_features < self._num_data
        ):  # if m < n  then calculate posterior in design space (an m*m matrix inversion)
            self._theta_posterior = self._prepare_theta_posterior_in_design_space()
        else:  # if n <= m  then calculate posterior in gram space (an n*n matrix inversion)
            self._theta_posterior = self._prepare_theta_posterior_in_gram_space()

        self._pre_calc = False  # Flag so we only calculate the posterior for the weights once.

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}(
        {self._sample_size!r},
        {self._model!r},
        {self._sample_min_value},
        {self._num_features!r})
        """

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

    def get_trajectory(self) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) by sampling weights
        and evaluating the feature functions.

        :return: A trajectory function representing an approximate trajectory from the Gaussian
            process, taking an input of shape `[N, D]` and returning shape `[N, 1]`
        """

        theta_sample = self._theta_posterior.sample(1)  # [1, m]

        def trajectory(x: TensorType) -> TensorType:
            feature_evaluations = self._feature_functions(x)  # [N, m]
            return tf.matmul(feature_evaluations, theta_sample, transpose_b=True)  # [N,1]

        return trajectory

    def sample(self, at: TensorType) -> TensorType:
        """
        Return approximate samples from either the objective function's minimser or its minimal
        value over the candidate set `at`.

        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :return: The samples, of shape `[S, D]` (where `S` is the `sample_size`) if sampling
            the function's minimser or shape `[S, 1]` if sampling the function's mimimal value.
        :raise ValueError (or InvalidArgumentError): If ``at`` has an invalid shape.
        """
        tf.debugging.assert_shapes([(at, ["N", None])])

        if self._sample_min_value:
            thompson_samples = tf.zeros([0, 1], dtype=at.dtype)  # [0,1]
        else:
            thompson_samples = tf.zeros([0, tf.shape(at)[1]], dtype=at.dtype)  # [0,D]

        for _ in tf.range(self._sample_size):
            sampled_trajectory = self.get_trajectory()
            evaluated_trajectory = sampled_trajectory(at)  # [N, 1]
            if self._sample_min_value:
                sample = tf.reduce_min(evaluated_trajectory, keepdims=True)  # [1, 1]
            else:
                sample = tf.gather(at, tf.math.argmin(evaluated_trajectory))  # [1, D]

            thompson_samples = tf.concat([thompson_samples, sample], axis=0)

        return thompson_samples  # [S, D] or [S, 1]

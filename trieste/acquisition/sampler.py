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

import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import bisect

from ..models import ProbabilisticModel
from ..type import TensorType
from ..utils import DEFAULTS


class Sampler(ABC):
    r"""
    An :class:`Sampler` samples a specific quantity according to an
    underlying :class:`ProbabilisticModel`.
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


class DiscreteThompsonSampler(Sampler):
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


class GumbelSampler(Sampler):
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

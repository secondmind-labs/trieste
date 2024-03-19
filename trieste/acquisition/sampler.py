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
acquisition functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Generic

import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import bisect

from ..models import ProbabilisticModel
from ..models.interfaces import HasTrajectorySampler, ProbabilisticModelType, SupportsPredictY
from ..types import TensorType
from .utils import select_nth_output


class ThompsonSampler(ABC, Generic[ProbabilisticModelType]):
    r"""
    A :class:`ThompsonSampler` samples either the minimum values or minimisers of a function
    modeled by an underlying :class:`ProbabilisticModel` across a  discrete set of points.
    """

    def __init__(self, sample_min_value: bool = False):
        """
        :sample_min_value: If True then sample from the minimum value of the function,
            else sample the function's minimiser.
        """
        self._sample_min_value = sample_min_value

    @property
    def sample_min_value(self) -> bool:
        """Whether this samples from the minimum value of the function
        (as opposed to the function's minimiser)."""
        return self._sample_min_value

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}(
        {self._sample_min_value})
        """

    @abstractmethod
    def sample(
        self,
        model: ProbabilisticModelType,
        sample_size: int,
        at: TensorType,
        select_output: Callable[[TensorType], TensorType] = select_nth_output,
    ) -> TensorType:
        """
        :param model: The model to sample from.
        :param sample_size: The desired number of samples.
        :param at: Input points that define the sampler.
        :param select_output: A method that returns the desired output from the model sampler, with
            shape `[S, N]` where `S` is the number of samples and `N` is the number of locations.
            Defaults to the :func:~`trieste.acquisition.utils.select_nth_output` function with
            output dimension 0.
        :return: Samples.
        """


class ExactThompsonSampler(ThompsonSampler[ProbabilisticModel]):
    r"""
    This sampler provides exact Thompson samples of the objective function's
    minimiser :math:`x^*` over a discrete set of input locations.
    Although exact Thompson sampling is costly (incuring with an :math:`O(N^3)` complexity to
    sample over a set of `N` locations), this method can be used for any probabilistic model
    with a sampling method.
    """

    def sample(
        self,
        model: ProbabilisticModel,
        sample_size: int,
        at: TensorType,
        select_output: Callable[[TensorType], TensorType] = select_nth_output,
    ) -> TensorType:
        """
        Return exact samples from either the objective function's minimiser or its minimal value
        over the candidate set `at`. Note that minimiser ties aren't broken randomly.

        :param model: The model to sample from.
        :param sample_size: The desired number of samples.
        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :param select_output: A method that returns the desired output from the model sampler, with
            shape `[S, N]` where `S` is the number of samples and `N` is the number of locations.
            Defaults to the :func:~`trieste.acquisition.utils.select_nth_output` function with
            output dimension 0.
        :return: The samples, of shape `[S, D]` (where `S` is the `sample_size`) if sampling
            the function's minimiser or shape `[S, 1]` if sampling the function's mimimal value.
        :raise ValueError: If ``at`` has an invalid shape or if ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_shapes([(at, ["N", None])])

        samples = select_output(model.sample(at, sample_size))[..., None]  # [S, N, 1]

        if self._sample_min_value:
            thompson_samples = tf.reduce_min(samples, axis=1)  # [S, 1]
        else:
            samples_2d = tf.squeeze(samples, -1)  # [S, N]
            indices = tf.math.argmin(samples_2d, axis=1)
            thompson_samples = tf.gather(at, indices)  # [S, D]

        return thompson_samples


class GumbelSampler(ThompsonSampler[ProbabilisticModel]):
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

    def __init__(self, sample_min_value: bool = False):
        """
        :sample_min_value: If True then sample from the minimum value of the function,
            else sample the function's minimiser.
        """

        if not sample_min_value:
            raise ValueError(
                f"""
                Gumbel samplers can only sample a function's minimal value,
                however received sample_min_value={sample_min_value}
                """
            )

        super().__init__(sample_min_value)

    def sample(
        self,
        model: ProbabilisticModel,
        sample_size: int,
        at: TensorType,
        select_output: Callable[[TensorType], TensorType] = select_nth_output,
    ) -> TensorType:
        """
        Return approximate samples from of the objective function's minimum value.

        :param model: The model to sample from.
        :param sample_size: The desired number of samples.
        :param at: Points at where to fit the Gumbel distribution, with shape `[N, D]`, for points
            of dimension `D`. We recommend scaling `N` with search space dimension.
        :param select_output: A method that returns the desired output from the model sampler, with
            shape `[S, N]` where `S` is the number of samples and `N` is the number of locations.
            Currently unused.
        :return: The samples, of shape `[S, 1]`, where `S` is the `sample_size`.
        :raise ValueError: If ``at`` has an invalid shape or if ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_shapes([(at, ["N", None])])

        if isinstance(model, SupportsPredictY):
            fmean, fvar = model.predict_y(at)
        else:
            fmean, fvar = model.predict(at)

        fsd = tf.math.sqrt(fvar)

        def probf(y: tf.Tensor) -> tf.Tensor:  # Build empirical CDF for Pr(y*^hat<y)
            unit_normal = tfp.distributions.Normal(
                tf.constant(0, fmean.dtype), tf.constant(1, fmean.dtype)
            )
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

        uniform_samples = tf.random.uniform([sample_size], dtype=fmean.dtype)
        gumbel_samples = log(-log(1 - uniform_samples)) * tf.cast(b, fmean.dtype) + tf.cast(
            a, fmean.dtype
        )
        gumbel_samples = tf.expand_dims(gumbel_samples, axis=-1)  # [S, 1]
        return gumbel_samples


class ThompsonSamplerFromTrajectory(ThompsonSampler[HasTrajectorySampler]):
    r"""
    This sampler provides approximate Thompson samples of the objective function's
    minimiser :math:`x^*` by minimizing approximate trajectories sampled from the
    underlying probabilistic model. This sampling method can be used for any
    probabilistic model with a :meth:`trajectory_sampler` method.
    """

    def sample(
        self,
        model: ProbabilisticModel,
        sample_size: int,
        at: TensorType,
        select_output: Callable[[TensorType], TensorType] = select_nth_output,
    ) -> TensorType:
        """
        Return approximate samples from either the objective function's minimser or its minimal
        value over the candidate set `at`. Note that minimiser ties aren't broken randomly.

        :param model: The model to sample from.
        :param sample_size: The desired number of samples.
        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :param select_output: A method that returns the desired output from the model sampler, with
            shape `[S, N]` where `S` is the number of samples and `N` is the number of locations.
            Defaults to the :func:~`trieste.acquisition.utils.select_nth_output` function with
            output dimension 0.
        :return: The samples, of shape `[S, D]` (where `S` is the `sample_size`) if sampling
            the function's minimser or shape `[S, 1]` if sampling the function's mimimal value.
        :raise ValueError: If ``at`` has an invalid shape or if ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_shapes([(at, ["N", None])])

        if not isinstance(model, HasTrajectorySampler):
            raise ValueError(
                f"Thompson sampling from trajectory only supports models with a trajectory_sampler "
                f"method; received {model!r}"
            )

        trajectory_sampler = model.trajectory_sampler()

        if self._sample_min_value:
            thompson_samples = tf.zeros([0, 1], dtype=at.dtype)  # [0,1]
        else:
            thompson_samples = tf.zeros([0, tf.shape(at)[1]], dtype=at.dtype)  # [0,D]

        for _ in tf.range(sample_size):
            sampled_trajectory = trajectory_sampler.get_trajectory()
            expanded_at = tf.expand_dims(at, -2)  # [N, 1, D]
            evaluated_trajectory = select_output(sampled_trajectory(expanded_at))  # [N, 1]
            if self._sample_min_value:
                sample = tf.reduce_min(evaluated_trajectory, keepdims=True)  # [1, 1]
            else:
                sample = tf.gather(at, tf.math.argmin(evaluated_trajectory))  # [1, D]

            thompson_samples = tf.concat([thompson_samples, sample], axis=0)

        return thompson_samples  # [S, D] or [S, 1]

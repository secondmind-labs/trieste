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
This module contains acquisition function builders and acquisition functions for Bayesian active
learning.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...models import ProbabilisticModel
from ...types import TensorType
from ...utils import DEFAULTS
from ..interface import AcquisitionFunction, SingleModelAcquisitionBuilder


class PredictiveVariance(SingleModelAcquisitionBuilder):
    """
    Builder for the determinant of the predictive covariance matrix over the batch points.
    For a batch of size 1 it is the same as maximizing the predictive variance.
    """

    def __init__(self, jitter: float = DEFAULTS.JITTER) -> None:
        """
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        """
        super().__init__()

        self._jitter = jitter

    def __repr__(self) -> str:
        """"""
        return f"PredictiveVariance(jitter={self._jitter!r})"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: Unused.

        :return: The determinant of the predictive function.
        """

        return predictive_variance(model, self._jitter)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: Unused.
        """
        return function  # no need to update anything


def predictive_variance(model: ProbabilisticModel, jitter: float) -> TensorType:
    """
    The predictive variance acquisition function for active learning, based on
    the determinant of the covariance (see :cite:`MacKay1992` for details).
    Note that the model needs to supply covariance of the joint marginal distribution,
    which can be expensive to compute.

    :param model: The model of the objective function.
    :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
    """

    @tf.function
    def acquisition(x: TensorType) -> TensorType:

        try:
            _, covariance = model.predict_joint(x)
        except NotImplementedError:
            raise ValueError(
                """
                PredictiveVariance only supports models with a predict_joint method.
                """
            )
        return tf.exp(tf.linalg.logdet(covariance + jitter))

    return acquisition


class ExpectedFeasibility(SingleModelAcquisitionBuilder):
    """
    Builder for the Expected feasibility acquisition function for identifying a failure or
    feasibility region. It implements two related sampling strategies called *bichon* criterion
    (:cite:`bichon2008efficient`) and *ranjan* criterion (:cite:`ranjan2008sequential`). The goal
    of both criteria is to sample points with a mean close to the threshold and a high variance.
    """

    def __init__(self, threshold: float, alpha: float = 1, delta: int = 1) -> None:
        """
        :param threshold: The failure or feasibility threshold.
        :param alpha: The parameter which determines the neighbourhood around the estimated contour
            line as a percentage of the posterior variance in which to allocate new points. Defaults
            to value of 1.
        :param delta: The parameter identifying which criterion is used, *bichon* for value of 1
            (default) and *ranjan* for value of 2.
        :raise ValueError (or InvalidArgumentError): If arguments are not a scalar, or `alpha` is
            not positive, or `delta` is not 1 or 2.
        """
        tf.debugging.assert_scalar(threshold)
        tf.debugging.assert_scalar(alpha)
        tf.debugging.assert_positive(alpha, message="Parameter alpha must be positive.")
        tf.debugging.assert_scalar(delta)
        tf.debugging.Assert(delta in [1, 2], [delta])

        super().__init__()

        self._threshold = threshold
        self._alpha = alpha
        self._delta = delta

    def __repr__(self) -> str:
        """"""
        return (
            f"ExpectedFeasibility(threshold={self._threshold!r}, alpha={self._alpha!r},"
            f" delta={self._delta!r})"
        )

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: Unused.
        :return: The expected feasibility function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """
        return bichon_ranjan_criterion(model, self._threshold, self._alpha, self._delta)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:

        return function  # no need to update anything


def bichon_ranjan_criterion(
    model: ProbabilisticModel,
    threshold: float,
    alpha: float,
    delta: int,
) -> TensorType:
    r"""
    Return the *bichon* criterion (:cite:`bichon2008efficient`) and *ranjan* criterion
    (:cite:`ranjan2008sequential`) used in Expected feasibility acquisition function for active
    learning of failure or feasibility regions.

    The problem of identifying a failure or feasibility region of a function :math:`f` can be
    formalized as estimating the excursion set, :math:`\Gamma^* = \{ x \in X: f(x) \ge T\}`, or
    estimating the contour line, :math:`C^* = \{ x \in X: f(x) = T\}`, for some threshold :math:`T`
    (see :cite:`bect2012sequential` for more details).

    It turns out that probabilistic models can be used as classifiers for identifying where
    excursion probability is larger than 1/2 and this idea is used to build many sequential
    sampling strategies. We follow :cite:`bect2012sequential` and use a formulation which provides
    a common expression for these two criteria:

    .. math:: \mathbb{E}[\max(0, (\alpha s(x))^\delta - |T - m(x)|^\delta)]

    Here :math:`m(x)` and :math:`s(x)` are the mean and standard deviation of the predictive
    posterior of a probabilistic model. *Bichon* criterion is obtained when :math:`\delta = 1` while
    *ranjan* criterion is obtained when :math:`\delta = 2`. :math:`\alpha>0` is another parameter
    that acts as a percentage of standard deviation of the posterior around the current boundary
    estimate where we want to sample. The goal is to sample a point with a mean close to the
    threshold :math:`T` and a high variance, so that the positive difference in the equation above
    is as large as possible.

    Note that only batches of size 1 are allowed.

    :param model: The probabilistic model of the objective function.
    :param threshold: The failure or feasibility threshold.
    :param alpha: The parameter which determines the neighbourhood around the estimated contour
        line as a percentage of the posterior variance in which to allocate new points.
    :param delta: The parameter identifying which criterion is used, *bichon* for value of 1
        and *ranjan* for value of 2.
    """

    @tf.function
    def acquisition(x: TensorType) -> TensorType:

        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )

        mean, variance = model.predict(tf.squeeze(x, -2))
        stdev = tf.sqrt(variance)
        t = (threshold - mean) / stdev
        t_plus = t + alpha
        t_minus = t - alpha
        normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))

        if delta == 1:
            G = (
                alpha * (normal.cdf(t_plus) - normal.cdf(t_minus))
                - t * (2 * normal.cdf(t) - normal.cdf(t_plus) - normal.cdf(t_minus))
                - (2 * normal.prob(t) - normal.prob(t_plus) - normal.prob(t_minus))
            )
            tf.debugging.check_numerics(G, "NaN or Inf values encountered in criterion")
            criterion = G * stdev
        elif delta == 2:
            G = (
                (alpha ** 2 - 1 - t ** 2) * (normal.cdf(t_plus) - normal.cdf(t_minus))
                - 2 * t * (normal.prob(t_plus) - normal.prob(t_minus))
                + t_plus * normal.prob(t_plus)
                - t_minus * normal.prob(t_minus)
            )
            tf.debugging.check_numerics(G, "NaN or Inf values encountered in criterion")
            criterion = G * variance

        return criterion

    return acquisition

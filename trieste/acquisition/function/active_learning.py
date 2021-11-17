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
    
    that the
    interesting points xn+1 ∈ X to evaluate f at are the points having both a high kriging
    variance and an excursion probability close to 1/2.

    Criteria proposed by Ranjan et al.
    (2008), Bichon et al. (2008)

    Aim is not to find the optimum of f, but to estimate the excursion set Γ∗ = {x ∈ X : f(x) ≥ T}, or the contour line C*:= {x ∈ X : f(x) = T}, where T is a fixed threshold.
    """

    def __init__(self, threshold: float, alpha: float = 1, delta: int = 1) -> None:
        """
        :param threshold: The (scalar) probability of feasibility threshold.
        :param alpha: ??
        :param delta: ??
        :raise ValueError (or InvalidArgumentError): If ``threshold`` is not a scalar.
        """
        tf.debugging.assert_scalar(threshold)
        tf.debugging.assert_scalar(alpha)
        tf.debugging.assert_positive(
            alpha, message="Parameter alpha must be positive."
        )
        tf.debugging.assert_scalar(delta)
        tf.debugging.Assert(delta in [1,2], [delta])

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
    """
    Calculations detailed in Bect et al. (2012) w

    :param model: The model of the objective function.
    :param threshold: ??
    :param alpha: alpha as percentage of stdev
    :param delta: ??
    """

    @tf.function
    def acquisition(x: TensorType) -> TensorType:
        
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )

        mean, variance = model.predict(tf.squeeze(x, -2))
        stdev = tf.sqrt(variance)
        t = (mean - threshold)/stdev
        t_plus = t + alpha
        t_minus = t - alpha
        normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))

        if delta == 1:
            G = alpha*(normal.cdf(t_plus) - normal.cdf(t_minus)) - \
                t*(2*normal.cdf(t) - normal.cdf(t_plus) - normal.cdf(t_minus)) - \
                (2*normal.prob(t) - normal.prob(t_plus) - normal.prob(t_minus))
            criterion = G*stdev
        elif delta == 2:
            G = (alpha**2 - 1 - t**2)*(normal.cdf(t_plus) - normal.cdf(t_minus)) - \
                2*t*(normal.prob(t_plus) - normal.prob(t_minus)) + \
                t_plus*normal.prob(t_plus) - t_minus*normal.prob(t_minus)
            criterion = G*variance
        
        return criterion

    return acquisition

# we are MAXIMISING this, so we should return the negative?
# bichon suggests a stopping criterion of 0.001
# is jitter needed anywhere, stdev close to 0?
# tf.cast(threshold, x.dtype)?

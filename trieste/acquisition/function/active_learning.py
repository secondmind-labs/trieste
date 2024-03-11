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

import math
from typing import Optional, Sequence, Union

import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...models import ProbabilisticModel
from ...models.interfaces import FastUpdateModel, SupportsPredictJoint
from ...types import TensorType
from ...utils import DEFAULTS
from ..interface import AcquisitionFunction, AcquisitionFunctionClass, SingleModelAcquisitionBuilder


class PredictiveVariance(SingleModelAcquisitionBuilder[SupportsPredictJoint]):
    """
    Builder for the determinant of the predictive covariance matrix over the batch points.
    For a batch of size 1 it is the same as maximizing the predictive variance.
    """

    def __init__(self, jitter: float = DEFAULTS.JITTER) -> None:
        """
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        """
        self._jitter = jitter

    def __repr__(self) -> str:
        """"""
        return f"PredictiveVariance(jitter={self._jitter!r})"

    def prepare_acquisition_function(
        self,
        model: SupportsPredictJoint,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: Unused.

        :return: The determinant of the predictive function.
        """
        if not isinstance(model, SupportsPredictJoint):
            raise NotImplementedError(
                f"PredictiveVariance only works with models that support "
                f"predict_joint; received {model!r}"
            )

        return predictive_variance(model, self._jitter)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: SupportsPredictJoint,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: Unused.
        """
        return function  # no need to update anything


def predictive_variance(model: SupportsPredictJoint, jitter: float) -> AcquisitionFunction:
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


class ExpectedFeasibility(SingleModelAcquisitionBuilder[ProbabilisticModel]):
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
) -> AcquisitionFunction:
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
        normal = tfp.distributions.Normal(tf.constant(0, x.dtype), tf.constant(1, x.dtype))

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
                (alpha**2 - 1 - t**2) * (normal.cdf(t_plus) - normal.cdf(t_minus))
                - 2 * t * (normal.prob(t_plus) - normal.prob(t_minus))
                + t_plus * normal.prob(t_plus)
                - t_minus * normal.prob(t_minus)
            )
            tf.debugging.check_numerics(G, "NaN or Inf values encountered in criterion")
            criterion = G * variance

        return criterion

    return acquisition


class IntegratedVarianceReduction(SingleModelAcquisitionBuilder[FastUpdateModel]):
    """
    Builder for the reduction of the integral of the predicted variance over the search
    space given a batch of query points.
    """

    def __init__(
        self,
        integration_points: TensorType,
        threshold: Optional[Union[float, Sequence[float], TensorType]] = None,
    ) -> None:
        """
        :param integration_points: set of points to integrate the prediction variance over.
        :param threshold: either None, a float or a sequence of 1 or 2 float values.
        """
        self._integration_points = integration_points
        self._threshold = threshold

    def __repr__(self) -> str:
        """"""
        return f"IntegratedVarianceReduction(threshold={self._threshold!r})"

    def prepare_acquisition_function(
        self,
        model: FastUpdateModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: Unused.

        :return: The integral of the predictive variance.
        """
        if not isinstance(model, FastUpdateModel):
            raise NotImplementedError(
                f"PredictiveVariance only works with FastUpdateModel models; received {model!r}"
            )

        return integrated_variance_reduction(model, self._integration_points, self._threshold)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: FastUpdateModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: Unused.
        """
        return function  # no need to update anything


class integrated_variance_reduction(AcquisitionFunctionClass):
    r"""
    The reduction of the (weighted) average of the predicted variance over the integration points
    (a.k.a. Integrated Means Square Error or IMSE criterion).
    See :cite:`Picheny2010` for details.

    The criterion (to maximise) writes as:

        .. math:: \int_x (v_{old}(x) - v_{new}(x)) * weights(x),

    where :math:`v_{old}(x)` is the predictive variance of the model at :math:`x`, and
    :math:`v_{new}(x)` is the updated predictive variance, given that the GP is further
    conditioned on the query points.

    Note that since :math:`v_{old}(x)` is constant w.r.t. the query points, this function
    only returns :math:`-\int_x v_{new}(x) * weights(x)`.

    If no threshold is provided, the goal is to learn a globally accurate model, and
    the predictive variance (:math:`v_{new}`) is used. Otherwise, learning is 'targeted'
    towards regions where the GP is close to particular values, and the variance is weighted
    by the posterior GP pdf evaluated at the threshold T (if a single value is given) or by the
    probability that the GP posterior belongs to the interval between the 2 thresholds T1 and T2
    (note the slightly different parametrisation compared to :cite:`Picheny2010` in that case).

    This criterion allows batch size > 1. Note that the computational cost grows cubically with
    the batch size.

    This criterion requires a method (conditional_predict_f) to compute the new predictive variance
    given that query points are added to the data.
    """

    def __init__(
        self,
        model: FastUpdateModel,
        integration_points: TensorType,
        threshold: Optional[Union[float, Sequence[float], TensorType]] = None,
    ):
        """
        :param model: The model of the objective function.
        :param integration_points: Points over which to integrate the objective prediction variance.
        :param threshold: Either None, a float or a sequence of 1 or 2 float values.
            See class docs for details.
        :raise ValueError (or InvalidArgumentError): If ``threshold`` has more than 2 values.
        """
        self._model = model

        tf.debugging.assert_equal(
            len(tf.shape(integration_points)),
            2,
            message="integration_points must be of shape [N, D]",
        )

        tf.debugging.assert_positive(
            tf.shape(integration_points)[0],
            message="integration_points should contain at least one point",
        )

        self._integration_points = integration_points

        if threshold is None:
            self._weights = tf.constant(1.0, integration_points.dtype)

        else:
            if isinstance(threshold, float):
                t_threshold = tf.constant([threshold], integration_points.dtype)
            else:
                t_threshold = tf.cast(threshold, integration_points.dtype)

                tf.debugging.assert_rank(
                    t_threshold,
                    1,
                    message=f"threshold should be a float, a sequence "
                    f"or a rank 1 tensor, received {tf.shape(t_threshold)}",
                )
                tf.debugging.assert_less_equal(
                    tf.size(t_threshold),
                    2,
                    message=f"threshold should have one or two values,"
                    f" received {tf.size(t_threshold)}",
                )
                tf.debugging.assert_greater_equal(
                    tf.size(t_threshold),
                    1,
                    message=f"threshold should have one or two values,"
                    f" received {tf.size(t_threshold)}",
                )
                if tf.size(t_threshold) > 1:
                    tf.debugging.assert_greater_equal(
                        t_threshold[1],
                        t_threshold[0],
                        message=f"threshold values should be in increasing order,"
                        f" received {t_threshold}",
                    )

            if tf.size(t_threshold) == 1:
                mean_old, var_old = self._model.predict(query_points=integration_points)
                distr = tfp.distributions.Normal(mean_old, tf.sqrt(var_old))
                self._weights = distr.prob(t_threshold[0])
            else:
                mean_old, var_old = self._model.predict(query_points=integration_points)
                distr = tfp.distributions.Normal(mean_old, tf.sqrt(var_old))
                self._weights = distr.cdf(t_threshold[1]) - distr.cdf(t_threshold[0])

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        additional_data = Dataset(x, tf.ones_like(x[..., 0:1]))

        _, variance = self._model.conditional_predict_f(
            query_points=self._integration_points, additional_data=additional_data
        )

        return -tf.reduce_mean(variance * self._weights, axis=-2)


class BayesianActiveLearningByDisagreement(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    """
    Builder for the *Bayesian Active Learning By Disagreement* acquisition function defined in
    :cite:`houlsby2011bayesian`.
    """

    def __init__(self, jitter: float = DEFAULTS.JITTER) -> None:
        """
        :param jitter: The size of the jitter to avoid numerical problem caused by the
                log operation if variance is close to zero.
        """
        self._jitter = jitter

    def __repr__(self) -> str:
        """"""
        return f"BayesianActiveLearningByDisagreement(jitter={self._jitter!r})"

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

        return bayesian_active_learning_by_disagreement(model, self._jitter)

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


class bayesian_active_learning_by_disagreement(AcquisitionFunctionClass):
    def __init__(self, model: ProbabilisticModel, jitter: float):
        r"""
        The Bayesian active learning by disagrement acquisition function computes
        the information gain of the predictive entropy :cite:`houlsby2011bayesian`.
        the acquisiton function is calculated by:

        .. math::
            \mathrm{h}\left(\Phi\left(\frac{\mu_{\boldsymbol{x}, \mathcal{D}}}
            {\sqrt{\sigma_{\boldsymbol{x}, \mathcal{D}}^{2}+1}}\right)\right)
            -\frac{C \exp \left(-\frac{\mu_{\boldsymbol{x}, \mathcal{D}}^{2}}
            {2\left(\sigma_{\boldsymbol{w}, \mathcal{D}}^{+C^{2}}\right)}\right)}
            {\sqrt{\sigma_{\boldsymbol{x}, \mathcal{D}}^{2}+C^{2}}}

        Here :math:`\mathrm{h}(p)` is defined as:

        .. math::
            \mathrm{h}(p)=-p \log p-(1-p) \log (1-p)

        This acquisition function is intended to use for Binary Gaussian Process Classification
        model with Bernoulli likelihood. It is designed for VGP but other Gaussian approximation
        of the posterior can be used. SVGP for instance, or some other model that is not currently
        supported by Trieste. Integrating over nuisance parameters is currently not
        supported (see equation 6 of the paper).

        :param model: The model of the objective function.
        :param jitter: The size of the jitter to avoid numerical problem caused by the
                log operation if variance is close to zero.
        :return: The Bayesian Active Learning By Disagreement acquisition function.
        """
        tf.debugging.assert_positive(jitter, message="Jitter must be positive.")

        self._model = model
        self._jitter = jitter

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        mean, variance = self._model.predict(tf.squeeze(x, -2))
        variance = tf.maximum(variance, self._jitter)

        normal = tfp.distributions.Normal(tf.constant(0, mean.dtype), tf.constant(1, mean.dtype))
        p = normal.cdf((mean / tf.sqrt(variance + 1)))

        C2 = (math.pi * tf.math.log(tf.constant(2, mean.dtype))) / 2
        Ef = (tf.sqrt(C2) / tf.sqrt(variance + C2)) * tf.exp(-(mean**2) / (2 * (variance + C2)))

        return -p * tf.math.log(p + self._jitter) - (1 - p) * tf.math.log(1 - p + self._jitter) - Ef

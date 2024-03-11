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
This module contains acquisition function builders, which build and define our acquisition
functions --- functions that estimate the utility of evaluating sets of candidate points.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, cast

import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import check_shapes

from ...data import Dataset
from ...models import ProbabilisticModel, ReparametrizationSampler
from ...models.interfaces import (
    HasReparamSampler,
    SupportsGetObservationNoise,
    SupportsReparamSamplerObservationNoise,
)
from ...space import SearchSpace
from ...types import Tag, TensorType
from ...utils import DEFAULTS
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    AcquisitionFunctionClass,
    ProbabilisticModelType,
    SingleModelAcquisitionBuilder,
    SingleModelVectorizedAcquisitionBuilder,
)
from .utils import MultivariateNormalCDF


class ProbabilityOfImprovement(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    """
    Builder for the probability of improvement function, where the "best" value
    is taken to be the minimum of the posterior mean at observed points.
    """

    def __repr__(self) -> str:
        """"""
        return "ProbabilityOfImprovement()"

    def prepare_acquisition_function(
        self, model: ProbabilisticModel, dataset: Optional[Dataset] = None
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :return: The probability of improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)[0]
        return probability_below_threshold(model, eta)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer.  Must be populated.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, probability_below_threshold), [tf.constant([])])
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)[0]
        function.update(eta)  # type: ignore
        return function


class ExpectedImprovement(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    """
    Builder for the expected improvement function where the "best" value is taken to be the minimum
    of the posterior mean at observed points.

    In the presence of constraints in the search_space the "best" value is computed only at the
    feasible query points. If there are no feasible points, the "best" value is instead taken to be
    the maximum of the posterior mean at all observed points.
    """

    def __init__(self, search_space: Optional[SearchSpace] = None):
        """
        :param search_space: The global search space over which the optimisation is defined. This is
            only used to determine explicit constraints.
        """
        self._search_space = search_space

    def __repr__(self) -> str:
        """"""
        return f"ExpectedImprovement({self._search_space!r})"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :return: The expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        # Check feasibility against any explicit constraints in the search space.
        if self._search_space is not None and self._search_space.has_constraints:
            is_feasible = self._search_space.is_feasible(dataset.query_points)
            if not tf.reduce_any(is_feasible):
                query_points = dataset.query_points
            else:
                query_points = tf.boolean_mask(dataset.query_points, is_feasible)
        else:
            is_feasible = tf.constant([True], dtype=bool)
            query_points = dataset.query_points

        mean, _ = model.predict(query_points)
        if not tf.reduce_any(is_feasible):
            eta = tf.reduce_max(mean, axis=0)
        else:
            eta = tf.reduce_min(mean, axis=0)

        return expected_improvement(model, eta)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer.  Must be populated.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, expected_improvement), [tf.constant([])])

        # Check feasibility against any explicit constraints in the search space.
        if self._search_space is not None and self._search_space.has_constraints:
            is_feasible = self._search_space.is_feasible(dataset.query_points)
            if not tf.reduce_any(is_feasible):
                query_points = dataset.query_points
            else:
                query_points = tf.boolean_mask(dataset.query_points, is_feasible)
        else:
            is_feasible = tf.constant([True], dtype=bool)
            query_points = dataset.query_points

        mean, _ = model.predict(query_points)
        if not tf.reduce_any(is_feasible):
            eta = tf.reduce_max(mean, axis=0)
        else:
            eta = tf.reduce_min(mean, axis=0)

        function.update(eta)  # type: ignore
        return function


class expected_improvement(AcquisitionFunctionClass):
    def __init__(self, model: ProbabilisticModel, eta: TensorType):
        r"""
        Return the Expected Improvement (EI) acquisition function for single-objective global
        optimization. Improvement is with respect to the current "best" observation ``eta``, where
        an improvement moves towards the objective function's minimum and the expectation is
        calculated with respect to the ``model`` posterior. For model posterior :math:`f`, this is

        .. math:: x \mapsto \mathbb E \left[ \max (\eta - f(x), 0) \right]

        This function was introduced by Mockus et al, 1975. See :cite:`Jones:1998` for details.

        :param model: The model of the objective function.
        :param eta: The "best" observation.
        :return: The expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """
        self._model = model
        self._eta = tf.Variable(eta)

    def update(self, eta: TensorType) -> None:
        """Update the acquisition function with a new eta value."""
        self._eta.assign(eta)

    @tf.function
    @check_shapes(
        "x: [N..., 1, D] # This acquisition function only supports batch sizes of one",
        "return: [N..., L]",
    )
    def __call__(self, x: TensorType) -> TensorType:
        mean, variance = self._model.predict(tf.squeeze(x, -2))
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        return (self._eta - mean) * normal.cdf(self._eta) + variance * normal.prob(self._eta)


class AugmentedExpectedImprovement(SingleModelAcquisitionBuilder[SupportsGetObservationNoise]):
    """
    Builder for the augmented expected improvement function for optimization single-objective
    optimization problems with high levels of observation noise.
    """

    def __repr__(self) -> str:
        """"""
        return "AugmentedExpectedImprovement()"

    def prepare_acquisition_function(
        self,
        model: SupportsGetObservationNoise,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :return: The expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """
        if not isinstance(model, SupportsGetObservationNoise):
            raise NotImplementedError(
                f"AugmentedExpectedImprovement only works with models that support "
                f"get_observation_noise; received {model!r}"
            )
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        return augmented_expected_improvement(model, eta)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: SupportsGetObservationNoise,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, augmented_expected_improvement), [tf.constant([])])
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        function.update(eta)  # type: ignore
        return function


class augmented_expected_improvement(AcquisitionFunctionClass):
    def __init__(self, model: SupportsGetObservationNoise, eta: TensorType):
        r"""
        Return the Augmented Expected Improvement (AEI) acquisition function for single-objective
        global optimization under homoscedastic observation noise.
        Improvement is with respect to the current "best" observation ``eta``, where an
        improvement moves towards the objective function's minimum and the expectation is calculated
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
            greater than one or a model without homoscedastic observation noise.
        """
        self._model = model
        self._eta = tf.Variable(eta)
        self._noise_variance = tf.Variable(model.get_observation_noise())

    def update(self, eta: TensorType) -> None:
        """Update the acquisition function with a new eta value and noise variance."""
        self._eta.assign(eta)
        self._noise_variance.assign(self._model.get_observation_noise())

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        mean, variance = self._model.predict(tf.squeeze(x, -2))
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        ei = (self._eta - mean) * normal.cdf(self._eta) + variance * normal.prob(self._eta)
        augmentation = 1 - (tf.math.sqrt(self._noise_variance)) / (
            tf.math.sqrt(self._noise_variance + variance)
        )
        return ei * augmentation


class NegativeLowerConfidenceBound(SingleModelAcquisitionBuilder[ProbabilisticModel]):
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
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: Unused.
        :return: The negative lower confidence bound function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise ValueError: If ``beta`` is negative.
        """
        lcb = lower_confidence_bound(model, self._beta)
        return tf.function(lambda at: -lcb(at))

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


class NegativePredictiveMean(NegativeLowerConfidenceBound):
    """
    Builder for the negative of the predictive mean. The predictive mean is minimised on minimising
    the objective function. The negative predictive mean is therefore maximised.
    """

    def __init__(self) -> None:
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
    :raise tf.errors.InvalidArgumentError: If ``beta`` is negative.
    """
    tf.debugging.assert_non_negative(
        beta, message="Standard deviation scaling parameter beta must not be negative"
    )

    @tf.function
    def acquisition(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        mean, variance = model.predict(tf.squeeze(x, -2))
        return mean - beta * tf.sqrt(variance)

    return acquisition


class ProbabilityOfFeasibility(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    r"""
    Uses the :func:`probability_below_threshold` function to build a
    probability of feasiblity acquisition function, defined in :cite:`gardner14` as

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

        self._threshold = threshold

    def __repr__(self) -> str:
        """"""
        return f"ProbabilityOfFeasibility({self._threshold!r})"

    @property
    def threshold(self) -> float | TensorType:
        """The probability of feasibility threshold."""
        return self._threshold

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: Unused.
        :return: The probability of feasibility function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """
        return probability_below_threshold(model, self.threshold)

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


class probability_below_threshold(AcquisitionFunctionClass):
    def __init__(self, model: ProbabilisticModel, threshold: float | TensorType):
        r"""
        The probability of being below the threshold. This brings together commonality
        between probability of improvement and probability of feasiblity.
        Probability is is caculated with respect to the `model` posterior.
        For model posterior :math:`f`, this is
        .. math:: x \mapsto \mathbb P \left (f(x) < \eta)\right]
        where :math:`\eta` is the threshold.
        :param model: The model of the objective function.
        :param threshold: The (scalar) probability of feasibility threshold.
        :return: The probability of feasibility function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
        :raise ValueError or tf.errors.InvalidArgumentError: If ``threshold`` is not a scalar.
        """
        tf.debugging.assert_scalar(threshold)
        self._model = model
        self._threshold = tf.Variable(threshold)

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        mean, var = self._model.predict(tf.squeeze(x, -2))
        distr = tfp.distributions.Normal(mean, tf.sqrt(var))
        return distr.cdf(tf.cast(self._threshold, x.dtype))

    def update(self, threshold: TensorType) -> None:
        """Update the acquisition function with a new threshold value."""
        self._threshold.assign(threshold)


class FastConstraintsFeasibility(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    """
    Builds a feasiblity acquisition function from the residuals of explicit constraints defined in
    the search space.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        smoothing_function: Optional[Callable[[TensorType], TensorType]] = None,
    ):
        """
        :param search_space: The global search space over which the feasibility of the constraints
            is defined.
        :param smoothing_function: The smoothing function used for constraints residuals. The
            default is CDF of the Normal distribution with a scale of `1e-3`.
        :raise NotImplementedError: If the `search_space` does not have constraints.
        """
        if not search_space.has_constraints:
            raise NotImplementedError(
                "FastConstraintsFeasibility requires constraints in the search space."
            )

        self._search_space = search_space
        self._smoothing_function = smoothing_function

    def __repr__(self) -> str:
        """"""
        return f"FastConstraintsFeasibility({self._search_space!r}, {self._smoothing_function!r})"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: Unused.
        :param dataset: Unused.
        :return: The function for feasibility of constraints.
        """
        return fast_constraints_feasibility(self._search_space, self._smoothing_function)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: Unused.
        :param dataset: Unused.
        :return: The function for feasibility of constraints.
        """
        return function  # No need to update anything.


def fast_constraints_feasibility(
    search_space: SearchSpace,
    smoothing_function: Optional[Callable[[TensorType], TensorType]] = None,
) -> AcquisitionFunction:
    """
    Returns a feasiblity acquisition function from the residuals of explicit constraints defined in
    the search space.

    :param search_space: The global search space over which the feasibility of the constraints
        is defined.
    :param smoothing_function: The smoothing function used for constraints residuals. The
        default is CDF of the Normal distribution with a scale of `1e-3`.
    :return: The function for feasibility of constraints.
    :raise NotImplementedError: If the `search_space` does not have constraints.
    """

    if not search_space.has_constraints:
        raise NotImplementedError(
            "fast_constraints_feasibility requires constraints in the search space."
        )

    @tf.function
    def acquisition(x: TensorType) -> TensorType:
        if smoothing_function is None:
            _smoothing_function = tfp.distributions.Normal(
                tf.constant(0.0, x.dtype), tf.constant(1e-3, x.dtype)
            ).cdf
        else:
            _smoothing_function = smoothing_function
        residuals = search_space.constraints_residuals(x)
        return tf.math.reduce_prod(_smoothing_function(residuals), axis=-1)

    return acquisition


class ExpectedConstrainedImprovement(AcquisitionFunctionBuilder[ProbabilisticModelType]):
    """
    Builder for the *expected constrained improvement* acquisition function defined in
    :cite:`gardner14`. The acquisition function computes the expected improvement from the best
    feasible point, where feasible points are those that (probably) satisfy some constraint. Where
    there are no feasible points, this builder simply builds the constraint function.
    """

    def __init__(
        self,
        objective_tag: Tag,
        constraint_builder: AcquisitionFunctionBuilder[ProbabilisticModelType],
        min_feasibility_probability: float | TensorType = 0.5,
        search_space: Optional[SearchSpace] = None,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_builder: The builder for the constraint function.
        :param min_feasibility_probability: The minimum probability of feasibility for a
            "best point" to be considered feasible.
        :param search_space: The global search space over which the optimisation is defined. This is
            only used to determine explicit constraints.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``min_feasibility_probability``
            is not a scalar in the unit interval :math:`[0, 1]`.
        """
        tf.debugging.assert_scalar(min_feasibility_probability)

        if isinstance(min_feasibility_probability, (int, float)):
            tf.debugging.assert_greater_equal(float(min_feasibility_probability), 0.0)
            tf.debugging.assert_less_equal(float(min_feasibility_probability), 1.0)
        else:
            dtype = min_feasibility_probability.dtype
            tf.debugging.assert_greater_equal(min_feasibility_probability, tf.constant(0, dtype))
            tf.debugging.assert_less_equal(min_feasibility_probability, tf.constant(1, dtype))

        self._objective_tag = objective_tag
        self._constraint_builder = constraint_builder
        self._search_space = search_space
        self._min_feasibility_probability = min_feasibility_probability
        self._constraint_fn: Optional[AcquisitionFunction] = None
        self._expected_improvement_fn: Optional[AcquisitionFunction] = None
        self._constrained_improvement_fn: Optional[AcquisitionFunction] = None

    def __repr__(self) -> str:
        """"""
        return (
            f"ExpectedConstrainedImprovement({self._objective_tag!r}, {self._constraint_builder!r},"
            f" {self._min_feasibility_probability!r}, {self._search_space!r})"
        )

    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param models: The models over each tag.
        :param datasets: The data from the observer.
        :return: The expected constrained improvement acquisition function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise KeyError: If `objective_tag` is not found in ``datasets`` and ``models``.
        :raise tf.errors.InvalidArgumentError: If the objective data is empty.
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        datasets = cast(Mapping[Tag, Dataset], datasets)

        objective_model = models[self._objective_tag]
        objective_dataset = datasets[self._objective_tag]

        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Expected improvement is defined with respect to existing points in the"
            " objective data, but the objective data is empty.",
        )

        self._constraint_fn = self._constraint_builder.prepare_acquisition_function(
            models, datasets=datasets
        )
        pof = self._constraint_fn(objective_dataset.query_points[:, None, ...])
        is_feasible = tf.squeeze(pof >= self._min_feasibility_probability, axis=-1)

        # Check feasibility against any explicit constraints in the search space.
        if self._search_space is not None and self._search_space.has_constraints:
            ss_is_feasible = self._search_space.is_feasible(objective_dataset.query_points)
            is_feasible = tf.logical_and(is_feasible, ss_is_feasible)

        if not tf.reduce_any(is_feasible):
            return self._constraint_fn

        feasible_query_points = tf.boolean_mask(objective_dataset.query_points, is_feasible)
        feasible_mean, _ = objective_model.predict(feasible_query_points)
        self._update_expected_improvement_fn(objective_model, feasible_mean)

        @tf.function
        def constrained_function(x: TensorType) -> TensorType:
            return cast(AcquisitionFunction, self._expected_improvement_fn)(x) * cast(
                AcquisitionFunction, self._constraint_fn
            )(x)

        self._constrained_improvement_fn = constrained_function
        return constrained_function

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param models: The models for each tag.
        :param datasets: The data from the observer.
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        datasets = cast(Mapping[Tag, Dataset], datasets)

        objective_model = models[self._objective_tag]
        objective_dataset = datasets[self._objective_tag]

        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Expected improvement is defined with respect to existing points in the"
            " objective data, but the objective data is empty.",
        )
        tf.debugging.Assert(self._constraint_fn is not None, [tf.constant([])])

        constraint_fn = cast(AcquisitionFunction, self._constraint_fn)
        self._constraint_builder.update_acquisition_function(
            constraint_fn, models, datasets=datasets
        )
        pof = constraint_fn(objective_dataset.query_points[:, None, ...])
        is_feasible = tf.squeeze(pof >= self._min_feasibility_probability, axis=-1)

        # Check feasibility against any explicit constraints in the search space.
        if self._search_space is not None and self._search_space.has_constraints:
            ss_is_feasible = self._search_space.is_feasible(objective_dataset.query_points)
            is_feasible = tf.logical_and(is_feasible, ss_is_feasible)

        if not tf.reduce_any(is_feasible):
            return constraint_fn

        feasible_query_points = tf.boolean_mask(objective_dataset.query_points, is_feasible)
        feasible_mean, _ = objective_model.predict(feasible_query_points)
        self._update_expected_improvement_fn(objective_model, feasible_mean)

        if self._constrained_improvement_fn is not None:
            return self._constrained_improvement_fn

        @tf.function
        def constrained_function(x: TensorType) -> TensorType:
            return cast(AcquisitionFunction, self._expected_improvement_fn)(x) * cast(
                AcquisitionFunction, self._constraint_fn
            )(x)

        self._constrained_improvement_fn = constrained_function
        return self._constrained_improvement_fn

    def _update_expected_improvement_fn(
        self, objective_model: ProbabilisticModelType, feasible_mean: TensorType
    ) -> None:
        """
        Set or update the unconstrained expected improvement function.

        :param objective_model: The objective model.
        :param feasible_mean: The mean of the feasible query points.
        """
        eta = tf.reduce_min(feasible_mean, axis=0)

        if self._expected_improvement_fn is None:
            self._expected_improvement_fn = expected_improvement(objective_model, eta)
        else:
            tf.debugging.Assert(
                isinstance(self._expected_improvement_fn, expected_improvement), [tf.constant([])]
            )
            self._expected_improvement_fn.update(eta)  # type: ignore


class MonteCarloExpectedImprovement(SingleModelAcquisitionBuilder[HasReparamSampler]):
    """
    Builder for a Monte Carlo-based expected improvement function for use with a model without
    analytical expected improvement (e.g. a deep GP). The "best" value is taken to be
    the minimum of the posterior mean at observed points. See
    :class:`monte_carlo_expected_improvement` for details.
    """

    def __init__(self, sample_size: int, *, jitter: float = DEFAULTS.JITTER):
        """
        :param sample_size: The number of samples for each batch of points.
        :param jitter: The jitter for the reparametrization sampler.
        :raise tf.errors.InvalidArgumentError: If ``sample_size`` is not positive, or ``jitter`` is
            negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        super().__init__()

        self._sample_size = sample_size
        self._jitter = jitter

    def __repr__(self) -> str:
        """"""
        return f"MonteCarloExpectedImprovement({self._sample_size!r}, jitter={self._jitter!r})"

    def prepare_acquisition_function(
        self,
        model: HasReparamSampler,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model over the specified ``dataset``. Must have output dimension [1].
        :param dataset: The data from the observer. Cannot be empty.
        :return: The estimated *expected improvement* acquisition function.
        :raise ValueError (or InvalidArgumentError): If ``dataset`` is not populated, ``model``
            does not have an output dimension of [1] or does not have a ``reparam_sample`` method.
        """
        if not isinstance(model, HasReparamSampler):
            raise ValueError(
                f"MonteCarloExpectedImprovement only supports models with a reparam_sampler method;"
                f"received {model!r}"
            )

        sampler = model.reparam_sampler(self._sample_size)

        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        samples_at_query_points = sampler.sample(
            dataset.query_points[..., None, :], jitter=self._jitter
        )
        mean = tf.reduce_mean(samples_at_query_points, axis=-3, keepdims=True)  # [N, 1, 1, L]

        tf.debugging.assert_shapes(
            [(mean, [..., 1])], message="Expected model with output dimension [1]."
        )

        eta = tf.squeeze(tf.reduce_min(mean, axis=0))

        return monte_carlo_expected_improvement(sampler, eta)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: HasReparamSampler,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model. Must have output dimension [1]. Unused here.
        :param dataset: The data from the observer. Cannot be empty
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(
            isinstance(function, monte_carlo_expected_improvement), [tf.constant([])]
        )
        sampler = function._sampler  # type: ignore
        sampler.reset_sampler()
        samples_at_query_points = sampler.sample(
            dataset.query_points[..., None, :], jitter=self._jitter
        )
        mean = tf.reduce_mean(samples_at_query_points, axis=-3, keepdims=True)

        tf.debugging.assert_shapes(
            [(mean, [..., 1])], message="Expected model with output dimension [1]."
        )

        eta = tf.squeeze(tf.reduce_min(mean, axis=0))
        function.update(eta)  # type: ignore
        return function


class monte_carlo_expected_improvement(AcquisitionFunctionClass):
    r"""
    Return a Monte Carlo based Expected Improvement (EI) acquisition function for
    single-objective global optimization. Improvement is with respect to the current "best"
    observation ``eta``, where an improvement moves towards the objective function's minimum
    and the expectation is calculated with respect to the ``model`` posterior. For model
    posterior :math:`f`, this is

    .. math:: x \mapsto \mathbb E \left[ \max (\eta - f(x), 0) \right].

    For the Monte Carlo version, the expectation is calculated by samples that we save. See
    :cite:`wilson2018maximizing` for details.
    """

    def __init__(self, sampler: ReparametrizationSampler[HasReparamSampler], eta: TensorType):
        r"""
        :param sampler: The model sampler of the objective function.
        :param eta: The "best" observation.
        :return: The Monte Carlo expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """
        self._sampler = sampler
        self._eta = tf.Variable(eta)

    def update(self, eta: TensorType) -> None:
        """Update the acquisition function with a new eta value."""
        self._eta.assign(eta)

    @tf.function
    def __call__(self, at: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(at, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        samples = tf.squeeze(self._sampler.sample(at), axis=-1)  # [..., S, 1]
        improvement = tf.maximum(self._eta - samples, 0.0)  # [..., S, 1]
        return tf.reduce_mean(improvement, axis=-2)  # [..., 1]


class MonteCarloAugmentedExpectedImprovement(
    SingleModelAcquisitionBuilder[SupportsReparamSamplerObservationNoise]
):
    """
    Builder for a Monte Carlo-based augmented expected improvement function for use with a model
    without analytical augmented expected improvement (e.g. a deep GP). The "best" value is taken to
    be the minimum of the posterior mean at observed points. See
    :class:`monte_carlo_augmented_expected_improvement` for details.
    """

    def __init__(self, sample_size: int, *, jitter: float = DEFAULTS.JITTER):
        """
        :param sample_size: The number of samples for each batch of points.
        :param jitter: The jitter for the reparametrization sampler.
        :raise tf.errors.InvalidArgumentError: If ``sample_size`` is not positive, or ``jitter`` is
            negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        super().__init__()

        self._sample_size = sample_size
        self._jitter = jitter

    def __repr__(self) -> str:
        """"""
        return (
            f"MonteCarloAugmentedExpectedImprovement({self._sample_size!r}, "
            f"jitter={self._jitter!r})"
        )

    def prepare_acquisition_function(
        self,
        model: SupportsReparamSamplerObservationNoise,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model over the specified ``dataset``. Must have output dimension [1].
        :param dataset: The data from the observer. Cannot be empty.
        :return: The estimated *expected improvement* acquisition function.
        :raise ValueError (or InvalidArgumentError): If ``dataset`` is not populated, ``model``
            does not have an output dimension of [1], does not have a ``reparam_sample`` method, or
            does not support observation noise.
        """
        if not isinstance(model, SupportsReparamSamplerObservationNoise):
            raise ValueError(
                f"MonteCarloAugmentedExpectedImprovement only supports models with a "
                f"reparam_sampler method and that support observation noise; received "
                f"{model!r}."
            )

        sampler = model.reparam_sampler(self._sample_size)

        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        samples_at_query_points = sampler.sample(
            dataset.query_points[..., None, :], jitter=self._jitter
        )
        mean = tf.reduce_mean(samples_at_query_points, axis=-3, keepdims=True)  # [N, 1, 1, L]

        tf.debugging.assert_shapes(
            [(mean, [..., 1])], message="Expected model with output dimension [1]."
        )

        eta = tf.squeeze(tf.reduce_min(mean, axis=0))

        return monte_carlo_augmented_expected_improvement(model, sampler, eta)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: SupportsReparamSamplerObservationNoise,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model. Must have output dimension [1]. Unused here
        :param dataset: The data from the observer. Cannot be empty.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(
            isinstance(function, monte_carlo_augmented_expected_improvement), [tf.constant([])]
        )
        sampler = function._sampler  # type: ignore
        sampler.reset_sampler()
        samples_at_query_points = sampler.sample(
            dataset.query_points[..., None, :], jitter=self._jitter
        )
        mean = tf.reduce_mean(samples_at_query_points, axis=-3, keepdims=True)  # [N, 1, 1, L]

        tf.debugging.assert_shapes(
            [(mean, [..., 1])], message="Expected model with output dimension [1]."
        )

        eta = tf.squeeze(tf.reduce_min(mean, axis=0))
        function.update(eta)  # type: ignore
        return function


class monte_carlo_augmented_expected_improvement(AcquisitionFunctionClass):
    r"""
    Return a Monte Carlo based Augmented Expected Improvement (AEI) acquisition function for
    single-objective global optimization with high levels of observation noise. See
    :cite:`wilson2018maximizing` for details on using the reparametrization trick for optimizing
    acquisition functions and :cite:`Huang:2006`: for details of AEI.
    """

    def __init__(
        self,
        model: SupportsReparamSamplerObservationNoise,
        sampler: ReparametrizationSampler[SupportsReparamSamplerObservationNoise],
        eta: TensorType,
    ):
        r"""
        :param model: The model of the objective function.
        :param sampler: The model sampler of the objective function.
        :param eta: The "best" observation.
        :return: The Monte Carlo expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """
        self._model = model
        self._sampler = sampler
        self._eta = tf.Variable(eta)
        self._noise_variance = tf.Variable(model.get_observation_noise())

    def update(self, eta: TensorType) -> None:
        """Update the acquisition function with a new eta and noise variance"""
        self._eta.assign(eta)
        self._noise_variance.assign(self._model.get_observation_noise())

    @tf.function
    def __call__(self, at: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(at, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        samples = tf.squeeze(self._sampler.sample(at), axis=-1)  # [..., S, 1]
        improvement = tf.maximum(self._eta - samples, 0.0)  # [..., S, 1]
        variance = tf.math.reduce_variance(samples, -2)  # [..., 1]
        augmentation = 1 - (
            tf.math.sqrt(self._noise_variance) / tf.math.sqrt(self._noise_variance + variance)
        )
        return augmentation * tf.reduce_mean(improvement, axis=-2)  # [..., 1]


class BatchMonteCarloExpectedImprovement(SingleModelAcquisitionBuilder[HasReparamSampler]):
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
        :raise tf.errors.InvalidArgumentError: If ``sample_size`` is not positive, or ``jitter``
            is negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        self._sample_size = sample_size
        self._jitter = jitter

    def __repr__(self) -> str:
        """"""
        return f"BatchMonteCarloExpectedImprovement({self._sample_size!r}, jitter={self._jitter!r})"

    def prepare_acquisition_function(
        self,
        model: HasReparamSampler,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model. Must have event shape [1].
        :param dataset: The data from the observer. Must be populated.
        :return: The batch *expected improvement* acquisition function.
        :raise ValueError (or InvalidArgumentError): If ``dataset`` is not populated, or ``model``
            does not have an event shape of [1].
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        mean, _ = model.predict(dataset.query_points)

        tf.debugging.assert_shapes(
            [(mean, ["_", 1])], message="Expected model with event shape [1]."
        )

        eta = tf.reduce_min(mean, axis=0)
        return batch_monte_carlo_expected_improvement(self._sample_size, model, eta, self._jitter)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: HasReparamSampler,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model. Must have event shape [1].
        :param dataset: The data from the observer. Must be populated.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(
            isinstance(function, batch_monte_carlo_expected_improvement), [tf.constant([])]
        )
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        function.update(eta)  # type: ignore
        return function


class batch_monte_carlo_expected_improvement(AcquisitionFunctionClass):
    def __init__(self, sample_size: int, model: HasReparamSampler, eta: TensorType, jitter: float):
        """
        :param sample_size: The number of Monte-Carlo samples.
        :param model: The model of the objective function.
        :param eta: The "best" observation.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :return: The expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """
        self._sample_size = sample_size

        if not isinstance(model, HasReparamSampler):
            raise ValueError(
                f"The batch Monte-Carlo expected improvement acquisition function only supports "
                f"models that implement a reparam_sampler method; received {model!r}"
            )

        sampler = model.reparam_sampler(self._sample_size)

        self._sampler = sampler
        self._eta = tf.Variable(eta)
        self._jitter = jitter

    def update(self, eta: TensorType) -> None:
        """Update the acquisition function with a new eta value and reset the reparam sampler."""
        self._eta.assign(eta)
        self._sampler.reset_sampler()

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        samples = tf.squeeze(self._sampler.sample(x, jitter=self._jitter), axis=-1)  # [..., S, B]
        min_sample_per_batch = tf.reduce_min(samples, axis=-1)  # [..., S]
        batch_improvement = tf.maximum(self._eta - min_sample_per_batch, 0.0)  # [..., S]
        return tf.reduce_mean(batch_improvement, axis=-1, keepdims=True)  # [..., 1]


class BatchExpectedImprovement(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    """Accurate approximation of the batch expected improvement, using the
    method of Chvallier and Ginsbourger :cite:`chevalier2013fast`.

    Internally, this uses a highly accurate approximation of the cumulative
    density function of the multivariate Gaussian, developed by Alan Genz
    :cite:`genz2016numerical`.
    """

    def __init__(
        self,
        sample_size: int,
        *,
        jitter: float = DEFAULTS.JITTER,
    ):
        """Initialise the BatchExpectedImprovement instance.

        :param sample_size: int, number of Sobol samples to use.
        :param jitter: float, amount of jitter for Cholesky factorisations.
        """

        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        self._sample_size = sample_size
        self._jitter = jitter

    def __repr__(self) -> str:
        """"""

        return f"BatchExpectedImprovement({self._sample_size!r}, jitter={self._jitter!r})"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model. Must have event shape [1].
        :param dataset: The data from the observer. Must be populated.
        :return: The batch *expected improvement* acquisition function.
        :raise ValueError (or InvalidArgumentError): If ``dataset`` is not populated, or ``model``
            does not have an event shape of [1].
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        # Get mean and covariance
        mean, _ = model.predict(dataset.query_points)

        tf.debugging.assert_shapes(
            [(mean, ["_", 1])],
            message="Expected model with event shape [1].",
        )

        eta = tf.reduce_min(mean, axis=0)

        acquisition_function = batch_expected_improvement(
            self._sample_size,
            model,
            eta,
            self._jitter,
        )

        return acquisition_function

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model. Must have event shape [1].
        :param dataset: The data from the observer. Must be populated.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, batch_expected_improvement), [])

        # Get mean and covariance
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)

        function.update(eta=eta)  # type: ignore

        return function


class batch_expected_improvement(AcquisitionFunctionClass):
    def __init__(
        self,
        sample_size: int,
        model: ProbabilisticModel,
        eta: TensorType,
        jitter: float,
    ):
        """Initialise the batch_expected_improvement instance.

        :param sample_size: int, number of samples to use.
        :param model: Gaussian process regression model.
        :param eta: Tensor of shape (,), expected improvement threshold. This
            is the best value observed so far durin the BO loop.
        :param jitter: float, amount of jitter for Cholesky factorisations.
        """

        self._sample_size = sample_size
        self._jitter = jitter
        self._eta = tf.Variable(eta)
        self._model = model

        self._mvn_cdf_1: Optional[MultivariateNormalCDF] = None
        self._mvn_cdf_2: Optional[MultivariateNormalCDF] = None

        self._num_sobol_skip = int(tf.math.floor(10**9 * tf.random.uniform((), dtype=tf.float32)))

    def update(self, eta: TensorType) -> None:
        """Update the acquisition function with a new eta value and reset the
        reparam sampler.
        """
        self._eta.assign(eta)
        self._num_sobol_skip = int(tf.math.floor(10**9 * tf.random.uniform((), dtype=tf.float32)))

    def _compute_bm(
        self,
        mean: tf.Tensor,
        threshold: tf.Tensor,
    ) -> TensorType:
        """Helper function for the batch expected improvement, which computes
        the tensors b and m as detailed in Chevalier and Ginsbourger
        :cite:`chevalier2013fast`.

        :param mean: Tensor of shape (B, Q)
        :param threshold: Tensor of shape (B,)

        :returns b: Tensor of shape (B, Q, Q)
        :returns m: Tensor of shape (B, Q, Q)
        """

        # Check shapes of input tensors
        tf.debugging.assert_shapes(
            [
                (mean, ("B", "Q")),
                (threshold, ("B",)),
            ]
        )

        # Unpack tensor shape and data type
        B, Q = mean.shape
        dtype = mean.dtype

        # Compute b tensor
        threshold = tf.tile(threshold[:, None], (1, Q))
        threshold = tf.linalg.diag(threshold)  # (B, Q, Q)

        b = tf.zeros(shape=(B, Q, Q), dtype=dtype)
        b = b - threshold

        # Compute m tensor
        m = mean[:, None, :] - mean[:, :, None]  # (B, Q, Q)
        m = m - tf.linalg.diag(mean)  # (B, Q, Q)

        return b, m

    def _delta(
        self, idx: int, dim: int, B: int, transpose: bool, dtype: tf.DType
    ) -> TensorType:  # pragma: no cover (tf.map_fn)
        """Helper function for the _compute_Sigma function, which computes a
        *delta* tensor of shape (B, idx, idx) such that

            delta[B, i, :] = 1 if i == idx
            delta[B, i, :] = 0 otherwise.

        If transpose == True, then the last two dimensions of the tensor are
        transposed, in which case

            delta[B, :, i] = 1 if i == idx
            delta[B, :, i] = 0 otherwise.

        :param idx: Index for entries equal to 1.
        :param dim: Dimension of the last and second to last axes.
        :param B: Leading dimension of tensor.
        :param transpose: Whether to transpose the last two dimensions or not.
        :param dtype: The dtype of the tensor, either tf.float32 or tf.float64.
        """

        # Check input parameters
        tf.debugging.assert_non_negative(idx)
        tf.debugging.assert_non_negative(dim)
        tf.debugging.assert_positive(B)

        o1 = tf.ones(shape=(B, idx, dim), dtype=dtype)
        z1 = tf.zeros(shape=(B, 1, dim), dtype=dtype)
        o2 = tf.ones(shape=(B, dim - idx - 1, dim), dtype=dtype)

        delta = tf.concat([o1, z1, o2], axis=1)
        delta = tf.transpose(delta, perm=[0, 2, 1]) if transpose else delta

        return delta

    def _compute_Sigma(
        self,
        covariance: tf.Tensor,
    ) -> TensorType:
        """Helper function for the batch expected improvement, which computes
        the tensor Sigma, as detailed in Chevalier and Ginsbourger
        :cite:`chevalier2013fast`.

        :param covariance: Tensor of shape (B, Q, Q)
        :returns Sigma: Tensor of shape (B, Q, Q, Q)
        """

        # Check shapes of covariance tensor
        tf.debugging.assert_shapes([(covariance, ("B", "Q", "Q"))])

        # Unpack tensor shape and dtype
        B, Q, _ = covariance.shape
        dtype = covariance.dtype

        Sigma = tf.zeros(shape=(B, Q, Q, Q))

        def compute_single_slice(q: int) -> TensorType:  # pragma: no cover (tf.map_fn)
            diq = self._delta(q, Q, B, transpose=False, dtype=dtype)
            dqj = self._delta(q, Q, B, transpose=True, dtype=dtype)

            Sigma_ij = covariance[:, :, :]
            Sigma_iq = covariance[:, :, q : q + 1]
            Sigma_qj = covariance[:, q : q + 1, :]
            Sigma_qq = covariance[:, q : q + 1, q : q + 1]

            cov = Sigma_ij * diq * dqj - Sigma_iq * diq - Sigma_qj * dqj + Sigma_qq

            return cov

        Sigma = tf.map_fn(
            compute_single_slice,
            tf.range(Q),
            fn_output_signature=dtype,
        )

        Sigma = tf.transpose(Sigma, perm=[1, 0, 2, 3])

        return Sigma

    def _compute_p(
        self,
        m_reshaped: tf.Tensor,
        b_reshaped: tf.Tensor,
        Sigma_reshaped: tf.Tensor,
        mvn_cdf: Callable[[TensorType, TensorType, TensorType, float], TensorType],
    ) -> TensorType:
        """Helper function for the batch expected improvement, which computes
        the tensor p, as detailed in Chevalier and Ginsbourger
        :cite:`chevalier2013fast`.

        :param m_reshaped: Tensor of shape (BQ, Q)
        :param b_reshaped: Tensor of shape (BQ, Q)
        :param Sigma_reshaped: Tensor of shape (BQ, Q, Q)
        :returns p: Tensor of shape (B, Q)
        """

        # Check shapes of covariance tensor
        tf.debugging.assert_shapes(
            [
                (m_reshaped, ("BQ", "Q")),
                (b_reshaped, ("BQ", "Q")),
                (Sigma_reshaped, ("BQ", "Q", "Q")),
            ]
        )

        # Unpack dtype and mean shape
        dtype = m_reshaped.dtype
        BQ, Q = m_reshaped.shape  # (B*Q, Q)

        if BQ % Q == 0:
            B = BQ // Q

        else:
            raise ValueError(
                f"Expected size of dimension 0 of m_reshaped tensor to be "
                f"divisible by size of dimension 1, instead found "
                f"{m_reshaped.shape[0]} and {m_reshaped.shape[1]}."
            )

        # Compute mean, covariance and x for p mvn normal cdf
        p_cdf_mean = tf.zeros(shape=(BQ, Q), dtype=dtype)  # (B*Q, Q)
        p_cdf_cov = Sigma_reshaped  # (B*Q, Q, Q)

        p_cdf_x = b_reshaped - m_reshaped  # (B*Q, Q)

        p = mvn_cdf(  # type: ignore
            x=p_cdf_x,
            mean=p_cdf_mean,
            cov=p_cdf_cov,
        )  # (B*Q,)

        p = tf.reshape(p, shape=(B, Q))  # (B, Q)

        return p

    def _compute_c(
        self,
        m_reshaped: tf.Tensor,
        b_reshaped: tf.Tensor,
        Sigma_reshaped: tf.Tensor,
    ) -> TensorType:
        """Helper function for the batch expected improvement, which computes
        the tensor c, which is the c^{(i)} tensor detailed in Chevalier and
        Ginsbourger :cite:`chevalier2013fast`.

        :param m_reshaped: Tensor of shape (BQ, Q)
        :param b_reshaped: Tensor of shape (BQ, Q)
        :param Sigma_reshaped: Tensor of shape (BQ, Q, Q)
        :returns c: Tensor of shape (B, Q, Q-1)
        """

        # Check shapes of covariance tensor
        tf.debugging.assert_shapes(
            [
                (m_reshaped, ("BQ", "Q")),
                (b_reshaped, ("BQ", "Q")),
                (Sigma_reshaped, ("BQ", "Q", "Q")),
            ]
        )

        # Unpack tensor shape
        BQ, Q = m_reshaped.shape

        # Compute difference between b and m tensors
        diff = b_reshaped - m_reshaped  # (B*Q, Q)

        # Compute c, including the ith entry, which we want to remove
        cov_ratio = Sigma_reshaped / tf.linalg.diag_part(Sigma_reshaped)[:, :, None]  # (B*Q, Q, Q)
        c = diff[:, None, :] - diff[:, :, None] * cov_ratio  # (B*Q, Q, Q)

        # Remove the ith entry by masking c with a boolean mask with False across
        # the diagonal and True in the off-diagonal terms
        mask = tf.math.logical_not(tf.cast(tf.eye(Q, dtype=tf.int32), dtype=tf.bool))
        mask = tf.tile(mask[None, :, :], (c.shape[0], 1, 1))

        c = tf.ragged.boolean_mask(c, mask).to_tensor()

        return c

    def _compute_R(
        self,
        Sigma_reshaped: tf.Tensor,
    ) -> TensorType:
        """Helper function for the batch expected improvement, which computes
        the tensor R, which is the Sigma^{(i)} tensor detailed in Chevalier
        and Ginsbourger :cite:`chevalier2013fast`.

        :param Sigma_reshaped: Tensor of shape (BQ, Q, Q)
        :returns R: Tensor of shape (B, Q-1, Q-1)
        """

        # Check shapes of covariance tensor
        tf.debugging.assert_shapes([(Sigma_reshaped, ("BQ", "Q", "Q"))])

        # Unpack tensor shape
        BQ, Q, _ = Sigma_reshaped.shape

        Sigma_uv = tf.tile(Sigma_reshaped[:, None, :, :], (1, Q, 1, 1))
        Sigma_iu = tf.tile(Sigma_reshaped[:, :, :, None], (1, 1, 1, Q))
        Sigma_iv = tf.tile(Sigma_reshaped[:, :, None, :], (1, 1, Q, 1))
        Sigma_ii = tf.linalg.diag_part(Sigma_reshaped)[:, :, None, None]

        R_whole = Sigma_uv - Sigma_iu * Sigma_iv / Sigma_ii

        def create_blocks(q: int) -> TensorType:  # pragma: no cover (tf.map_fn)
            block1 = tf.concat(
                [
                    R_whole[:, q, :q, :q],
                    R_whole[:, q, q + 1 :, :q],
                ],
                axis=1,
            )

            block2 = tf.concat(
                [
                    R_whole[:, q, :q, q + 1 :],
                    R_whole[:, q, q + 1 :, q + 1 :],
                ],
                axis=1,
            )

            R_block = tf.concat([block1, block2], axis=2)

            return R_block

        R = tf.map_fn(
            create_blocks,
            tf.range(Q),
            fn_output_signature=R_whole.dtype,
        )
        R = tf.transpose(R, perm=[1, 0, 2, 3])

        return R

    def _compute_Phi(
        self,
        c: tf.Tensor,
        R: tf.Tensor,
        mvn_cdf: Callable[[TensorType, TensorType, TensorType, float], TensorType],
    ) -> TensorType:
        """Helper function for the batch expected improvement, which computes
        the tensor Phi, which is the tensor of multivariate Gaussian CDFs, in
        the inner sum of the equation (3) in Chevalier and Ginsbourger
        :cite:`chevalier2013fast`.

        :param c: Tensor of shape (BQ, Q, Q-1).
        :param R: Tensor of shape (BQ, Q, Q-1, Q-1).
        :param mvn_cdf: Multivariate Gaussian CDF, made using MultivariateNormalCDF.
        :returns Phi: Tensor of multivariate Gaussian CDFs.
        """

        # Check shapes of covariance tensor
        tf.debugging.assert_shapes(
            [
                (c, ("BQ", "Q", "Q_")),
                (R, ("BQ", "Q", "Q_", "Q_")),
            ]
        )

        # Unpack tensor shape and data type
        BQ, Q, _, Q_ = R.shape
        dtype = R.dtype

        try:
            assert BQ % Q == 0

        except AssertionError:
            raise ValueError(
                f"Expected size of dimension 0 of R tensor to be "
                f"divisible by size of dimension 1, instead found "
                f"{R.shape[0]} and {R.shape[1]}."
            )

        # Compute parallelisation dimension from batch size
        B = BQ // Q

        c_reshaped = tf.reshape(c, (BQ * Q, Q - 1))
        R_reshaped = tf.reshape(R, (BQ * Q, Q - 1, Q - 1))

        # Compute mean, covariance and x for Phi mvn normal cdf
        Phi_cdf_x = c_reshaped  # (B*Q, Q-1)
        Phi_cdf_mean = tf.zeros(shape=(BQ * Q, Q - 1), dtype=dtype)  # (B*Q*Q, Q)
        Phi_cdf_cov = R_reshaped  # (B*Q*Q, Q-1, Q-1)

        # Compute multivariate cdfs
        mvn_cdfs = mvn_cdf(  # type: ignore
            x=Phi_cdf_x,
            mean=Phi_cdf_mean,
            cov=Phi_cdf_cov,
        )
        mvn_cdfs = tf.reshape(mvn_cdfs, (B, Q, Q))  # (B, Q, Q)

        return mvn_cdfs

    def _compute_batch_expected_improvement(
        self,
        mean: tf.Tensor,
        covariance: tf.Tensor,
        threshold: tf.Tensor,
        mvn_cdf_1: Callable[[TensorType, TensorType, TensorType, float], TensorType],
        mvn_cdf_2: Callable[[TensorType, TensorType, TensorType, float], TensorType],
    ) -> TensorType:
        """Accurate Monte Carlo approximation of the batch expected
        improvement, using the method of Chevalier and Ginsbourger
        :cite:`chevalier2013fast`.

        :param mean: Tensor of shape (B, Q).
        :param covariance: Tensor of shape (B, Q, Q).
        :param threshold: Tensor of shape (B, Q).
        :param mvn_cdf_1: Callable computing the multivariate CDF of a Q-dimensional Gaussian.
        :param mvn_cdf_2: Callable computing the multivariate CDF of a (Q-1)-dimensional Gaussian.
        :returns ei: Tensor of shape (B,), expected improvement.
        """

        # Check shapes of covariance tensor
        tf.debugging.assert_shapes(
            [
                (mean, ("B", "Q")),
                (covariance, ("B", "Q", "Q")),
                (threshold, ("B",)),
            ]
        )

        # Unpack and mean shape
        B, Q = mean.shape

        # Compute b and m tensors
        b, m = self._compute_bm(
            mean=mean,
            threshold=threshold,
        )  # (B, Q, Q), (B, Q, Q)

        # Compute Sigma
        Sigma = self._compute_Sigma(covariance=covariance)  # (B, Q, Q, Q)

        # Reshape all tensors, for batching
        b_reshaped = tf.reshape(b, (B * Q, Q))
        m_reshaped = tf.reshape(m, (B * Q, Q))
        Sigma_reshaped = tf.reshape(Sigma, (B * Q, Q, Q))

        # Compute p tensor
        p = self._compute_p(
            m_reshaped=m_reshaped,
            b_reshaped=b_reshaped,
            Sigma_reshaped=Sigma_reshaped,
            mvn_cdf=mvn_cdf_1,
        )

        # Compute c
        c = self._compute_c(
            m_reshaped=m_reshaped,
            b_reshaped=b_reshaped,
            Sigma_reshaped=Sigma_reshaped,
        )  # (B*Q, Q, Q-1)

        # Compute Sigma_i
        R = self._compute_R(
            Sigma_reshaped=Sigma_reshaped,
        )  # (B*Q, Q, Q-1, Q-1)

        # Compute Q-1 multivariate CDFs
        Phi_mvn_cdfs = self._compute_Phi(
            c=c,
            R=R,
            mvn_cdf=mvn_cdf_2,
        )

        # Compute univariate pdfs
        S_diag = tf.linalg.diag_part(Sigma)
        normal = tfp.distributions.Normal(loc=m, scale=S_diag**0.5)
        uvn_pdfs = tf.math.exp(normal.log_prob(b))  # (B, Q, Q)

        Sigma_diag = tf.linalg.diag_part(tf.transpose(Sigma, perm=[0, 2, 1, 3]))
        Sigma_diag = tf.transpose(Sigma_diag, perm=[0, 2, 1])

        T = tf.tile(threshold[:, None], (1, Q))

        mean_T_term = (mean - T) * p

        # Compute inner sum
        sum_term = tf.reduce_sum(
            Sigma_diag * uvn_pdfs * Phi_mvn_cdfs,
            axis=2,
        )

        # Compute outer sum
        ei = tf.reduce_sum(mean_T_term + sum_term, axis=1)

        return ei

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        """Computes the accurate approximation of the multi-point expected
        improvement.

        :param x: Tensor of shape (B, Q, D).
        :returns ei: Tensor of shape (B,), expected improvement.
        """

        if self._mvn_cdf_1 is None:
            self._mvn_cdf_1 = MultivariateNormalCDF(
                sample_size=self._sample_size,
                dim=x.shape[1],
                dtype=x.dtype,
                num_sobol_skip=self._num_sobol_skip,
            )

        if self._mvn_cdf_2 is None:
            self._mvn_cdf_2 = MultivariateNormalCDF(
                sample_size=self._sample_size,
                dim=x.shape[1] - 1,
                dtype=x.dtype,
                num_sobol_skip=self._num_sobol_skip,
            )

        mean, covariance = self._model.predict_joint(x)  # type: ignore

        mean = mean[:, :, 0]
        covariance = covariance[:, 0, :, :]
        covariance = (
            covariance
            + 1e-6
            * tf.eye(
                covariance.shape[-1],
                dtype=covariance.dtype,
            )[None, :, :]
        )

        threshold = tf.tile(self._eta, (mean.shape[0],))

        # Check shapes of x, mean, covariance and threshold tensors
        tf.debugging.assert_shapes(
            [
                (x, ("B", "Q", "D")),
                (mean, ("B", "Q")),
                (covariance, ("B", "Q", "Q")),
                (threshold, ("B",)),
            ]
        )

        ei = self._compute_batch_expected_improvement(
            mean=-mean,
            covariance=covariance,
            threshold=-threshold,
            mvn_cdf_1=self._mvn_cdf_1,
            mvn_cdf_2=self._mvn_cdf_2,
        )[:, None]

        return ei


class MultipleOptimismNegativeLowerConfidenceBound(
    SingleModelVectorizedAcquisitionBuilder[ProbabilisticModel]
):
    """
    A simple parallelization of the lower confidence bound acquisition function that produces
    a vectorized acquisition function which can efficiently optimized even for large batches.

    See :cite:`torossian2020bayesian` for details.
    """

    def __init__(self, search_space: SearchSpace):
        """
        :param search_space: The global search space over which the optimisation is defined.
        """
        self._search_space = search_space

    def __repr__(self) -> str:
        """"""
        return f"MultipleOptimismNegativeLowerConfidenceBound({self._search_space!r})"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: Unused.
        :return: The multiple optimism negative lower confidence bound function.
        """
        return multiple_optimism_lower_confidence_bound(model, self._search_space.dimension)

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
        tf.debugging.Assert(
            isinstance(function, multiple_optimism_lower_confidence_bound), [tf.constant([])]
        )
        return function  # nothing to update


class multiple_optimism_lower_confidence_bound(AcquisitionFunctionClass):
    r"""
    The multiple optimism lower confidence bound (MOLCB) acquisition function for single-objective
    global optimization.

    Each batch dimension of this acquisiton function correponds to a lower confidence bound
    acquisition function with different beta values, i.e. each point in a batch chosen by this
    acquisition function lies on a gradient of exploration/exploitation trade-offs.

    We choose the different beta values following the cdf method of :cite:`torossian2020bayesian`.
    See their paper for more details.
    """

    def __init__(self, model: ProbabilisticModel, search_space_dim: int):
        """
        :param model: The model of the objective function.
        :param search_space_dim: The dimensions of the optimisation problem's search space.
        :raise tf.errors.InvalidArgumentError: If ``search_space_dim`` is not postive.
        """

        tf.debugging.assert_positive(search_space_dim)
        self._search_space_dim = search_space_dim

        self._model = model
        self._initialized = tf.Variable(False)  # Keep track of when we need to resample
        self._betas = tf.Variable(tf.ones([0], dtype=tf.float64), shape=[None])  # [0] lazy init

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        batch_size = tf.shape(x)[-2]
        tf.debugging.assert_positive(batch_size)

        if self._initialized:  # check batch size hasnt changed during BO
            tf.debugging.assert_equal(
                batch_size,
                tf.shape(self._betas)[0],
                f"{type(self).__name__} requires a fixed batch size. Got batch size {batch_size}"
                f" but previous batch size was {tf.shape(self._betas)[0]}.",
            )

        if not self._initialized:
            normal = tfp.distributions.Normal(
                tf.constant(0.0, dtype=x.dtype), tf.constant(1.0, dtype=x.dtype)
            )
            spread = 0.5 + 0.5 * tf.range(1, batch_size + 1, dtype=x.dtype) / (
                tf.cast(batch_size, dtype=x.dtype) + 1.0
            )  # [B]
            betas = normal.quantile(spread)  # [B]
            scaled_betas = 5.0 * tf.cast(self._search_space_dim, dtype=x.dtype) * betas  # [B]
            self._betas.assign(scaled_betas)  # [B]
            self._initialized.assign(True)

        mean, variance = self._model.predict(x)  # [..., B, 1]
        mean, variance = tf.squeeze(mean, -1), tf.squeeze(variance, -1)
        return -mean + tf.sqrt(variance) * self._betas  # [..., B]


class MakePositive(SingleModelAcquisitionBuilder[ProbabilisticModelType]):
    r"""
    Converts an acquisition function builder into one that only returns positive values, via
    :math:`x \mapsto \log(1 + \exp(x))`.

    This is sometimes a useful transformation: for example, converting non-batch acquisition
    functions into batch acquisition functions with local penalization requires functions
    that only return positive values.
    """

    def __init__(
        self,
        base_acquisition_function_builder: SingleModelAcquisitionBuilder[ProbabilisticModelType],
    ) -> None:
        """
        :param base_acquisition_function_builder: Base acquisition function to be made positive.
        """
        self._base_builder = base_acquisition_function_builder

    def __repr__(self) -> str:
        """"""
        return f"MakePositive({self._base_builder})"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data to use to build the acquisition function (optional).
        :return: An acquisition function.
        """
        self._base_function = self._base_builder.prepare_acquisition_function(model, dataset)

        @tf.function
        def acquisition(x: TensorType) -> TensorType:
            return tf.math.log(1 + tf.math.exp(self._base_function(x)))

        return acquisition

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer (optional).
        :return: The updated acquisition function.
        """
        up_fn = self._base_builder.update_acquisition_function(self._base_function, model, dataset)
        if up_fn is self._base_function:
            return function
        else:
            self._base_function = up_fn

            @tf.function
            def acquisition(x: TensorType) -> TensorType:
                return tf.math.log(1 + tf.math.exp(self._base_function(x)))

            return acquisition

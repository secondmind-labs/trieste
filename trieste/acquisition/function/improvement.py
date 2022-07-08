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
This module contains improvement based acquisition function builders, which build and define our
acquisition functions --- functions that estimate the utility of evaluating sets of candidate
points.
"""

from __future__ import annotations

from typing import Mapping, Optional, cast

import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...models import ProbabilisticModel, ReparametrizationSampler
from ...models.interfaces import (
    HasReparamSampler,
    SupportsGetObservationNoise,
    SupportsReparamSamplerObservationNoise,
)
from ...types import TensorType
from ...utils import DEFAULTS
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    AcquisitionFunctionClass,
    ProbabilisticModelType,
    SingleModelAcquisitionBuilder,
)


class ExpectedImprovement(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    """
    Builder for the expected improvement function where the "best" value is taken to be the minimum
    of the posterior mean at observed points.
    """

    def __repr__(self) -> str:
        """"""
        return "ExpectedImprovement()"

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
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)
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
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, expected_improvement), [])
        mean, _ = model.predict(dataset.query_points)
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
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
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
                f"get_observation_noise; received {model.__repr__()}"
            )
        tf.debugging.Assert(dataset is not None, [])
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
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, augmented_expected_improvement), [])
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
        expected_improvement = (self._eta - mean) * normal.cdf(self._eta) + variance * normal.prob(
            self._eta
        )

        augmentation = 1 - (tf.math.sqrt(self._noise_variance)) / (
            tf.math.sqrt(self._noise_variance + variance)
        )
        return expected_improvement * augmentation


class ExpectedConstrainedImprovement(AcquisitionFunctionBuilder[ProbabilisticModelType]):
    """
    Builder for the *expected constrained improvement* acquisition function defined in
    :cite:`gardner14`. The acquisition function computes the expected improvement from the best
    feasible point, where feasible points are those that (probably) satisfy some constraint. Where
    there are no feasible points, this builder simply builds the constraint function.
    """

    def __init__(
        self,
        objective_tag: str,
        constraint_builder: AcquisitionFunctionBuilder[ProbabilisticModelType],
        min_feasibility_probability: float | TensorType = 0.5,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_builder: The builder for the constraint function.
        :param min_feasibility_probability: The minimum probability of feasibility for a
            "best point" to be considered feasible.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``min_feasibility_probability``
            is not a scalar in the unit interval :math:`[0, 1]`.
        """
        tf.debugging.assert_scalar(min_feasibility_probability)

        if isinstance(min_feasibility_probability, (int, float)):
            tf.debugging.assert_greater_equal(float(min_feasibility_probability), 0.0)
            tf.debugging.assert_less_equal(float(min_feasibility_probability), 1.0)
        else:
            dtype = min_feasibility_probability.dtype
            tf.debugging.assert_greater_equal(min_feasibility_probability, tf.cast(0, dtype))
            tf.debugging.assert_less_equal(min_feasibility_probability, tf.cast(1, dtype))

        self._objective_tag = objective_tag
        self._constraint_builder = constraint_builder
        self._min_feasibility_probability = min_feasibility_probability
        self._constraint_fn: Optional[AcquisitionFunction] = None
        self._expected_improvement_fn: Optional[AcquisitionFunction] = None
        self._constrained_improvement_fn: Optional[AcquisitionFunction] = None

    def __repr__(self) -> str:
        """"""
        return (
            f"ExpectedConstrainedImprovement({self._objective_tag!r}, {self._constraint_builder!r},"
            f" {self._min_feasibility_probability!r})"
        )

    def prepare_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
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
        tf.debugging.Assert(datasets is not None, [])
        datasets = cast(Mapping[str, Dataset], datasets)

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
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param models: The models for each tag.
        :param datasets: The data from the observer.
        """
        tf.debugging.Assert(datasets is not None, [])
        datasets = cast(Mapping[str, Dataset], datasets)

        objective_model = models[self._objective_tag]
        objective_dataset = datasets[self._objective_tag]

        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Expected improvement is defined with respect to existing points in the"
            " objective data, but the objective data is empty.",
        )
        tf.debugging.Assert(self._constraint_fn is not None, [])

        constraint_fn = cast(AcquisitionFunction, self._constraint_fn)
        self._constraint_builder.update_acquisition_function(
            constraint_fn, models, datasets=datasets
        )
        pof = constraint_fn(objective_dataset.query_points[:, None, ...])
        is_feasible = tf.squeeze(pof >= self._min_feasibility_probability, axis=-1)

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
            tf.debugging.Assert(isinstance(self._expected_improvement_fn, expected_improvement), [])
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
                f"received {model.__repr__()}"
            )

        sampler = model.reparam_sampler(self._sample_size)

        tf.debugging.Assert(dataset is not None, [])
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
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, monte_carlo_expected_improvement), [])
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
                f"{model.__repr__()}."
            )

        sampler = model.reparam_sampler(self._sample_size)

        tf.debugging.Assert(dataset is not None, [])
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
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, monte_carlo_augmented_expected_improvement), [])
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
        tf.debugging.Assert(dataset is not None, [])
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
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, batch_monte_carlo_expected_improvement), [])
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        function.update(eta)  # type: ignore
        return function


class batch_monte_carlo_expected_improvement(AcquisitionFunctionClass):
    def __init__(self, sample_size: int, model: HasReparamSampler, eta: TensorType, jitter: float):
        """
        :param sample_size: The number of Monte-Carlo samples.
        :param model: The model of the objective function.
        :param sampler:  ReparametrizationSampler.
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
                f"models that implement a reparam_sampler method; received {model.__repr__()}"
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

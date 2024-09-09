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
This module contains entropy-based acquisition function builders.
"""
from __future__ import annotations

from typing import List, Optional, TypeVar, cast, overload

import tensorflow as tf
import tensorflow_probability as tfp
from typing_extensions import Protocol, runtime_checkable

from ...data import Dataset, add_fidelity_column
from ...models import ProbabilisticModel
from ...models.gpflow.interface import SupportsCovarianceBetweenPoints
from ...models.interfaces import (
    HasTrajectorySampler,
    SupportsCovarianceWithTopFidelity,
    SupportsGetObservationNoise,
    SupportsPredictY,
)
from ...space import SearchSpace
from ...types import TensorType
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionClass,
    PenalizationFunction,
    ProbabilisticModelType,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
    UpdatablePenalizationFunction,
)
from ..sampler import ExactThompsonSampler, ThompsonSampler

CLAMP_LB = 1e-8


class MinValueEntropySearch(SingleModelAcquisitionBuilder[ProbabilisticModelType]):
    r"""
    Builder for the max-value entropy search acquisition function modified for objective
    minimisation. :class:`MinValueEntropySearch` estimates the information in the distribution
    of the objective minimum that would be gained by evaluating the objective at a given point.

    This implementation largely follows :cite:`wang2017max` and samples the objective's minimum
    :math:`y^*` across a large set of sampled locations via either a Gumbel sampler, an exact
    Thompson sampler or an approximate random Fourier feature-based Thompson sampler, with the
    Gumbel sampler being the cheapest but least accurate. Default behavior is to use the
    exact Thompson sampler.
    """

    @overload
    def __init__(
        self: "MinValueEntropySearch[ProbabilisticModel]",
        search_space: SearchSpace,
        num_samples: int = 5,
        grid_size: int = 1000,
        min_value_sampler: None = None,
    ): ...

    @overload
    def __init__(
        self: "MinValueEntropySearch[ProbabilisticModelType]",
        search_space: SearchSpace,
        num_samples: int = 5,
        grid_size: int = 1000,
        min_value_sampler: Optional[ThompsonSampler[ProbabilisticModelType]] = None,
    ): ...

    def __init__(
        self,
        search_space: SearchSpace,
        num_samples: int = 5,
        grid_size: int = 1000,
        min_value_sampler: Optional[ThompsonSampler[ProbabilisticModelType]] = None,
    ):
        """
        :param search_space: The global search space over which the optimisation is defined.
        :param num_samples: Number of samples to draw from the distribution over the minimum of the
            objective function.
        :param grid_size: Size of the grid from which to sample the min-values. We recommend
            scaling this with search space dimension.
        :param min_value_sampler: Sampler which samples minimum values.
        :raise tf.errors.InvalidArgumentError: If

            - ``num_samples`` or ``grid_size`` are negative.
        """
        tf.debugging.assert_positive(num_samples)
        tf.debugging.assert_positive(grid_size)

        if min_value_sampler is not None:
            if not min_value_sampler.sample_min_value:
                raise ValueError(
                    """
                    Minvalue Entropy Search requires a min_value_sampler that samples minimum
                    values, however the passed sampler has sample_min_value=False.
                    """
                )
        else:
            min_value_sampler = ExactThompsonSampler(sample_min_value=True)

        self._min_value_sampler = min_value_sampler
        self._search_space = search_space
        self._num_samples = num_samples
        self._grid_size = grid_size

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer.
        :return: The max-value entropy search acquisition function modified for objective
            minimisation. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        query_points = self._search_space.sample(num_samples=self._grid_size)
        tf.debugging.assert_same_float_dtype([dataset.query_points, query_points])
        query_points = tf.concat([dataset.query_points, query_points], 0)
        min_value_samples = self._min_value_sampler.sample(model, self._num_samples, query_points)

        return min_value_entropy_search(model, min_value_samples)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, min_value_entropy_search), [tf.constant([])])

        query_points = self._search_space.sample(num_samples=self._grid_size)
        tf.debugging.assert_same_float_dtype([dataset.query_points, query_points])
        query_points = tf.concat([dataset.query_points, query_points], 0)
        min_value_samples = self._min_value_sampler.sample(model, self._num_samples, query_points)
        function.update(min_value_samples)  # type: ignore
        return function


class min_value_entropy_search(AcquisitionFunctionClass):
    def __init__(self, model: ProbabilisticModel, samples: TensorType):
        r"""
        Return the max-value entropy search acquisition function (adapted from :cite:`wang2017max`),
        modified for objective minimisation. This function calculates the information gain (or
        change in entropy) in the distribution over the objective minimum :math:`y^*`, if we were
        to evaluate the objective at a given point.

        :param model: The model of the objective function.
        :param samples: Samples from the distribution over :math:`y^*`.
        :return: The max-value entropy search acquisition function modified for objective
            minimisation. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        :raise ValueError or tf.errors.InvalidArgumentError: If ``samples`` has rank less than two,
            or is empty.
        """
        tf.debugging.assert_rank(samples, 2)
        tf.debugging.assert_positive(len(samples))

        self._model = model
        self._samples = tf.Variable(samples)

    def update(self, samples: TensorType) -> None:
        """Update the acquisition function with new samples."""
        tf.debugging.assert_rank(samples, 2)
        tf.debugging.assert_positive(len(samples))
        self._samples.assign(samples)

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        fmean, fvar = self._model.predict(tf.squeeze(x, -2))
        fsd = tf.math.sqrt(fvar)
        fsd = tf.clip_by_value(
            fsd, CLAMP_LB, fmean.dtype.max
        )  # clip below to improve numerical stability

        normal = tfp.distributions.Normal(tf.constant(0, fmean.dtype), tf.constant(1, fmean.dtype))
        gamma = (tf.squeeze(self._samples) - fmean) / fsd

        log_minus_cdf = normal.log_cdf(-gamma)
        ratio = tf.math.exp(normal.log_prob(gamma) - log_minus_cdf)
        f_acqu_x = -gamma * ratio / 2 - log_minus_cdf

        return tf.math.reduce_mean(f_acqu_x, axis=1, keepdims=True)


@runtime_checkable
class SupportsCovarianceObservationNoise(
    SupportsCovarianceBetweenPoints, SupportsGetObservationNoise, Protocol
):
    """A model that supports both covariance_between_points and get_observation_noise."""


class SupportsCovarianceObservationNoiseTrajectory(
    HasTrajectorySampler, SupportsCovarianceObservationNoise, Protocol
):
    """A model that supports covariance_between_points and get_observation_noise, and also
    has an associated trajectory sampler."""


GIBBONModelType = TypeVar(
    "GIBBONModelType", bound=SupportsCovarianceObservationNoise, contravariant=True
)
""" Type variable bound to :class:`~trieste.models.SupportsCovarianceObservationNoise`. """


class GIBBON(SingleModelGreedyAcquisitionBuilder[GIBBONModelType]):
    r"""
    The General-purpose Information-Based Bayesian Optimisation (GIBBON) acquisition function
    of :cite:`Moss:2021`. :class:`GIBBON` provides a computationally cheap approximation of the
    information gained about (i.e the change in entropy of) the objective function's minimum by
    evaluating a batch of candidate points. Batches are built in a greedy manner.

    This implementation follows :cite:`Moss:2021` but is modified for function
    minimisation (rather than maximisation). We sample the objective's minimum
    :math:`y^*` across a large set of sampled locations via either a Gumbel sampler, an exact
    Thompson sampler or an approximate random Fourier feature-based Thompson sampler, with the
    Gumbel sampler being the cheapest but least accurate. Default behavior is to use the
    exact Thompson sampler.
    """

    @overload
    def __init__(
        self: "GIBBON[SupportsCovarianceObservationNoise]",
        search_space: SearchSpace,
        num_samples: int = 5,
        grid_size: int = 1000,
        min_value_sampler: None = None,
        rescaled_repulsion: bool = True,
    ): ...

    @overload
    def __init__(
        self: "GIBBON[GIBBONModelType]",
        search_space: SearchSpace,
        num_samples: int = 5,
        grid_size: int = 1000,
        min_value_sampler: Optional[ThompsonSampler[GIBBONModelType]] = None,
        rescaled_repulsion: bool = True,
    ): ...

    def __init__(
        self,
        search_space: SearchSpace,
        num_samples: int = 5,
        grid_size: int = 1000,
        min_value_sampler: Optional[ThompsonSampler[GIBBONModelType]] = None,
        rescaled_repulsion: bool = True,
    ):
        """
        :param search_space: The global search space over which the optimisation is defined.
        :param num_samples: Number of samples to draw from the distribution over the minimum of
            the objective function.
        :param grid_size: Size of the grid from which to sample the min-values. We recommend
            scaling this with search space dimension.
        :param min_value_sampler: Sampler which samples minimum values.
        :param rescaled_repulsion: If True, then downweight GIBBON's repulsion term to improve
            batch optimization performance.
        :raise tf.errors.InvalidArgumentError: If

            - ``num_samples`` is not positive, or
            - ``grid_size`` is not positive.
        """
        tf.debugging.assert_positive(num_samples)
        tf.debugging.assert_positive(grid_size)

        if min_value_sampler is not None:
            if not min_value_sampler.sample_min_value:
                raise ValueError(
                    """
                    GIBBON requires a min_value_sampler that samples minimum values,
                    however the passed sampler has sample_min_value=False.
                    """
                )
        else:
            min_value_sampler = ExactThompsonSampler(sample_min_value=True)

        self._min_value_sampler = min_value_sampler
        self._search_space = search_space
        self._num_samples = num_samples
        self._grid_size = grid_size
        self._rescaled_repulsion = rescaled_repulsion

        self._min_value_samples: Optional[TensorType] = None
        self._quality_term: Optional[gibbon_quality_term] = None
        self._diversity_term: Optional[gibbon_repulsion_term] = None
        self._gibbon_acquisition: Optional[AcquisitionFunction] = None

    def prepare_acquisition_function(
        self,
        model: GIBBONModelType,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: The points we penalize with respect to.
        :return: The GIBBON acquisition function modified for objective minimisation.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """
        if not isinstance(model, SupportsCovarianceObservationNoise):
            raise NotImplementedError(
                f"GIBBON only works with models that support "
                f"covariance_between_points and get_observation_noise; received {model!r}"
            )

        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        acq = self._update_quality_term(dataset, model)
        if pending_points is not None and len(pending_points) != 0:
            acq = self._update_repulsion_term(acq, dataset, model, pending_points)

        return acq

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: GIBBONModelType,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, or to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: The updated acquisition function.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(self._quality_term is not None, [tf.constant([])])

        if new_optimization_step:
            self._update_quality_term(dataset, model)

        if pending_points is None:
            # no repulsion term required if no pending_points.
            return cast(AcquisitionFunction, self._quality_term)

        return self._update_repulsion_term(function, dataset, model, pending_points)

    def _update_repulsion_term(
        self,
        function: Optional[AcquisitionFunction],
        dataset: Dataset,
        model: GIBBONModelType,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        tf.debugging.assert_rank(pending_points, 2)

        if self._gibbon_acquisition is not None and isinstance(
            self._diversity_term, gibbon_repulsion_term
        ):
            # if possible, just update the repulsion term
            self._diversity_term.update(pending_points)
            return self._gibbon_acquisition
        else:
            # otherwise construct a new repulsion term and acquisition function
            self._diversity_term = gibbon_repulsion_term(
                model, pending_points, rescaled_repulsion=self._rescaled_repulsion
            )
            self._gibbon_acquisition = GibbonAcquisition(
                cast(AcquisitionFunction, self._quality_term), self._diversity_term
            )
            return self._gibbon_acquisition

    def _update_quality_term(self, dataset: Dataset, model: GIBBONModelType) -> AcquisitionFunction:
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        query_points = self._search_space.sample(num_samples=self._grid_size)
        tf.debugging.assert_same_float_dtype([dataset.query_points, query_points])
        query_points = tf.concat([dataset.query_points, query_points], 0)
        self._min_value_samples = self._min_value_sampler.sample(
            model, self._num_samples, query_points
        )

        if self._quality_term is not None:  # if possible, just update the quality term
            self._quality_term.update(self._min_value_samples)
        else:  # otherwise build quality term
            self._quality_term = gibbon_quality_term(model, self._min_value_samples)
        return cast(AcquisitionFunction, self._quality_term)


class GibbonAcquisition:
    """Class representing a GIBBON acquisition function."""

    # (note that this needs to be defined as a top level class make it pickleable)
    def __init__(self, quality_term: AcquisitionFunction, diversity_term: PenalizationFunction):
        """
        :param quality_term: Quality term.
        :param diversity_term: Diversity term.
        """
        self._quality_term = quality_term
        self._diversity_term = diversity_term

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        return self._diversity_term(x) + self._quality_term(x)


class gibbon_quality_term(AcquisitionFunctionClass):
    def __init__(self, model: SupportsCovarianceObservationNoise, samples: TensorType):
        """
        GIBBON's quality term measures the amount of information that each individual
        batch element provides about the objective function's minimal value :math:`y^*` (ensuring
        that evaluations are targeted in promising areas of the space).

        :param model: The model of the objective function. GIBBON requires a model with
            a :method:covariance_between_points method and so GIBBON only
            supports :class:`GaussianProcessRegression` models.
        :param samples: Samples from the distribution over :math:`y^*`.
        :return: GIBBON's quality term. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        :raise ValueError or tf.errors.InvalidArgumentError: If ``samples`` does not have rank two,
            or is empty, or if ``model`` has no homoscedastic observation noise.
        :raise AttributeError: If ``model`` doesn't implement covariance_between_points method.
        """
        tf.debugging.assert_rank(samples, 2)
        tf.debugging.assert_positive(len(samples))

        try:
            model.get_observation_noise()
        except NotImplementedError:
            raise ValueError(
                """
                GIBBON only currently supports homoscedastic gpflow models
                with a likelihood.variance attribute.
                """
            )

        self._model = model
        self._samples = tf.Variable(samples)

    def update(self, samples: TensorType) -> None:
        """Update the acquisition function with new samples."""
        tf.debugging.assert_rank(samples, 2)
        tf.debugging.assert_positive(len(samples))
        self._samples.assign(samples)

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:  # [N, D] -> [N, 1]
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )

        fmean, fvar = self._model.predict(tf.squeeze(x, -2))
        noise_variance = self._model.get_observation_noise()
        yvar = fvar + tf.cast(noise_variance, fmean.dtype)  # predictive variance of observations

        rho_squared = fvar / yvar  # squared correlation between observations and latent function
        fsd = tf.clip_by_value(
            tf.math.sqrt(fvar), CLAMP_LB, fmean.dtype.max
        )  # clip below to improve numerical stability
        gamma = (tf.squeeze(self._samples) - fmean) / fsd

        normal = tfp.distributions.Normal(tf.constant(0, fmean.dtype), tf.constant(1, fmean.dtype))
        log_minus_cdf = normal.log_cdf(-gamma)
        ratio = tf.math.exp(normal.log_prob(gamma) - log_minus_cdf)
        inner_log = 1 + rho_squared * ratio * (gamma - ratio)

        return -0.5 * tf.math.reduce_mean(tf.math.log(inner_log), axis=1, keepdims=True)  # [N, 1]


class gibbon_repulsion_term(UpdatablePenalizationFunction):
    def __init__(
        self,
        model: SupportsCovarianceObservationNoise,
        pending_points: TensorType,
        rescaled_repulsion: bool = True,
    ):
        r"""
        GIBBON's repulsion term encourages diversity within the batch
        (achieving high values for points with low predictive correlation).

        The repulsion term :math:`r=\log |C|` is given by the log determinant of the predictive
        correlation matrix :math:`C` between the `m` pending points and the current candidate.
        The predictive covariance :math:`V` can be expressed as :math:V = [[v, A], [A, B]]` for a
        tensor :math:`B` with shape [`m`,`m`] and so we can efficiently calculate :math:`|V|` using
        the formula for the determinant of block matrices, i.e
        :math:`|V| = (v - A^T * B^{-1} * A) * |B|`.
        Note that when using GIBBON for purely sequential optimization, the repulsion term is
        not required.

        As GIBBON's batches are built in a greedy manner, i.e sequentially adding points to build a
        set of `m` pending points, we need only ever calculate the entropy reduction provided by
        adding the current candidate point to the current pending points, not the full information
        gain provided by evaluating all the pending points. This allows for a modest computational
        saving.

        When performing batch BO, GIBBON's approximation can sometimes become
        less accurate as its repulsion term dominates. Therefore, we follow the
        arguments of :cite:`Moss:2021` and divide GIBBON's repulsion term by :math:`B^{2}`. This
        behavior can be deactivated by setting `rescaled_repulsion` to False.

        :param model: The model of the objective function. GIBBON requires a model with
            a :method:covariance_between_points method and so GIBBON only
            supports :class:`GaussianProcessRegression` models.
        :param pending_points: The points already chosen in the current batch.
        :param rescaled_repulsion: If True, then downweight GIBBON's repulsion term to improve
            batch optimization performance.
        :return: GIBBON's repulsion term. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        :raise ValueError or tf.errors.InvalidArgumentError: If ``pending_points`` does not have
            rank two, or is empty, or if ``model`` has no homoscedastic observation noise.
        :raise AttributeError: If ``model`` doesn't implement covariance_between_points method.
        """
        tf.debugging.assert_rank(pending_points, 2)
        tf.debugging.assert_positive(len(pending_points))

        try:
            model.get_observation_noise()
        except NotImplementedError:
            raise ValueError(
                """
                GIBBON only currently supports homoscedastic gpflow models
                with a likelihood.variance attribute.
                """
            )

        if not hasattr(model, "covariance_between_points"):
            raise AttributeError(
                """
                GIBBON only supports models with a covariance_between_points method.
                """
            )

        self._model = model
        self._pending_points = tf.Variable(pending_points, shape=[None, *pending_points.shape[1:]])
        self._rescaled_repulsion = rescaled_repulsion

    def update(
        self,
        pending_points: TensorType,
        lipschitz_constant: TensorType = None,
        eta: TensorType = None,
    ) -> None:
        """Update the repulsion term with new variable values."""
        self._pending_points.assign(pending_points)

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This penalization function cannot be calculated for batches of points.",
        )

        fmean, fvar = self._model.predict(tf.squeeze(x, -2))
        noise_variance = self._model.get_observation_noise()
        yvar = fvar + noise_variance  # need predictive variance of observations

        _, B = self._model.predict_joint(self._pending_points)  # [1, m, m]

        B_shape = tf.shape(B)
        noise = noise_variance * tf.eye(
            B_shape[-2], batch_shape=B_shape[:-2], dtype=B.dtype
        )  # need predictive variance of observations

        L = tf.linalg.cholesky(B + noise)

        A = tf.squeeze(
            tf.expand_dims(
                self._model.covariance_between_points(tf.squeeze(x, -2), self._pending_points),
                axis=-1,
            ),
            axis=0,
        )  # [N, m, 1]
        L_inv_A = tf.linalg.triangular_solve(L, A)
        V_det = yvar - tf.squeeze(
            tf.matmul(L_inv_A, L_inv_A, transpose_a=True), -1
        )  # equation for determinant of block matrices
        repulsion = 0.5 * (tf.math.log(V_det) - tf.math.log(yvar))

        if self._rescaled_repulsion:
            batch_size = tf.cast(tf.shape(self._pending_points)[0], dtype=fmean.dtype)
            repulsion_weight = (1 / batch_size) ** (2)
        else:
            repulsion_weight = 1.0

        return repulsion_weight * repulsion


@runtime_checkable
class SupportsCovarianceWithTopFidelityPredictY(
    SupportsCovarianceWithTopFidelity, SupportsPredictY, Protocol
):
    """A model that is both multifidelity and supports predict_y."""


MUMBOModelType = TypeVar(
    "MUMBOModelType", bound=SupportsCovarianceWithTopFidelityPredictY, contravariant=True
)
""" Type variable bound to :class:`~trieste.models.SupportsCovarianceWithTopFidelityPredictY`. """


class MUMBO(MinValueEntropySearch[MUMBOModelType]):
    r"""
    Builder for the MUlti-task Max-value Bayesian Optimization MUMBO acquisition function modified
    for objective minimisation. :class:`MinValueEntropySearch` estimates the information in the
    distribution of the objective minimum that would be gained by evaluating the objective at a
    given point on a given fidelity level.

    This implementation largely follows :cite:`moss2021mumbo` and samples the objective's minimum
    :math:`y^*` across a large set of sampled locations via either a Gumbel sampler, an exact
    Thompson sampler or an approximate random Fourier feature-based Thompson sampler, with the
    Gumbel sampler being the cheapest but least accurate. Default behavior is to use the
    exact Thompson sampler.
    """

    @overload
    def __init__(
        self: "MUMBO[SupportsCovarianceWithTopFidelityPredictY]",
        search_space: SearchSpace,
        num_samples: int = 5,
        grid_size: int = 1000,
        min_value_sampler: None = None,
    ): ...

    @overload
    def __init__(
        self: "MUMBO[MUMBOModelType]",
        search_space: SearchSpace,
        num_samples: int = 5,
        grid_size: int = 1000,
        min_value_sampler: Optional[ThompsonSampler[MUMBOModelType]] = None,
    ): ...

    def __init__(
        self,
        search_space: SearchSpace,
        num_samples: int = 5,
        grid_size: int = 1000,
        min_value_sampler: Optional[ThompsonSampler[MUMBOModelType]] = None,
    ):
        super().__init__(search_space, num_samples, grid_size, min_value_sampler)

    def prepare_acquisition_function(
        self,
        model: MUMBOModelType,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The multifidelity model.
        :param dataset: The data from the observer.
        :return: The max-value entropy search acquisition function modified for objective
            minimisation. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        min_value_samples = self.get_min_value_samples_on_top_fidelity(model, dataset)
        return mumbo(model, min_value_samples)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: MUMBOModelType,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        min_value_samples = self.get_min_value_samples_on_top_fidelity(model, dataset)
        function.update(min_value_samples)  # type: ignore
        return function

    def get_min_value_samples_on_top_fidelity(
        self, model: MUMBOModelType, dataset: Dataset
    ) -> TensorType:
        """
        :param model: The model.
        :param dataset: The data from the observer.
        """
        query_points = self._search_space.sample(num_samples=self._grid_size)
        tf.debugging.assert_same_float_dtype([dataset.query_points, query_points])
        query_points = tf.concat([dataset.query_points, query_points], 0)
        query_points_on_top_fidelity = add_fidelity_column(
            query_points[:, :-1], model.num_fidelities - 1
        )
        return self._min_value_sampler.sample(
            model, self._num_samples, query_points_on_top_fidelity
        )


class mumbo(AcquisitionFunctionClass):
    def __init__(self, model: MUMBOModelType, samples: TensorType):
        r"""
        The MUMBO acquisition function of :cite:`moss2021mumbo`, modified for objective
        minimisation. This function calculates the information gain (or change in entropy) in the
        distribution over the objective minimum :math:`y^*`, if we were to evaluate the objective
        at a given point on a given fidelity level.

        To speed up calculations, we use a trick from :cite:`Moss:2021` and use moment-matching to
        calculate MUMBO's entropy terms rather than numerical integration.

        :param model: The model of the objective function.
        :param samples: Samples from the distribution over :math:`y^*`.
        :return: The MUMBO acquisition function modified for objective
            minimisation. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        :raise ValueError or tf.errors.InvalidArgumentError: If ``samples`` has rank less than two,
            or is empty.
        """
        tf.debugging.assert_rank(samples, 2)
        tf.debugging.assert_positive(len(samples))

        self._model = model
        self._samples = tf.Variable(samples)

    def update(self, samples: TensorType) -> None:
        """Update the acquisition function with new samples."""
        tf.debugging.assert_rank(samples, 2)
        tf.debugging.assert_positive(len(samples))
        self._samples.assign(samples)

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )

        x_squeezed = tf.squeeze(x, -2)
        x_on_top_fidelity = add_fidelity_column(x_squeezed[:, :-1], self._model.num_fidelities - 1)

        fmean, fvar = self._model.predict(x_on_top_fidelity)
        fsd = tf.clip_by_value(
            tf.math.sqrt(fvar), CLAMP_LB, fmean.dtype.max
        )  # clip below to improve numerical stability
        ymean, yvar = self._model.predict_y(x_squeezed)
        cov = self._model.covariance_with_top_fidelity(x_squeezed)

        # calculate squared correlation between observations and high-fidelity latent function
        rho_squared = (cov**2) / (fvar * yvar)
        rho_squared = tf.clip_by_value(rho_squared, 0.0, 1.0)

        normal = tfp.distributions.Normal(tf.constant(0, fmean.dtype), tf.constant(1, fmean.dtype))
        gamma = (tf.squeeze(self._samples) - fmean) / fsd
        log_minus_cdf = normal.log_cdf(-gamma)
        ratio = tf.math.exp(normal.log_prob(gamma) - log_minus_cdf)

        inner_log = 1 + rho_squared * ratio * (gamma - ratio)

        return -0.5 * tf.math.reduce_mean(tf.math.log(inner_log), axis=1, keepdims=True)  # [N, 1]


class CostWeighting(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    def __init__(self, fidelity_costs: List[float]):
        """
        Builder for a cost-weighted acquisition function which returns the reciprocal of the cost
        associated with the fidelity of each input.

        Note that the fidelity level is assumed to be contained in the inputs final dimension.

        The primary use of this acquisition function is to be used as a product with
        multi-fidelity acquisition functions.
        """

        self._fidelity_costs = fidelity_costs
        self._num_fidelities = len(self._fidelity_costs)

    def prepare_acquisition_function(
        self, model: ProbabilisticModel, dataset: Optional[Dataset] = None
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Not actually used here.
        :return: The reciprocal of the costs corresponding to the fidelity level of each input.
        """

        @tf.function
        def acquisition(x: TensorType) -> TensorType:
            tf.debugging.assert_shapes(
                [(x, [..., 1, None])],
                message="This acquisition function only supports batch sizes of one.",
            )
            fidelities = x[..., -1]  # [..., 1]
            tf.debugging.assert_greater(
                tf.cast(self._num_fidelities, fidelities.dtype),
                tf.reduce_max(fidelities),
                message="You are trying to use more fidelity levels than cost levels.",
            )

            costs = tf.gather(self._fidelity_costs, tf.cast(fidelities, tf.int32))

            return tf.cast(1.0 / costs, x.dtype)  # [N, 1]

        return acquisition

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        Nothing to do here, so just return previous cost function.

        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer.
        """
        return function

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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

import gpflow
import tensorflow as tf
from typing_extensions import Protocol, runtime_checkable

from ..data import Dataset
from ..types import TensorType
from ..utils import DEFAULTS

ProbabilisticModelType = TypeVar(
    "ProbabilisticModelType", bound="ProbabilisticModel", contravariant=True
)
""" Contravariant type variable bound to :class:`~trieste.models.ProbabilisticModel`.
This is used to specify classes such as samplers and acquisition function builders that
take models as input parameters and might ony support models with certain features. """


@runtime_checkable
class ProbabilisticModel(Protocol):
    """A probabilistic model.

    NOTE: This and its subclasses are defined as Protocols rather than ABCs in order to allow
    acquisition functions to depend on the intersection of different model types. As a result, it
    is also possible to pass models to acquisition functions that don't explicitly inherit from
    this class, as long as they implement all the necessary methods. This may change in future if
    https://github.com/python/typing/issues/213 is implemented.
    """

    @abstractmethod
    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Return the mean and variance of the independent marginal distributions at each point in
        ``query_points``.

        This is essentially a convenience method for :meth:`predict_joint`, where non-event
        dimensions of ``query_points`` are all interpreted as broadcasting dimensions instead of
        batch dimensions, and the covariance is squeezed to remove redundant nesting.

        :param query_points: The points at which to make predictions, of shape [..., D].
        :return: The mean and variance of the independent marginal distributions at each point in
            ``query_points``. For a predictive distribution with event shape E, the mean and
            variance will both have shape [...] + E.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples from the independent marginal distributions at
        ``query_points``.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples. For a predictive distribution with event shape E, this has shape
            [..., S, N] + E, where S is the number of samples.
        """
        raise NotImplementedError

    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Return the mean and variance of the independent marginal distributions at each point in
        ``query_points`` for the observations, including noise contributions.

        Note that this is not supported by all models.

        :param query_points: The points at which to make predictions, of shape [..., D].
        :return: The mean and variance of the independent marginal distributions at each point in
            ``query_points``. For a predictive distribution with event shape E, the mean and
            variance will both have shape [...] + E.
        """
        raise NotImplementedError(
            f"Model {self!r} does not support predicting observations, just the latent function"
        )

    def log(self) -> None:
        """
        Log model-specific information at a given optimization step.
        """
        pass


@runtime_checkable
class TrainableProbabilisticModel(ProbabilisticModel, Protocol):
    """A trainable probabilistic model."""

    @abstractmethod
    def update(self, dataset: Dataset) -> None:
        """
        Update the model given the specified ``dataset``. Does not train the model.

        :param dataset: The data with which to update the model.
        """
        raise NotImplementedError

    @abstractmethod
    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the model objective with respect to (hyper)parameters given the specified
        ``dataset``.

        :param dataset: The data with which to train the model.
        """
        raise NotImplementedError


@runtime_checkable
class SupportsPredictJoint(ProbabilisticModel, Protocol):
    """A probabilistic model that supports predict_joint."""

    @abstractmethod
    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        :param query_points: The points at which to make predictions, of shape [..., B, D].
        :return: The mean and covariance of the joint marginal distribution at each batch of points
            in ``query_points``. For a predictive distribution with event shape E, the mean will
            have shape [..., B] + E, and the covariance shape [...] + E + [B, B].
        """
        raise NotImplementedError


@runtime_checkable
class SupportsGetKernel(ProbabilisticModel, Protocol):
    """A probabilistic model that supports get_kernel."""

    @abstractmethod
    def get_kernel(self) -> gpflow.kernels.Kernel:
        """
        Return the kernel of the model.
        :return: The kernel.
        """
        raise NotImplementedError


@runtime_checkable
class SupportsGetObservationNoise(ProbabilisticModel, Protocol):
    """A probabilistic model that supports get_observation_noise."""

    @abstractmethod
    def get_observation_noise(self) -> TensorType:
        """
        Return the variance of observation noise.

        :return: The observation noise.
        """
        raise NotImplementedError


@runtime_checkable
class SupportsInternalData(ProbabilisticModel, Protocol):
    """A probabilistic model that stores and has access to its own training data."""

    @abstractmethod
    def get_internal_data(self) -> Dataset:
        """
        Return the model's training data.

        :return: The model's training data.
        """
        raise NotImplementedError


@runtime_checkable
class FastUpdateModel(ProbabilisticModel, Protocol):
    """A model with the ability to predict based on (possibly fantasized) supplementary data."""

    @abstractmethod
    def conditional_predict_f(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        """
        Return the mean and variance of the independent marginal distributions at each point in
        ``query_points``, given an additional batch of (possibly fantasized) data.

        :param query_points: The points at which to make predictions, of shape [M, D].
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :return: The mean and variance of the independent marginal distributions at each point in
            ``query_points``, with shape [..., L, M, M].
        """
        raise NotImplementedError

    @abstractmethod
    def conditional_predict_joint(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        """
        :param query_points: The points at which to make predictions, of shape [M, D].
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :return: The mean and covariance of the joint marginal distribution at each batch of points
            in ``query_points``, with shape [..., L, M, M].
        """
        raise NotImplementedError

    @abstractmethod
    def conditional_predict_f_sample(
        self, query_points: TensorType, additional_data: Dataset, num_samples: int
    ) -> TensorType:
        """
        Return ``num_samples`` samples from the independent marginal distributions at
        ``query_points``, given an additional batch of (possibly fantasized) data.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :param num_samples: The number of samples at each point.
        :return: The samples. For a predictive distribution with event shape E, this has shape
            [..., S, N] + E, where S is the number of samples.
        """
        raise NotImplementedError

    def conditional_predict_y(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        """
        Return the mean and variance of the independent marginal distributions at each point in
        ``query_points`` for the observations, including noise contributions, given an additional
        batch of (possibly fantasized) data.

        Note that this is not supported by all models.

        :param query_points: The points at which to make predictions, of shape [M, D].
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :return: The mean and variance of the independent marginal distributions at each point in
            ``query_points``.
        """
        raise NotImplementedError(
            f"Model {self!r} does not support predicting observations, just the latent function"
        )


@runtime_checkable
class EnsembleModel(ProbabilisticModel, Protocol):
    """
    This is an interface for ensemble types of models. These models can act as probabilistic models
    by deriving estimates of epistemic uncertainty from the diversity of predictions made by
    individual models in the ensemble.
    """

    @abstractmethod
    def ensemble_size(self) -> int:
        """
        Returns the size of the ensemble, that is, the number of base learners or individual
        models in the ensemble.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_index(self, size: int) -> TensorType:
        """
        Returns indices of individual models in the ensemble sampled randomly with replacement.

        :param size: The number of samples to take.
        :return: A tensor with indices
        """
        raise NotImplementedError

    @abstractmethod
    def predict_ensemble(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Returns mean and variance at ``query_points`` for each member of the ensemble. First tensor
        is the mean and second is the variance, where each has shape [..., M, N, 1], where M is
        the ``ensemble_size``.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points`` for each member of the ensemble.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_ensemble(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples at ``query_points`` where each sample is taken from a
        distribution given by a randomly chosen model in the ensemble.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples. For a predictive distribution with event shape E, this has shape
            [..., S, N] + E, where S is the number of samples.
        """
        raise NotImplementedError


@runtime_checkable
class HasTrajectorySampler(ProbabilisticModel, Protocol):
    """A probabilistic model that has an associated trajectory sampler."""

    def trajectory_sampler(
        self: ProbabilisticModelType,
    ) -> TrajectorySampler[ProbabilisticModelType]:
        """
        Return a trajectory sampler that supports this model.

        :return: The trajectory sampler.
        """
        raise NotImplementedError


@runtime_checkable
class HasReparamSampler(ProbabilisticModel, Protocol):
    """A probabilistic model that has an associated reparametrization sampler."""

    def reparam_sampler(
        self: ProbabilisticModelType, num_samples: int
    ) -> ReparametrizationSampler[ProbabilisticModelType]:
        """
        Return a reparametrization sampler providing `num_samples` samples.

        :param num_samples: The desired number of samples.
        :return: The reparametrization sampler.
        """
        raise NotImplementedError


class ModelStack(ProbabilisticModel, Generic[ProbabilisticModelType]):
    r"""
    A :class:`ModelStack` is a wrapper around a number of :class:`ProbabilisticModel`\ s of type
    :class:`ProbabilisticModelType`. It combines the outputs of each model for predictions and
    sampling.

    **Note:** Only supports vector outputs (i.e. with event shape [E]). Outputs for any two models
    are assumed independent. Each model may itself be single- or multi-output, and any one
    multi-output model may have dependence between its outputs. When we speak of *event size* in
    this class, we mean the output dimension for a given :class:`ProbabilisticModel`,
    whether that is the :class:`ModelStack` itself, or one of the subsidiary
    :class:`ProbabilisticModel`\ s within the :class:`ModelStack`. Of course, the event
    size for a :class:`ModelStack` will be the sum of the event sizes of each subsidiary model.
    """

    def __init__(
        self,
        model_with_event_size: tuple[ProbabilisticModelType, int],
        *models_with_event_sizes: tuple[ProbabilisticModelType, int],
    ):
        r"""
        The order of individual models specified at :meth:`__init__` determines the order of the
        :class:`ModelStack` output dimensions.

        :param model_with_event_size: The first model, and the size of its output events.
            **Note:** This is a separate parameter to ``models_with_event_sizes`` simply so that the
            method signature requires at least one model. It is not treated specially.
        :param \*models_with_event_sizes: The other models, and sizes of their output events.
        """
        self._models, self._event_sizes = zip(*(model_with_event_size,) + models_with_event_sizes)

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        :param query_points: The points at which to make predictions, of shape [..., D].
        :return: The predictions from all the wrapped models, concatenated along the event axis in
            the same order as they appear in :meth:`__init__`. If the wrapped models have predictive
            distributions with event shapes [:math:`E_i`], the mean and variance will both have
            shape [..., :math:`\sum_i E_i`].
        """
        means, vars_ = zip(*[model.predict(query_points) for model in self._models])
        return tf.concat(means, axis=-1), tf.concat(vars_, axis=-1)

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        r"""
        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples from all the wrapped models, concatenated along the event axis. For
            wrapped models with predictive distributions with event shapes [:math:`E_i`], this has
            shape [..., S, N, :math:`\sum_i E_i`], where S is the number of samples.
        """
        samples = [model.sample(query_points, num_samples) for model in self._models]
        return tf.concat(samples, axis=-1)

    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        :param query_points: The points at which to make predictions, of shape [..., D].
        :return: The predictions from all the wrapped models, concatenated along the event axis in
            the same order as they appear in :meth:`__init__`. If the wrapped models have predictive
            distributions with event shapes [:math:`E_i`], the mean and variance will both have
            shape [..., :math:`\sum_i E_i`].
        :raise NotImplementedError: If any of the models don't implement predict_y.
        """
        means, vars_ = zip(*[model.predict_y(query_points) for model in self._models])
        return tf.concat(means, axis=-1), tf.concat(vars_, axis=-1)

    def log(self) -> None:
        """
        Log model-specific information at a given optimization step.
        """
        for i, model in enumerate(self._models):
            with tf.name_scope(f"{i}"):
                model.log()


class TrainableModelStack(ModelStack[TrainableProbabilisticModel], TrainableProbabilisticModel):
    r"""
    A :class:`TrainableModelStack` is a wrapper around a number of
    :class:`TrainableProbabilisticModel`\ s.
    It delegates training data to each model for updates and optimization.

    :class:`TrainableProbabilisticModel`\ s within the :class:`TrainableModelStack`.
    Of course, the event size for a :class:`TrainableModelStack` will be the sum of the
    event sizes of each subsidiary model.
    """

    def update(self, dataset: Dataset) -> None:
        """
        Update all the wrapped models on their corresponding data. The data for each model is
        extracted by splitting the observations in ``dataset`` along the event axis according to the
        event sizes specified at :meth:`__init__`.

        :param dataset: The query points and observations for *all* the wrapped models.
        """
        observations = tf.split(dataset.observations, self._event_sizes, axis=-1)

        for model, obs in zip(self._models, observations):
            model.update(Dataset(dataset.query_points, obs))

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize all the wrapped models on their corresponding data. The data for each model is
        extracted by splitting the observations in ``dataset`` along the event axis according to the
        event sizes specified at :meth:`__init__`.

        :param dataset: The query points and observations for *all* the wrapped models.
        """
        observations = tf.split(dataset.observations, self._event_sizes, axis=-1)

        for model, obs in zip(self._models, observations):
            model.optimize(Dataset(dataset.query_points, obs))


class HasReparamSamplerModelStack(ModelStack[HasReparamSampler], HasReparamSampler):
    r"""
    A :class:`PredictJointModelStack` is a wrapper around a number of
    :class:`HasReparamSampler`\ s.
    It provides a  :meth:`reparam_sampler` method only if all the submodel samplers
    are the same.
    """

    def reparam_sampler(self, num_samples: int) -> ReparametrizationSampler[HasReparamSampler]:
        """
        Return a reparameterization sampler providing `num_samples` samples across
        all the models in the model stack. This is currently only implemented for
        stacks made from models that have a :class:`BatchReparametrizationSampler`
        as their reparameterization sampler.

        :param num_samples: The desired number of samples.
        :return: The reparametrization sampler.
        :raise NotImplementedError: If the models in the stack do not share the
            same :meth:`reparam_sampler`.
        """

        samplers = [model.reparam_sampler(num_samples) for model in self._models]
        unique_sampler_types = set(type(sampler) for sampler in samplers)
        if len(unique_sampler_types) == 1:
            # currently assume that all sampler constructors look the same
            shared_sampler_type = type(samplers[0])
            return shared_sampler_type(num_samples, self)
        else:
            raise NotImplementedError(
                f"""
                Reparameterization sampling is only currently supported for model
                stacks built from models that use the same reparameterization sampler,
                however, received samplers of types {unique_sampler_types}.
                """
            )


class PredictJointModelStack(ModelStack[SupportsPredictJoint], SupportsPredictJoint):
    r"""
    A :class:`PredictJointModelStack` is a wrapper around a number of
    :class:`SupportsPredictJoint`\ s.
    It delegates :meth:`predict_joint` to each model.
    """

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        :param query_points: The points at which to make predictions, of shape [..., B, D].
        :return: The predictions from all the wrapped models, concatenated along the event axis in
            the same order as they appear in :meth:`__init__`. If the wrapped models have predictive
            distributions with event shapes [:math:`E_i`], the mean will have shape
            [..., B, :math:`\sum_i E_i`], and the covariance shape
            [..., :math:`\sum_i E_i`, B, B].
        """
        means, covs = zip(*[model.predict_joint(query_points) for model in self._models])
        return tf.concat(means, axis=-1), tf.concat(covs, axis=-3)


# It's useful, though a bit ugly, to define the stack constructors for some model type combinations
class TrainableSupportsPredictJoint(TrainableProbabilisticModel, SupportsPredictJoint, Protocol):
    """A model that is both trainable and supports predict_joint."""

    pass


class TrainablePredictJointModelStack(
    TrainableModelStack, PredictJointModelStack, ModelStack[TrainableSupportsPredictJoint]
):
    """A stack of models that are both trainable and support predict_joint."""

    pass


class TrainableSupportsPredictJointHasReparamSampler(
    TrainableSupportsPredictJoint, HasReparamSampler, Protocol
):
    """A model that is trainable, supports predict_joint and has a reparameterization sampler."""

    pass


class TrainablePredictJointReparamModelStack(
    TrainablePredictJointModelStack,
    HasReparamSamplerModelStack,
    ModelStack[TrainableSupportsPredictJointHasReparamSampler],
):
    """A stack of models that are both trainable and support predict_joint."""

    pass


class ReparametrizationSampler(ABC, Generic[ProbabilisticModelType]):
    r"""
    These samplers employ the *reparameterization trick* to draw samples from a
    :class:`ProbabilisticModel`\ 's predictive distribution  across a discrete set of
    points. See :cite:`wilson2018maximizing` for details.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModelType):
        r"""
        Note that our :class:`TrainableModelStack` currently assumes that
        all :class:`ReparametrizationSampler` constructors have **only** these inputs
        and so will not work with more complicated constructors.

        :param sample_size: The desired number of samples.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)

        self._sample_size = sample_size
        self._model = model
        self._initialized = tf.Variable(False)  # Keep track of when we need to resample

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._sample_size!r}, {self._model!r})"

    @abstractmethod
    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        :param at: Input points that define the sampler of shape `[N, D]`.
        :param jitter: The size of the jitter to use when stabilizing the Cholesky
            decomposition of the covariance matrix.
        :return: Samples of shape `[sample_size, D]`.
        """

        raise NotImplementedError

    def reset_sampler(self) -> None:
        """
        Reset the sampler so that new samples are drawn at the next :meth:`sample` call.
        """
        self._initialized.assign(False)


TrajectoryFunction = Callable[[TensorType], TensorType]
"""
Type alias for trajectory functions. These have essentially the same behavior as an
:const:`AcquisitionFunction` but have additional sampling properties.

An :const:`TrajectoryFunction` evaluates a batch of `B` samples, each across different sets
of `N` query points (of dimension `D`) i.e. takes input of shape `[N, B, D]` and returns
shape `[N, B]`.

A key property of these trajectory functions is that the same sample draw is evaluated
for all queries. This property is known as consistency.
"""


class TrajectoryFunctionClass(ABC):
    """
    An :class:`TrajectoryFunctionClass` is a trajectory function represented using a class
    rather than as a standalone function. Using a class to represent a trajectory function
    makes it easier to update and resample without having to retrace the function.
    """

    @abstractmethod
    def __call__(self, x: TensorType) -> TensorType:
        """Call trajectory function."""


class TrajectorySampler(ABC, Generic[ProbabilisticModelType]):
    r"""
    This class builds functions that approximate a trajectory sampled from an
    underlying :class:`ProbabilisticModel`.

    Unlike the :class:`ReparametrizationSampler`, a :class:`TrajectorySampler` provides
    consistent samples (i.e ensuring that the same sample draw is used for all evaluations
    of a particular trajectory function).
    """

    def __init__(self, model: ProbabilisticModelType):
        """
        :param model: The model to sample from.
        """
        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._model!r})"

    @abstractmethod
    def get_trajectory(self) -> TrajectoryFunction:
        """
        Sample a batch of `B` trajectories. Note that the batch size `B` is determined
        by the first call of the :const:`TrajectoryFunction`. To change the batch size
        of a :const:`TrajectoryFunction` after initialization, you must
        recall :meth:`get_trajecotry`.

        :return: A trajectory function representing an approximate trajectory
            from the model, taking an input of shape `[N, B, D]` and returning shape `[N, B]`.
        """
        raise NotImplementedError

    def resample_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        A :const:`TrajectoryFunction` can often be efficiently updated in-place to provide
        a new sample without retracing. Note that if the underlying :class:`ProbabilisticModel`
        has been updated, then we must call :meth:`update_trajectory` to get a new sample from
        the new model.

        Efficient implementations of a :class:`TrajectorySampler` will have a custom method here
        to allow in-place resampling. However, the default behavior is just to make a new
        trajectory from scratch.

        :param trajectory: The trajectory function to be resampled.
        :return: The new resampled trajectory function.
        """
        return self.get_trajectory()

    def update_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        Update a :const:`TrajectoryFunction` to reflect an update in its
        underlying :class:`ProbabilisticModel` and resample accordingly.

        Efficient implementations will have a custom method here to allow in-place resampling
        and updating. However, the default behavior is just to make a new trajectory from scratch.

        :param trajectory: The trajectory function to be resampled.
        :return: The new trajectory function updated for a new model
        """
        return self.get_trajectory()

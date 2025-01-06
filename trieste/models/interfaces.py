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
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, overload

import gpflow
import tensorflow as tf
from check_shapes import check_shapes, inherit_check_shapes
from typing_extensions import Protocol, final, runtime_checkable

from ..data import Dataset
from ..space import EncoderFunction
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
    @check_shapes(
        "query_points: [batch..., D]",
        "return[0]: [batch..., E...]",
        "return[1]: [batch..., E...]",
    )
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
    @check_shapes(
        "query_points: [batch..., N, D]",
        "return: [batch..., S, N, E...]",
    )
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

    @abstractmethod
    def log(self, dataset: Optional[Dataset] = None) -> None:
        """
        Log model-specific information at a given optimization step.

        :param dataset: Optional data that can be used to log additional data-based model summaries.
        """
        raise NotImplementedError


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
    def optimize(self, dataset: Dataset) -> Any:
        """
        Optimize the model objective with respect to (hyper)parameters given the specified
        ``dataset``.

        :param dataset: The data with which to train the model.
        :return: Any (optimizer-specific) optimization result.
        """
        raise NotImplementedError


@runtime_checkable
class SupportsPredictJoint(ProbabilisticModel, Protocol):
    """A probabilistic model that supports predict_joint."""

    @abstractmethod
    @check_shapes(
        "query_points: [batch..., B, D]",
        "return[0]: [batch..., B, E...]",
        "return[1]: [batch..., E..., B, B]",
    )
    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        :param query_points: The points at which to make predictions, of shape [..., B, D].
        :return: The mean and covariance of the joint marginal distribution at each batch of points
            in ``query_points``. For a predictive distribution with event shape E, the mean will
            have shape [..., B] + E, and the covariance shape [...] + E + [B, B].
        """
        raise NotImplementedError


@runtime_checkable
class SupportsPredictY(ProbabilisticModel, Protocol):
    """A probabilistic model that supports predict_y."""

    @abstractmethod
    @check_shapes(
        "query_points: [broadcast batch..., D]",
        "return[0]: [batch..., E...]",
        "return[1]: [batch..., E...]",
    )
    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Return the mean and variance of the independent marginal distributions at each point in
        ``query_points`` for the observations, including noise contributions.

        :param query_points: The points at which to make predictions, of shape [..., D].
        :return: The mean and variance of the independent marginal distributions at each point in
            ``query_points``. For a predictive distribution with event shape E, the mean and
            variance will both have shape [...] + E.
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
class TrainableSupportsGetKernel(TrainableProbabilisticModel, SupportsGetKernel, Protocol):
    """A trainable probabilistic model that supports get_kernel."""


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
class SupportsGetInternalData(ProbabilisticModel, Protocol):
    """A probabilistic model that stores and has access to its own training data."""

    @abstractmethod
    def get_internal_data(self) -> Dataset:
        """
        Return the model's training data.

        :return: The model's training data.
        """
        raise NotImplementedError


@runtime_checkable
class SupportsGetMeanFunction(ProbabilisticModel, Protocol):
    """A probabilistic model that makes use of a mean function."""

    @abstractmethod
    def get_mean_function(self) -> Callable[[TensorType], TensorType]:
        """
        Return the model's mean function, i.e. a parameterized function that can explain
        coarse scale variations in the data, leaving just the residuals to be explained by
        our model.

        :return: The model's mean function.
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


@runtime_checkable
class SupportsReparamSamplerObservationNoise(
    HasReparamSampler, SupportsGetObservationNoise, Protocol
):
    """A model that supports both reparam_sampler and get_observation_noise."""


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

    # NB we don't use @inherit_shapes below as some classes break the shape API (👀 fantasizer)
    # instead we rely on the shape checking inside the submodels

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

    def log(self, dataset: Optional[Dataset] = None) -> None:
        """
        Log model-specific information at a given optimization step.

        :param dataset: Optional data that can be used to log additional data-based model summaries.
        """
        for i, model in enumerate(self._models):
            with tf.name_scope(f"{i}"):
                model.log(dataset)


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

    def optimize(self, dataset: Dataset) -> Sequence[Any]:
        """
        Optimize all the wrapped models on their corresponding data. The data for each model is
        extracted by splitting the observations in ``dataset`` along the event axis according to the
        event sizes specified at :meth:`__init__`.

        :param dataset: The query points and observations for *all* the wrapped models.
        """
        observations = tf.split(dataset.observations, self._event_sizes, axis=-1)
        results = []

        for model, obs in zip(self._models, observations):
            results.append(model.optimize(Dataset(dataset.query_points, obs)))

        return results


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


class PredictYModelStack(ModelStack[SupportsPredictY], SupportsPredictY):
    r"""
    A :class:`PredictYModelStack` is a wrapper around a number of
    :class:`SupportsPredictY`\ s.
    It delegates :meth:`predict_y` to each model.
    """

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


# It's useful, though a bit ugly, to define the stack constructors for some model type combinations
class TrainableSupportsPredictJoint(TrainableProbabilisticModel, SupportsPredictJoint, Protocol):
    """A model that is both trainable and supports predict_joint."""


class TrainablePredictJointModelStack(
    TrainableModelStack, PredictJointModelStack, ModelStack[TrainableSupportsPredictJoint]
):
    """A stack of models that are both trainable and support predict_joint."""


class TrainableSupportsPredictY(TrainableProbabilisticModel, SupportsPredictY, Protocol):
    """A model that is both trainable and supports predict_y."""


class TrainablePredictYModelStack(
    TrainableModelStack, PredictYModelStack, ModelStack[TrainableSupportsPredictY]
):
    """A stack of models that are both trainable and support predict_y."""


class SupportsPredictJointPredictY(SupportsPredictJoint, SupportsPredictY, Protocol):
    """A model that supports both predict_joint and predict_y."""


class PredictJointPredictYModelStack(
    PredictJointModelStack, PredictYModelStack, ModelStack[SupportsPredictJointPredictY]
):
    """A stack of models that support both predict_joint and predict_y."""


class TrainableSupportsPredictJointHasReparamSampler(
    TrainableSupportsPredictJoint, HasReparamSampler, Protocol
):
    """A model that is trainable, supports predict_joint and has a reparameterization sampler."""


class TrainablePredictJointReparamModelStack(
    TrainablePredictJointModelStack,
    HasReparamSamplerModelStack,
    ModelStack[TrainableSupportsPredictJointHasReparamSampler],
):
    """A stack of models that are both trainable and support predict_joint."""


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
        :param at: Where to sample the predictive distribution, with shape `[..., 1, D]`, for points
            of dimension `D`.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :return: The samples, of shape `[..., S, B, L]`, where `S` is the `sample_size`, `B` is
            the number of points per batch, and `L` is the number of latent model dimensions.
        """

        raise NotImplementedError

    def reset_sampler(self) -> None:
        """
        Reset the sampler so that new samples are drawn at the next :meth:`sample` call.
        """
        self._initialized.assign(False)


TrajectoryFunction = Callable[[TensorType], TensorType]
"""
Type alias for trajectory functions. These have similar behaviour to an :const:`AcquisitionFunction`
but have additional sampling properties and support multiple model outputs.

An :const:`TrajectoryFunction` evaluates a batch of `B` samples, each across different sets
of `N` query points (of dimension `D`) i.e. takes input of shape `[N, B, D]` and returns
shape `[N, B, L]`, where `L` is the number of outputs of the model. Note that we require the `L`
dimension to be present, even if there is only one output.

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
        recall :meth:`get_trajectory`.

        :return: A trajectory function representing an approximate trajectory
            from the model, taking an input of shape `[N, B, D]` and returning shape `[N, B, L]`,
            where `L` is the number of outputs of the model.
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


@runtime_checkable
class SupportsGetInducingVariables(ProbabilisticModel, Protocol):
    """A probabilistic model uses and has access to an inducing point approximation."""

    @abstractmethod
    def get_inducing_variables(self) -> tuple[TensorType, TensorType, TensorType, bool]:
        """
        Return the model's inducing variables.

        :return: Tensors containing: the inducing points (i.e. locations of the inducing
            variables); the variational mean q_mu; the Cholesky decomposition of the
            variational covariance q_sqrt; and a bool denoting if we are using whitened
            or not whitened representations.
        """
        raise NotImplementedError


@runtime_checkable
class SupportsCovarianceWithTopFidelity(ProbabilisticModel, Protocol):
    """A probabilistic model is multifidelity and has access to a method to calculate the
    covariance between a point and the same point at the top fidelity"""

    @property
    @abstractmethod
    def num_fidelities(self) -> int:
        """
        The number of fidelities
        """
        raise NotImplementedError

    @abstractmethod
    def covariance_with_top_fidelity(self, query_points: TensorType) -> TensorType:
        """
        Calculate the covariance of the output at `query_point` and a given fidelity with the
        highest fidelity output at the same `query_point`.

        :param query_points: The query points to calculate the covariance for, of shape [N, D+1],
            where the final column of the final dimension contains the fidelity of the query point
        :return: The covariance with the top fidelity for the `query_points`, of shape [N, P]
        """
        raise NotImplementedError


def encode_dataset(dataset: Dataset, encoder: EncoderFunction) -> Dataset:
    """Return a new Dataset with the query points encoded using the given encoder."""
    return Dataset(encoder(dataset.query_points), dataset.observations)


class EncodedProbabilisticModel(ProbabilisticModel):
    """A probabilistic model with an associated query point encoder.

    Classes that inherit from this (or the other associated mixins below) should implement the
    relevant _encoded methods (e.g. predict_encoded instead of predict), to which the public
    methods delegate after encoding their input. Take care to use the correct methods internally
    to avoid encoding twice accidentally.
    """

    @property
    @abstractmethod
    def encoder(self) -> EncoderFunction | None:
        """Query point encoder."""

    @overload
    def encode(self, points: TensorType) -> TensorType: ...

    @overload
    def encode(self, points: Dataset) -> Dataset: ...

    def encode(self, points: Dataset | TensorType) -> Dataset | TensorType:
        """Encode points or a Dataset using the query point encoder."""
        if self.encoder is None:
            return points
        elif isinstance(points, Dataset):
            return encode_dataset(points, self.encoder)
        else:
            return self.encoder(points)

    @abstractmethod
    def predict_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """Implementation of predict on encoded query points."""

    @abstractmethod
    def sample_encoded(self, query_points: TensorType, num_samples: int) -> TensorType:
        """Implementation of sample on encoded query points."""

    @final
    @inherit_check_shapes
    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.predict_encoded(self.encode(query_points))

    @final
    @inherit_check_shapes
    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        return self.sample_encoded(self.encode(query_points), num_samples)


class EncodedTrainableProbabilisticModel(EncodedProbabilisticModel, TrainableProbabilisticModel):
    """A trainable probabilistic model with an associated query point encoder."""

    @abstractmethod
    def update_encoded(self, dataset: Dataset) -> None:
        """Implementation of update on the encoded dataset."""

    @abstractmethod
    def optimize_encoded(self, dataset: Dataset) -> Any:
        """Implementation of optimize on the encoded dataset."""

    @final
    def update(self, dataset: Dataset) -> None:
        return self.update_encoded(self.encode(dataset))

    @final
    def optimize(self, dataset: Dataset) -> Any:
        return self.optimize_encoded(self.encode(dataset))


class EncodedSupportsPredictJoint(EncodedProbabilisticModel, SupportsPredictJoint):
    """A probabilistic model that supports predict_joint with an associated query point encoder."""

    @abstractmethod
    def predict_joint_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """Implementation of predict_joint on encoded query points."""

    @final
    @inherit_check_shapes
    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.predict_joint_encoded(self.encode(query_points))


class EncodedSupportsPredictY(EncodedProbabilisticModel, SupportsPredictY):
    """A probabilistic model that supports predict_y with an associated query point encoder."""

    @abstractmethod
    def predict_y_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """Implementation of predict_y on encoded query points."""

    @final
    @inherit_check_shapes
    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.predict_y_encoded(self.encode(query_points))


class EncodedFastUpdateModel(EncodedProbabilisticModel, FastUpdateModel):
    """A fast update model with an associated query point encoder."""

    @abstractmethod
    def conditional_predict_f_encoded(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        """Implementation of conditional_predict_f on encoded query points."""

    @abstractmethod
    def conditional_predict_joint_encoded(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        """Implementation of conditional_predict_joint on encoded query points."""

    @abstractmethod
    def conditional_predict_f_sample_encoded(
        self, query_points: TensorType, additional_data: Dataset, num_samples: int
    ) -> TensorType:
        """Implementation of conditional_predict_f_sample on encoded query points."""

    @abstractmethod
    def conditional_predict_y_encoded(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        """Implementation of conditional_predict_y on encoded query points."""

    @final
    def conditional_predict_f(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        return self.conditional_predict_f_encoded(
            self.encode(query_points), self.encode(additional_data)
        )

    @final
    def conditional_predict_joint(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        return self.conditional_predict_joint_encoded(
            self.encode(query_points), self.encode(additional_data)
        )

    @final
    def conditional_predict_f_sample(
        self, query_points: TensorType, additional_data: Dataset, num_samples: int
    ) -> TensorType:
        return self.conditional_predict_f_sample_encoded(
            self.encode(query_points), self.encode(additional_data), num_samples
        )

    @final
    def conditional_predict_y(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        return self.conditional_predict_y_encoded(
            self.encode(query_points), self.encode(additional_data)
        )


def get_encoder(model: ProbabilisticModel) -> EncoderFunction | None:
    """Helper function for getting an encoder from model (which may or may not have one)."""
    if isinstance(model, EncodedProbabilisticModel):
        return model.encoder
    return None

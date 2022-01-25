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

r"""
This module contains registry for supported models and config related classes and functions.
Configs allow expert users to build model as a dictionary of model and optimizer arguments,
rather than working with interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Type, Union
from warnings import warn

import gpflow
import tensorflow as tf

from .interfaces import TrainableProbabilisticModel
from .optimizer import BatchOptimizer, Optimizer


class ModelRegistry:
    """
    This is a registry of all supported models with their corresponding wrappers, and model
    optimizers with their corresponding wrapppers.

    A single entry per model and optimizer is maintained, if same model is registered again it will
    overwrite the previous entry. Registry is primarily used by :class:`ModelConfig` and
    :func:`create_model` to facilitate building models by specifying a simple dictionary of model
    and optimizer arguments.

    Note that you do not need to register your custom model if you will provide an instance of
    :class:`~trieste.models.TrainableProbabilisticModel` directly to the
    :class:`~trieste.BayesianOptimizer`. Registering is required if you intend to build your custom
    model as a dictionary of arguments for the wrapper and the optimizer, or as a
    :class:`ModelConfig`.
    """

    _MODEL_TO_WRAPPER: Dict[Type[Any], Type[TrainableProbabilisticModel]] = dict()
    _OPTIMIZER_TO_WRAPPER: Dict[Type[Any], Type[Optimizer]] = dict()

    @classmethod
    def get_model_wrapper(cls, model_type: Type[Any]) -> Type[TrainableProbabilisticModel]:
        """
        Get a Trieste model wrapper for a given model type.

        :param model_type: The model type.
        :return: The wrapper which builds a model.
        """
        # Check against all the supertypes in order, from most specific to most general
        for _model_type in model_type.__mro__:
            if _model_type in cls._MODEL_TO_WRAPPER:
                return cls._MODEL_TO_WRAPPER[_model_type]

        raise ValueError(
            f"Model {model_type} currently not supported. "
            f"We only support the following model types: "
            f"{list(k.__name__ for k in cls._MODEL_TO_WRAPPER.keys())}. "
            f"You can register more via ModelRegistry.register_model."
        )

    @classmethod
    def get_optimizer_wrapper(cls, optimizer_type: Type[Any]) -> Type[Optimizer]:
        """
        Get a Trieste model optimizer wrapper for a given optimizer type.

        :param optimizer_type: The optimizer type.
        :return: The optimizer wrapper to be used with the optimizer type.
        """
        # Check against all the supertypes in order, from most specific to most general
        for _optimizer_type in optimizer_type.__mro__:
            if _optimizer_type in cls._OPTIMIZER_TO_WRAPPER:
                return cls._OPTIMIZER_TO_WRAPPER[_optimizer_type]

        raise ValueError(
            f"Model {optimizer_type} currently not supported. "
            f"We only support the following optimizer types: "
            f"{list(k.__name__ for k in cls._OPTIMIZER_TO_WRAPPER.keys())}. "
            f"You can register more via ModelRegistry.register_optimizer."
        )

    @classmethod
    def register_model(
        cls,
        model_type: Type[Any],
        wrapper_type: Type[TrainableProbabilisticModel],
    ) -> None:
        """
        Register a new model type. Note that this will overwrite a registry
        entry if the model has already been registered.

        :param model_type: The model type.
        :param wrapper_type: The model wrapper to be used with the model type.
        """
        if model_type in cls._MODEL_TO_WRAPPER.keys():
            warn(f"Model {model_type} has already been registered, you have now overwritten it.")

        cls._MODEL_TO_WRAPPER[model_type] = wrapper_type

    @classmethod
    def register_optimizer(
        cls,
        optimizer_type: Type[Any],
        wrapper_type: Type[Optimizer],
    ) -> None:
        """
        Register a new optimizer type. Note that this will overwrite a registry
        entry if the optimizer has already been registered.

        :param optimizer_type: The optimizer type.
        :param wrapper_type: The optimier wrapper to be used with the optimizer type.
        """
        if optimizer_type in cls._OPTIMIZER_TO_WRAPPER.keys():
            warn(
                f"Optimizer {optimizer_type} has already been registered, "
                f"you have now overwritten it. "
            )

        cls._OPTIMIZER_TO_WRAPPER[optimizer_type] = wrapper_type

    @classmethod
    def get_registered_models(cls) -> Iterable[Any]:
        """
        Provides a generator with all supported model types.
        """
        yield from cls._MODEL_TO_WRAPPER.keys()

    @classmethod
    def get_registered_optimizers(cls) -> Iterable[Any]:
        """
        Provides a generator with all supported optimizer types.
        """
        yield from cls._OPTIMIZER_TO_WRAPPER.keys()


# We register supported optimizers and their wrappers for usage with ModelConfig.
# GPflow's Scipy optimizer is not used in the batch mode and hence uses Optimizer wrapper
ModelRegistry.register_optimizer(gpflow.optimizers.Scipy, Optimizer)


# TensorFlow optimizers are stochastic gradient descent variants and should be used
# with BatchOptimizer wrapper
ModelRegistry.register_optimizer(tf.optimizers.Optimizer, BatchOptimizer)


@dataclass(frozen=True)
class ModelConfig:
    """
    This class is a specification for building a
    :class:`~trieste.models.TrainableProbabilisticModel`. It is not meant to be used by itself,
    it implements methods that facilitate building a Trieste model as a dictionary of model and
    optimizer arguments with :func:`create_model`.
    """

    model: Any
    """
    The low-level model to pass to the :class:`~trieste.models.TrainableProbabilisticModel`
    wrapper registered with the `model` via :class:`ModelRegistry`. The model has to be one of
    the supported models, that is, registered via :class:`ModelRegistry`. We use type `Any` here as
    this can be either a model that is supported by default (for example, GPflow- or GPflux-based
    models) or a user-defined model that has been registered.
    """

    model_args: dict[str, Any] = field(default_factory=lambda: {})
    """
    The keyword arguments to pass to the model wrapper
    :class:`~trieste.models.TrainableProbabilisticModel` registered with the `model` via
    :class:`ModelRegistry`.
    """

    optimizer: Any = None
    """
    The low-level optimizer to pass to the :class:`~trieste.models.Optimizer` wrapper
    registered with the `model` via :class:`ModelRegistry`, with which to train the model (by
    minimizing its loss function). The model has to be one of the supported models, that is,
    registered via :class:`ModelRegistry`. We use type `Any` here as this can be either an
    optimizer that is supported by default (for example, GPflow or TensorFlow) or a user-defined
    optimizer that has been registered.
    """

    optimizer_args: dict[str, Any] = field(default_factory=lambda: {})
    """
    The keyword arguments to pass to the optimizer wrapper :class:`~trieste.models.Optimizer`
    registered with the `model` via :class:`ModelRegistry`.
    """

    def __post_init__(self) -> None:
        self._check_model_type()
        self._check_optimizer_type()

    def _check_model_type(self) -> None:
        if type(self.model) in ModelRegistry.get_registered_models():
            return
        raise NotImplementedError(f"Not supported type {type(self.model)}")

    def _check_optimizer_type(self) -> None:
        if self.optimizer is not None:
            if ModelRegistry.get_optimizer_wrapper(type(self.optimizer)):
                return
            raise NotImplementedError(f"Not supported type {type(self.optimizer)}")
        else:
            return

    def build_model(self) -> TrainableProbabilisticModel:
        """
        Builds a Trieste model from the model and optimizer configuration.
        """
        model_interface = ModelRegistry.get_model_wrapper(type(self.model))

        if self.optimizer is not None:
            optimizer_wrapper = ModelRegistry.get_optimizer_wrapper(type(self.optimizer))
            optimizer = optimizer_wrapper(self.optimizer, **self.optimizer_args)
            return model_interface(self.model, optimizer, **self.model_args)  # type: ignore
        else:
            return model_interface(self.model, **self.model_args)  # type: ignore


ModelDictConfig = Dict[str, Any]
""" Type alias for a config type specification of a model. """

ModelConfigType = Union[ModelDictConfig, ModelConfig]
""" Type alias for any config type that can be used to fully specify a model. """

ModelSpec = Union[ModelConfigType, TrainableProbabilisticModel]
""" Type alias for any type that can be used to fully specify a model. """


def create_model(
    config: ModelSpec,
) -> TrainableProbabilisticModel:
    """
    Build a model in a flexible way by providing a dictionary of model and optimizer arguments, a
    :class:`ModelConfig`, or a :class:`~trieste.models.TrainableProbabilisticModel`. This function
    is primarily used by :class:`~trieste.BayesianOptimizer` to build a model.

    :param config: A configuration for building a Trieste model.
    :return: A Trieste model built according to ``config``.
    """
    if isinstance(config, ModelConfig):
        return config.build_model()
    elif isinstance(config, dict):
        return ModelConfig(**config).build_model()
    elif isinstance(config, TrainableProbabilisticModel):
        return config
    raise NotImplementedError("Unknown format passed to create a TrainableProbabilisticModel.")

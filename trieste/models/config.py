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
from typing import Any, Dict, Iterable, Tuple, Type, Union
from warnings import warn

import tensorflow as tf

from .interfaces import TrainableProbabilisticModel
from .optimizer import Optimizer


class ModelRegistry:
    """
    This is a registry of all supported models with their corresponding interfaces and optimizers.
    A single entry per model is maintained, if same model is registered again it will overwrite the
    previous entry. Registry is primarily used by :class:`ModelConfig` and :func:`create_model` to
    facilitate building models by specifying a simple dictionary of model and optimizer arguments.

    Note that you do not need to register your custom model if you will provide an instance of
    :class:`~trieste.models.TrainableProbabilisticModel` directly to the
    :class:`~trieste.BayesianOptimizer`. Registering is required if you intend to build your custom
    model as a dictionary of arguments for the interface and the optimizer, or as a
    :class:`ModelConfig`.
    """

    _REGISTRY: Dict[Type[Any], Tuple[Type[TrainableProbabilisticModel], Type[Optimizer]]] = dict()

    @classmethod
    def get_interface(cls, model_type: Type[Any]) -> Type[TrainableProbabilisticModel]:
        """
        Get a Trieste model interface for a given model type.

        :param model_type: The model type.
        :return: The interface which builds a model.
        """
        try:
            return cls._REGISTRY[model_type][0]
        except KeyError as e:
            raise ValueError(
                f"Model {type(model_type)} currently not supported. "
                f"We only support the following model types: "
                f"{list(k.__name__ for k in cls._REGISTRY.keys())}. "
                f"You can register more via ModelRegistry.register_model."
            ) from e

    @classmethod
    def get_optimizer(cls, model_type: Type[Any]) -> Type[Optimizer]:
        """
        Get a Trieste model optimizer for a given model type.

        :param model_type: The model type.
        :return: The optimizer to be used with the model type.
        """
        try:
            return cls._REGISTRY[model_type][1]
        except KeyError as e:
            raise ValueError(
                f"Model {type(model_type)} currently not supported. "
                f"We only support the following model types: "
                f"{list(k.__name__ for k in cls._REGISTRY.keys())}. "
                f"You can register more via ModelRegistry.register_model."
            ) from e

    @classmethod
    def register_model(
        cls,
        model_type: Type[Any],
        interface_type: Type[TrainableProbabilisticModel],
        optimizer_type: Type[Optimizer],
    ) -> None:
        """
        Register a new model type. Note that this will overwrite a registry
        entry if the model has already been registered.

        :param model_type: The model type.
        :param interface_type: The interface to be used with the model type.
        :param optimizer_type: The optimizer to be used with the model type.
        """
        if model_type in cls._REGISTRY.keys():
            warn(f"Model {model_type} has already been registered, you have now overwritten it. ")

        cls._REGISTRY[model_type] = (interface_type, optimizer_type)

    @classmethod
    def get_registered_models(cls) -> Iterable[Any]:
        """
        Provides a generator with all supported model types.
        """
        yield from cls._REGISTRY.keys()


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
    interface registered with the `model` via :class:`ModelRegistry`. The model has to be one of
    the supported models, that is, registered via :class:`ModelRegistry`. We use type `Any` here as
    this can be either a model that is supported by default (for example, GPflow- or GPflux-based
    models) or a user-defined model that has been registered.
    """

    model_args: dict[str, Any] = field(default_factory=lambda: {})
    """
    The keyword arguments to pass to the model interface
    :class:`~trieste.models.TrainableProbabilisticModel` registered with the `model` via
    :class:`ModelRegistry`.
    """

    optimizer: Any = field(default_factory=lambda: tf.optimizers.Adam())
    """
    The low-level optimizer to pass to the :class:`~trieste.models.Optimizer` interface
    registered with the `model` via :class:`ModelRegistry`, with which to train the model (by
    minimizing its loss function).
    """

    optimizer_args: dict[str, Any] = field(default_factory=lambda: {})
    """
    The keyword arguments to pass to the optimizer interface :class:`~trieste.models.Optimizer`
    registered with the `model` via :class:`ModelRegistry`.
    """

    def __post_init__(self) -> None:
        self._check_model_type()

    def _check_model_type(self) -> None:
        if type(self.model) in ModelRegistry.get_registered_models():
            return

        raise NotImplementedError(f"Not supported type {type(self.model)}")

    def build_model(self) -> TrainableProbabilisticModel:
        """
        Builds a Trieste model from the model and optimizer configuration.
        """
        model_interface = ModelRegistry.get_interface(type(self.model))
        model_optimizer = ModelRegistry.get_optimizer(type(self.model))
        optimizer = model_optimizer(self.optimizer, **self.optimizer_args)

        return model_interface(self.model, optimizer, **self.model_args)  # type: ignore


ModelDictConfig = Dict[str, Any]
""" Type alias for a config type specification of a model. """

ModelSpec = Union[ModelDictConfig, ModelConfig, TrainableProbabilisticModel]
""" Type alias for any type that can be used to fully specify a model. """


def create_model(config: ModelSpec) -> TrainableProbabilisticModel:
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

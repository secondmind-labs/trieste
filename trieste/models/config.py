# Copyright 2020 The Trieste Contributors
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
from dataclasses import dataclass, field
from typing import Any, Dict, Union

import gpflow
import tensorflow as tf

from .model_interfaces import ModelInterface, Optimizer, supported_models


@dataclass(frozen=True)
class ModelConfig:
    """ Specification for building a :class:`~trieste.models.ModelInterface`. """

    model: Union[tf.Module, ModelInterface]
    """ The :class:`~trieste.models.ModelInterface`, or the model to wrap in one. """

    optimizer: Optimizer = field(default_factory=lambda: gpflow.optimizers.Scipy())
    """ The optimizer with which to train the model (by minimizing its loss function). """

    optimizer_args: Dict[str, Any] = field(default_factory=lambda: {})
    """ The keyword arguments to pass to the optimizer when training the model. """

    def __post_init__(self) -> None:
        self._check_model_type()

    @staticmethod
    def create_from_dict(d: Dict[str, Any]) -> ModelConfig:
        """
        :param d: A dictionary from which to construct this :class:`ModelConfig`.
        :return: A :class:`ModelConfig` built from ``d``.
        :raise TypeError: If the keys in ``d`` do not correspond to the parameters of
            :class:`ModelConfig`.
        """
        return ModelConfig(**d)

    def _check_model_type(self) -> None:
        if isinstance(self.model, ModelInterface):
            return

        for model_type in supported_models:
            if isinstance(self.model, model_type):
                return

        raise NotImplementedError(f"Not supported type {type(self.model)}")

    def create_model_interface(self) -> ModelInterface:
        """
        :return: A model built from this model configuration.
        """
        if isinstance(self.model, ModelInterface):
            return self.model

        for model_type, model_interface in supported_models.items():
            if isinstance(self.model, model_type):
                mi = model_interface(self.model)
                mi.set_optimizer(self.optimizer)
                mi.set_optimizer_args(self.optimizer_args)
                return mi

        raise NotImplementedError(f"Not supported type {type(self.model)}")


ModelSpec = Union[Dict[str, Any], ModelConfig, ModelInterface]
""" Type alias for any type that can be used to fully specify a model. """


def create_model_interface(config: ModelSpec) -> ModelInterface:
    """
    :param config: A :class:`ModelInterface` or configuration of a model.
    :return: A :class:`~trieste.models.ModelInterface` build according to ``config``.
    """
    if isinstance(config, ModelConfig):
        return config.create_model_interface()
    elif isinstance(config, dict):
        return ModelConfig(**config).create_model_interface()
    elif isinstance(config, ModelInterface):
        return config
    raise NotImplementedError("Unknown format passed to create a ModelInterface.")

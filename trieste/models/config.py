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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Union

import tensorflow as tf

from .interfaces import TrainableProbabilisticModel
from .optimizer import Optimizer, create_optimizer


@dataclass(frozen=True)
class ModelConfig:
    """
    This class is a specification for building a
    :class:`~trieste.models.TrainableProbabilisticModel`. It is not meant to be used by itself,
    subclasses that implement the missing asbtract methods should be used instead. These abstract
    methods define a default optimizer and all models supported by a specific model type (e.g.
    Gaussian processes implementation). Note that subclasses should also be frozen dataclasses.
    """

    model: tf.Module | TrainableProbabilisticModel
    """ The :class:`~trieste.models.TrainableProbabilisticModel`, or the model to wrap in one. """

    model_args: dict[str, Any] = field(default_factory=lambda: {})
    """ The keyword arguments to pass to the model wrapper. """

    optimizer: Any = field(default_factory=lambda: tf.optimizers.Adam())
    """ The optimizer with which to train the model (by minimizing its loss function). """

    optimizer_args: dict[str, Any] = field(default_factory=lambda: {})
    """ The keyword arguments to pass to the optimizer wrapper. """

    def __post_init__(self) -> None:
        self._check_model_type()

    def supported_models(
        self,
    ) -> dict[Any, Callable[[Any, Optimizer], TrainableProbabilisticModel]]:
        """
        Defines all models supported by certain model type (e.g. Gaussian process implementation).
        This method has to be specified by a model type specific subclass.

        :return: A mapping of third-party model types to :class:`CustomTrainable` classes that wrap
            models of those types.
        """
        raise NotImplementedError

    def _check_model_type(self) -> None:
        if isinstance(self.model, TrainableProbabilisticModel):
            return

        for model_type in self.supported_models():
            if isinstance(self.model, model_type):
                return

        raise NotImplementedError(f"Not supported type {type(self.model)}")

    def create_model_interface(self) -> TrainableProbabilisticModel:
        """
        :return: A model built from this model configuration.
        """
        if isinstance(self.model, TrainableProbabilisticModel):
            return self.model

        optimizer = create_optimizer(self.optimizer, self.optimizer_args)

        for model_type, model_interface in self.supported_models().items():
            if isinstance(self.model, model_type):
                return model_interface(self.model, optimizer, **self.model_args)  # type: ignore

        raise NotImplementedError(f"Not supported type {type(self.model)}")


ModelSpec = Union[ModelConfig, TrainableProbabilisticModel]
""" Type alias for any type that can be used to fully specify a model. """


def create_model(config: ModelSpec) -> TrainableProbabilisticModel:
    """
    :param config: A :class:`TrainableProbabilisticModel` or configuration of a model.
    :return: A :class:`~trieste.models.TrainableProbabilisticModel` build according to ``config``.
    """
    if isinstance(config, ModelConfig):
        return config.create_model_interface()
    elif isinstance(config, TrainableProbabilisticModel):
        return config
    raise NotImplementedError("Unknown format passed to create a TrainableProbabilisticModel.")

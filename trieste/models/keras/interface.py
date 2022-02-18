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
from typing import Optional

import tensorflow as tf

from ...types import TensorType
from ..interfaces import ProbabilisticModel
from ..optimizer import KerasOptimizer


class KerasPredictor(ProbabilisticModel, ABC):
    """
    This is an interface for trainable wrappers of TensorFlow and Keras neural network models.
    """

    def __init__(self, optimizer: Optional[KerasOptimizer] = None):
        """
        :param optimizer: The optimizer wrapper containing the optimizer with which to train the
            model and arguments for the wrapper and the optimizer. The optimizer must
            be an instance of a :class:`~tf.optimizers.Optimizer`. Defaults to
            :class:`~tf.optimizers.Adam` optimizer with default parameters.
        :raise ValueError: If the optimizer is not an instance of :class:`~tf.optimizers.Optimizer`.
        """
        if optimizer is None:
            optimizer = KerasOptimizer(tf.optimizers.Adam())
        self._optimizer = optimizer

        if not isinstance(optimizer.optimizer, tf.optimizers.Optimizer):
            raise ValueError(
                f"Optimizer for `KerasPredictor` models must be an instance of a "
                f"`tf.optimizers.Optimizer`, received {type(optimizer.optimizer)} instead."
            )

    @property
    @abstractmethod
    def model(self) -> tf.keras.Model:
        """The compiled Keras model."""
        raise NotImplementedError

    @property
    def optimizer(self) -> KerasOptimizer:
        """The optimizer wrapper for training the model."""
        return self._optimizer

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.model.predict(query_points)

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        raise NotImplementedError(
            """
            KerasPredictor does not implement sampling. Acquisition
            functions relying on it cannot be used with this class by default. Certain
            types of neural networks might be able to generate samples and
            such subclasses should overwrite this method.
            """
        )

    def __deepcopy__(self, memo: dict[int, object]) -> KerasPredictor:
        raise NotImplementedError(
            """
            KerasPredictor does not support deepcopy at the moment. For this reason,
            ``track_state`` argument when calling
            :meth:`~trieste.bayesian_optimizer.BayesianOptimizer.optimize` method should be set to
            `False`. This means that the model cannot be saved during Bayesian optimization, only
            the final model will be available.
            """
        )

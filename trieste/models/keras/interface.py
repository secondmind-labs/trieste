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

from abc import ABC, abstractproperty
from typing import Optional, Union

import tensorflow as tf

from ...types import TensorType
from ..interfaces import ProbabilisticModel
from ..optimizer import BatchOptimizer


class NeuralNetworkPredictor(ProbabilisticModel, tf.Module, ABC):
    """
    This is an interface for trainable wrappers of TensorFlow and Keras neural network models.
    We recommend to set `tf.keras.backend.set_floatx(tf.float64)` for alignment with the Trieste
    toolbox.
    """

    def __init__(self, optimizer: Optional[BatchOptimizer] = None):
        """
        :param optimizer: The optimizer wrapper containing the optimizer with which to train the
            model and arguments for the wrapper and the optimizer. The optimizer must
            be an instance of a :class:`~tf.optimizers.Optimizer`. Defaults to
            :class:`~tf.optimizers.Adam` optimizer with default parameters.
        """
        super().__init__()

        if optimizer is None:
            optimizer = BatchOptimizer(tf.optimizers.Adam())
        self._optimizer = optimizer

        if not isinstance(self._optimizer.optimizer, tf.optimizers.Optimizer):
            raise ValueError(
                f"Optimizer for `NeuralNetworkPredictor` models must be an instance of a "
                f"`tf.optimizers.Optimizer` or `tf.keras.optimizers.Optimizer`, "
                f"received {type(optimizer.optimizer)} instead."
            )

    @property
    @abstractproperty
    def model(self) -> Union[tf.keras.Model, tf.Module]:
        """The compiled Keras or generic TensorFlow neural network model."""
        raise NotImplementedError

    @property
    def optimizer(self) -> BatchOptimizer:
        """The optimizer wrapper for training the model."""
        return self._optimizer

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.model.predict(query_points)

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        raise NotImplementedError(
            """
            NeuralNetworkPredictor class does not implement joint predictions. Acquisition
            functions relying on it cannot be used with this class by default. Certain
            types of neural networks might be able to generate joint predictions and
            such subclasses should overwrite this method.
            """
        )

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        raise NotImplementedError(
            """
            NeuralNetworkPredictor class does not implement sampling. Acquisition
            functions relying on it cannot be used with this class by default. Certain
            types of neural networks might be able to generate samples and
            such subclasses should overwrite this method.
            """
        )

    def __deepcopy__(self, memo: dict[int, object]) -> NeuralNetworkPredictor:
        raise NotImplementedError("`deepcopy` not yet supported for `NeuralNetworkPredictor`")

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

import tensorflow as tf
from gpflow.base import Module

from ...types import TensorType
from ..interfaces import ProbabilisticModel
from ..optimizer import Optimizer


class GPfluxPredictor(ProbabilisticModel, tf.Module, ABC):
    """A trainable wrapper for a GPflux deep Gaussian process model. The code assumes subclasses
    will use the Keras `fit` method for training, and so they should provide access to both a
    `model_keras` and `model_gpflux`. Note: due to Keras integration, the user should remember to
    use `tf.keras.backend.set_floatx()` with the desired value (consistent with GPflow) to avoid
    dtype errors."""

    def __init__(self, optimizer: Optimizer | None = None):
        """
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~tf.optimizers.Adam`.
        """
        super().__init__()

        if optimizer is None:
            optimizer = Optimizer(tf.optimizers.Adam())

        self._optimizer = optimizer

        if not isinstance(self._optimizer.optimizer, tf.optimizers.Optimizer):
            raise ValueError(
                f"Optimizer for `DeepGaussianProcess` must be an instance of a "
                f"`tf.optimizers.Optimizer` or `tf.keras.optimizers.Optimizer`, "
                f"received {type(optimizer.optimizer)} instead."
            )

    @property
    @abstractmethod
    def model_gpflux(self) -> Module:
        """The underlying GPflux model."""

    @property
    @abstractmethod
    def model_keras(self) -> tf.keras.Model:
        """Returns the compiled Keras model for training."""

    @property
    def optimizer(self) -> Optimizer:
        """The optimizer with which to train the model."""
        return self._optimizer

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """Note: unless otherwise noted, this returns the mean and variance of the last layer
        conditioned on one sample from the previous layers."""
        return self.model_gpflux.predict_f(query_points)

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        raise NotImplementedError("Joint prediction not implemented for deep GPs")

    @abstractmethod
    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        raise NotImplementedError

    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """Note: unless otherwise noted, this will return the prediction conditioned on one sample
        from the lower layers."""
        f_mean, f_var = self.model_gpflux.predict_f(query_points)
        return self.model_gpflux.likelihood_layer.likelihood.predict_mean_and_var(f_mean, f_var)

    def get_observation_noise(self) -> TensorType:
        """
        Return the variance of observation noise for homoscedastic likelihoods.

        :return: The observation noise.
        :raise NotImplementedError: If the model does not have a homoscedastic likelihood.
        """
        try:
            noise_variance = self.model_gpflux.likelihood_layer.likelihood.variance
        except AttributeError:
            raise NotImplementedError(f"Model {self!r} does not have scalar observation noise")

        return noise_variance

    def __deepcopy__(self, memo: dict[int, object]) -> GPfluxPredictor:
        raise NotImplementedError("`deepcopy` not yet supported for `GPfluxPredictor`")

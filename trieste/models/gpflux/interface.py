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
from ..interfaces import SupportsGetObservationNoise
from ..optimizer import BatchOptimizer


class GPfluxPredictor(SupportsGetObservationNoise, ABC):
    """A trainable wrapper for a GPflux deep Gaussian process model. The code assumes subclasses
    will use the Keras `fit` method for training, and so they should provide access to both a
    `model_keras` and `model_gpflux`. Note: due to Keras integration, the user should remember to
    use `tf.keras.backend.set_floatx()` with the desired value (consistent with GPflow) to avoid
    dtype errors."""

    def __init__(self, optimizer: BatchOptimizer | None = None):
        """
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.BatchOptimizer` with :class:`~tf.optimizers.Adam`.
        """
        if optimizer is None:
            optimizer = BatchOptimizer(tf.optimizers.Adam())

        self._optimizer = optimizer

    @property
    @abstractmethod
    def model_gpflux(self) -> Module:
        """The underlying GPflux model."""

    @property
    @abstractmethod
    def model_keras(self) -> tf.keras.Model:
        """Returns the compiled Keras model for training."""

    @property
    def optimizer(self) -> BatchOptimizer:
        """The optimizer with which to train the model."""
        return self._optimizer

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """Note: unless otherwise noted, this returns the mean and variance of the last layer
        conditioned on one sample from the previous layers."""
        return self.model_gpflux.predict_f(query_points)

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
        raise NotImplementedError(
            """
            GPfluxPredictor does not support deepcopy at the moment. For this reason,
            ``track_state`` argument when calling
            :meth:`~trieste.bayesian_optimizer.BayesianOptimizer.optimize` method should be set to
            `False`. This means that the model cannot be saved during Bayesian optimization, only
            the final model will be available.
            """
        )

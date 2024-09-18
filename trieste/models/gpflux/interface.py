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

from gpflow.base import Module
from gpflow.keras import tf_keras

from ...space import EncoderFunction
from ...types import TensorType
from ..interfaces import EncodedSupportsPredictY, SupportsGetObservationNoise
from ..optimizer import KerasOptimizer


class GPfluxPredictor(SupportsGetObservationNoise, EncodedSupportsPredictY, ABC):
    """
    A trainable wrapper for a GPflux deep Gaussian process model. The code assumes subclasses
    will use the Keras `fit` method for training, and so they should provide access to both a
    `model_keras` and `model_gpflux`.
    """

    def __init__(
        self, optimizer: KerasOptimizer | None = None, encoder: EncoderFunction | None = None
    ):
        """
        :param optimizer: The optimizer wrapper containing the optimizer with which to train the
            model and arguments for the wrapper and the optimizer. The optimizer must
            be an instance of a :class:`~tf.optimizers.Optimizer`. Defaults to
            :class:`~tf.optimizers.Adam` optimizer with 0.01 learning rate.
        :param encoder: Optional encoder with which to transform query points before
            generating predictions.
        """
        if optimizer is None:
            optimizer = KerasOptimizer(tf_keras.optimizers.Adam(0.01))

        self._optimizer = optimizer
        self._encoder = encoder

    @property
    def encoder(self) -> EncoderFunction | None:
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: EncoderFunction | None) -> None:
        self._encoder = encoder

    @property
    @abstractmethod
    def model_gpflux(self) -> Module:
        """The underlying GPflux model."""

    @property
    @abstractmethod
    def model_keras(self) -> tf_keras.Model:
        """Returns the compiled Keras model for training."""

    @property
    def optimizer(self) -> KerasOptimizer:
        """The optimizer wrapper for training the model."""
        return self._optimizer

    def predict_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """Note: unless otherwise noted, this returns the mean and variance of the last layer
        conditioned on one sample from the previous layers."""
        return self.model_gpflux.predict_f(query_points)

    @abstractmethod
    def sample_encoded(self, query_points: TensorType, num_samples: int) -> TensorType:
        raise NotImplementedError

    def predict_y_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """Note: unless otherwise noted, this will return the prediction conditioned on one sample
        from the lower layers."""
        f_mean, f_var = self.model_gpflux.predict_f(query_points)
        return self.model_gpflux.likelihood_layer.likelihood.predict_mean_and_var(
            query_points, f_mean, f_var
        )

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

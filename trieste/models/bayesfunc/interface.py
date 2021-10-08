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

import torch as t
import torch.nn as nn

import tensorflow as tf
from gpflow.base import Module

from ...types import TensorType
from ..interfaces import ProbabilisticModel


class BayesFuncPredictor(ProbabilisticModel, tf.Module, ABC):
    """A trainable wrapper for a GPflux deep Gaussian process model. The code assumes subclasses
    will use the Keras `fit` method for training, and so they should provide access to both a
    `model_keras` and `model_gpflux`. Note: due to Keras integration, the user should remember to
    use `tf.keras.backend.set_floatx()` with the desired value (consistent with GPflow) to avoid
    dtype errors."""

    def __init__(self):
        """
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.TFOptimizer` with :class:`~tf.optimizers.Adam` with
            batch size 100. Optimizer is required to be a TFOptimizer, attempts to use other
            optimizers will result in an error.
        """
        super().__init__()

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """The underlying bayesfunc model."""

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """Note: unless otherwise noted, this returns the mean and variance of the last layer
        conditioned on one sample from the previous layers."""
        raise NotImplementedError("Predict f not implemented for BayesFunc models")

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        raise NotImplementedError("Joint prediction not implemented for deep GPs")

    @abstractmethod
    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        raise NotImplementedError

    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """Note: unless otherwise noted, this will return the prediction conditioned on one sample
        from the lower layers."""
        raise NotImplementedError

    def get_observation_noise(self) -> TensorType:
        """
        Return the variance of observation noise for homoscedastic likelihoods.
        :return: The observation noise.
        :raise NotImplementedError: If the model does not have a homoscedastic likelihood.
        """
        raise NotImplementedError(f"bayesfunc does not provide access to noise variance")

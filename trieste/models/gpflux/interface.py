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

import gpflux
import gpflow
import tensorflow as tf
from gpflux.models import DeepGP

from ...data import Dataset
from ...types import TensorType
from ..interfaces import ProbabilisticModel
from ..optimizer import Optimizer, TFOptimizer
from ..gpflow.utils import module_deepcopy


class GPfluxPredictor(ProbabilisticModel, tf.Module, ABC):
    """A trainable wrapper for a GPflux deep Gaussian process model. Note that Scipy optimizer
    is not supported for deep GP models."""

    def __init__(self, optimizer: TFOptimizer | None = None):
        """
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.TFOptimizer` with :class:`~tf.optimizers.Adam` with
            batch size 100. Note: using Scipy optimizer will raise ValueError as it is not
            supported.
        """
        super().__init__()

        if optimizer is None:
            optimizer = TFOptimizer(tf.optimizers.Adam(), batch_size=100)

        if isinstance(optimizer.optimizer, gpflow.optimizers.Scipy):
            raise ValueError("Cannot use Scipy optimizer for GPflux models")

        self._optimizer = optimizer

    @property
    def optimizer(self) -> Optimizer:
        """The optimizer with which to train the model."""
        return self._optimizer

    @property
    @abstractmethod
    def model(self) -> DeepGP:
        """The underlying GPflux model."""

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """Note: unless otherwise noted, this returns the mean and variance of the last layer
        conditioned on one sample from the previous layers."""
        return self.model.predict_f(query_points)

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        raise NotImplementedError("Joint prediction not implemented for deep GPs")

    @abstractmethod
    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        raise NotImplementedError

    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """Note: unless otherwise noted, this will return the prediction conditioned on one sample
        from the lower layers."""
        f_mean, f_var = self.model.predict_f(query_points)
        return self.model.likelihood_layer.likelihood.predict_mean_and_var(f_mean, f_var)

    def get_kernel(self) -> gpflow.kernels.Kernel:
        """
        Return the kernel of the model.

        :return: The kernel.
        """
        raise NotImplementedError("Deep GPs do not have a single kernel")

    def get_observation_noise(self) -> TensorType:
        """
        Return the variance of observation noise for homoscedastic likelihoods.
        :return: The observation noise.
        :raise NotImplementedError: If the model does not have a homoscedastic likelihood.
        """
        try:
            noise_variance = self.model.likelihood_layer.likelihood.variance
        except AttributeError:
            raise NotImplementedError(f"Model {self!r} does not have scalar observation noise")

        return noise_variance

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the model with the specified `dataset`.

        :param dataset: The data with which to optimize the `model`.
        """
        self.optimizer.optimize(self.model, dataset)

    __deepcopy__ = module_deepcopy


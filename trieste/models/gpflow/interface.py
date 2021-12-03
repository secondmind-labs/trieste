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

import gpflow
import tensorflow as tf
from gpflow.models import GPModel

from ...data import Dataset
from ...logging import get_step_number, get_tensorboard_writer
from ...types import TensorType
from ..interfaces import ProbabilisticModel
from ..optimizer import Optimizer


class GPflowPredictor(ProbabilisticModel, tf.Module, ABC):
    """A trainable wrapper for a GPflow Gaussian process model."""

    def __init__(self, optimizer: Optimizer | None = None):
        """
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        """
        super().__init__()

        if optimizer is None:
            optimizer = Optimizer(gpflow.optimizers.Scipy())

        self._optimizer = optimizer

    @property
    def optimizer(self) -> Optimizer:
        """The optimizer with which to train the model."""
        return self._optimizer

    @property
    @abstractmethod
    def model(self) -> GPModel:
        """The underlying GPflow model."""

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.model.predict_f(query_points)

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.model.predict_f(query_points, full_cov=True)

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        return self.model.predict_f_samples(query_points, num_samples)

    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.model.predict_y(query_points)

    def get_kernel(self) -> gpflow.kernels.Kernel:
        """
        Return the kernel of the model.

        :return: The kernel.
        """
        return self.model.kernel

    def get_observation_noise(self) -> TensorType:
        """
        Return the variance of observation noise for homoscedastic likelihoods.
        :return: The observation noise.
        :raise NotImplementedError: If the model does not have a homoscedastic likelihood.
        """
        try:
            noise_variance = self.model.likelihood.variance
        except AttributeError:
            raise NotImplementedError(f"Model {self!r} does not have scalar observation noise")

        return noise_variance

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the model with the specified `dataset`.

        :param dataset: The data with which to optimize the `model`.
        """
        self.optimizer.optimize(self.model, dataset)

    def log(self) -> None:
        """
        Log model-specific information at a given optimization step.
        """
        summary_writer = get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=get_step_number()):
                tf.summary.scalar("kernel.variance", self.get_kernel().variance)
                lengthscales = self.get_kernel().lengthscales
                if tf.rank(lengthscales) == 0:
                    tf.summary.scalar("kernel.lengthscale", lengthscales)
                elif tf.rank(lengthscales) == 1:
                    for i, lengthscale in enumerate(lengthscales):
                        tf.summary.scalar(f"kernel.lengthscale.{i}", lengthscale)

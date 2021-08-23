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

import warnings

import tensorflow as tf
from gpflux.layers import GPLayer
from gpflux.models import DeepGP
from gpflux.models.deep_gp import sample_dgp

from ...data import Dataset
from ...types import TensorType
from ...utils import jit
from ..interfaces import TrainableProbabilisticModel
from ..optimizer import Optimizer, TFOptimizer
from .interface import GPfluxPredictor


class DeepGaussianProcess(GPfluxPredictor, TrainableProbabilisticModel):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflux :class:`~gpflow.models.DeepGP` with
    only standard :class:`GPLayer`: this class does not support keras layers, latent variable
    layers, etc.
    """

    def __init__(self, model: DeepGP, optimizer: Optimizer | None = None):
        """
        :param model: The underlying GPflux deep Gaussian process model.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.TFOptimizer` with :class:`~tf.optimizers.Adam` with
            batch size 100.
        """

        if optimizer is None:
            optimizer = TFOptimizer(tf.optimizers.Adam(), batch_size=100)

        if not isinstance(optimizer, TFOptimizer):
            raise ValueError("Optimizer must be a TFOptimizer for deep GPs")

        if not isinstance(optimizer.optimizer, tf.optimizers.Optimizer):
            raise ValueError("Model optimizer must be a tf.optimizers.Optimizer for deep GPs")

        super().__init__(optimizer)

        if not all([isinstance(layer, GPLayer) for layer in model.f_layers]):
            raise ValueError("`DeepGaussianProcess` can only be built out of `GPLayer`")

        for layer in model.f_layers:
            if layer.whiten:
                warnings.warn(
                    "Sampling cannot be currently used with whitening in layers", UserWarning
                )

        self._model = model

        self._keras_model = model.as_training_model()
        self._keras_model.compile(optimizer.optimizer)

    def __repr__(self) -> str:
        """"""
        return f"DeepGaussianProcess({self._model!r}, {self.optimizer!r})"

    @property
    def model(self) -> DeepGP:
        return self._model

    @property
    def keras_model(self) -> tf.keras.Model:
        return self._keras_model

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        sampler = sample_dgp(self.model)

        @jit(apply=self.optimizer.compile)
        def get_samples(query_points: TensorType, num_samples: int) -> TensorType:
            samples = []
            for _ in range(num_samples):
                samples.append(sampler(query_points))
            return tf.stack(samples)

        return get_samples(query_points, num_samples)

    def update(self, dataset: Dataset) -> None:
        new_num_data = dataset.query_points.shape[0]
        self.model.num_data = new_num_data

        # Update num_data for each layer, as well as make sure dataset shapes are ok
        for i, layer in enumerate(self.model.f_layers):
            layer.num_data = new_num_data
            if i == 0:
                if (
                    dataset.query_points.shape[-1]
                    != layer.inducing_variable.inducing_variable.Z.shape[-1]
                ):
                    raise ValueError(
                        f"Shape {dataset.query_points.shape} of new query points is incompatible"
                        f" with shape {layer.inducing_variable.inducing_variable.Z.shape} of "
                        f" existing query points. Trailing dimensions must match."
                    )
            elif i == len(self.model.f_layers) - 1:
                if dataset.observations.shape[-1] != layer.q_mu.shape[-1]:
                    raise ValueError(
                        f"Shape {dataset.observations.shape} of new observations is incompatible"
                        f" with shape {layer.q_mu.shape} of existing observations. Trailing"
                        f" dimensions must match."
                    )

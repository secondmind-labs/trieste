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

from typing import Any, Dict

import tensorflow as tf
from gpflow.inducing_variables import InducingPoints
from gpflux.layers import GPLayer, LatentVariableLayer
from gpflux.models import DeepGP

from ...data import Dataset
from ...types import TensorType
from ..interfaces import TrainableProbabilisticModel
from .interface import GPfluxPredictor
from .utils import sample_dgp


class DeepGaussianProcess(GPfluxPredictor, TrainableProbabilisticModel):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflux :class:`~gpflux.models.DeepGP` with
    :class:`GPLayer` or :class:`LatentVariableLayer`: this class does not support e.g. keras layers.
    We provide simple architectures that can be used with this class in the `architectures.py` file.
    Note: the user should remember to set `tf.keras.backend.set_floatx()` with the desired value
    (consistent with GPflow) so that dtype errors do not occur.
    """

    def __init__(
        self,
        model: DeepGP,
        optimizer: tf.optimizers.Optimizer | None = None,
        fit_args: Dict[str, Any] | None = None,
    ):
        """
        :param model: The underlying GPflux deep Gaussian process model.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.TFOptimizer` with :class:`~tf.optimizers.Adam`. Only
            the optimizer itself is used; other args relevant for fitting should be passed as part
            of `fit_args`.
        :param fit_args: A dictionary of arguments to be used in the Keras `fit` method. Default to
            using 100 epochs, batch size 100, and verbose 0.
        """

        super().__init__()

        if optimizer is None:
            self._optimizer = tf.optimizers.Adam()
        else:
            self._optimizer = optimizer

        if not isinstance(self._optimizer, tf.keras.optimizers.Optimizer):
            raise ValueError("Must use a Keras/TF optimizer for DGPs, not wrapped in TFOptimizer")

        self.original_lr = self._optimizer.lr.numpy()

        if fit_args is None:
            self.fit_args = dict(
                {
                    "verbose": 0,
                    "epochs": 100,
                    "batch_size": 100,
                }
            )
        else:
            self.fit_args = fit_args

        if not all([isinstance(layer, (GPLayer, LatentVariableLayer)) for layer in model.f_layers]):
            raise ValueError(
                "`DeepGaussianProcess` can only be built out of `GPLayer` or"
                "`LatentVariableLayer`"
            )

        self._model_gpflux = model

        self._model_keras = model.as_training_model()
        self._model_keras.compile(self._optimizer)

    def __repr__(self) -> str:
        """"""
        return f"DeepGaussianProcess({self._model_gpflux!r}, {self.optimizer!r})"

    @property
    def model_gpflux(self) -> DeepGP:
        return self._model_gpflux

    @property
    def model_keras(self) -> tf.keras.Model:
        return self._model_keras

    @property
    def optimizer(self) -> tf.keras.optimizers.Optimizer:
        return self._optimizer

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        def get_samples(query_points: TensorType, num_samples: int) -> TensorType:
            samples = []
            for _ in range(num_samples):
                samples.append(sample_dgp(self.model_gpflux)(query_points))
            return tf.stack(samples)

        return get_samples(query_points, num_samples)

    def update(self, dataset: Dataset) -> None:
        inputs = dataset.query_points
        new_num_data = inputs.shape[0]
        self.model_gpflux.num_data = new_num_data

        # Update num_data for each layer, as well as make sure dataset shapes are ok
        for i, layer in enumerate(self.model_gpflux.f_layers):
            if hasattr(layer, "num_data"):
                layer.num_data = new_num_data

            if isinstance(layer, LatentVariableLayer):
                inputs = layer(inputs)
                continue

            if isinstance(layer.inducing_variable, InducingPoints):
                inducing_variable = layer.inducing_variable
            else:
                inducing_variable = layer.inducing_variable.inducing_variable

            if inputs.shape[-1] != inducing_variable.Z.shape[-1]:
                raise ValueError(
                    f"Shape {inputs.shape} of input to layer {layer} is incompatible with shape"
                    f" {inducing_variable.Z.shape} of that layer. Trailing dimensions must match."
                )

            if i == len(self.model_gpflux.f_layers) - 1:
                if dataset.observations.shape[-1] != layer.q_mu.shape[-1]:
                    raise ValueError(
                        f"Shape {dataset.observations.shape} of new observations is incompatible"
                        f" with shape {layer.q_mu.shape} of existing observations. Trailing"
                        f" dimensions must match."
                    )

            inputs = layer(inputs)

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the model with the specified `dataset`.
        :param dataset: The data with which to optimize the `model`.
        """
        self.model_keras.fit(
            {"inputs": dataset.query_points, "targets": dataset.observations}, **self.fit_args
        )

        # Reset lr in case there was an lr schedule
        self.optimizer.lr.assign(self.original_lr)

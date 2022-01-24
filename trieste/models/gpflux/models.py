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
from ..optimizer import BatchOptimizer
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
        optimizer: BatchOptimizer | None = None,
        continuous_optimisation: bool = True,
    ):
        """
        :param model: The underlying GPflux deep Gaussian process model.
        :param optimizer: The optimizer configuration for training the model. Defaults to
            :class:`~trieste.models.optimizer.BatchOptimizer` wrapper with
            :class:`~tf.optimizers.Adam`.
            This wrapper itself is not used, instead only its `optimizer` and `minimize_args` are
            used. Its optimizer is used when compiling a Keras GPflux model and `minimize_args` is
            a dictionary of arguments to be used in the Keras `fit` method. Defaults to
            using 100 epochs, batch size 100, and verbose 0. See
            https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
            arguments.
        :param continuous_optimisation: if True (default), the optimizer will keep track of the
            number of epochs across BO iterations and use this number as initial_epoch. This is
            essential to allow monitoring of model training across BO iterations.
        """
        for layer in model.f_layers:
            if not isinstance(layer, (GPLayer, LatentVariableLayer)):
                raise ValueError(
                    f"`DeepGaussianProcess` can only be built out of `GPLayer` or"
                    f"`LatentVariableLayer`, received {type(layer)} instead."
                )

        super().__init__(optimizer)

        if not isinstance(self.optimizer.optimizer, tf.optimizers.Optimizer):
            raise ValueError(
                f"Optimizer for `DeepGaussianProcess` must be an instance of a "
                f"`tf.optimizers.Optimizer` or `tf.keras.optimizers.Optimizer`, "
                f"received {type(self.optimizer.optimizer)} instead."
            )

        self.original_lr = self.optimizer.optimizer.lr.numpy()

        if not self.optimizer.minimize_args:
            self._fit_args: Dict[str, Any] = {
                "verbose": 0,
                "epochs": 100,
                "batch_size": 100,
            }
        else:
            self._fit_args = self.optimizer.minimize_args

        self._model_gpflux = model

        self._model_keras = model.as_training_model()
        self._model_keras.compile(self.optimizer.optimizer)
        self._absolute_epochs = 0
        self._continuous_optimisation = continuous_optimisation

    def __repr__(self) -> str:
        """"""
        return f"DeepGaussianProcess({self.model_gpflux!r}, {self.optimizer.optimizer!r})"

    @property
    def model_gpflux(self) -> DeepGP:
        return self._model_gpflux

    @property
    def model_keras(self) -> tf.keras.Model:
        return self._model_keras

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        samples = []
        for _ in range(num_samples):
            samples.append(sample_dgp(self.model_gpflux)(query_points))
        return tf.stack(samples)

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

            if (
                i == len(self.model_gpflux.f_layers) - 1
                and dataset.observations.shape[-1] != layer.q_mu.shape[-1]
            ):
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
        fit_args = dict(self._fit_args)

        # Tell optimizer how many epochs have been used before: the optimizer will "continue"
        # optimization across multiple BO iterations rather than start fresh at each iteration.
        # This allows us to monitor training across iterations.

        if "epochs" in fit_args:
            fit_args["epochs"] = fit_args["epochs"] + self._absolute_epochs

        hist = self.model_keras.fit(
            {"inputs": dataset.query_points, "targets": dataset.observations},
            **fit_args,
            initial_epoch=self._absolute_epochs,
        )

        if self._continuous_optimisation:
            self._absolute_epochs = self._absolute_epochs + len(hist.history["loss"])

        # Reset lr in case there was an lr schedule: a schedule will have change the learning rate,
        # so that the next time we call `optimize` the starting learning rate would be different.
        # Therefore we make sure the learning rate is set back to its initial value.
        self.optimizer.optimizer.lr.assign(self.original_lr)

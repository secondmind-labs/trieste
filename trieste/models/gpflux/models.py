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

import gpflow
from gpflow.base import Module
import tensorflow as tf
from gpflow.inducing_variables import InducingPoints
from gpflux.layers import GPLayer, LatentVariableLayer, GIGPLayer
from gpflux.models import DeepGP, GIDeepGP

from ...data import Dataset
from ...types import TensorType
from ..interfaces import TrainableProbabilisticModel
from .architectures import ModifiedLatentVariableLayer
from .interface import GPfluxPredictor
from .utils import sample_dgp


class DeepGaussianProcess(GPfluxPredictor, TrainableProbabilisticModel):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflux :class:`~gpflux.models.DeepGP` with
    :class:`GPLayer` or :class:`ModifiedLatentVariableLayer`: this class does not support
    e.g. keras layers

    Note: the user should remember to set `tf.keras.backend.set_floatx()` with the desired value
    (consistent with GPflow) so that dtype errors do not occur.
    """

    def __init__(
        self,
        model: DeepGP,
        optimizer: tf.optimizers.Optimizer | None = None,
        fit_args: Dict[Any] | None = None,
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

        if not all(
            [isinstance(layer, (GPLayer, ModifiedLatentVariableLayer)) for layer in model.f_layers]
        ):
            raise ValueError(
                "`DeepGaussianProcess` can only be built out of `GPLayer` or"
                "`ModifiedLatentVariableLayer`"
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

    def mc_posterior_mean(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return a Monte Carlo estimate of the posterior mean at the ``query_points``, based off
        ``num_samples`` samples.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The Monte Carlo estimate of the posterior mean. For a predictive distribution with
            event shape E, this has shape [..., N] + E.
        """
        samples = tf.tile(
            tf.expand_dims(query_points, 0), [num_samples, *[1] * tf.rank(query_points).numpy()]
        )
        for layer in self.model_gpflux.f_layers[:-1]:
            if isinstance(layer, LatentVariableLayer):
                samples = layer.compositor([samples, layer.prior.sample(tf.shape(samples)[:-1])])
                continue

            mean, var = layer.predict(samples, full_cov=False, full_output_cov=False)

            samples = mean + tf.sqrt(var) * tf.random.normal(
                [num_samples, 1, tf.shape(mean)[-1]], dtype=gpflow.default_float()
            )

        last_layer = self.model_gpflux.f_layers[-1]
        if isinstance(last_layer, LatentVariableLayer):
            mean_samples = last_layer.compositor(
                [samples, last_layer.prior.sample(tf.shape(samples)[:-1])]
            )
        else:
            mean_samples, _ = self.model_gpflux.f_layers[-1].predict(
                samples, full_cov=False, full_output_cov=False
            )

        return tf.reduce_mean(mean_samples, axis=0)

    def update(self, dataset: Dataset) -> None:
        inputs = dataset.query_points
        new_num_data = inputs.shape[0]
        self.model_gpflux.num_data = new_num_data

        # Update num_data for each layer, as well as make sure dataset shapes are ok
        for i, layer in enumerate(self.model_gpflux.f_layers):
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


class GlobalInducingDeepGaussianProcess(GPfluxPredictor, TrainableProbabilisticModel):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflux :class:`~gpflux.models.GIDeepGP`
    with :class:`GIGPLayer`: this class does not support e.g. keras layers or latent variable layers

    Note: the user should remember to set `tf.keras.backend.set_floatx()` with the desired value
    (consistent with GPflow) so that dtype errors do not occur.
    """

    def __init__(
        self,
        model: GIDeepGP,
        optimizer: tf.optimizers.Optimizer | None = None,
        fit_args: Dict[Any] | None = None,
    ):
        """
        :param model: The underlying GPflux global inducing deep Gaussian process model.
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

        if not all(
            [isinstance(layer, GIGPLayer) for layer in model.f_layers]
        ):
            raise ValueError(
                "`GlobalInducingDeepGaussianProcess` can only be built out of `GIGPLayer`"
            )

        self._model_gpflux = model

        self._model_keras = model.as_training_model()
        self._model_keras.compile(self._optimizer)

    def __repr__(self) -> str:
        """"""
        return f"GlobalInducingDeepGaussianProcess({self._model_gpflux!r}, {self.optimizer!r})"

    @property
    def model_gpflux(self) -> Module:
        return self._model_gpflux

    @property
    def model_keras(self) -> tf.keras.Model:
        return self._model_keras

    @property
    def optimizer(self) -> tf.keras.optimizers.Optimizer:
        return self._optimizer

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        return self.model_gpflux.sample(query_points, num_samples, consistent=True)

    def mc_posterior_mean(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return a Monte Carlo estimate of the posterior mean at the ``query_points``, based off
        ``num_samples`` samples.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The Monte Carlo estimate of the posterior mean. For a predictive distribution with
            event shape E, this has shape [..., N] + E.
        """
        samples = self.model_gpflux.sample(query_points, num_samples, consistent=False)

        return tf.reduce_mean(samples, axis=0)

    def update(self, dataset: Dataset) -> None:
        inputs = dataset.query_points
        new_num_data = inputs.shape[0]
        self.model_gpflux.num_data = new_num_data

        # Update num_data for each layer, as well as make sure dataset shapes are ok
        for i, layer in enumerate(self.model_gpflux.f_layers):
            layer.num_data = new_num_data

            if isinstance(layer, LatentVariableLayer):
                raise NotImplementedError("Currently we do not support latent variable layers for "
                                          "`GlobalInducingDeepGaussianProcess` models.")

        if tf.shape(inputs)[-1] != tf.shape(self.model_gpflux.inducing_data)[-1]:
            raise ValueError(
                f"Shape {inputs.shape} of input to the model is incompatible with shape"
                f" {self.model_gpflux.inducing_data.shape} of the model. Trailing dimensions must"
                f" match."
            )

        if tf.shape(dataset.observations)[-1] != tf.shape(self.model_gpflux.f_layers[-1].v)[-1]:
            raise ValueError(
                f"Shape {dataset.observations.shape} of new observations is incompatible with shape"
                f" {self.model_gpflux.f_layers[-1].v.shape} of existing observations. Trailing "
                f" dimensions must match."
            )

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

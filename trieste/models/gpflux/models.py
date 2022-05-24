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

import tensorflow as tf
from gpflow.inducing_variables import InducingPoints
from gpflux.layers import GPLayer, LatentVariableLayer
from gpflux.models import DeepGP

from ...data import Dataset
from ...types import TensorType
from ..interfaces import (
    HasReparamSampler,
    HasTrajectorySampler,
    ReparametrizationSampler,
    TrainableProbabilisticModel,
    TrajectorySampler,
)
from ..optimizer import KerasOptimizer
from .interface import GPfluxPredictor
from .sampler import (
    DeepGaussianProcessDecoupledTrajectorySampler,
    DeepGaussianProcessReparamSampler,
)


class DeepGaussianProcess(
    GPfluxPredictor, TrainableProbabilisticModel, HasReparamSampler, HasTrajectorySampler
):
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
        optimizer: KerasOptimizer | None = None,
        num_rff_features: int = 1000,
        continuous_optimisation: bool = True,
    ):
        """
        :param model: The underlying GPflux deep Gaussian process model.
        :param optimizer: The optimizer configuration for training the model. Defaults to
            :class:`~trieste.models.optimizer.KerasOptimizer` wrapper with
            :class:`~tf.optimizers.Adam` optimizer. The ``optimizer`` argument to the wrapper is
            used when compiling the model and ``fit_args`` is a dictionary of arguments to be used
            in the Keras ``fit`` method. Defaults to 400 epochs, batch size of 1000, and verbose 0.
            A custom callback that reduces the optimizer learning rate is used as well. See
            https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
            arguments.
        :param num_rff_features: The number of random Fourier features used to approximate the
            kernel when calling :meth:`trajectory_sampler`. We use a default of 1000 as it typically
            performs well for a wide range of kernels. Note that very smooth kernels (e.g. RBF) can
            be well-approximated with fewer features.
        :param continuous_optimisation: if True (default), the optimizer will keep track of the
            number of epochs across BO iterations and use this number as initial_epoch. This is
            essential to allow monitoring of model training across BO iterations.
        :raise ValueError: If ``model`` has unsupported layers, ``num_rff_features`` is less than 0,
            or if the ``optimizer`` is not of a supported type.
        """
        for layer in model.f_layers:
            if not isinstance(layer, (GPLayer, LatentVariableLayer)):
                raise ValueError(
                    f"`DeepGaussianProcess` can only be built out of `GPLayer` or"
                    f"`LatentVariableLayer`, received {type(layer)} instead."
                )

        super().__init__(optimizer)

        if num_rff_features <= 0:
            raise ValueError(
                f"num_rff_features must be greater or equal to zero, got {num_rff_features}."
            )
        self._num_rff_features = num_rff_features

        if not isinstance(self.optimizer.optimizer, tf.optimizers.Optimizer):
            raise ValueError(
                f"Optimizer for `DeepGaussianProcess` must be an instance of a "
                f"`tf.optimizers.Optimizer` or `tf.keras.optimizers.Optimizer`, "
                f"received {type(self.optimizer.optimizer)} instead."
            )

        self.original_lr = self.optimizer.optimizer.lr.numpy()

        epochs = 400

        def scheduler(epoch: int, lr: float) -> float:
            if epoch == epochs // 2:
                return lr * 0.1
            else:
                return lr

        if not self.optimizer.fit_args:
            self.optimizer.fit_args = {
                "verbose": 0,
                "epochs": epochs,
                "batch_size": 1000,
                "callbacks": [tf.keras.callbacks.LearningRateScheduler(scheduler)],
            }

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
        trajectory = self.trajectory_sampler().get_trajectory()
        expanded_query_points = tf.expand_dims(query_points, -2)
        tiled_query_points = tf.tile(expanded_query_points, [1, num_samples, 1])
        return tf.expand_dims(tf.transpose(trajectory(tiled_query_points), [1, 0]), -1)

    def reparam_sampler(self, num_samples: int) -> ReparametrizationSampler[GPfluxPredictor]:
        """
        Return a reparametrization sampler for a :class:`DeepGaussianProcess` model.

        :param num_samples: The number of samples to obtain.
        :return: The reparametrization sampler.
        """
        return DeepGaussianProcessReparamSampler(num_samples, self)

    def trajectory_sampler(self) -> TrajectorySampler[GPfluxPredictor]:
        """
        Return a trajectory sampler. For :class:`DeepGaussianProcess`, we build
        trajectories using the GPflux default sampler.

        :return: The trajectory sampler.
        """
        return DeepGaussianProcessDecoupledTrajectorySampler(self, self._num_rff_features)

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
        fit_args = dict(self.optimizer.fit_args)

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

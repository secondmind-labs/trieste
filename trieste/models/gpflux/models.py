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

from typing import Any, Callable, Optional

import dill
import gpflow
import tensorflow as tf
from gpflow.inducing_variables import InducingPoints
from gpflow.utilities.traversal import _merge_leaf_components, leaf_components
from gpflux.layers import GPLayer, LatentVariableLayer
from gpflux.models import DeepGP
from tensorflow.python.keras.callbacks import Callback

from ... import logging
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
    """

    def __init__(
        self,
        model: DeepGP | Callable[[], DeepGP],
        optimizer: KerasOptimizer | None = None,
        num_rff_features: int = 1000,
        continuous_optimisation: bool = True,
    ):
        """
        :param model: The underlying GPflux deep Gaussian process model. Passing in a named closure
            rather than a model can help when copying or serialising.
        :param optimizer: The optimizer wrapper with necessary specifications for compiling and
            training the model. Defaults to :class:`~trieste.models.optimizer.KerasOptimizer` with
            :class:`~tf.optimizers.Adam` optimizer, mean squared error metric and a dictionary of
            default arguments for the Keras `fit` method: 400 epochs, batch size of 1000, and
            verbose 0. A custom callback that reduces the optimizer learning rate is used as well.
            See https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
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
        if isinstance(model, DeepGP):
            self._model_closure = None
        else:
            self._model_closure = model
            model = model()

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

        if self.optimizer.metrics is None:
            self.optimizer.metrics = ["mse"]

        self._model_gpflux = model
        # inputs and targets need to be redone with a float64 dtype to avoid setting the keras
        # backend to float64, this is likely to be fixed in GPflux, see issue:
        # https://github.com/secondmind-labs/GPflux/issues/76
        self._model_gpflux.inputs = tf.keras.Input(
            tuple(self._model_gpflux.inputs.shape[:-1]),
            name=self._model_gpflux.inputs.name,
            dtype=tf.float64,
        )
        self._model_gpflux.targets = tf.keras.Input(
            tuple(self._model_gpflux.targets.shape[:-1]),
            name=self._model_gpflux.targets.name,
            dtype=tf.float64,
        )
        self._model_keras = model.as_training_model()
        self._model_keras.compile(self.optimizer.optimizer, metrics=self.optimizer.metrics)
        self._absolute_epochs = 0
        self._continuous_optimisation = continuous_optimisation

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()

        # when using a model closure, store the model parameters, not the model itself
        if self._model_closure is not None:
            state["_model_gpflux"] = gpflow.utilities.parameter_dict(self._model_gpflux)
            state["_model_keras"] = gpflow.utilities.parameter_dict(self._model_keras)

        # use to_json and get_weights to save any optimizer fit_arg callback models
        callbacks: list[Callback] = self._optimizer.fit_args.get("callbacks", [])
        callback: Callback
        saved_models: list[KerasOptimizer] = []
        tensorboard_writers: list[dict[str, Any]] = []
        try:
            for callback in callbacks:
                # serialize the callback models before pickling the optimizer
                saved_models.append(callback.model)
                if callback.model is self._model_keras:
                    # no need to serialize the main model, just use a special value instead
                    callback.model = ...
                elif callback.model:
                    callback.model = (callback.model.to_json(), callback.model.get_weights())
                # don't pickle tensorboard writers either; they'll be recreated when needed
                if isinstance(callback, tf.keras.callbacks.TensorBoard):
                    tensorboard_writers.append(callback._writers)
                    callback._writers = {}
            state["_optimizer"] = dill.dumps(state["_optimizer"])
        except Exception as e:
            raise NotImplementedError(
                "Failed to copy DeepGaussianProcess optimizer due to unsupported callbacks."
            ) from e
        finally:
            # revert original state, even if the pickling failed
            for callback, model in zip(self._optimizer.fit_args.get("callbacks", []), saved_models):
                callback.model = model
            for callback, writers in zip(
                (cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.TensorBoard)),
                tensorboard_writers,
            ):
                callback._writers = writers

        # do the same thing for the history callback
        if self._model_keras.history:
            history_model = self._model_keras.history.model
            try:
                if history_model is self._model_keras:
                    # no need to serialize the main model, just use a special value instead
                    self._model_keras.history.model = ...
                elif history_model:
                    self._model_keras.history.model = (
                        history_model.to_json(),
                        history_model.get_weights(),
                    )
                state["_history"] = dill.dumps(self._model_keras.history)
            finally:
                self._model_keras.history.model = history_model

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

        # regenerate the models using the model closure
        if self._model_closure is not None:
            dgp: DeepGP = state["_model_closure"]()
            self._model_gpflux = dgp
            # inputs and targets need to be redone with a float64 dtype to avoid setting the keras
            # backend to float64, this is likely to be fixed in GPflux, see issue:
            # https://github.com/secondmind-labs/GPflux/issues/76
            self._model_gpflux.inputs = tf.keras.Input(
                tuple(self._model_gpflux.inputs.shape[:-1]),
                name=self._model_gpflux.inputs.name,
                dtype=tf.float64,
            )
            self._model_gpflux.targets = tf.keras.Input(
                tuple(self._model_gpflux.targets.shape[:-1]),
                name=self._model_gpflux.targets.name,
                dtype=tf.float64,
            )
            self._model_keras = dgp.as_training_model()

        # unpickle the optimizer, and restore all the callback models
        self._optimizer = dill.loads(self._optimizer)
        for callback in self._optimizer.fit_args.get("callbacks", []):
            if callback.model is ...:
                callback.set_model(self._model_keras)
            elif callback.model:
                model_json, weights = callback.model
                model = tf.keras.models.model_from_json(model_json)
                model.set_weights(weights)
                callback.set_model(model)

        # recompile the model
        self._model_keras.compile(self._optimizer.optimizer)

        # assign the model parameters
        if self._model_closure is not None:
            gpflow.utilities.multiple_assign(self._model_gpflux, state["_model_gpflux"])
            gpflow.utilities.multiple_assign(self._model_keras, state["_model_keras"])

        # restore the history (including any model it contains)
        if "_history" in state:
            self._model_keras.history = dill.loads(state["_history"])
            if self._model_keras.history.model is ...:
                self._model_keras.history.set_model(self._model_keras)
            elif self._model_keras.history.model:
                model_json, weights = self._model_keras.history.model
                model = tf.keras.models.model_from_json(model_json)
                model.set_weights(weights)
                self._model_keras.history.set_model(model)

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
        expanded_query_points = tf.expand_dims(query_points, -2)  # [N, 1, D]
        tiled_query_points = tf.tile(expanded_query_points, [1, num_samples, 1])  # [N, S, D]
        return tf.transpose(trajectory(tiled_query_points), [1, 0, 2])  # [S, N, L]

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

    def log(self, dataset: Optional[Dataset] = None) -> None:
        """
        Log model training information at a given optimization step to the Tensorboard.
        We log a few summary statistics of losses, layer KL divergences and metrics (as provided in
        ``optimizer``): ``final`` value at the end of the training, ``diff`` value as a difference
        between inital and final epoch. We also log epoch statistics, but as histograms, rather
        than time series. We also log several training data based metrics, such as root mean square
        error between predictions and observations and several others.

        For custom logs user will need to subclass the model and overwrite this method.

        :param dataset: Optional data that can be used to log additional data-based model summaries.
        """
        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                logging.scalar("epochs/num_epochs", len(self.model_keras.history.epoch))
                # kernel parameters
                for idx, layer in enumerate(self.model_gpflux.f_layers):
                    kernel_components = _merge_leaf_components(leaf_components(layer.kernel))
                    for k, v in kernel_components.items():
                        if v.trainable:
                            if tf.rank(v) == 0:
                                logging.scalar(f"layer[{idx}]/kernel.{k}", v)
                            elif tf.rank(v) == 1:
                                for i, vi in enumerate(v):
                                    logging.scalar(f"layer[{idx}]/kernel.{k}[{i}]", vi)
                # likelihood parameters
                likelihood_components = _merge_leaf_components(
                    leaf_components(self.model_gpflux.likelihood_layer.likelihood)
                )
                for k, v in likelihood_components.items():
                    if v.trainable:
                        logging.scalar(f"likelihood/{k}", v)

                # losses and metrics
                for k, v in self.model_keras.history.history.items():
                    logging.histogram(f"{k}/epoch", lambda: v)
                    logging.scalar(f"{k}/final", lambda: v[-1])
                    logging.scalar(f"{k}/diff", lambda: v[0] - v[-1])
                # training data based diagnostics
                if dataset:
                    predict = self.predict(dataset.query_points)
                    # training accuracy
                    diffs = dataset.observations - predict[0]
                    z_residuals = diffs / tf.math.sqrt(predict[1])
                    logging.histogram("accuracy/absolute_errors", tf.math.abs(diffs))
                    logging.histogram("accuracy/z_residuals", z_residuals)
                    logging.scalar(
                        "accuracy/root_mean_square_error", tf.math.sqrt(tf.reduce_mean(diffs ** 2))
                    )
                    logging.scalar(
                        "accuracy/mean_absolute_error", tf.reduce_mean(tf.math.abs(diffs))
                    )
                    logging.scalar("accuracy/z_residuals_std", tf.math.reduce_std(z_residuals))
                    # training variance
                    variance_error = predict[1] - diffs ** 2
                    logging.histogram("variance/predict_variance", predict[1])
                    logging.histogram("variance/variance_error", variance_error)
                    logging.scalar("variance/predict_variance_mean", tf.reduce_mean(predict[1]))
                    logging.scalar(
                        "variance/root_mean_variance_error",
                        tf.math.sqrt(tf.reduce_mean(variance_error ** 2)),
                    )
                    # data stats
                    empirical_variance = tf.math.reduce_variance(dataset.observations)
                    logging.scalar("variance/empirical", empirical_variance)

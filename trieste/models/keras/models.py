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

import re
from typing import Any, Dict, Mapping, Optional

import dill
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
from check_shapes import check_shapes
from gpflow.keras import tf_keras
from tensorflow.python.keras.callbacks import Callback

from ... import logging
from ...data import Dataset
from ...space import EncoderFunction
from ...types import TensorType
from ...utils import flatten_leading_dims
from ..interfaces import (
    EncodedSupportsPredictY,
    EncodedTrainableProbabilisticModel,
    HasTrajectorySampler,
    TrajectorySampler,
)
from ..optimizer import KerasOptimizer
from ..utils import write_summary_data_based_metrics
from .architectures import KerasEnsemble, MultivariateNormalTriL
from .interface import DeepEnsembleModel, KerasPredictor
from .sampler import DeepEnsembleTrajectorySampler
from .utils import negative_log_likelihood, sample_model_index, sample_with_replacement


class DeepEnsemble(
    KerasPredictor,
    EncodedTrainableProbabilisticModel,
    DeepEnsembleModel,
    HasTrajectorySampler,
    EncodedSupportsPredictY,
):
    """
    A :class:`~trieste.model.TrainableProbabilisticModel` wrapper for deep ensembles built using
    Keras.

    Deep ensembles are ensembles of deep neural networks that have been found to have good
    representation of uncertainty in practice (<cite data-cite="lakshminarayanan2017simple"/>).
    This makes them a potentially attractive model for Bayesian optimization for use-cases with
    large number of observations, non-stationary objective functions and need for fast predictions,
    in which standard Gaussian process models are likely to struggle. The model consists of simple
    fully connected multilayer probabilistic networks as base learners, with Gaussian distribution
    as a final layer, using the negative log-likelihood loss for training the networks. The
    model relies on differences in random initialization of weights for generating diversity among
    base learners.

    The original formulation of the model does not include boostrapping of the data. The authors
    found that it does not improve performance the model. We include bootstrapping as an option
    as later work that more precisely measured uncertainty quantification found that boostrapping
    does help with uncertainty representation (see <cite data-cite="osband2021epistemic"/>).

    We provide classes for constructing ensembles using Keras
    (:class:`~trieste.models.keras.KerasEnsemble`) in the `architectures` package that should be
    used with the :class:`~trieste.models.keras.DeepEnsemble` wrapper. There we also provide a
    :class:`~trieste.models.keras.GaussianNetwork` base learner following the original
    formulation in <cite data-cite="lakshminarayanan2017simple"/>, but any user-specified network
    can be supplied, as long as it has a Gaussian distribution as a final layer and follows the
    :class:`~trieste.models.keras.KerasEnsembleNetwork` interface.

    A word of caution in case a learning rate scheduler is used in ``fit_args`` to
    :class:`KerasOptimizer` optimizer instance. Typically one would not want to continue with the
    reduced learning rate in the subsequent Bayesian optimization step. Hence, we reset the
    learning rate to the original one after calling the ``fit`` method. In case this is not the
    behaviour you would like, you will need to subclass the model and overwrite the
    :meth:`optimize` method.

    Currently, we do not support setting up the model with dictionary config.
    """

    def __init__(
        self,
        model: KerasEnsemble,
        optimizer: Optional[KerasOptimizer] = None,
        bootstrap: bool = False,
        diversify: bool = False,
        continuous_optimisation: bool = True,
        compile_args: Optional[Mapping[str, Any]] = None,
        encoder: EncoderFunction | None = None,
    ) -> None:
        """
        :param model: A Keras ensemble model with probabilistic networks as ensemble members. The
            model has to be built but not compiled.
        :param optimizer: The optimizer wrapper with necessary specifications for compiling and
            training the model. Defaults to :class:`~trieste.models.optimizer.KerasOptimizer` with
            :class:`~tf.optimizers.Adam` optimizer, negative log likelihood loss, mean squared
            error metric and a dictionary of default arguments for Keras `fit` method: 3000 epochs,
            batch size 16, early stopping callback with patience of 50, and verbose 0.
            See https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
            arguments.
        :param bootstrap: Sample with replacement data for training each network in the ensemble.
            By default, set to `False`.
        :param diversify: Whether to use quantiles from the approximate Gaussian distribution of
            the ensemble as trajectories instead of mean predictions when calling
            :meth:`trajectory_sampler`. This mode can be used to increase the diversity
            in case of optimizing very large batches of trajectories. By
            default, set to `False`.
        :param continuous_optimisation: If True (default), the optimizer will keep track of the
            number of epochs across BO iterations and use this number as initial_epoch. This is
            essential to allow monitoring of model training across BO iterations.
        :param compile_args: Keyword arguments to pass to the ``compile`` method of the
            Keras model (:class:`~tf.keras.Model`).
            See https://keras.io/api/models/model_training_apis/#compile-method for a
            list of possible arguments. The ``optimizer``, ``loss`` and ``metrics`` arguments
            must not be included.
        :param encoder: Optional encoder with which to transform query points before
            generating predictions.
        :raise ValueError: If ``model`` is not an instance of
            :class:`~trieste.models.keras.KerasEnsemble`, or ensemble has less than two base
            learners (networks), or `compile_args` contains disallowed arguments.
        """
        if model.ensemble_size < 2:
            raise ValueError(f"Ensemble size must be greater than 1 but got {model.ensemble_size}.")

        super().__init__(optimizer, encoder)

        if compile_args is None:
            compile_args = {}

        if not {"optimizer", "loss", "metrics"}.isdisjoint(compile_args):
            raise ValueError(
                "optimizer, loss and metrics arguments must not be included in compile_args."
            )

        if not self.optimizer.fit_args:
            self.optimizer.fit_args = {
                "verbose": 0,
                "epochs": 3000,
                "batch_size": 16,
                "callbacks": [
                    tf_keras.callbacks.EarlyStopping(
                        monitor="loss", patience=50, restore_best_weights=True
                    )
                ],
            }

        if self.optimizer.loss is None:
            self.optimizer.loss = negative_log_likelihood

        if self.optimizer.metrics is None:
            self.optimizer.metrics = ["mse"]

        model.model.compile(
            optimizer=self.optimizer.optimizer,
            loss=[self.optimizer.loss] * model.ensemble_size,
            metrics=[self.optimizer.metrics] * model.ensemble_size,
            **compile_args,
        )

        if not isinstance(
            self.optimizer.optimizer.lr, tf_keras.optimizers.schedules.LearningRateSchedule
        ):
            self.original_lr = self.optimizer.optimizer.lr.numpy()
        self._absolute_epochs = 0
        self._continuous_optimisation = continuous_optimisation

        self._model = model
        self._bootstrap = bootstrap
        self._diversify = diversify

    def __repr__(self) -> str:
        """"""
        return (
            f"DeepEnsemble({self.model!r}, {self.optimizer!r}, {self._bootstrap!r}, "
            f"{self._diversify!r})"
        )

    @property
    def model(self) -> tf_keras.Model:
        """Returns compiled Keras ensemble model."""
        return self._model.model

    @property
    def ensemble_size(self) -> int:
        """
        Returns the size of the ensemble, that is, the number of base learners or individual neural
        network models in the ensemble.
        """
        return self._model.ensemble_size

    @property
    def num_outputs(self) -> int:
        """
        Returns the number of outputs trained on by each member network.
        """
        return self._model.num_outputs

    def prepare_dataset(
        self, dataset: Dataset
    ) -> tuple[Dict[str, TensorType], Dict[str, TensorType]]:
        """
        Transform ``dataset`` into inputs and outputs with correct names that can be used for
        training the :class:`KerasEnsemble` model.

        If ``bootstrap`` argument in the :class:`~trieste.models.keras.DeepEnsemble` is set to
        `True`, data will be additionally sampled with replacement, independently for
        each network in the ensemble.

        :param dataset: A dataset with ``query_points`` and ``observations`` tensors.
        :return: A dictionary with input data and a dictionary with output data.
        """
        inputs = {}
        outputs = {}
        for index in range(self.ensemble_size):
            if self._bootstrap:
                resampled_data = sample_with_replacement(dataset)
            else:
                resampled_data = dataset
            input_name = self.model.input_names[index]
            output_name = self.model.output_names[index]
            inputs[input_name], outputs[output_name] = resampled_data.astuple()

        return inputs, outputs

    def prepare_query_points(self, query_points: TensorType) -> Dict[str, TensorType]:
        """
        Transform ``query_points`` into inputs with correct names that can be used for
        predicting with the model.

        :param query_points: A tensor with ``query_points``.
        :return: A dictionary with query_points prepared for predictions.
        """
        inputs = {}
        for index in range(self.ensemble_size):
            inputs[self.model.input_names[index]] = query_points

        return inputs

    def ensemble_distributions(self, query_points: TensorType) -> tuple[tfd.Distribution, ...]:
        """
        Return distributions for each member of the ensemble.

        :param query_points: The points at which to return distributions.
        :return: The distributions for the observations at the specified
            ``query_points`` for each member of the ensemble.
        """
        x_transformed: dict[str, TensorType] = self.prepare_query_points(query_points)
        return self._model.model(x_transformed)

    def predict_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        Returns mean and variance at ``query_points`` for the whole ensemble. Variance here is
        the epistemic uncertainty, computed as the variance of the ensemble means.

        Following <cite data-cite="lakshminarayanan2017simple"/> we treat the ensemble as a
        uniformly-weighted Gaussian mixture model and combine the predictions as

        .. math:: p(y|\mathbf{x}) = M^{-1} \Sum_{m=1}^M \mathcal{N}
            (\mu_{\theta_m}(\mathbf{x}),\,\sigma_{\theta_m}^{2}(\mathbf{x}))

        We further approximate the ensemble prediction as a Gaussian whose mean and variance
        are respectively the mean and variance of the mixture, given by

        .. math:: \mu_{*}(\mathbf{x}) = M^{-1} \Sum_{m=1}^M \mu_{\theta_m}(\mathbf{x})

        .. math:: \sigma^2_{*}(\mathbf{x}) = M^{-1} \Sum_{m=1}^M (\mu^2_{\theta_m}(\mathbf{x}))
            - \mu^2_{*}(\mathbf{x})

        This method assumes that the final layer in each member of the ensemble is
        probabilistic, an instance of :class:`~tfp.distributions.Distribution`. In particular, given
        the nature of the approximations stated above the final layer should be a Gaussian
        distribution with `mean` and `variance` methods.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points``.
        """
        ensemble_means, _ = self.predict_ensemble_encoded(query_points)
        predicted_means = tf.math.reduce_mean(ensemble_means, axis=-3)
        epistemic_variance = tf.math.reduce_variance(ensemble_means, axis=-3)

        return predicted_means, epistemic_variance

    def predict_y_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        Returns mean and variance at ``query_points`` for the whole ensemble. Variance here is
        the total uncertainty, computed as the sum of the epistemic and aleatoric uncertainties.

        Following <cite data-cite="lakshminarayanan2017simple"/> we treat the ensemble as a
        uniformly-weighted Gaussian mixture model and combine the predictions as

        .. math:: p(y|\mathbf{x}) = M^{-1} \Sum_{m=1}^M \mathcal{N}
            (\mu_{\theta_m}(\mathbf{x}),\,\sigma_{\theta_m}^{2}(\mathbf{x}))

        We further approximate the ensemble prediction as a Gaussian whose mean and variance
        are respectively the mean and variance of the mixture, given by

        .. math:: \mu_{*}(\mathbf{x}) = M^{-1} \Sum_{m=1}^M \mu_{\theta_m}(\mathbf{x})

        .. math:: \sigma^2_{*}(\mathbf{x}) = M^{-1} \Sum_{m=1}^M (\sigma_{\theta_m}^{2}(\mathbf{x})
            + \mu^2_{\theta_m}(\mathbf{x})) - \mu^2_{*}(\mathbf{x})

        This method assumes that the final layer in each member of the ensemble is
        probabilistic, an instance of :class:`~tfp.distributions.Distribution`. In particular, given
        the nature of the approximations stated above the final layer should be a Gaussian
        distribution with `mean` and `variance` methods.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points``.
        """
        ensemble_means, ensemble_vars = self.predict_ensemble_encoded(query_points)
        predicted_means = tf.math.reduce_mean(ensemble_means, axis=-3)
        epistemic_variance = tf.math.reduce_variance(ensemble_means, axis=-3)
        aleatoric_variance = tf.math.reduce_mean(ensemble_vars, axis=-3)

        return predicted_means, epistemic_variance + aleatoric_variance

    def predict_noise_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Returns mean and variance of the noise at ``query_points`` for the whole ensemble.
        Mean is the mean of the variance across the ensemble, variance is the variance of the
        variance across the ensemble. Assumes that the data is encoded already.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the aleatoric noise at the specified
            ``query_points``.
        """
        _, ensemble_vars = self.predict_ensemble_encoded(query_points)
        aleatoric_variance_mean = tf.math.reduce_mean(ensemble_vars, axis=-3)
        aleatoric_variance_variance = tf.math.reduce_variance(ensemble_vars, axis=-3)

        return aleatoric_variance_mean, aleatoric_variance_variance

    @check_shapes(
        "query_points: [batch..., D]",
        "return[0]: [batch..., E...]",
        "return[1]: [batch..., E...]",
    )
    def predict_noise(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Returns mean and variance of the noise at ``query_points`` for the whole ensemble.
        Mean is the mean of the variance across the ensemble, variance is the variance of the
        variance across the ensemble. Data is encoded before prediction if encoder is provided.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the aleatoric noise at the specified
            ``query_points``.
        """
        return self.predict_noise_encoded(self.encode(query_points))

    @property
    def dtype(self) -> tf.DType:
        """The prediction dtype."""
        return self._model.output_dtype

    def predict_ensemble_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Returns mean and variance at ``query_points`` for each member of the ensemble. First tensor
        is the mean and second is the variance, where each has shape [..., M, N, 1], where M is
        the ``ensemble_size``.

        This method assumes that the final layer in each member of the ensemble is
        probabilistic, an instance of :class:`¬tfp.distributions.Distribution`, in particular
        `mean` and `variance` methods should be available.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points`` for each member of the ensemble.
        """
        input_dims = min(len(query_points.shape), len(self.model.input_shape[0]))
        flat_x, unflatten = flatten_leading_dims(query_points, output_dims=input_dims)
        ensemble_distributions = self.ensemble_distributions(flat_x)
        predicted_means = tf.stack(
            [unflatten(dist.mean()) for dist in ensemble_distributions], axis=-3
        )
        predicted_vars = tf.stack(
            [unflatten(dist.variance()) for dist in ensemble_distributions], axis=-3
        )

        return predicted_means, predicted_vars

    def predict_ensemble(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.predict_ensemble_encoded(self.encode(query_points))

    def sample_encoded(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples at ``query_points``. We use the mixture approximation in
        :meth:`predict` for ``query_points`` and sample ``num_samples`` times from a Gaussian
        distribution given by the predicted mean and variance.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples. For a predictive distribution with event shape E, this has shape
            [..., S, N] + E, where S is the number of samples.
        """

        predicted_means, predicted_vars = self.predict_encoded(query_points)
        normal = tfp.distributions.Normal(predicted_means, tf.sqrt(predicted_vars))
        samples = normal.sample(num_samples)

        return samples  # [num_samples, len(query_points), 1]

    def sample_ensemble_encoded(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples at ``query_points``. Each sample is taken from a Gaussian
        distribution given by the predicted mean and variance of a randomly chosen network in the
        ensemble. This avoids using the Gaussian mixture approximation and samples directly from
        individual Gaussian distributions given by each network in the ensemble.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples. For a predictive distribution with event shape E, this has shape
            [..., S, N] + E, where S is the number of samples.
        """
        ensemble_distributions = self.ensemble_distributions(query_points)
        network_indices = sample_model_index(self.ensemble_size, num_samples)

        stacked_samples = []
        for i in range(num_samples):
            stacked_samples.append(ensemble_distributions[network_indices[i]].sample())
        samples = tf.stack(stacked_samples, axis=0)

        return samples  # [num_samples, len(query_points), 1]

    def sample_ensemble(self, query_points: TensorType, num_samples: int) -> TensorType:
        return self.sample_ensemble_encoded(self.encode(query_points), num_samples)

    def trajectory_sampler(self) -> TrajectorySampler[DeepEnsemble]:
        """
        Return a trajectory sampler. For :class:`DeepEnsemble`, we use an ensemble
        sampler that randomly picks a network from the ensemble and uses its predicted means
        for generating a trajectory, or optionally randomly sampled quantiles rather than means.

        :return: The trajectory sampler.
        """
        return DeepEnsembleTrajectorySampler(self, self._diversify)

    def update_encoded(self, dataset: Dataset) -> None:
        """
        Neural networks are parametric models and do not need to update data.
        `TrainableProbabilisticModel` interface, however, requires an update method, so
        here we simply pass the execution.
        """
        return

    def optimize_encoded(self, dataset: Dataset) -> tf_keras.callbacks.History:
        """
        Optimize the underlying Keras ensemble model with the specified ``dataset``.

        Optimization is performed by using the Keras `fit` method, rather than applying the
        optimizer and using the batches supplied with the optimizer wrapper. User can pass
        arguments to the `fit` method through ``minimize_args`` argument in the optimizer wrapper.
        These default to using 100 epochs, batch size 100, and verbose 0. See
        https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
        arguments.

        Note that optimization does not return the result, instead optimization results are
        stored in a history attribute of the model object.

        :param dataset: The data with which to optimize the model.
        """
        fit_args = dict(self.optimizer.fit_args)

        # Tell optimizer how many epochs have been used before: the optimizer will "continue"
        # optimization across multiple BO iterations rather than start fresh at each iteration.
        # This allows us to monitor training across iterations.

        if "epochs" in fit_args:
            fit_args["epochs"] = fit_args["epochs"] + self._absolute_epochs

        x, y = self.prepare_dataset(dataset)
        history = self.model.fit(
            x=x,
            y=y,
            **fit_args,
            initial_epoch=self._absolute_epochs,
        )
        if self._continuous_optimisation:
            self._absolute_epochs = self._absolute_epochs + len(history.history["loss"])

        # Reset lr in case there was an lr schedule: a schedule will have changed the learning
        # rate, so that the next time we call `optimize` the starting learning rate would be
        # different. Therefore, we make sure the learning rate is set back to its initial value.
        # However, this is not needed for `LearningRateSchedule` instances.
        if not isinstance(
            self.optimizer.optimizer.lr, tf_keras.optimizers.schedules.LearningRateSchedule
        ):
            self.optimizer.optimizer.lr.assign(self.original_lr)

        return history

    def log(self, dataset: Optional[Dataset] = None) -> None:
        """
        Log model training information at a given optimization step to the Tensorboard.
        We log several summary statistics of losses and metrics given in ``fit_args`` to
        ``optimizer`` (final, difference between inital and final loss, min and max). We also log
        epoch statistics, but as histograms, rather than time series. We also log several training
        data based metrics, such as root mean square error between predictions and observations,
        and several others.

        We do not log statistics of individual models in the ensemble unless specifically switched
        on with ``trieste.logging.set_summary_filter(lambda name: True)``.

        For custom logs user will need to subclass the model and overwrite this method.

        :param dataset: Optional data that can be used to log additional data-based model summaries.
        """
        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                logging.scalar("epochs/num_epochs", len(self.model.history.epoch))
                for k, v in self.model.history.history.items():
                    KEY_SPLITTER = {
                        # map history keys to prefix and suffix
                        "loss": ("loss", ""),
                        r"(?P<model>model_\d+)_output_loss": ("loss", r"_\g<model>"),
                        r"(?P<model>model_\d+)_output_(?P<metric>.+)": (
                            r"\g<metric>",
                            r"_\g<model>",
                        ),
                    }
                    for pattern, (pre_sub, post_sub) in KEY_SPLITTER.items():
                        if re.match(pattern, k):
                            pre = re.sub(pattern, pre_sub, k)
                            post = re.sub(pattern, post_sub, k)
                            break
                    else:
                        # unrecognised history key; ignore
                        continue
                    if "model" in post:
                        if not logging.include_summary("_ensemble"):
                            break
                        pre = pre + "/_ensemble"
                    logging.histogram(f"{pre}/epoch{post}", lambda: v)
                    logging.scalar(f"{pre}/final{post}", lambda: v[-1])
                    logging.scalar(f"{pre}/diff{post}", lambda: v[0] - v[-1])
                    logging.scalar(f"{pre}/min{post}", lambda: tf.reduce_min(v))
                    logging.scalar(f"{pre}/max{post}", lambda: tf.reduce_max(v))
                if dataset:
                    write_summary_data_based_metrics(
                        dataset=dataset, model=self, prefix="training_"
                    )
                    if logging.include_summary("_ensemble"):
                        predict_ensemble_variance = self.predict_ensemble(dataset.query_points)[1]
                        for i in range(predict_ensemble_variance.shape[0]):
                            logging.histogram(
                                f"variance/_ensemble/predict_variance_model_{i}",
                                predict_ensemble_variance[i, ...],
                            )
                            logging.scalar(
                                f"variance/_ensemble/predict_variance_mean_model_{i}",
                                tf.reduce_mean(predict_ensemble_variance[i, ...]),
                            )

    def __getstate__(self) -> dict[str, Any]:
        # use to_json and get_weights to save any optimizer fit_arg callback models
        state = self.__dict__.copy()
        if self._optimizer:
            callbacks: list[Callback] = self._optimizer.fit_args.get("callbacks", [])
            saved_models: list[KerasOptimizer] = []
            tensorboard_writers: list[dict[str, Any]] = []
            try:
                for callback in callbacks:
                    # serialize the callback models before pickling the optimizer
                    saved_models.append(callback.model)
                    if callback.model is self.model:
                        # no need to serialize the main model, just use a special value instead
                        callback.model = ...
                    elif callback.model:
                        callback.model = (callback.model.to_json(), callback.model.get_weights())
                    # don't pickle tensorboard writers either; they'll be recreated when needed
                    if isinstance(callback, tf_keras.callbacks.TensorBoard):
                        tensorboard_writers.append(callback._writers)
                        callback._writers = {}
                state["_optimizer"] = dill.dumps(state["_optimizer"])
            except Exception as e:
                raise NotImplementedError(
                    "Failed to copy DeepEnsemble optimizer due to unsupported callbacks."
                ) from e
            finally:
                # revert original state, even if the pickling failed
                for callback, model in zip(callbacks, saved_models):
                    callback.model = model
                for callback, writers in zip(
                    (cb for cb in callbacks if isinstance(cb, tf_keras.callbacks.TensorBoard)),
                    tensorboard_writers,
                ):
                    callback._writers = writers

        # don't serialize any history optimization result
        if isinstance(state.get("_last_optimization_result"), tf_keras.callbacks.History):
            state["_last_optimization_result"] = ...

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Restore optimizer and callback models after depickling, and recompile.
        self.__dict__.update(state)

        # Unpickle the optimizer, and restore all the callback models
        self._optimizer = dill.loads(self._optimizer)
        for callback in self._optimizer.fit_args.get("callbacks", []):
            if callback.model is ...:
                callback.set_model(self.model)
            elif callback.model:
                model_json, weights = callback.model
                model = tf_keras.models.model_from_json(
                    model_json,
                    custom_objects={"MultivariateNormalTriL": MultivariateNormalTriL},
                )
                model.set_weights(weights)
                callback.set_model(model)

        # Recompile the model
        self.model.compile(
            self.optimizer.optimizer,
            loss=[self.optimizer.loss] * self._model.ensemble_size,
            metrics=[self.optimizer.metrics] * self._model.ensemble_size,
        )

        # recover optimization result if necessary (and possible)
        if state.get("_last_optimization_result") is ...:
            self._last_optimization_result = getattr(self.model, "history")

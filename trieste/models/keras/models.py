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

from typing import Any, Dict, Optional

import dill
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
from tensorflow.python.keras.callbacks import Callback

from ...data import Dataset
from ...types import TensorType
from ..interfaces import HasTrajectorySampler, TrainableProbabilisticModel, TrajectorySampler
from ..optimizer import KerasOptimizer
from .architectures import KerasEnsemble, MultivariateNormalTriL
from .interface import DeepEnsembleModel, KerasPredictor
from .sampler import DeepEnsembleTrajectorySampler
from .utils import negative_log_likelihood, sample_model_index, sample_with_replacement


class DeepEnsemble(
    KerasPredictor, TrainableProbabilisticModel, DeepEnsembleModel, HasTrajectorySampler
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

    Currently we do not support setting up the model with dictionary configs and saving
    the model during Bayesian optimization loop (``track_state`` argument in
    :meth:`~trieste.bayesian_optimizer.BayesianOptimizer.optimize` method should be set to `False`).
    """

    def __init__(
        self,
        model: KerasEnsemble,
        optimizer: Optional[KerasOptimizer] = None,
        bootstrap: bool = False,
        diversify: bool = False,
    ) -> None:
        """
        :param model: A Keras ensemble model with probabilistic networks as ensemble members. The
            model has to be built but not compiled.
        :param optimizer: The optimizer wrapper with necessary specifications for compiling and
            training the model. Defaults to :class:`~trieste.models.optimizer.KerasOptimizer` with
            :class:`~tf.optimizers.Adam` optimizer, negative log likelihood loss and a dictionary
            of default arguments for Keras `fit` method: 1000 epochs, batch size 16, early stopping
            callback with patience of 50, and verbose 0.
            See https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
            arguments.
        :param bootstrap: Sample with replacement data for training each network in the ensemble.
            By default set to `False`.
        :param diversify: Whether to use quantiles from final probabilistic layer as trajectories
            instead of mean predictions when calling :meth:`trajectory_sampler`. Quantiles are
            sampled uniformly from a unit interval. This mode can be used to increase the diversity
            in case of optimizing very large batches of trajectories. When batch size is larger
            than the ensemble size, multiple quantiles will be used with the same network. By
            default set to `False`.
        :raise ValueError: If ``model`` is not an instance of
            :class:`~trieste.models.keras.KerasEnsemble` or ensemble has less than two base
            learners (networks).
        """
        if model.ensemble_size < 2:
            raise ValueError(f"Ensemble size must be greater than 1 but got {model.ensemble_size}.")

        super().__init__(optimizer)

        if not self.optimizer.fit_args:
            self.optimizer.fit_args = {
                "verbose": 0,
                "epochs": 1000,
                "batch_size": 16,
                "callbacks": [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="loss", patience=50, restore_best_weights=True
                    )
                ],
            }

        if self.optimizer.loss is None:
            self.optimizer.loss = negative_log_likelihood

        model.model.compile(
            self.optimizer.optimizer,
            loss=[self.optimizer.loss] * model.ensemble_size,
            metrics=[self.optimizer.metrics] * model.ensemble_size,
        )
        self.original_lr = self.optimizer.optimizer.lr.numpy()

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
    def model(self) -> tf.keras.Model:
        """ " Returns compiled Keras ensemble model."""
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

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        Returns mean and variance at ``query_points`` for the whole ensemble.

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
        probabilistic, an instance of :class:`¬tfp.distributions.Distribution`. In particular, given
        the nature of the approximations stated above the final layer should be a Gaussian
        distribution with `mean` and `variance` methods.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points``.
        """
        ensemble_distributions = self.ensemble_distributions(query_points)
        predicted_means = tf.math.reduce_mean(
            [dist.mean() for dist in ensemble_distributions], axis=0
        )
        predicted_vars = (
            tf.math.reduce_mean(
                [dist.variance() + dist.mean() ** 2 for dist in ensemble_distributions], axis=0
            )
            - predicted_means ** 2
        )

        return predicted_means, predicted_vars

    def predict_ensemble(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
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
        ensemble_distributions = self.ensemble_distributions(query_points)
        predicted_means = tf.convert_to_tensor([dist.mean() for dist in ensemble_distributions])
        predicted_vars = tf.convert_to_tensor([dist.variance() for dist in ensemble_distributions])

        return predicted_means, predicted_vars

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples at ``query_points``. We use the mixture approximation in
        :meth:`predict` for ``query_points`` and sample ``num_samples`` times from a Gaussian
        distribution given by the predicted mean and variance.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples. For a predictive distribution with event shape E, this has shape
            [..., S, N] + E, where S is the number of samples.
        """

        predicted_means, predicted_vars = self.predict(query_points)
        normal = tfp.distributions.Normal(predicted_means, tf.sqrt(predicted_vars))
        samples = normal.sample(num_samples)

        return samples  # [num_samples, len(query_points), 1]

    def sample_ensemble(self, query_points: TensorType, num_samples: int) -> TensorType:
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

    def trajectory_sampler(self) -> TrajectorySampler[DeepEnsemble]:
        """
        Return a trajectory sampler. For :class:`DeepEnsemble`, we use an ensemble
        sampler that randomly picks a network from the ensemble and uses its predicted means
        for generating a trajectory, or optionally randomly sampled quantiles rather than means.

        At the moment only models with single output are supported.

        :return: The trajectory sampler.
        :raise NotImplementedError: If we try to use the sampler with a model that has more than
            one output.
        """
        if self.num_outputs > 1:
            raise NotImplementedError(
                f"""
                Trajectory sampler does not currently support models with multiple outputs,
                however received a model with {self.num_outputs} outputs.
                """
            )

        return DeepEnsembleTrajectorySampler(self, self._diversify)

    def update(self, dataset: Dataset) -> None:
        """
        Neural networks are parametric models and do not need to update data.
        `TrainableProbabilisticModel` interface, however, requires an update method, so
        here we simply pass the execution.
        """
        pass

    def optimize(self, dataset: Dataset) -> None:
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

        x, y = self.prepare_dataset(dataset)
        self.model.fit(x=x, y=y, **self.optimizer.fit_args)

        # Reset lr in case there was an lr schedule: a schedule will have change the learning rate,
        # so that the next time we call `optimize` the starting learning rate would be different.
        # Therefore we make sure the learning rate is set back to its initial value.
        self.optimizer.optimizer.lr.assign(self.original_lr)

    def __getstate__(self) -> dict[str, Any]:
        # When pickling use to_json to save any optimizer fit_arg callback models
        state = self.__dict__.copy()
        if self._optimizer:
            # jsonify all the callback models, pickle the optimizer(!), and revert (ugh!)
            callback: Callback
            saved_models: list[KerasOptimizer] = []
            for callback in self._optimizer.fit_args["callbacks"]:
                saved_models.append(callback.model)
                callback.model = callback.model and callback.model.to_json()
            state["_optimizer"] = dill.dumps(state["_optimizer"])
            for callback, model in zip(self._optimizer.fit_args["callbacks"], saved_models):
                callback.model = model
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Restore optimizer and callback models after depickling, and recompile.
        self.__dict__.update(state)
        if self._optimizer:
            # unpickle the optimizer, and restore all the callback models
            self._optimizer = dill.loads(self._optimizer)
            for callback in self._optimizer.fit_args.get("callbacks", []):
                if callback.model:
                    callback.model = tf.keras.models.model_from_json(
                        callback.model,
                        custom_objects={"MultivariateNormalTriL": MultivariateNormalTriL},
                    )
        # Recompile the model
        self._model.model.compile(
            self.optimizer.optimizer,
            loss=[self.optimizer.loss] * self._model.ensemble_size,
            metrics=[self.optimizer.metrics] * self._model.ensemble_size,
        )

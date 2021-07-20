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

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, TypeVar
from warnings import warn

import tensorflow as tf
from numpy import inf

from ...data import Dataset
from ...type import TensorType
from ...utils import DEFAULTS
from .data import EnsembleDataTransformer
from .networks import KerasNetwork
from ..optimizer import Optimizer, TFOptimizer, TFKerasOptimizer
from ..model_interfaces import ProbabilisticModel, TrainableProbabilisticModel


class NeuralNetworkPredictor(ProbabilisticModel, tf.Module, ABC):
    """ A trainable wrapper for a Keras neural network models. """

    def __init__(self, optimizer: TFKerasOptimizer | None = None):
        """
        :param optimizer: The optimizer with which to train the model. Defaults
            to :class:`~trieste.models.optimizer.TFKerasOptimizer` with
            default arguments to the :meth:`fit` method.
        """
        super().__init__()

        if optimizer is None:
            optimizer = TFKerasOptimizer()

        self._optimizer = optimizer

    @property
    def optimizer(self) -> TFKerasOptimizer:
        """ The optimizer with which to train the model. """
        return self._optimizer

    @property
    @abstractmethod
    def model(self) -> NeuralNetworkPredictor:
        """ The underlying neural network model. """

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.model.predict(query_points)

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        raise Error(
            """
            NeuralNetworkPredictor class does not implement 'predict_joint' method. Acquisition
            rules relying on it cannot be used with this class by default. Certain types of neural
            networks might be able to generate 'predict_join' and such subclasses should
            overwrite this method.
            """
        )

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        return self.model.sample(query_points, num_samples)

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the model with the specified `dataset`.

        :param dataset: The data with which to optimize the `model`.
        """
        self.optimizer.optimize(self.model, dataset)


class NeuralNetworkEnsemble(NeuralNetworkPredictor, TrainableProbabilisticModel):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a Keras
    :class:`~trieste.models.keras_networks.KerasNetwork`.
    """

    def __init__(
        self,
        networks: List[KerasNetwork],
        optimizer: TFKerasOptimizer | None = None,
        dataset_transformer: EnsembleDataTransformer = None,
    ):
        """
        :param networks: A list of `KerasNetwork` objects. The ensemble
            will consist of this collection of networks.
        :param ensemble_size: Number of functions in the ensemble.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~tensorflow.keras.optimizers.Adam`.
        """

        if optimizer is None:
            self._optimizer_keras = tf.keras.optimizers.Adam()
        else:
            self._optimizer_keras = optimizer.optimizer

        super().__init__(optimizer)

        if dataset_transformer is None:
            self._dataset_transformer = EnsembleDataTransformer(networks)
        else:
            self._dataset_transformer = dataset_transformer
        self._networks = networks

        self._ensemble_size = len(networks)
        if self._ensemble_size == 1:
            warn(
                f"""A single network was passed to the class while ensemble as a rule should
                consist of more than a single models, results are unlikely to be meaningful."""
            )

        self._indices = tf.Variable(
            0, name="sampling_indices", dtype=tf.int32, shape=tf.TensorShape(None)
        )
        self._resample_indices(5)

        self._model = self._build_ensemble()
        self.set_input_output_names_data_transformer()
        self._compile_ensemble()

    def __repr__(self) -> str:
        """"""
        return f"NeuralNetworkEnsemble({self._model!r}, {self.optimizer!r})"

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    def get_input_output_names_from_model(self) -> Dict[str, List[str]]:
        names = {'inputs':[], 'outputs': []}
        for i in range(self._ensemble_size):
             names['inputs'].append(self.model.layers[i].name)
             names['outputs'].append(self.model.layers[-(self._ensemble_size-i)].name)
        return names

    def set_input_output_names_data_transformer(self) -> None:
        names = self.get_input_output_names_from_model()
        self._dataset_transformer.input_output_names = names

    def _resample_indices(self, num_search_space_samples: int):
        self._indices.assign(
            tf.random.uniform(
                shape=(num_search_space_samples,),
                maxval=self._ensemble_size,
                dtype=tf.int32,  # pylint: disable=all
            )
        )

    def _build_ensemble(self) -> tf.keras.Model:
        """
        Defines and returns model.
        This method uses each of the `KerasModel` objects to build an element of the
        ensemble. All of these networks are collected together into the model.
        :return: The model.
        """
        inputs = []
        outputs = []
        for index, network in enumerate(self._networks):

            name = network.input_tensor_spec.name + "_" + str(index)
            input_tensor = network.gen_input_tensor(name)
            inputs.append(input_tensor)

            input_layer = tf.keras.layers.Flatten(dtype=network.input_tensor_spec.dtype)(
                input_tensor
            )
            output_layer = network.build_model(input_layer)
            outputs.append(output_layer)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _compile_ensemble(self) -> tf.keras.Model:
        """
        Compiles the model, with the loss function, optimizer and metrics from each of the
        individual networks. Optimizer is shared among the networks.
        """
        losses = [network.loss() for network in self._networks]
        metrics = [network.metrics() for network in self._networks]
        self._model.compile(optimizer=self._optimizer_keras, loss=losses, metrics=metrics)

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Returns means and variance over predictions of all members of the ensemble. In case ensemble
        consists of a single network, predicted means will be equal to the predictions from that
        network, while predicted variances will be set to infinity.

        Note that raw `query_points` need to be transformed to be used in the ensemble,
        `dataset_transformer` is used for that purpose, which essentially replicates the input such
        that each network in the ensemble receives appropriate inputs.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points``.
        """
        # query_points_transformed = self._dataset_transformer(query_points)
        # ensemble_predictions = self._model.predict(query_points_transformed)
        ensemble_predictions = self._model.predict(query_points)
        if self._ensemble_size == 1:
            predicted_means = tf.convert_to_tensor(ensemble_predictions)
            predicted_vars = tf.constant(inf, shape=predicted_means.shape)
        else:
            predicted_means = tf.math.reduce_mean(ensemble_predictions, axis=0)
            predicted_vars = tf.math.reduce_variance(ensemble_predictions, axis=0)
        return (predicted_means, predicted_vars)

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples at ``query_points``. Some versions of
        non probabilistic models can still sample from a distribution even if
        not having direct access to it. Non probabilistic models that cannot
        sample should simply pass

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples. For a predictive distribution with event shape E, this has shape
            [..., S, N] + E, where S is the number of samples.
        """
        stacked_samples = []
        for _ in range(num_samples):
            self._resample_indices(len(query_points))
            inputs = tf.dynamic_partition(query_points, self._indices, self._ensemble_size)
            partitioned_samples = self._model(inputs)

            if self._ensemble_size > 1:
                merge_indices = tf.dynamic_partition(
                    tf.range(len(query_points)), self._indices, self._ensemble_size
                )
                partitioned_samples = tf.dynamic_stitch(merge_indices, partitioned_samples)

            stacked_samples.append(partitioned_samples)

        samples = tf.stack(stacked_samples, axis=0)
        return samples  # [num_samples, len(query_points), 1]

    def update(self, dataset: Dataset) -> None:
        """
        Neural networks are parametric models and do not need to save update data. However,
        Bayesian optimization loop requires an update method, so here we simply pass the execution.
        """
        pass

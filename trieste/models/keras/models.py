# Copyright 2020 The Trieste Contributors
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
from typing import Any, List, TypeVar

import tensorflow as tf

from ...data import Dataset
from ...type import TensorType
from ...utils import DEFAULTS
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

    # __deepcopy__ = module_deepcopy


# class EnsembleSampler:

#     def __init__(self, num_search_space_samples: int, num_query_points: int):
#         """
#         :param num_search_space_samples: The number of points at which to sample the posterior.
#         :param num_query_points: The number of points to acquire at each point.
#         """
#         if not num_search_space_samples > 0:
#             raise ValueError(f"Search space must be greater than 0, got {num_search_space_samples}")

#         if not num_query_points > 0:
#             raise ValueError(
#                 f"Number of query points must be greater than 0, got {num_query_points}"
#             )

#         self._ensemble_size = ensemble_size
#         self._num_search_space_samples = num_search_space_samples
#         self._num_query_points = num_query_points
#         self._indices = tf.Variable(
#             0, name="sampling_indices", dtype=tf.int32, shape=tf.TensorShape(None)
#         )

#     def resample_indices(self):
#         self._indices.assign(
#             tf.random.uniform(
#                 shape=(self._num_search_space_samples, self._num_query_points),
#                 maxval=self._ensemble_size,
#                 dtype=tf.int32,  # pylint: disable=all
#             )
#         )


class NeuralNetworkEnsemble(NeuralNetworkPredictor, TrainableProbabilisticModel):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a Keras
    :class:`~trieste.models.keras_networks.KerasNetwork`.
    """

    def __init__(
        self,
        networks: List[KerasNetwork],
        optimizer: TFKerasOptimizer | None = None,
        # ensemble_sampler: EnsembleSampler | None = None,
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

        # if ensemble_sampler is None:
        #     ensemble_sampler = EnsembleSampler(100,1)
        # self._ensemble_sampler = ensemble_sampler

        self._networks = networks
        self._ensemble_size = len(networks)

        self._indices = tf.Variable(
            0, name="sampling_indices", dtype=tf.int32, shape=tf.TensorShape(None)
        )
        self._resample_indices(500)

        self._model = self._build_ensemble()
        self._compile_ensemble()

    def __repr__(self) -> str:
        """"""
        return f"NeuralNetworkEnsemble({self._model!r}, {self.optimizer!r})"

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    # def _resample_indices(self, num_search_space_samples: int, num_query_points: int):
    #     # self._indices.assign(
    #     indices = tf.random.uniform(
    #             shape=(num_search_space_samples, num_query_points),
    #             maxval=self._ensemble_size,
    #             dtype=tf.int32,  # pylint: disable=all
    #         )
    #     # )
    #     return indices

    def _resample_indices(self, num_search_space_samples: int):
        self._indices.assign(
            tf.random.uniform(
                shape=(num_search_space_samples,),
                maxval=self._ensemble_size,
                dtype=tf.int32,  # pylint: disable=all
            )
        )

    # def _resample_indices(self, num_search_space_samples: int):
    #     # self._indices.assign(
    #     indices = tf.random.uniform(
    #             shape=(num_search_space_samples,),
    #             maxval=self._ensemble_size,
    #             dtype=tf.int32,  # pylint: disable=all
    #         )
    #     # )
    #     return indices

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
            # breakpoint()
            # example_model=tf.keras.Model(inputs=input_tensor, outputs=output_layer)
            # tf.keras.utils.plot_model(example_model, show_shapes=True, to_file='model.png'    )
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
        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points``.
        """
        ensemble_predictions = self._model.predict(query_points)
        breakpoint()
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

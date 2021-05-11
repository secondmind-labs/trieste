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
from typing import List, Optional

import numpy as np
import tensorflow as tf

from ..data import Dataset
# from ..type import TensorType
# from ..utils import DEFAULTS
# from .optimizer import Optimizer


def size(tensor_spec: tf.TensorSpec) -> int:
    """
    Equivalent to `np.size` for `TensorSpec` objects.
    """
    return int(np.prod(tensor_spec.shape))


def get_tensor_spec_from_data(data: Dataset) -> tuple(tf.TensorSpec, tf.TensorSpec):
    """
    Extract tensor specifications for neural network inputs and outputs based on the data.
    """
    input_tensor_spec = tf.TensorSpec(
        shape=(data.query_points.shape[-1],),
        dtype=data.query_points.dtype,
        name="query_points",
    )
    output_tensor_spec = tf.TensorSpec(
        shape=(data.observations.shape[-1],),
        dtype=data.observations.dtype,
        name="observations",
    )
    return input_tensor_spec, output_tensor_spec


def sample_with_replacement(dataset: Dataset) -> Dataset:
    """
    Create a new ``dataset`` with data sampled with replacement. This
    function is useful for creating bootstrap samples of data for training ensembles.
    :param dataset: The data whose observations should be sampled.
    :return: A (new) ``dataset`` with sampled data.
    """
    # transition.observation has shape [batch_size,] + observation_space_spec.shape
    n_rows = dataset.observations.shape[0]

    index_tensor = tf.random.uniform(
        (n_rows,), maxval=n_rows, dtype=tf.dtypes.int64
    )  # pylint: disable=all

    observations = tf.gather(dataset.observations, index_tensor)  # pylint: disable=all
    query_points = tf.gather(dataset.query_points, index_tensor)  # pylint: disable=all

    return Dataset(query_points=query_points, observations=observations)


class KerasNetwork(ABC):
    """
    This class defines the structure and essential methods for a transition network. The transition
    network is a sequential, feed forward neural network. It also makes it easy to create networks
    where data is bootstrapped for each new network in an ensemble.
    Subclasses of this class should define the structure of the network by implementing the
    `build_model` method. The output layer should be reshaped to match the observation tensors from
    the environment. The loss function of the network should be specified by implementing the
    `loss` method. The training data can be manipulated by overriding the `transform_training_data`
    method with the appropriate transformation.
    """

    def __init__(
        self,
        input_tensor_spec: tf.TensorSpec,
        output_tensor_spec: tf. TensorSpec,
        bootstrap_data: bool = False,
    ):
        """
        :param bootstrap_data: Create an ensemble version of the network where data is resampled with
                               replacement.
        """
        assert isinstance(input_tensor_spec, tf.TensorSpec)
        assert isinstance(output_tensor_spec, tf.TensorSpec)

        self._input_tensor_spec = input_tensor_spec
        self._output_tensor_spec = output_tensor_spec
        self._bootstrap_data = bootstrap_data

    @property
    def input_tensor_spec(self) -> tf.TensorSpec:
        return self._input_tensor_spec

    @property
    def output_tensor_spec(self) -> tf.TensorSpec:
        return self._output_tensor_spec

    def gen_input_tensor(self, name: str = None, batch_size: int = None) -> tf.keras.Input:
        input_tensor_spec = self._input_tensor_spec
        if name is None:
            name = input_tensor_spec.name
        input_tensor = tf.keras.Input(
            shape=input_tensor_spec.shape,
            batch_size=batch_size,
            dtype=input_tensor_spec.dtype,
            name=name
        )
        # breakpoint()
        return input_tensor

    @abstractmethod
    def build_model(self, inputs: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        """
        Define the layers of the sequential, feed-forward neural network from the layer
        `input_layer`.

        :param inputs: Input layer.
        :return: outputs layer.
        """
        pass

    def loss(self) -> tf.keras.losses.Loss:
        """
        Define the loss function for training the network, in standard Bayesian optimization
        applications mean squared error is used and we use it here as a default. Method should be
        overwritten for custom losses.
        :return: Return the loss function for this network.
        """
        return tf.keras.losses.MeanSquaredError()

    def metrics(self) -> tf.keras.metrics.Metric:
        """
        Defines metrics for monitoring the training of the network. Method should be overwritten
        for custom metrics.
        :return: Return the metrics for this network.
        """
        metrics = [
            tf.keras.metrics.RootMeanSquaredError(name="RMSE"),
            tf.keras.metrics.MeanAbsoluteError(name="MAE"),
        ]
        return metrics

    def transform_training_data(self, dataset: Dataset) -> Dataset:
        """
        This network will be trained on the data in `Transition`. This method ensures the training
        data can be transformed before it is used in training. Also, when ensembles are used
        this method can use the `bootstrap_data` flag to use bootstrap samples of the data for
        each model in the ensemble.
        :param dataset: A `Transition` object, consisting of states, actions and successor
                           states.
        :return: Return a (new) `Transition` object.
        """
        if self._bootstrap_data:
            return sample_with_replacement(dataset)
        else:
            return dataset


class MultilayerFcNetwork(KerasNetwork):
    """
    This class defines a multilayer transition model using Keras, fully connected type. If defined
    with zero layers (default) we obtain a network equivalent of a linear regression.
    If number of hidden layers is one or more then all arguments to the dense Keras layer
    can be set individually for each layer.
    """

    def __init__(
        self,
        input_tensor_spec: tf.TensorSpec,
        output_tensor_spec: tf. TensorSpec,
        num_hidden_layers: int = 0,
        units: Optional[List[int]] = None,
        activation: Optional[List[Callable]] = None,
        use_bias: Optional[List[bool]] = None,
        kernel_initializer: Optional[List[Callable]] = None,
        bias_initializer: Optional[List[Callable]] = None,
        kernel_regularizer: Optional[List[Callable]] = None,
        bias_regularizer: Optional[List[Callable]] = None,
        activity_regularizer: Optional[List[Callable]] = None,
        kernel_constraint: Optional[List[Callable]] = None,
        bias_constraint: Optional[List[Callable]] = None,
        bootstrap_data: bool = False,
    ):
        """
        :param input_tensor_spec: Environment observation specifications.
        :param num_hidden_layers: A number of hidden layers in the network. If larger than zero
            (default), then all other arguments to the `Dense` Keras layer have to have
            same length, if specified.
        :param units: Number of nodes in each hidden layer.
        :param activation: Activation function of the hidden nodes.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param kernel_initializer: Initializer for the kernel weights matrix.
        :param bias_initializer: Initializer for the bias vector.
        :param kernel_regularizer: Regularizer function applied to the kernel weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        :param activity_regularizer: Regularizer function applied to the output of the layer.
        :param kernel_constraint: Constraint function applied to the kernel weights matrix.
        :param bias_constraint: Constraint function applied to the bias vector.
        :param bootstrap_data: Re-sample data with replacement.
        """
        assert num_hidden_layers >= 0, "num_hidden_layers must be an integer >= 0"
        if num_hidden_layers > 0:
            assert units is not None, "if num_hidden_layers > 0, units cannot be None"
            assert num_hidden_layers == len(units), (
                "if num_hidden_layers > 0, units has to be a list with a "
                + "number of elements equal to num_hidden_layers"
            )
            for i in [
                activation,
                use_bias,
                kernel_initializer,
                bias_initializer,
                kernel_regularizer,
                bias_regularizer,
                activity_regularizer,
                kernel_constraint,
                bias_constraint,
            ]:
                if i is not None:
                    assert num_hidden_layers == len(i)

        super().__init__(input_tensor_spec, output_tensor_spec, bootstrap_data)

        self._input_tensor_spec = input_tensor_spec
        self._output_tensor_spec = output_tensor_spec
        self._num_hidden_layers = num_hidden_layers
        self._units = units
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activity_regularizer = activity_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint

    def gen_hidden_dense_layers(self, hidden_layer):
        """Generate a sequence of dense Keras layers"""
        if self._num_hidden_layers > 0:
            for id_layer in range(self._num_hidden_layers):
                hidden_layer = tf.keras.layers.Dense(units=self._units[id_layer])(hidden_layer)
                if self._activation is not None:
                    hidden_layer.activation = self._activation[id_layer]
                if self._use_bias is not None:
                    hidden_layer.use_bias = self._use_bias[id_layer]
                if self._kernel_initializer is not None:
                    hidden_layer.kernel_initializer = self._kernel_initializer[id_layer]
                if self._bias_initializer is not None:
                    hidden_layer.bias_initializer = self._bias_initializer[id_layer]
                if self._kernel_regularizer is not None:
                    hidden_layer.kernel_regularizer = self._kernel_regularizer[id_layer]
                if self._bias_regularizer is not None:
                    hidden_layer.bias_regularizer = self._bias_regularizer[id_layer]
                if self._activity_regularizer is not None:
                    hidden_layer.activity_regularizer = self._activity_regularizer[id_layer]
                if self._kernel_constraint is not None:
                    hidden_layer.kernel_constraint = self._kernel_constraint[id_layer]
                if self._bias_constraint is not None:
                    hidden_layer.bias_constraint = self._bias_constraint[id_layer]
        return hidden_layer

    def build_model(self, input_layer: tf.keras.layers.Layer = None) -> tf.keras.layers.Layer:
        if input_layer is None:
            input_tensor = self.gen_input_tensor()
            input_layer = tf.keras.layers.Flatten(
                dtype=self._input_tensor_spec.dtype
            )(input_tensor)
        output_space_nodes = size(self._output_tensor_spec)
        hidden_layer = input_layer
        hidden_layer = self.gen_hidden_dense_layers(hidden_layer)
        output_layer = tf.keras.layers.Dense(output_space_nodes, activation="linear")(
            hidden_layer
        )
        return output_layer

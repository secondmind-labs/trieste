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

from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import repeat
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from .utils import sample_with_replacement, size


class KerasNetwork(ABC):
    """
    This class defines the structure and essential methods for a neural network using Keras. It
    also includes a method that facilitates creating ensembles where data is bootstrapped for each
    network in an ensemble.

    Subclasses of this class should define the structure of the network by implementing the
    `build_model` method. The loss function of the network should be specified by implementing the
    `loss` method. The training data can be manipulated by overriding the `transform_training_data`
    method with the appropriate transformation.
    """

    def __init__(
        self,
        input_tensor_spec: tf.TensorSpec,
        output_tensor_spec: tf.TensorSpec,
        bootstrap_data: bool = False,
    ):
        """
        :param input_tensor_spec: Input tensor specification.
        :param output_tensor_spec: Output tensor specification.
        :param bootstrap_data: Resample data with replacement.
        """
        assert isinstance(input_tensor_spec, tf.TensorSpec)
        assert isinstance(output_tensor_spec, tf.TensorSpec)

        self._input_tensor_spec = input_tensor_spec
        self._output_tensor_spec = output_tensor_spec
        self._flattened_output_shape = size(output_tensor_spec)
        self._bootstrap_data = bootstrap_data

    @property
    def input_tensor_spec(self) -> tf.TensorSpec:
        return self._input_tensor_spec

    @property
    def output_tensor_spec(self) -> tf.TensorSpec:
        return self._output_tensor_spec

    def gen_input_tensor(self, name: str = None, batch_size: int = None) -> tf.keras.Input:
        """
        Generate input tensor based on input tensor specification.Define the layers of the sequential, feed-forward neural network from the layer
        `input_layer`.

        :param name: Optional name for the input layer.
        :param batch_size: Optional batch size for the input layer.
        :return: Input layer.
        """
        input_tensor_spec = self._input_tensor_spec
        if name is None:
            name = input_tensor_spec.name
        input_tensor = tf.keras.Input(
            shape=input_tensor_spec.shape,
            batch_size=batch_size,
            dtype=input_tensor_spec.dtype,
            name=name,
        )
        return input_tensor

    @abstractmethod
    def build_model(self, inputs: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        """
        Define the layers of the neural network from the layer `inputs`.

        :param inputs: Input layer.
        :return: Output layer.
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
        Defines metrics for monitoring the training of the network. Uses Root mean square error
        (RMSE) and Mean absolute error (MAE) as a default. Method should be overwritten for custom
        metrics.

        :return: Return the metrics for this network.
        """
        metrics = [
            tf.keras.metrics.RootMeanSquaredError(name="RMSE"),
            tf.keras.metrics.MeanAbsoluteError(name="MAE"),
        ]
        return metrics

    def transform_training_data(self, dataset: Dataset) -> Dataset:
        """
        This method ensures the training data can be transformed before it is used in training.
        Also, when ensembles are used this method can use the `bootstrap_data` flag to use
        bootstrap samples of the data for each model in the ensemble.

        :param dataset: A `Dataset` object consisting of `query_points` and `observations`. 
        :return: Return a (new) `Dataset` object.
        """
        if self._bootstrap_data:
            return sample_with_replacement(dataset)
        else:
            return dataset


class MultilayerFcNetwork(KerasNetwork):
    """
    This class defines a multilayer fully-connected feed-forward network. If defined
    with zero layers (default) we obtain a network equivalent of a linear regression.
    If number of hidden layers is one or more then all arguments to the dense Keras layer
    should be set individually for each layer.
    """

    def __init__(
        self,
        input_tensor_spec: tf.TensorSpec,
        output_tensor_spec: tf.TensorSpec,
        num_hidden_layers: int = 0,
        units: Optional[List[int]] = None,
        activation: Optional[List[Union[Callable, str]]] = None,
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

        self._activation = activation or repeat(None, num_hidden_layers)
        self._use_bias = use_bias or repeat(True, num_hidden_layers)
        self._kernel_initializer = kernel_initializer or repeat('glorot_uniform', num_hidden_layers)
        self._bias_initializer = bias_initializer or repeat('zeros', num_hidden_layers)
        self._kernel_regularizer = kernel_regularizer or repeat(None, num_hidden_layers)
        self._bias_regularizer = bias_regularizer or repeat(None, num_hidden_layers)
        self._activity_regularizer = activity_regularizer or repeat(None, num_hidden_layers)
        self._kernel_constraint = kernel_constraint or repeat(None, num_hidden_layers)
        self._bias_constraint = bias_constraint or repeat(None, num_hidden_layers)

    def gen_hidden_dense_layers(self, hidden_layer):
        """Generate a sequence of dense Keras layers"""
        if self._num_hidden_layers > 0:
            layer_args = zip(
                self._units,
                self._activation,
                self._use_bias,
                self._kernel_initializer,
                self._bias_initializer,
                self._activity_regularizer,
                self._kernel_constraint,
                self._bias_constraint
            )

            for dense_layer_args in layer_args:
                layer = tf.keras.layers.Dense(*dense_layer_args)
                hidden_layer = layer(hidden_layer)

        return hidden_layer
    
    def gen_output_layer(self, input_layer: tf.keras.layers.Layer = None) -> tf.keras.layers.Layer:
        output_layer = tf.keras.layers.Dense(
            self._flattened_output_shape, activation="linear"
        )(input_layer)
        # output_layer_reshaped = tf.keras.layers.Reshape(
        #     self._output_tensor_spec.shape
        # )(output_layer)
        # return output_layer_reshaped
        return output_layer

    def build_model(self, input_layer: tf.keras.layers.Layer = None) -> tf.keras.layers.Layer:
        if input_layer is None:
            input_tensor = self.gen_input_tensor()
            input_layer = tf.keras.layers.Flatten(dtype=self._input_tensor_spec.dtype)(input_tensor)
        hidden_layer = input_layer
        hidden_layer = self.gen_hidden_dense_layers(hidden_layer)
        output_layer = self.gen_output_layer(hidden_layer)
        return output_layer


class LinearNetwork(MultilayerFcNetwork):
    """
    This class defines a linear network using Keras, i.e. a neural network with no
    hidden layers, equivalent to a linear regression.
    """

    def __init__(
        self,
        input_tensor_spec: tf.TensorSpec,
        output_tensor_spec: tf.TensorSpec,
        bootstrap_data: bool = False,
    ):
        super().__init__(
            input_tensor_spec,
            output_tensor_spec,
            num_hidden_layers=0,
            bootstrap_data=bootstrap_data
        )


negloglik = lambda y, p_y: -p_y.log_prob(y)
negloglik.__doc__ = """define general log-likelihood loss for distribution Lambda"""

class GaussianNetwork(MultilayerFcNetwork):
    """
    This class defines a probabilistic neural network using Keras, with Gaussian distributed
    outputs.
    """

    def gen_output_layer(self, input_layer: tf.keras.layers.Layer = None) -> tf.keras.layers.Layer:
        mvn_shape = tfp.layers.MultivariateNormalTriL.params_size(self._flattened_output_shape)
        output_layer = tf.keras.layers.Dense(mvn_shape)(input_layer)
        dist = tfp.layers.MultivariateNormalTriL(
            event_size=self._flattened_output_shape,
            convert_to_tensor_fn=self._convert_to_tensor_fn,
        )(output_layer)
        return dist

    def _convert_to_tensor_fn(self, distribution: tfp.distributions.Distribution) -> tf.Tensor:
        output_tensor_spec_shape = (-1,) + tuple(self._output_tensor_spec.shape)
        return tf.reshape(distribution.sample(), output_tensor_spec_shape)

    def loss(self):
        return lambda y, p_y: negloglik(y, p_y)


class DiagonalGaussianNetwork(GaussianNetwork):
    """
    This class defines a probabilistic neural network using Keras, with Gaussian distributed
    outputs, but modelling only variances on the diagonal and assuming zero covariances elsewhere.
    """

    def gen_output_layer(self, input_layer: tf.keras.layers.Layer = None) -> tf.keras.layers.Layer:
        mvn_shape = tfp.layers.IndependentNormal.params_size(self._flattened_output_shape)
        output_layer = tf.keras.layers.Dense(mvn_shape)(input_layer)
        dist = tfp.layers.IndependentNormal(
            event_shape=self._flattened_output_shape,
            convert_to_tensor_fn=self._convert_to_tensor_fn,
        )(output_layer)
        return dist

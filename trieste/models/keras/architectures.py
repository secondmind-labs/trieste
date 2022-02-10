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

"""
This file contains implementations of neural network architectures with Keras.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Sequence

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class KerasEnsemble:
    """
    This class builds an ensemble of neural networks, using Keras. Individual networks must
    be instance of :class:`~trieste.models.keras_networks.KerasEnsembleNetwork`. This class
    is meant to be used with :class:`~trieste.models.keras_networks.DeepEnsemble` model wrapper,
    which compiles the model.
    """

    def __init__(
        self,
        networks: Sequence[KerasEnsembleNetwork],
    ) -> None:
        """
        :param networks: A list of neural network specifications, one for each member of the
            ensemble. The ensemble will be built using these specifications.
        """
        for index, network in enumerate(networks):
            if not isinstance(network, KerasEnsembleNetwork):
                raise ValueError(
                    f"Individual networks must be an instance of KerasEnsembleNetwork, "
                    f"received {type(network)} instead."
                )
            networks[index].network_name = f"model_{index}_"
        self._networks = networks

        self._model = self._build_ensemble()

    def __repr__(self) -> str:
        """"""
        return f"KerasEnsemble({self._networks!r})"

    @property
    def model(self) -> tf.keras.Model:
        """Returns built but uncompiled Keras ensemble model."""
        return self._model

    @property
    def ensemble_size(self) -> int:
        """
        Returns the size of the ensemble, that is, the number of base learners or individual neural
        network models in the ensemble.
        """
        return len(self._networks)

    def _build_ensemble(self) -> tf.keras.Model:
        """
        Builds the ensemble model by combining all the individual networks in a single Keras model.
        This method relies on ``connect_layers`` method of :class:`KerasEnsembleNetwork` objects
        to construct individual networks.

        :return: The Keras model.
        """
        inputs, outputs = zip(*[network.connect_layers() for network in self._networks])

        return tf.keras.Model(inputs=inputs, outputs=outputs)


class KerasEnsembleNetwork:
    """
    This class is an interface that defines necessary attributes and methods for neural networks
    that are meant to be used for building ensembles by
    :class:`~trieste.models.keras_networks.KerasEnsemble`. Subclasses are not meant to
    build and compile Keras models, instead they are providing specification that
    :class:`~trieste.models.keras_networks.KerasEnsemble` will use to build the Keras model.
    """

    def __init__(
        self,
        input_tensor_spec: tf.TensorSpec,
        output_tensor_spec: tf.TensorSpec,
        network_name: str = "",
    ):
        """
        :param input_tensor_spec: Tensor specification for the input to the network.
        :param output_tensor_spec: Tensor specification for the output of the network.
        :param network_name: The name to be used when building the network.
        """
        if not isinstance(input_tensor_spec, tf.TensorSpec):
            raise ValueError(
                f"input_tensor_spec must be an instance of tf.TensorSpec, "
                f"received {type(input_tensor_spec)} instead."
            )
        if not isinstance(output_tensor_spec, tf.TensorSpec):
            raise ValueError(
                f"output_tensor_spec must be an instance of tf.TensorSpec, "
                f"received {type(output_tensor_spec)} instead."
            )

        self.input_tensor_spec = input_tensor_spec
        self.output_tensor_spec = output_tensor_spec
        self.network_name = network_name

    @property
    def input_layer_name(self) -> str:
        return self.network_name + "input"

    @property
    def output_layer_name(self) -> str:
        return self.network_name + "output"

    @property
    def flattened_output_shape(self) -> int:
        return int(np.prod(self.output_tensor_spec.shape))

    @abstractmethod
    def connect_layers(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Connects the layers of the neural network. Architecture, layers and layer specifications
        need to be defined by the subclasses.

        :return: Input and output tensor of the network, required by :class:`tf.keras.Model` to
            build a model.
        """
        raise NotImplementedError


class GaussianNetwork(KerasEnsembleNetwork):
    """
    This class defines layers of a probabilistic neural network using Keras. The network
    architecture is a multilayer fully-connected feed-forward network, with Gaussian
    distribution as an output. The layers are meant to be built as an ensemble model by
    :class:`KerasEnsemble`. Note that this is not a Bayesian neural network.
    """

    def __init__(
        self,
        input_tensor_spec: tf.TensorSpec,
        output_tensor_spec: tf.TensorSpec,
        hidden_layer_args: Sequence[dict[str, Any]] = (
            {"units": 50, "activation": "relu"},
            {"units": 50, "activation": "relu"},
        ),
        independent: bool = False,
    ):
        """
        :param input_tensor_spec: Tensor specification for the input to the network.
        :param output_tensor_spec: Tensor specification for the output of the network.
        :param hidden_layer_args: Specification for building dense hidden layers. Each element in
            the sequence should be a dictionary containing arguments (keys) and their values for a
            :class:`~tf.keras.layers.Dense` hidden layer. Please check Keras Dense layer API for
            available arguments. Objects in the sequence will sequentially be used to add
            :class:`~tf.keras.layers.Dense` layers. Length of this sequence determines the number of
            hidden layers in the network. Default value is two hidden layers, 50 nodes each, with
            ReLu activation functions. Empty sequence needs to be passed to have no hidden layers.
        :param independent: If set to `True` then :class:`~tfp.layers.IndependentNormal` layer
            is used as the output layer. This models outputs as independent, only the diagonal
            elements of the covariance matrix are parametrized. If left as the default `False`,
            then :class:`~tfp.layers.MultivariateNormalTriL` layer is used where correlations
            between outputs are learned as well.
        :raise ValueError: If objects in ``hidden_layer_args`` are not dictionaries.
        """
        super().__init__(input_tensor_spec, output_tensor_spec)

        self._hidden_layer_args = hidden_layer_args
        self._independent = independent

    def _gen_input_tensor(self) -> tf.keras.Input:

        input_tensor = tf.keras.Input(
            shape=self.input_tensor_spec.shape,
            dtype=self.input_tensor_spec.dtype,
            name=self.input_layer_name,
        )
        return input_tensor

    def _gen_hidden_layers(self, input_tensor: tf.Tensor) -> tf.Tensor:

        for index, hidden_layer_args in enumerate(self._hidden_layer_args):
            layer_name = f"{self.network_name}dense_{index}"
            layer = tf.keras.layers.Dense(**hidden_layer_args, name=layer_name)
            input_tensor = layer(input_tensor)

        return input_tensor

    def _gen_output_layer(self, input_tensor: tf.Tensor) -> tf.Tensor:

        dist_layer = (
            tfp.layers.IndependentNormal if self._independent else tfp.layers.MultivariateNormalTriL
        )
        n_params = dist_layer.params_size(self.flattened_output_shape)

        parameter_layer = tf.keras.layers.Dense(
            n_params, name=self.network_name + "dense_parameters"
        )(input_tensor)

        distribution = dist_layer(
            self.flattened_output_shape,
            lambda s: s.mean(),
            name=self.output_layer_name,
        )(parameter_layer)

        return distribution

    def connect_layers(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Connect all layers in the network. We start by generating an input tensor based on input
        tensor specification. Next we generate a sequence of hidden dense layers based on
        hidden layer arguments. Finally, we generate a dense layer whose nodes act as parameters of
        a Gaussian distribution in the final probabilistic layer.

        :return: Input and output tensor of the sequence of layers.
        """
        input_tensor = self._gen_input_tensor()
        hidden_tensor = self._gen_hidden_layers(input_tensor)
        output_tensor = self._gen_output_layer(hidden_tensor)

        return input_tensor, output_tensor

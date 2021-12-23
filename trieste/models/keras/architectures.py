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

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, List, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from .utils import get_tensor_spec_from_data


def build_vanilla_keras_ensemble(
    dataset: Dataset,
    ensemble_size: int = 5,
    num_hidden_layers: int = 2,
    units: int = 50,
    activation: Union[str, Callable] = "relu",
    independent_normal: bool = False,
) -> KerasEnsemble:

    """
    Builds a simple ensemble of neural networks in Keras where each network has the same
    architecture: number of hidden layers, nodes in hidden layers and activation function.

    :param dataset: Data for training, used for extracting input and output tensor specifications.
    :param ensemble_size: The size of the ensemble, that is, the number of base learners or
        individual neural networks in the ensemble.
    :param num_hidden_layers: The number of hidden layers in each network.
    :param units: The number of nodes in each hidden layer.
    :param activation: The activation function in each hidden layer.
    :param independent: If set to `True` then :class:`~tfp.layers.IndependentNormal` layer
        is used as the output layer. This models outputs as independent, only the diagonal
        elements of the covariance matrix are parametrized. If left as the default `False`,
        then :class:`~tfp.layers.MultivariateNormalTriL` layer is used where correlations
        between outputs are learned as well.
    :return: Keras ensemble model.
    """
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(dataset)

    hidden_layer_args = []
    for i in range(num_hidden_layers):
        hidden_layer_args.append({"units": units, "activation": activation})

    networks = [
        GaussianNetwork(
            input_tensor_spec,
            output_tensor_spec,
            hidden_layer_args,
            independent_normal,
        )
        for _ in range(ensemble_size)
    ]
    keras_ensemble = KerasEnsemble(networks)

    return keras_ensemble


class KerasEnsemble(tf.Module):
    """
    This class builds an ensemble of neural networks, using Keras. Individual networks must
    be instance of :class:`~trieste.models.keras_networks.KerasEnsembleNetwork`. This class
    is meant to be used with :class:`~trieste.models.keras_networks.DeepEnsemble` model wrapper,
    which compiles the model.
    """

    def __init__(
        self,
        networks: List[KerasEnsembleNetwork],
    ) -> None:
        """
        :param networks: A list of neural network specifications, one for each member of the
            ensemble. The ensemble will be built using these specifications.
        """

        super().__init__()

        for network in networks:
            if not isinstance(network, KerasEnsembleNetwork):
                raise ValueError(
                    f"Individual networks must be an instance of KerasEnsembleNetwork, "
                    f"received {type(network)} instead."
                )

        self._networks = networks
        self._model = self.build_ensemble()

    def __repr__(self) -> str:
        """"""
        return f"KerasEnsemble({self._networks!r})"

    @property
    def model(self) -> tf.keras.Model:
        """ " Returns built but uncompiled Keras ensemble model."""
        return self._model

    @property
    def ensemble_size(self) -> int:
        """
        Returns the size of the ensemble, that is, the number of base learners or individual neural
        network models in the ensemble.
        """
        return len(self._networks)

    def build_ensemble(self) -> tf.keras.Model:
        """
        Builds the ensemble model by combining all the individual networks in a single Keras model.
        This method relies on ``connect_layers`` method of :class:`KerasEnsembleNetwork` objects
        to construct individual networks.

        :return: The Keras model.
        """
        inputs = []
        outputs = []
        for index, network in enumerate(self._networks):
            network.network_name = "model_" + str(index) + "_"
            input_tensor, output_tensor = network.connect_layers()
            inputs.append(input_tensor)
            outputs.append(output_tensor)

        return tf.keras.Model(inputs=inputs, outputs=outputs)


class KerasEnsembleNetwork(tf.Module):
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
    This class defines layers of a probabilistic neural network using Keras, with Gaussian
    distribution as an output. The network architecture is a multilayer fully-connected
    feed-forward network. The network has to be built and compiled as an ensemble model,
    it should also be trained with maximum likelihood loss function.
    """

    def __init__(
        self,
        input_tensor_spec: tf.TensorSpec,
        output_tensor_spec: tf.TensorSpec,
        hidden_layer_args: Optional[List[dict[str, Any]]] = None,
        independent: bool = False,
    ):
        """
        :param input_tensor_spec: Tensor specification for the input to the network.
        :param output_tensor_spec: Tensor specification for the output of the network.
        :param hidden_layer_args: Specification for building dense hidden layers. Each element in
            the list should be a dictionary containing arguments (keys) and their values for a
            :class:`~tf.keras.layers.Dense` hidden layer. Please check Keras Dense layer API for
            available arguments. Objects in the list will sequentially be used to add
            :class:`~tf.keras.layers.Dense` layers. Length of this list determines the number of
            hidden layers in the network. Default value is two hidden layers, 50 nodes each, with
            ReLu activation functions. Empty list needs to be passed to have no hidden layers.
        :param independent: If set to `True` then :class:`~tfp.layers.IndependentNormal` layer
            is used as the output layer. This models outputs as independent, only the diagonal
            elements of the covariance matrix are parametrized. If left as the default `False`,
            then :class:`~tfp.layers.MultivariateNormalTriL` layer is used where correlations
            between outputs are learned as well.
        :raise ValueError: If objects in ``hidden_layer_args`` are not dictionaries.
        """
        if hidden_layer_args is None:
            hidden_layer_args = [
                {"units": 50, "activation": "relu"},
                {"units": 50, "activation": "relu"},
            ]
        if not isinstance(hidden_layer_args, list):
            raise ValueError(
                f"hidden_layer_args must be a list of dictionaries."
                f"Received {type(hidden_layer_args)} instead."
            )
        else:
            if len(hidden_layer_args) > 0:
                for layer in hidden_layer_args:
                    if not isinstance(layer, dict):
                        raise ValueError(
                            f"Objects in hidden_layer_args must be dictionaries. They should "
                            f"contain arguments for hidden dense layers, "
                            f"received {type(layer)} instead."
                        )

        super().__init__(input_tensor_spec, output_tensor_spec)

        self._num_hidden_layers = len(hidden_layer_args)
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

        if self._num_hidden_layers > 0:
            for index, hidden_layer_args in enumerate(self._hidden_layer_args):
                layer_name = self.network_name + "dense_" + str(index)
                layer = tf.keras.layers.Dense(**hidden_layer_args, name=layer_name)
                input_tensor = layer(input_tensor)

        return input_tensor

    def _gen_output_layer(self, input_tensor: tf.Tensor) -> tf.Tensor:

        if self._independent:
            n_params = tfp.layers.IndependentNormal.params_size(self.flattened_output_shape)
        else:
            n_params = tfp.layers.MultivariateNormalTriL.params_size(self.flattened_output_shape)

        parameter_layer = tf.keras.layers.Dense(
            n_params, name=self.network_name + "dense_parameters"
        )(input_tensor)

        if self._independent:
            distribution = tfp.layers.IndependentNormal(
                event_shape=self.flattened_output_shape,
                convert_to_tensor_fn=lambda s: s.mean(),
                name=self.output_layer_name,
            )(parameter_layer)
        else:
            distribution = tfp.layers.MultivariateNormalTriL(
                event_size=self.flattened_output_shape,
                convert_to_tensor_fn=lambda s: s.mean(),
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

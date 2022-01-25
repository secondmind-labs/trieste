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

""" This module contains functions that facilitate building neural network models. """

from __future__ import annotations

from typing import Union

import tensorflow as tf

from ...data import Dataset
from .architectures import GaussianNetwork, KerasEnsemble
from .utils import get_tensor_spec_from_data


def build_vanilla_keras_ensemble(
    dataset: Dataset,
    ensemble_size: int = 5,
    num_hidden_layers: int = 2,
    units: int = 50,
    activation: Union[str, tf.keras.layers.Activation] = "relu",
    independent_normal: bool = False,
) -> KerasEnsemble:

    """
    Builds a simple ensemble of neural networks in Keras where each network has the same
    architecture: number of hidden layers, nodes in hidden layers and activation function.

    Default ensemble size and activation function seem to work well in practice, in regression type
    of problems at least. Number of hidden layers and units per layer should be modified according
    to the dataset size and complexity of the function. Using the independent normal is relevant
    only if one is modelling multiple output variables, as it simplifies the distribution by
    ignoring correlations between outputs.

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

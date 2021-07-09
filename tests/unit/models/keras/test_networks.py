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

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from trieste.models.keras.networks import (
    DiagonalGaussianNetwork,
    GaussianNetwork,
    MultilayerFcNetwork,
)
from trieste.models.keras.utils import get_tensor_spec_from_data, size


def test_network_output_shape_matches_observations(neural_network, example_data):
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)
    network = neural_network(
        input_tensor_spec,
        output_tensor_spec,
    )
    network_output = network.build_model()
    # breakpoint()
    assert example_data.observations.shape[-1] == network_output[0].type_spec.shape[-1]


@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3])
@pytest.mark.parametrize("num_hidden_nodes", [1, 10])
def test_multilayer_fc_network_nparams(
    input_tensor_spec, output_tensor_spec, num_hidden_layers, num_hidden_nodes
):
    """
    Ensure we have a correct number of nodes/parameters in the network.
    """

    network = MultilayerFcNetwork(
        input_tensor_spec,
        output_tensor_spec,
        num_hidden_layers,
        [num_hidden_nodes] * num_hidden_layers,
    )
    input_tensor = network.gen_input_tensor()
    input_layer = tf.keras.layers.Flatten(dtype=input_tensor_spec.dtype)(input_tensor)
    network_output = network.build_model(input_layer)
    model = tf.keras.Model(inputs=input_tensor, outputs=network_output)

    # number of parameters
    nparams = model.count_params()

    # expected number of parameters
    input_nodes = size(input_tensor_spec)
    output_nodes = size(output_tensor_spec)
    if num_hidden_layers == 0:
        nparams_exp = input_nodes * output_nodes + output_nodes
    elif num_hidden_layers > 0:
        nparams_exp = (
            (input_nodes + 1) * num_hidden_nodes
            + (num_hidden_layers - 1) * num_hidden_nodes * (num_hidden_nodes + 1)
            + output_nodes * (num_hidden_nodes + 1)
        )

    assert nparams == nparams_exp


@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3])
@pytest.mark.parametrize("num_hidden_nodes", [1, 10])
@pytest.mark.parametrize(
    "prob_network", [GaussianNetwork]  #, DiagonalGaussianNetwork]
)
def test_multilayer_fc_probabilistic_network_nparams(
    input_tensor_spec, output_tensor_spec, num_hidden_layers, num_hidden_nodes, prob_network
):
    """
    Ensure we have a correct number of nodes/parameters in a probabilistic network.
    """
    
    network = prob_network(
        input_tensor_spec,
        output_tensor_spec,
        num_hidden_layers,
        [num_hidden_nodes] * num_hidden_layers,
    )
    input_tensor = network.gen_input_tensor()
    input_layer = tf.keras.layers.Flatten(dtype=input_tensor_spec.dtype)(input_tensor)
    network_output = network.build_model(input_layer)
    model = tf.keras.Model(inputs=input_tensor, outputs=network_output)

    # number of parameters
    nparams = model.count_params()

    # expected number of parameters
    input_nodes = size(input_tensor_spec)
    output_nodes = size(output_tensor_spec)
    if isinstance(network, GaussianNetwork):
        output_nodes = tfp.layers.MultivariateNormalTriL.params_size(output_nodes)
    elif isinstance(network, DiagonalGaussianNetwork):
        output_nodes = tfp.layers.IndependentNormal.params_size(output_nodes)
    if num_hidden_layers == 0:
        nparams_exp = input_nodes * output_nodes + output_nodes
    elif num_hidden_layers > 0:
        nparams_exp = (
            (input_nodes + 1) * num_hidden_nodes
            + (num_hidden_layers - 1) * num_hidden_nodes * (num_hidden_nodes + 1)
            + output_nodes * (num_hidden_nodes + 1)
        )

    assert nparams == nparams_exp
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

from typing import List

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import empty_dataset
from trieste.models.keras import GaussianNetwork, get_tensor_spec_from_data


@pytest.mark.parametrize(
    "query_point_shape, observation_shape",
    [
        ([1], [1]),
        ([5], [1]),
        ([5], [2]),
    ],
)
@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3])
def test_gaussian_network_is_correctly_constructed(
    query_point_shape: List[int], observation_shape: List[int], num_hidden_layers: int
) -> None:
    n_obs = 10
    example_data = empty_dataset(query_point_shape, observation_shape)
    query_points = tf.random.uniform([n_obs] + query_point_shape)

    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)
    hidden_layer_args = []
    for i in range(num_hidden_layers):
        hidden_layer_args.append({"units": 10, "activation": "relu"})
    network = GaussianNetwork(
        input_tensor_spec,
        output_tensor_spec,
        hidden_layer_args,
    )
    network_input, network_output = network.connect_layers()
    network_built = tf.keras.Model(inputs=network_input, outputs=network_output)

    # check input shape
    assert network_input.shape[1:] == tf.TensorShape(query_point_shape)

    # testing output shape is more complex as probabilistic layers don't have some properties
    # we make some predictions instead and then check the output is correct
    predictions = network_built.predict(query_points)
    assert predictions.shape == tf.TensorShape([n_obs] + observation_shape)

    # check layers
    assert isinstance(network_built.layers[0], tf.keras.layers.InputLayer)
    assert len(network_built.layers[1:-2]) == num_hidden_layers
    assert isinstance(network_built.layers[-1], tfp.layers.DistributionLambda)

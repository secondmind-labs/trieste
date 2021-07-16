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

import pytest
from collections.abc import Callable

import tensorflow as tf

from trieste.models.keras.networks import (
    LinearNetwork,
    DiagonalGaussianNetwork,
    GaussianNetwork,
    MultilayerFcNetwork,
)
from tests.util.trieste.utils.objectives import hartmann_6_dataset, branin_dataset


# @pytest.fixture(name="input_output_space_bound", params=[1.0, 2.5], scope="session")
# def _input_output_space_bound_fixture(request):
#     return request.param


@pytest.fixture(name="input_space_shape", params=[tuple(), (1,), (2,), (2, 5)], scope="session")
def _input_space_shape_fixture(request):
    return request.param


@pytest.fixture(name="output_space_shape", params=[(1,), (2,)], scope="session")
def _output_space_shape_fixture(request):
    return request.param


@pytest.fixture(name="input_tensor_spec", scope="session")
def _input_tensor_spec_fixture(input_space_shape):
    input_tensor_spec = tf.TensorSpec(
        shape=input_space_shape,
        dtype=tf.float64,
        name="query_points",
    )
    return input_tensor_spec


@pytest.fixture(name="output_tensor_spec", scope="session")
def _output_tensor_spec_fixture(output_space_shape):
    output_tensor_spec = tf.TensorSpec(
        shape=output_space_shape,
        dtype=tf.float64,
        name="observations",
    )
    return output_tensor_spec


@pytest.fixture(name="ensemble_size", params=[1, 3])
def _ensemble_size_fixture(request):
    return request.param


@pytest.fixture(name="num_hidden_layers", params=[0, 1, 3])
def _num_hidden_layers_fixture(request):
    return request.param


@pytest.fixture(name="num_hidden_nodes", params=[1, 10])
def _num_hidden_nodes_fixture(request):
    return request.param


_NEURAL_NETWORK_CLASSES = [
    LinearNetwork,
    DiagonalGaussianNetwork,
    GaussianNetwork,
    MultilayerFcNetwork,
]
@pytest.fixture(name="neural_network", params=_NEURAL_NETWORK_CLASSES)
def _neural_network_fixture(request):
    return request.param


@pytest.fixture(name="bootstrap_data", params=[False, True])
def _bootstrap_data_fixture(request):
    return request.param


@pytest.fixture(name="branin_dataset_function", scope="session")
def _branin_dataset_function_fixture() -> Callable:
    return branin_dataset


@pytest.fixture(name="hartmann_6_dataset_function", scope="session")
def _hartmann_6_dataset_function_fixture() -> Callable:
    return hartmann_6_dataset


_EXAMPLE_DATASET = [
    branin_dataset(200),
    hartmann_6_dataset(200),
]
@pytest.fixture(name="example_data", params=_EXAMPLE_DATASET, scope="session")
def _example_dataset_fixture(request):
    return request.param

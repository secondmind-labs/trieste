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
import tensorflow as tf

from trieste.models.keras.networks import (
    LinearNetwork,
    DiagonalGaussianNetwork,
    GaussianNetwork,
    MultilayerFcNetwork,
)
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset
from trieste.space import Box
from trieste.utils.objectives import branin, hartmann_6, mk_observer



# @pytest.fixture(name="gym_space_bound", params=[1.0, 2.5], scope="session")
# def _gym_space_bound_fixture(request):
#     return request.param


# @pytest.fixture(name="gym_space_shape", params=[tuple(), (1,), (5, 10)], scope="session")
# def _gym_space_shape_fixture(request):
#     return request.param


# @pytest.fixture(name="gym_space_shape_latent_obs", params=[(2, 3), (5, 10)], scope="session")
# def _gym_space_shape_latent_obs_fixture(request):
#     return request.param


# @pytest.fixture(name="observation_space", scope="session")
# def _observation_space_fixture(gym_space_bound, gym_space_shape):
#     gym_space = gym.spaces.Box(
#         low=-gym_space_bound, high=gym_space_bound, shape=gym_space_shape, dtype=np.float32
#     )
#     return tensor_spec.from_spec(spec_from_gym_space(gym_space, name="observation"))


# @pytest.fixture(name="action_space", scope="session")
# def _action_space_fixture(gym_space_bound, gym_space_shape):
#     gym_space = gym.spaces.Box(
#         low=-gym_space_bound, high=gym_space_bound, shape=gym_space_shape, dtype=np.float32
#     )
#     return tensor_spec.from_spec(spec_from_gym_space(gym_space, name="action"))


# @pytest.fixture(name="observation_space_latent_obs", scope="session")
# def _observation_space_latent_obs_fixture(gym_space_bound, gym_space_shape_latent_obs):
#     gym_space = gym.spaces.Box(
#         low=-gym_space_bound,
#         high=gym_space_bound,
#         shape=gym_space_shape_latent_obs,
#         dtype=np.float32,
#     )
#     return tensor_spec.from_spec(spec_from_gym_space(gym_space, name="observation"))


# @pytest.fixture(name="action_space_latent_obs", scope="session")
# def _action_space_latent_obs_fixture(gym_space_bound, gym_space_shape_latent_obs):
#     gym_space = gym.spaces.Box(
#         low=-gym_space_bound,
#         high=gym_space_bound,
#         shape=gym_space_shape_latent_obs,
#         dtype=np.float32,
#     )
#     return tensor_spec.from_spec(spec_from_gym_space(gym_space, name="action"))


@pytest.fixture(name="ensemble_size", params=[1, 3])
def _ensemble_size_fixture(request):
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


def _branin_example_data(num_query_points = 100) -> Dataset:

    search_space = Box([0, 0], [1, 1])
    query_points = search_space.sample(num_query_points)
    
    observer = mk_observer(branin, OBJECTIVE)
    data = observer(query_points)

    return data[OBJECTIVE]

@pytest.fixture(name="branin_example_data", scope="session")
def _branin_example_data_fixture(num_query_points = 100) -> Dataset:
    return _branin_example_data(num_query_points)


def _hartmann_6_example_data(num_query_points = 1000) -> Dataset:

    search_space = Box([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1])
    query_points = search_space.sample(num_query_points)
    
    observer = mk_observer(hartmann_6, OBJECTIVE)
    data = observer(query_points)

    return data[OBJECTIVE]

@pytest.fixture(name="hartmann_6_example_data", scope="session")
def _hartmann_6_example_data_fixture(num_query_points = 100) -> Dataset:
    return _hartmann_6_example_data(num_query_points)


_EXAMPLE_DATA = [
    _branin_example_data(),
    _hartmann_6_example_data(),
]
@pytest.fixture(name="example_data", params=_EXAMPLE_DATA, scope="session")
def _example_data_fixture(request):
    return request.param

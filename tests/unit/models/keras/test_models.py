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

from trieste.data import Dataset
from trieste.models.optimizer import TFKerasOptimizer
from trieste.models.keras.models import NeuralNetworkEnsemble
from trieste.models.keras.networks import MultilayerFcNetwork
from trieste.models.keras.utils import get_tensor_spec_from_data

_ENSEMBLE_SIZE = 5



# def test_step_call_shape(
#     transition_network,
#     observation_space,
#     action_space,
#     batch_size,
#     ensemble_size,
# ):
#     network_list = [
#         transition_network(observation_space, bootstrap_data=True)
#         for _ in range(ensemble_size)
#     ]
#     transition_model = KerasTransitionModel(
#         network_list,
#         observation_space,
#         action_space,
#         predict_state_difference=True,
#         trajectory_sampling_strategy=OneStepTrajectorySampling(batch_size, ensemble_size),
#     )
#     observation_distribution = create_uniform_distribution_from_spec(observation_space)
#     observations = observation_distribution.sample((batch_size,))
#     action_distribution = create_uniform_distribution_from_spec(action_space)
#     actions = action_distribution.sample((batch_size,))

#     next_observations = transition_model.step(observations, actions)

#     assert next_observations.shape == (batch_size,) + observation_space.shape
#     assert observation_space.is_compatible_with(next_observations[0])


# def test_step_call_goal_state_transform(
#     transition_network,
#     observation_space_latent_obs,
#     action_space_latent_obs,
#     batch_size,
#     ensemble_size,
# ):
#     latent_observation_space_spec = BoundedTensorSpec(
#         shape=observation_space_latent_obs.shape[:-1]
#         + [observation_space_latent_obs.shape[-1] - 1],
#         dtype=observation_space_latent_obs.dtype,
#         minimum=observation_space_latent_obs.minimum,
#         maximum=observation_space_latent_obs.maximum,
#         name=observation_space_latent_obs.name,
#     )
#     network_list = [
#         transition_network(latent_observation_space_spec, bootstrap_data=True)
#         for _ in range(ensemble_size)
#     ]
#     observation_transformation = GoalStateObservationTransformation(
#         latent_observation_space_spec=latent_observation_space_spec,
#         goal_state_start_index=-1,
#     )
#     transition_model = KerasTransitionModel(
#         network_list,
#         observation_space_latent_obs,
#         action_space_latent_obs,
#         predict_state_difference=True,
#         trajectory_sampling_strategy=OneStepTrajectorySampling(batch_size, ensemble_size),
#         observation_transformation=observation_transformation,
#     )
#     observation_distribution = create_uniform_distribution_from_spec(
#         observation_space_latent_obs
#     )
#     observations = observation_distribution.sample((batch_size,))
#     action_distribution = create_uniform_distribution_from_spec(action_space_latent_obs)
#     actions = action_distribution.sample((batch_size,))

#     next_observations = transition_model.step(observations, actions)

#     assert next_observations.shape == (batch_size,) + observation_space_latent_obs.shape
#     assert observation_space_latent_obs.is_compatible_with(next_observations[0])
#     tf.assert_equal(next_observations[..., -1], observations[..., -1])



# def test_mismatch_ensemble_size(
#     observation_space, action_space, trajectory_sampling_strategy_factory, batch_size
# ):
#     """
#     Ensure that the ensemble size specified in the trajectory sampling strategy is equal to the
#     number of networks in the models.
#     """
#     strategy = trajectory_sampling_strategy_factory(batch_size, 2)
#     if isinstance(strategy, SingleFunction):
#         pytest.skip("SingleFunction strategy is not an ensemble strategy.")

#     with pytest.raises(AssertionError):
#         KerasTransitionModel(
#             [LinearTransitionNetwork(observation_space)],
#             observation_space,
#             action_space,
#             trajectory_sampling_strategy=strategy,
#         )


# def test_fit_improves(training_data: Dataset):
    
#     input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(training_data)
#     networks = [
#         MultilayerFcNetwork(
#             input_tensor_spec,
#             output_tensor_spec,
#             num_hidden_layers=0,
#             bootstrap_data=False,
#         )
#         for _ in range(ensemble_size)
#     ]
#     breakpoint()
#     optimizer = tf.keras.optimizers.Adam()
#     fit_args = {
#         'batch_size': 16,
#         'epochs': 10,
#         'callbacks': [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)],
#         'validation_split': 0.1,
#         'verbose': 1,
#     }
#     model = NeuralNetworkEnsemble(networks, TFKerasOptimizer(optimizer, fit_args))

#     history = model.optimize(dataset)

#     assert history.history["loss"][-1] < history.history["loss"][0]


# @pytest.mark.parametrize("training_data", [branin_training_data, hartmann_6_training_data])
# def test_ensemble_model_close_to_actuals(training_data: Dataset):
    
#     input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(training_data)
#     networks = [
#         MultilayerFcNetwork(
#             input_tensor_spec,
#             output_tensor_spec,
#             num_hidden_layers=0,
#             bootstrap_data=False,
#         )
#         for _ in range(ensemble_size)
#     ]
#     breakpoint()
#     optimizer = tf.keras.optimizers.Adam()
#     fit_args = {
#         'batch_size': 16,
#         'epochs': 10,
#         'callbacks': [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)],
#         'validation_split': 0.1,
#         'verbose': 1,
#     }
#     model = NeuralNetworkEnsemble(networks, TFKerasOptimizer(optimizer, fit_args))


#     model.optimize(dataset)

#     assert_rollouts_are_close_to_actuals(model, max_steps=1)

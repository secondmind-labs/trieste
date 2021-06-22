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



@pytest.mark.skip("Stochastic failure to investigate.")
def test_fit_mountain_car_data(
    mountain_car_data, transition_network, bootstrap_data, batch_size, ensemble_size
):
    tf_env, trajectories = mountain_car_data

    network_list = [
        transition_network(tf_env.observation_spec(), bootstrap_data=bootstrap_data)
        for _ in range(ensemble_size)
    ]
    transition_model = KerasTransitionModel(
        network_list,
        tf_env.observation_spec(),
        tf_env.action_spec(),
        predict_state_difference=False,
        trajectory_sampling_strategy=OneStepTrajectorySampling(batch_size, ensemble_size),
    )

    training_spec = KerasTrainingSpec(
        epochs=10,
        training_batch_size=256,
        callbacks=[],
    )

    history = transition_model.train(trajectories, training_spec)

    assert history.history["loss"][-1] < history.history["loss"][0]


@pytest.mark.parametrize("training_data", [branin_training_data, hartmann_6_training_data])
def test_ensemble_model_close_to_actuals(training_data: Dataset):
    
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(training_data)
    networks = [
        MultilayerFcNetwork(
            input_tensor_spec,
            output_tensor_spec,
            num_hidden_layers=0,
            bootstrap_data=False,
        )
        for _ in range(ensemble_size)
    ]
    breakpoint()
    optimizer = tf.keras.optimizers.Adam()
    fit_args = {
        'batch_size': 16,
        'epochs': 10,
        'callbacks': [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)],
        'validation_split': 0.1,
        'verbose': 1,
    }
    model = NeuralNetworkEnsemble(networks, TFKerasOptimizer(optimizer, fit_args))


    model.optimize(dataset)

    assert_rollouts_are_close_to_actuals(model, max_steps=1)

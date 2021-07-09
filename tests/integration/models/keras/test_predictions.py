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
import numpy as np
import tensorflow as tf

from trieste.data import Dataset
from trieste.models.keras.data import EnsembleDataTransformer
from trieste.models.optimizer import TFKerasOptimizer
from trieste.models.keras.models import NeuralNetworkEnsemble
from trieste.models.keras.networks import MultilayerFcNetwork
from trieste.models.keras.utils import get_tensor_spec_from_data

_ENSEMBLE_SIZE = 5

tf.keras.backend.set_floatx('float64')


# def test_fit_mountain_car_data(
#     mountain_car_data, transition_network, bootstrap_data, batch_size, ensemble_size
# ):
#     tf_env, trajectories = mountain_car_data

#     network_list = [
#         transition_network(tf_env.observation_spec(), bootstrap_data=bootstrap_data)
#         for _ in range(ensemble_size)
#     ]
#     transition_model = KerasTransitionModel(
#         network_list,
#         tf_env.observation_spec(),
#         tf_env.action_spec(),
#         predict_state_difference=False,
#         trajectory_sampling_strategy=OneStepTrajectorySampling(batch_size, ensemble_size),
#     )

#     training_spec = KerasTrainingSpec(
#         epochs=10assert_rollouts_are_close_to_actuals,
#         training_batch_size=256,
#         callbacks=[],
#     )

#     history = transition_model.train(trajectories, training_spec)

#     assert history.history["loss"][-1] < history.history["loss"][0]


# @pytest.mark.parametrize("example_data", [hartmann_6_example_data])
def test_ensemble_model_close_to_actuals(example_data):
    
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)
    networks = [
        MultilayerFcNetwork(
            input_tensor_spec,
            output_tensor_spec,
            num_hidden_layers=3,
            units=[250,250,250],
            # activation=[None],
            # use_bias=[None],
            # kernel_initializer=[None],
            # bias_initializer=[None],
            # kernel_regularizer=[None],
            # bias_regularizer=[None],
            # activity_regularizer=[None],
            # kernel_constraint=[None],
            # bias_constraint=[None],
            bootstrap_data=False,
        )
        for _ in range(_ENSEMBLE_SIZE)
    ]
    # breakpoint()
    optimizer = tf.keras.optimizers.Adam()
    fit_args = {
        'batch_size': 256,
        'epochs': 100,
        'callbacks': [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)],
        'validation_split': 0.2,
        'verbose': 1,
    }
    dataset_builder = EnsembleDataTransformer(networks)
    model = NeuralNetworkEnsemble(
        networks,
        dataset_builder,
        TFKerasOptimizer(optimizer, fit_args, dataset_builder)
    )
    # breakpoint()

    history = model.optimize(example_data)

    x, y = dataset_builder(example_data)
    predicted_means, predicted_vars = model.predict(x)

    observations = y[list(y)[0]]
    breakpoint()
    # mean_absolute_error = tf.reduce_mean(tf.abs(predicted_means - observations))
    np.testing.assert_allclose(
        predicted_means, observations, atol=1e-1, rtol=2e-1
    )

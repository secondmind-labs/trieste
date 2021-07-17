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
from tests.util.misc import random_seed

tf.keras.backend.set_floatx('float64')


@random_seed
def test_ensemble_model_close_to_actuals(hartmann_6_dataset_function):
    """
    Ensure that ensemble model fits well and predictions are close to actual output values.
    """

    ensemble_size = 5
    dataset_size = 3000

    example_data = hartmann_6_dataset_function(dataset_size)
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)
    networks = [
        MultilayerFcNetwork(
            input_tensor_spec,
            output_tensor_spec,
            num_hidden_layers=3,
            units=[32, 32, 32],
            activation=['relu', 'relu', 'relu'],
            bootstrap_data=False,
        )
        for _ in range(ensemble_size)
    ]
    optimizer = tf.keras.optimizers.Adam()
    fit_args = {
        'batch_size': 32,
        'epochs': 200,
        'callbacks': [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)],
        'verbose': 0,
    }
    dataset_builder = EnsembleDataTransformer(networks)
    model = NeuralNetworkEnsemble(
        networks,
        TFKerasOptimizer(optimizer, fit_args, dataset_builder),
        dataset_builder,
    )
    model.optimize(example_data)

    x, y = dataset_builder(example_data)
    # model.predict(example_data.query_points)
    predicted_means, predicted_vars = model.predict(x)
    observations = y[list(y)[0]]

    np.testing.assert_allclose(predicted_means, observations, atol=0.1, rtol=0.2)


@random_seed
def test_ensemble_model_close_to_simple_implementation(hartmann_6_dataset_function):
    
    """
    Ensure that trieste implementation produces similar result to a direct keras implementation.
    """

    ensemble_size = 1
    dataset_size = 3000

    # dataset
    example_data = hartmann_6_dataset_function(dataset_size)
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)

    # trieste implementation
    networks = [
        MultilayerFcNetwork(
            input_tensor_spec,
            output_tensor_spec,
            num_hidden_layers=3,
            units=[32, 32, 32],
            activation=['relu', 'relu', 'relu'],
            bootstrap_data=False,
        )
        for _ in range(ensemble_size)
    ]
    optimizer = tf.keras.optimizers.Adam()
    fit_args = {
        'batch_size': 32,
        'epochs': 200,
        'callbacks': [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)],
        'verbose': 0,
    }
    dataset_builder = EnsembleDataTransformer(networks)
    trieste_model = NeuralNetworkEnsemble(
        networks,
        TFKerasOptimizer(optimizer, fit_args, dataset_builder),
        dataset_builder,
    )
    trieste_model.optimize(example_data)
    x, _ = dataset_builder(example_data)
    trieste_model_predictions = trieste_model.model.predict(x)

    # simpler but equal implementation
    simple_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=input_tensor_spec.shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_tensor_spec.shape[-1], activation='linear'),
    ])
    simple_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mse'],
    )
    x = tf.convert_to_tensor(example_data.query_points)
    y = tf.convert_to_tensor(example_data.observations)
    simple_model.fit(x, y, **fit_args)
    simple_model_predictions = simple_model.predict(x)
    
    # examine the match
    np.testing.assert_allclose(
        trieste_model_predictions,
        simple_model_predictions,
        atol=0.2,
        rtol=0.2,
    )

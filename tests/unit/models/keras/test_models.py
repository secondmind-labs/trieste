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
from trieste.models.keras.data import EnsembleDataTransformer
from trieste.models.keras.networks import LinearNetwork

_ENSEMBLE_SIZE = 3
_DATASET_SIZE = 1000


def _create_neural_network_ensemble_model(
    example_data, neural_network, ensemble_size, bootstrap_data
):

    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)

    if neural_network.__name__ == "LinearNetwork":
        networks = [
            neural_network(
                input_tensor_spec,
                output_tensor_spec,
                bootstrap_data=bootstrap_data,
            )
            for _ in range(ensemble_size)
        ]
    else:
        networks = [
            neural_network(
                input_tensor_spec,
                output_tensor_spec,
                num_hidden_layers=2,
                units=[32, 32],
                activation=['relu', 'relu'],
                bootstrap_data=bootstrap_data,
            )
            for _ in range(ensemble_size)
        ]
    optimizer = tf.keras.optimizers.Adam()
    fit_args = {
        'batch_size': 32,
        'epochs': 10,
        'callbacks': [],
        'verbose': 0,
    }
    dataset_builder = EnsembleDataTransformer(networks)
    model = NeuralNetworkEnsemble(
        networks,
        TFKerasOptimizer(optimizer, fit_args, dataset_builder),
        dataset_builder,
    )

    return model, dataset_builder


def test_neural_network_ensemble_predict_call_shape(
    hartmann_6_dataset_function, neural_network, ensemble_size, bootstrap_data
):
    example_data = hartmann_6_dataset_function(int(_DATASET_SIZE/10))
    model, dataset_builder = _create_neural_network_ensemble_model(example_data, neural_network, ensemble_size, bootstrap_data)
    x, y = dataset_builder(example_data)

    # model.predict(example_data.query_points)
    predicted_means, predicted_vars = model.predict(x)
    ensemble_predictions = model.model.predict(x)
    if ensemble_size == 1:
        assert tf.reduce_all(tf.math.is_inf(predicted_vars))
        assert ensemble_predictions.shape == example_data.observations.shape
    else:
        assert isinstance(ensemble_predictions, list)
        assert len(ensemble_predictions) == ensemble_size
    assert tf.is_tensor(predicted_vars)
    assert predicted_vars.shape == example_data.observations.shape
    assert tf.is_tensor(predicted_means)
    assert predicted_means.shape == example_data.observations.shape


# @random_seed
# def test_neural_network_ensemble_fit_improves(
#     hartmann_6_dataset_function, neural_network, bootstrap_data, ensemble_size
# ):
#     """
#     Ensure that fit improves with several epochs of optimization.
#     """

#     example_data = hartmann_6_dataset_function(_DATASET_SIZE)
#     model = _create_neural_network_ensemble_model(example_data, neural_network, ensemble_size)

#     model.optimize(example_data)
#     loss = model.model.history.history["loss"]

#     assert loss[-1] < loss[0]

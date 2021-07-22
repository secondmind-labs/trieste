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
from tests.util.misc import random_seed

tf.keras.backend.set_floatx('float64')

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
    dataset_builder = EnsembleDataTransformer(networks, bootstrap_data)
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
    model, dataset_builder = _create_neural_network_ensemble_model(
        example_data,
        neural_network,
        ensemble_size,
        bootstrap_data
    )
    query_points_for_internal_predict = dataset_builder.ensemblise_inputs(example_data.query_points)
    
    predicted_means, predicted_vars = model.predict(example_data.query_points)
    ensemble_predictions = model.model.predict(query_points_for_internal_predict)
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


@random_seed
def test_neural_network_ensemble_fit_improves(
    hartmann_6_dataset_function, neural_network, ensemble_size, bootstrap_data
):
    """
    Ensure that fit improves with several epochs of optimization.
    """

    example_data = hartmann_6_dataset_function(_DATASET_SIZE)
    model, dataset_builder = _create_neural_network_ensemble_model(
        example_data,
        neural_network,
        ensemble_size,
        bootstrap_data
    )   

    model.optimize(example_data)
    loss = model.model.history.history["loss"]

    assert loss[-1] < loss[0]






# @random_seed
# def test_gpflow_predictor_sample() -> None:
#     model = _QuadraticPredictor()
#     num_samples = 20_000
#     samples = model.sample(tf.constant([[2.5]], gpflow.default_float()), num_samples)

#     assert samples.shape == [num_samples, 1, 1]

#     sample_mean = tf.reduce_mean(samples, axis=0)
#     sample_variance = tf.reduce_mean((samples - sample_mean) ** 2)

#     linear_error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
#     npt.assert_allclose(sample_mean, [[6.25]], rtol=linear_error)
#     npt.assert_allclose(sample_variance, 1.0, rtol=2 * linear_error)


# def test_gpflow_predictor_sample_no_samples() -> None:
#     samples = _QuadraticPredictor().sample(tf.constant([[50.0]], gpflow.default_float()), 0)
#     assert samples.shape == (0, 1, 1)


# def test_sparse_variational_model_attribute() -> None:
#     model = _svgp(_mock_data()[0])
#     sv = SparseVariational(model, Dataset(*_mock_data()))
#     assert sv.model is model

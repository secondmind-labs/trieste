
from typing import Any, List, Tuple

import numpy as np
import pytest
import tensorflow as tf

from tests.util.misc import empty_dataset
from trieste.models.keras import (
    DropConnectNetwork,
    DropoutNetwork,
    get_tensor_spec_from_data,
    negative_log_likelihood,
)

@pytest.fixture(name="dropout_network", params=[DropConnectNetwork, DropoutNetwork])
def _dropout_network_fixture(request: Any) -> DropoutNetwork:
    return request.param

def test_dropout_network_model_attributes(dropout_network: DropoutNetwork) -> None:
    example_data = empty_dataset([1], [1])
    inputs, outputs = get_tensor_spec_from_data(example_data)
    dropout_nn = dropout_network(inputs, outputs)

    assert isinstance(dropout_nn.model, tf.keras.Model)


def test_dropout_network_build_seems_correct() -> None:
    ...

    # n_obs = 10
    # example_data = empty_dataset(query_point_shape, observation_shape)
    # query_points = tf.random.uniform([n_obs] + query_point_shape)
    # keras_ensemble = trieste_keras_ensemble_model(example_data, ensemble_size, independent_normal)

    # # basics
    # assert isinstance(keras_ensemble.model, tf.keras.Model)
    # assert keras_ensemble.model.built

    # # check input shape
    # for shape in keras_ensemble.model.input_shape:
    #     assert shape[1:] == tf.TensorShape(query_point_shape)

    # # testing output shape is more complex as probabilistic layers don't have some properties
    # # we make some predictions instead and then check the output is correct
    # predictions = keras_ensemble.model.predict([query_points] * ensemble_size)
    # assert len(predictions) == ensemble_size
    # for pred in predictions:
    #     assert pred.shape == tf.TensorShape([n_obs] + observation_shape)

    # # check input/output names
    # for ens in range(ensemble_size):
    #     ins = ["model_" + str(ens) in i_name for i_name in keras_ensemble.model.input_names]
    #     assert np.any(ins)
    #     outs = ["model_" + str(ens) in o_name for o_name in keras_ensemble.model.output_names]
    #     assert np.any(outs)

    # # check the model has not been compiled
    # assert keras_ensemble.model.compiled_loss is None
    # assert keras_ensemble.model.compiled_metrics is None
    # assert keras_ensemble.model.optimizer is None

    # # check correct number of layers
    # assert len(keras_ensemble.model.layers) == 2 * ensemble_size + 3 * ensemble_size


def test_dropout_network_can_be_compiled(dropout_network: DropoutNetwork) -> None:

    example_data = empty_dataset([1], [1])
    inputs, outputs = get_tensor_spec_from_data(example_data)

    dropout_nn = dropout_network(inputs, outputs)

    dropout_nn.model.compile(tf.optimizers.Adam(), tf.losses.MeanSquaredError())

    assert dropout_nn.model.compiled_loss is not None
    assert dropout_nn.model.compiled_metrics is not None
    assert dropout_nn.model.optimizer is not None

def test_dropout_network_is_correctly_constructed() -> None:
    ...

    # n_obs = 10
    # example_data = empty_dataset(query_point_shape, observation_shape)
    # query_points = tf.random.uniform([n_obs] + query_point_shape)

    # input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)
    # hidden_layer_args = []
    # for i in range(num_hidden_layers):
    #     hidden_layer_args.append({"units": 10, "activation": "relu"})
    # network = GaussianNetwork(
    #     input_tensor_spec,
    #     output_tensor_spec,
    #     hidden_layer_args,
    # )
    # network_input, network_output = network.connect_layers()
    # network_built = tf.keras.Model(inputs=network_input, outputs=network_output)

    # # check input shape
    # assert network_input.shape[1:] == tf.TensorShape(query_point_shape)

    # # testing output shape is more complex as probabilistic layers don't have some properties
    # # we make some predictions instead and then check the output is correct
    # predictions = network_built.predict(query_points)
    # assert predictions.shape == tf.TensorShape([n_obs] + observation_shape)

    # # check layers
    # assert isinstance(network_built.layers[0], tf.keras.layers.InputLayer)
    # assert len(network_built.layers[1:-2]) == num_hidden_layers
    # assert isinstance(network_built.layers[-1], tfp.layers.DistributionLambda)

#%%
from typing import Any, List, Tuple, Union

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
from trieste.models.keras.layers import DropConnect

@pytest.fixture(name="dropout_network", params=[DropoutNetwork, DropConnectNetwork])
def _dropout_network_fixture(request: Any) -> DropoutNetwork:
    return request.param

@pytest.fixture(name="query_point_shape", params = [[1], [5]])
def _query_point_shape_fixture(request: Any) -> List[int]:
    return request.param

@pytest.fixture(name="observation_shape", params = [[1], [2]])
def _observation_shape_fixture(request: Any) -> List[int]:
    return request.param

@pytest.mark.parametrize(
    "num_hidden_layers, rate", 
    [
        (1, 0.3),
        (3, 0.7),
        (3, [0.2, 0.3, 0.6, 0.5]),
        (5, [0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    ]
)
@pytest.mark.parametrize("units", [10, 50])
@pytest.mark.parametrize("activation", ["relu", tf.keras.activations.tanh])
def test_dropout_network_build_seems_correct(
    dropout_network: DropoutNetwork, 
    query_point_shape: List[int], 
    observation_shape: List[int], 
    num_hidden_layers: int,
    units: int,
    activation : Union[str, tf.keras.layers.Activation],
    rate: Union[float, int, List[Union[int, float]]]
) -> None:
    '''Tests the correct consturction of dropout network architectures'''
    
    example_data = empty_dataset(query_point_shape, observation_shape)
    inputs, outputs = get_tensor_spec_from_data(example_data)
    hidden_layer_args = [{"units": units, "activation": activation} for _ in range(num_hidden_layers)]
    
    dropout_nn = dropout_network(
        inputs, 
        outputs,
        hidden_layer_args,
        rate
    )

    if not isinstance(rate, list):
        rate = [rate for _ in range(num_hidden_layers + 1)]

    # basics
    assert isinstance(dropout_nn.model, tf.keras.Model)
    assert dropout_nn.model.built

    # check input and output shapes
    assert dropout_nn.model.input_shape[1:] == tf.TensorShape(query_point_shape)
    assert dropout_nn.model.output_shape[1:] == tf.TensorShape(observation_shape)

    # check the model has not been compiled
    assert dropout_nn.model.compiled_loss is None
    assert dropout_nn.model.compiled_metrics is None
    assert dropout_nn.model.optimizer is None

    # check input layer
    assert isinstance(dropout_nn.model.layers[0], tf.keras.layers.InputLayer)
    
    # check correct number of layers and proerply constructed
    if isinstance(dropout_nn, DropConnectNetwork):
        assert len(dropout_nn.model.layers) == 2 + num_hidden_layers
        
        for layer in dropout_nn.model.layers[1:-1]:
            assert isinstance(layer, DropConnect)
            assert layer.units == units
            assert layer.activation == activation or layer.activation.__name__ == activation
        
        assert isinstance(dropout_nn.model.layers[-1], DropConnect)
    
    elif isinstance(dropout_nn, DropoutNetwork):
        assert len(dropout_nn.model.layers) == 1 + 2 * (num_hidden_layers + 1)
        
        for i, layer in enumerate(dropout_nn.model.layers[1:-1]):
            if i % 2 == 0:
                isinstance(layer, tf.keras.layers.Dropout)
                layer.rate == rate[int(i/2)]
            elif i % 2 == 1:
                isinstance(layer, tf.keras.layers.Dense)
                assert layer.units == units
                assert layer.activation == activation or layer.activation.__name__ == activation
        
        assert isinstance(dropout_nn.model.layers[-1], tf.keras.layers.Dense)
    
    # check output layer activation
    assert dropout_nn.model.layers[-1].activation == tf.keras.activations.linear

def test_dropout_network_can_be_compiled(
    dropout_network: DropoutNetwork, 
    query_point_shape: List[int], 
    observation_shape: List[int]
 ) -> None:
    '''Checks that dropout networks are compilable.'''

    example_data = empty_dataset(query_point_shape, observation_shape)
    inputs, outputs = get_tensor_spec_from_data(example_data)

    dropout_nn = dropout_network(inputs, outputs)

    dropout_nn.model.compile(tf.optimizers.Adam(), tf.losses.MeanSquaredError())

    assert dropout_nn.model.compiled_loss is not None
    assert dropout_nn.model.compiled_metrics is not None
    assert dropout_nn.model.optimizer is not None
# %%

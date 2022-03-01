from typing import Union, Sequence

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import empty_dataset
from layers import DropConnect
from builders import build_vanilla_keras_mcdropout


@pytest.mark.parametrize("units, activation", [(10, "relu"), (50, tf.keras.activations.tanh)])
@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3])
@pytest.mark.parametrize("dropout_prob", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("dropout", ["standard", "dropconnect"])
def test_build_vanilla_keras_mcdropout(
    num_hidden_layers: int,
    units: int,
    activation: Union[str, tf.keras.layers.Activation],
    dropout_prob,
    dropout:str
) -> None:
    example_data = empty_dataset([1], [1])
    mcdropout = build_vanilla_keras_mcdropout(
        example_data,
        num_hidden_layers,
        units,
        activation,
        dropout_prob,
        dropout
    )

    assert len(mcdropout.layers) == num_hidden_layers + 1, f"Model does not have the correct number of layers. Expected {num_hidden_layers + 1} got {len(mcdropout.layers)}"
    if num_hidden_layers > 0:
        for i, layer in enumerate(mcdropout.layers):
            if dropout == "dropconnect":
                assert isinstance(layer, DropConnect) 
                assert layer.units == units
                assert layer.activation == activation
            elif dropout == "standard":
                if i % 2 == 0:
                    assert isinstance(layer, tf.keras.layers.Dropout)
                    assert layer.rate == dropout_prob
                if i % 2 == 1:
                    assert isinstance(layer, tf.keras.layers.Dense)
                    assert layer.units == units
                    assert layer.activation == activation
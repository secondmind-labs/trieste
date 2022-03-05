from typing import Union, Sequence

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import empty_dataset
from trieste.models.keras.architectures import DropoutNetwork, DropConnectNetwork
from trieste.models.keras.layers import DropConnect
from trieste.models.keras.builders import build_vanilla_keras_mcdropout


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
    if dropout == "standard":
        assert isinstance(mcdropout, DropoutNetwork), f"Model is not correct class. Got {mcdropout.model}."
        assert len(mcdropout.model.layers) == 2 * (num_hidden_layers + 1) + 1, f"Model does not have the correct number of layers. Expected {num_hidden_layers + 1} got {len(mcdropout.model.layers)}"
    elif dropout == "dropconnect":
        assert isinstance(mcdropout, DropConnectNetwork), f"Model is not correct class. Got {mcdropout.model}."
        assert len(mcdropout.model.layers) == num_hidden_layers + 2, f"Model does not have the correct number of layers. Expected {num_hidden_layers + 1} got {len(mcdropout.model.layers)}"
    if num_hidden_layers > 0:
        for i, layer in enumerate(mcdropout.model.layers[1:-2]):
            if dropout == "dropconnect":
                assert isinstance(layer, DropConnect) 
                assert layer.units == units
                assert layer.activation == activation or layer.activation.__name__ == activation
            elif dropout == "standard":
                if i % 2 == 0:
                    assert isinstance(layer, tf.keras.layers.Dropout)
                    assert layer.rate == dropout_prob
                elif i % 2 == 1:
                    assert isinstance(layer, tf.keras.layers.Dense)
                    assert layer.units == units
                    assert layer.activation == activation or layer.activation.__name__ == activation

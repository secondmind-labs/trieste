from typing import Union, Sequence

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import empty_dataset
from builders import build_vanilla_keras_mcdropout


@pytest.mark.parametrize("units, activation", [(10, "relu"), (50, tf.keras.activations.tanh)])
@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3])
@pytest.mark.parametrize("dropout_prob", [0.1, 0.5, 0.9])
def test_build_vanilla_keras_mcdropout(
    num_hidden_layers: int,
    units: int,
    activation: Union[str, tf.keras.layers.Activation],
    dropout_prob
) -> None:
    example_data = empty_dataset([1], [1])
    keras_ensemble = build_vanilla_keras_mcdropout(
        example_data,
        num_hidden_layers,
        units,
        activation,
        dropout_prob
    )

    

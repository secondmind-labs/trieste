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

from typing import Union

import pytest
import tensorflow_probability as tfp
from gpflow.keras import tf_keras

from tests.util.misc import empty_dataset
from tests.util.models.keras.ensemble_layers import (
    activation_matches,
    expected_ensemble_layer_count,
    is_vectorized_ensemble_model,
    per_member_hidden_layers,
    vectorized_hidden_layers,
)
from trieste.models.keras import build_keras_ensemble


@pytest.mark.parametrize("units, activation", [(10, "relu"), (50, tf_keras.activations.tanh)])
@pytest.mark.parametrize("ensemble_size", [2, 5])
@pytest.mark.parametrize("independent_normal", [False, True])
@pytest.mark.parametrize("num_hidden_layers", [0, 1, 3])
@pytest.mark.parametrize("num_outputs", [1, 3])
def test_build_keras_ensemble(
    num_outputs: int,
    ensemble_size: int,
    num_hidden_layers: int,
    units: int,
    activation: Union[str, tf_keras.layers.Activation],
    independent_normal: bool,
) -> None:
    example_data = empty_dataset([num_outputs], [num_outputs])
    keras_ensemble = build_keras_ensemble(
        example_data,
        ensemble_size,
        num_hidden_layers,
        units,
        activation,
        independent_normal,
    )

    assert keras_ensemble.ensemble_size == ensemble_size
    model = keras_ensemble.model
    vectorized = is_vectorized_ensemble_model(model)
    assert len(model.layers) == expected_ensemble_layer_count(
        ensemble_size, num_hidden_layers, vectorized=vectorized
    )
    if num_outputs > 1:
        if independent_normal:
            assert isinstance(model.layers[-1], tfp.layers.IndependentNormal)
        else:
            assert isinstance(model.layers[-1], tfp.layers.MultivariateNormalTriL)
    else:
        assert isinstance(model.layers[-1], tfp.layers.DistributionLambda)
    if num_hidden_layers > 0:
        hidden_layers = (
            vectorized_hidden_layers(model)
            if vectorized
            else per_member_hidden_layers(model, ensemble_size)
        )
        expected_hidden_count = (
            num_hidden_layers if vectorized else num_hidden_layers * ensemble_size
        )
        assert len(hidden_layers) == expected_hidden_count
        for layer in hidden_layers:
            assert layer.units == units
            assert activation_matches(layer, activation)

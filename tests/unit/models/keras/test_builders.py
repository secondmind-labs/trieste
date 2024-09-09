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
    assert len(keras_ensemble.model.layers) == num_hidden_layers * ensemble_size + 3 * ensemble_size
    if num_outputs > 1:
        if independent_normal:
            assert isinstance(keras_ensemble.model.layers[-1], tfp.layers.IndependentNormal)
        else:
            assert isinstance(keras_ensemble.model.layers[-1], tfp.layers.MultivariateNormalTriL)
    else:
        assert isinstance(keras_ensemble.model.layers[-1], tfp.layers.DistributionLambda)
    if num_hidden_layers > 0:
        for layer in keras_ensemble.model.layers[ensemble_size : -ensemble_size * 2]:
            assert layer.units == units
            assert layer.activation == activation or layer.activation.__name__ == activation

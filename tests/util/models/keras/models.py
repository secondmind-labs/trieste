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

"""
Utilities for creating (Keras) neural network models to be used in the tests.
"""

from __future__ import annotations

from typing import Union

import tensorflow as tf

from trieste.data import Dataset
from trieste.models.keras import (
    DeepEnsemble,
    GaussianNetwork,
    KerasEnsemble,
    get_tensor_spec_from_data,
    negative_log_likelihood,
    sample_with_replacement,
)
from trieste.models.optimizer import KerasOptimizer, TrainingData


def ensemblise_data(
    model: KerasEnsemble, data: Dataset, ensemble_size: int, bootstrap: bool = False
) -> TrainingData:
    inputs = {}
    outputs = {}
    for index in range(ensemble_size):
        if bootstrap:
            resampled_data = sample_with_replacement(data)
        else:
            resampled_data = data
        input_name = model.model.input_names[index]
        output_name = model.model.output_names[index]
        inputs[input_name], outputs[output_name] = resampled_data.astuple()

    return inputs, outputs


def trieste_keras_ensemble_model(
    example_data: Dataset,
    ensemble_size: int,
    independent_normal: bool = False,
) -> KerasEnsemble:

    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)

    networks = [
        GaussianNetwork(
            input_tensor_spec,
            output_tensor_spec,
            hidden_layer_args=[
                {"units": 32, "activation": "relu"},
                {"units": 32, "activation": "relu"},
            ],
            independent=independent_normal,
        )
        for _ in range(ensemble_size)
    ]
    keras_ensemble = KerasEnsemble(networks)

    return keras_ensemble


def trieste_deep_ensemble_model(
    example_data: Dataset,
    ensemble_size: int,
    bootstrap_data: bool = False,
    independent_normal: bool = False,
    return_all: bool = False,
    optimizer_default: bool = False,
) -> Union[DeepEnsemble, tuple[DeepEnsemble, KerasEnsemble, KerasOptimizer]]:

    keras_ensemble = trieste_keras_ensemble_model(example_data, ensemble_size, independent_normal)

    optimizer = tf.keras.optimizers.Adam()
    loss = negative_log_likelihood
    fit_args = {
        "batch_size": 32,
        "epochs": 10,
        "callbacks": [],
        "verbose": 0,
    }
    optimizer_wrapper = KerasOptimizer(optimizer, fit_args, loss)
    if optimizer_default:
        optimizer_wrapper = None

    model = DeepEnsemble(keras_ensemble, optimizer_wrapper, bootstrap_data)

    if return_all:
        return model, keras_ensemble, optimizer_wrapper
    else:
        return model

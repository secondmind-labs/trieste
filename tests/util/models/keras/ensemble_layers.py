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

"""Helpers for asserting Keras ensemble model layer structure."""

from typing import Any, Union

from gpflow.keras import tf_keras


def is_vectorized_ensemble_dense_layer(layer: Any) -> bool:
    """True for vectorized dense layers from either ``tf_keras`` or ``tensorflow.keras`` builds."""
    return layer.__class__.__name__ == "VectorizedEnsembleDenseLayer"


def is_vectorized_ensemble_model(model: tf_keras.Model) -> bool:
    """Return True if the model was built via the vectorized ensemble path."""
    return any(is_vectorized_ensemble_dense_layer(layer) for layer in model.layers)


def expected_ensemble_layer_count(
    ensemble_size: int,
    num_hidden_layers: int,
    *,
    vectorized: bool,
) -> int:
    """
    :param ensemble_size: Number of ensemble members.
    :param num_hidden_layers: Number of hidden dense layers per member.
    :param vectorized: Whether the vectorized build path was used.
    :return: Expected ``len(model.layers)``.
    """
    if vectorized:
        # input, reshape, H × vec_dense, vec_params, transpose, distribution
        return num_hidden_layers + 5
    return num_hidden_layers * ensemble_size + 3 * ensemble_size


def vectorized_hidden_layers(model: tf_keras.Model) -> list[Any]:
    """Hidden ``VectorizedEnsembleDenseLayer`` instances (excludes ``vec_params``)."""
    return [
        layer
        for layer in model.layers
        if is_vectorized_ensemble_dense_layer(layer) and layer.name.startswith("vec_dense_")
    ]


def per_member_hidden_layers(
    model: tf_keras.Model, ensemble_size: int
) -> list[tf_keras.layers.Dense]:
    """Hidden ``Dense`` layers from the per-member functional build path."""
    return [
        layer
        for layer in model.layers[ensemble_size : -ensemble_size * 2]
        if isinstance(layer, tf_keras.layers.Dense) and "dense_parameters" not in layer.name
    ]


def activation_matches(
    layer: Any,
    activation: Union[str, tf_keras.layers.Activation],
) -> bool:
    """Check layer activation matches the requested activation (string or callable)."""
    if is_vectorized_ensemble_dense_layer(layer):
        layer_activation: Any = layer._activation
    else:
        layer_activation = layer.activation

    if layer_activation == activation:
        return True
    if isinstance(activation, str):
        return getattr(layer_activation, "__name__", None) == activation
    return getattr(layer_activation, "__name__", None) == activation.__name__

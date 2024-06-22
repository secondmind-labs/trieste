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

from typing import Any, List, Tuple

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.keras import tf_keras

from tests.util.misc import empty_dataset
from tests.util.models.keras.models import trieste_keras_ensemble_model
from trieste.models.keras import (
    GaussianNetwork,
    KerasEnsembleNetwork,
    get_tensor_spec_from_data,
    negative_log_likelihood,
)

_ENSEMBLE_SIZE = 3


@pytest.fixture(name="ensemble_size", params=[2, 5])
def _ensemble_size_fixture(request: Any) -> int:
    return request.param


@pytest.fixture(name="independent_normal", params=[False, True])
def _independent_normal_fixture(request: Any) -> bool:
    return request.param


@pytest.fixture(name="num_hidden_layers", params=[0, 1, 3])
def _num_hidden_layers_fixture(request: Any) -> int:
    return request.param


def test_keras_ensemble_repr(
    ensemble_size: int,
    independent_normal: bool,
) -> None:
    example_data = empty_dataset([1], [1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, ensemble_size, independent_normal)

    expected_repr = f"KerasEnsemble({keras_ensemble._networks!r})"

    assert type(keras_ensemble).__name__ in repr(keras_ensemble)
    assert repr(keras_ensemble) == expected_repr


def test_keras_ensemble_model_attributes() -> None:
    example_data = empty_dataset([1], [1])
    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE)

    assert isinstance(keras_ensemble.model, tf_keras.Model)


def test_keras_ensemble_ensemble_size_attributes(ensemble_size: int) -> None:
    example_data = empty_dataset([1], [1])
    keras_ensemble = trieste_keras_ensemble_model(example_data, ensemble_size)

    assert keras_ensemble.ensemble_size == ensemble_size


@pytest.mark.parametrize(
    "query_point_shape, observation_shape",
    [
        ([1], [1]),
        ([5], [1]),
        ([5], [2]),
    ],
)
def test_keras_ensemble_build_ensemble_seems_correct(
    ensemble_size: int,
    independent_normal: bool,
    query_point_shape: List[int],
    observation_shape: List[int],
) -> None:
    n_obs = 10
    example_data = empty_dataset(query_point_shape, observation_shape)
    query_points = tf.random.uniform([n_obs] + query_point_shape)
    keras_ensemble = trieste_keras_ensemble_model(example_data, ensemble_size, independent_normal)

    # basics
    assert isinstance(keras_ensemble.model, tf_keras.Model)
    assert keras_ensemble.model.built

    # check ensemble size
    assert len(keras_ensemble.model.inputs) == ensemble_size
    assert len(keras_ensemble.model.input_names) == ensemble_size
    assert len(keras_ensemble.model.output_names) == ensemble_size

    # check input shape
    for shape in keras_ensemble.model.input_shape:
        assert shape[1:] == tf.TensorShape(query_point_shape)

    # testing output shape is more complex as probabilistic layers don't have some properties
    # we make some predictions instead and then check the output is correct
    predictions = keras_ensemble.model.predict([query_points] * ensemble_size)
    assert len(predictions) == ensemble_size
    for pred in predictions:
        assert pred.shape == tf.TensorShape([n_obs] + observation_shape)

    # check input/output names
    for ens in range(ensemble_size):
        ins = ["model_" + str(ens) in i_name for i_name in keras_ensemble.model.input_names]
        assert np.any(ins)
        outs = ["model_" + str(ens) in o_name for o_name in keras_ensemble.model.output_names]
        assert np.any(outs)

    # check the model has not been compiled
    assert keras_ensemble.model.compiled_loss is None
    assert keras_ensemble.model.compiled_metrics is None
    assert keras_ensemble.model.optimizer is None

    # check correct number of layers
    assert len(keras_ensemble.model.layers) == 2 * ensemble_size + 3 * ensemble_size


def test_keras_ensemble_can_be_compiled() -> None:
    example_data = empty_dataset([1], [1])
    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE)

    keras_ensemble.model.compile(tf_keras.optimizers.Adam(), negative_log_likelihood)

    assert keras_ensemble.model.compiled_loss is not None
    assert keras_ensemble.model.compiled_metrics is not None
    assert keras_ensemble.model.optimizer is not None


class _DummyKerasEnsembleNetwork(KerasEnsembleNetwork):
    def connect_layers(self) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError


def test_keras_ensemble_network_raises_on_incorrect_tensor_spec() -> None:
    with pytest.raises(ValueError):
        _DummyKerasEnsembleNetwork(
            [1],
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf_keras.losses.MeanSquaredError(),
        )

    with pytest.raises(ValueError):
        _DummyKerasEnsembleNetwork(
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            [1],
            tf_keras.losses.MeanSquaredError(),
        )


def test_keras_ensemble_network_network_and_layer_name() -> None:
    model = _DummyKerasEnsembleNetwork(
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )

    # check defaults
    assert model.network_name == ""
    assert model.input_layer_name == "input"
    assert model.output_layer_name == "output"

    # check that network name is changed
    model.network_name = "model_"
    assert model.network_name == "model_"
    assert model.input_layer_name == "model_" + "input"
    assert model.output_layer_name == "model_" + "output"


@pytest.mark.parametrize("n_dims", list(range(10)))
def test_keras_ensemble_network_flattened_output_shape(n_dims: int) -> None:
    shape = np.random.randint(1, 10, (n_dims,))
    tensor = np.random.randint(0, 1, shape)
    tensor_spec = tf.TensorSpec(shape)

    model = _DummyKerasEnsembleNetwork(
        tensor_spec,
        tensor_spec,
    )
    flattened_shape = model.flattened_output_shape

    assert flattened_shape == np.size(tensor)


def test_gaussian_network_check_default_hidden_layer_args() -> None:
    example_data = empty_dataset([1], [1])
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)

    network = GaussianNetwork(
        input_tensor_spec,
        output_tensor_spec,
    )
    default_args = ({"units": 50, "activation": "relu"}, {"units": 50, "activation": "relu"})

    assert network._hidden_layer_args == default_args


@pytest.mark.parametrize(
    "query_point_shape, observation_shape",
    [
        ([1], [1]),
        ([5], [1]),
        ([5], [2]),
    ],
)
def test_gaussian_network_is_correctly_constructed(
    query_point_shape: List[int], observation_shape: List[int], num_hidden_layers: int
) -> None:
    n_obs = 10
    example_data = empty_dataset(query_point_shape, observation_shape)
    query_points = tf.random.uniform([n_obs] + query_point_shape)

    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)
    hidden_layer_args = []
    for i in range(num_hidden_layers):
        hidden_layer_args.append({"units": 10, "activation": "relu"})
    network = GaussianNetwork(
        input_tensor_spec,
        output_tensor_spec,
        hidden_layer_args,
    )
    network_input, network_output = network.connect_layers()
    network_built = tf_keras.Model(inputs=network_input, outputs=network_output)

    # check input shape
    assert network_input.shape[1:] == tf.TensorShape(query_point_shape)

    # testing output shape is more complex as probabilistic layers don't have some properties
    # we make some predictions instead and then check the output is correct
    predictions = network_built.predict(query_points)
    assert predictions.shape == tf.TensorShape([n_obs] + observation_shape)

    # check layers
    assert isinstance(network_built.layers[0], tf_keras.layers.InputLayer)
    assert len(network_built.layers[1:-2]) == num_hidden_layers
    assert isinstance(network_built.layers[-1], tfp.layers.DistributionLambda)


def test_multivariatenormaltril_layer_fails_to_serialilze() -> None:
    # tfp.layers.MultivariateNormalTriL currently fails to serialize out of the box
    # (with different errors in TF2.4 and TF2.5). When that's fixed we can remove our workaround.
    layer = tfp.layers.MultivariateNormalTriL(1)
    with pytest.raises(Exception):
        serialized = tf_keras.utils.serialize_keras_object(layer)
        tf_keras.utils.deserialize_keras_object(
            serialized, custom_objects={"MultivariateNormalTriL": tfp.layers.MultivariateNormalTriL}
        )

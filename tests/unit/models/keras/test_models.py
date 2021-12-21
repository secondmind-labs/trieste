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

import gpflow
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import ShapeLike, empty_dataset, random_seed
from tests.util.models.keras.models import (
    ensemblise_data,
    trieste_deep_ensemble_model,
    trieste_keras_ensemble_model,
)
from tests.util.models.models import fnc_2sin_x_over_3
from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.keras import (
    DeepEnsemble,
    GaussianNetwork,
    get_tensor_spec_from_data,
    negative_log_likelihood,
)
from trieste.models.optimizer import KerasOptimizer

_ENSEMBLE_SIZE = 3


def _get_example_data(query_point_shape: ShapeLike) -> Dataset:
    qp = tf.random.uniform(tf.TensorShape(query_point_shape), dtype=tf.float64)
    obs = fnc_2sin_x_over_3(qp)
    return Dataset(qp, obs)


@pytest.mark.parametrize("optimizer", [tf.optimizers.Adam(), tf.optimizers.RMSprop()])
def test_deep_ensemble_repr(
    optimizer: tf.optimizers.Optimizer,
    bootstrap_data: bool,
):
    example_data = empty_dataset([1], [1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE)
    keras_ensemble.model.compile(optimizer, loss=negative_log_likelihood)
    optimizer_wrapper = KerasOptimizer(optimizer, loss=negative_log_likelihood)
    model = DeepEnsemble(keras_ensemble, optimizer_wrapper, bootstrap_data)

    expected_repr = (
        f"DeepEnsemble({keras_ensemble.model!r}, {optimizer_wrapper!r}, {bootstrap_data!r})"
    )

    assert type(model).__name__ in repr(model)
    assert repr(model) == expected_repr


def test_deep_ensemble_model_attributes() -> None:
    example_data = empty_dataset([1], [1])
    model, keras_ensemble, optimizer = trieste_deep_ensemble_model(
        example_data, _ENSEMBLE_SIZE, False, False, return_all=True
    )

    keras_ensemble.model.compile(optimizer=optimizer.optimizer, loss=optimizer.loss)

    assert model.model is keras_ensemble.model


def test_deep_ensemble_ensemble_size_attributes(ensemble_size: int) -> None:
    example_data = empty_dataset([1], [1])
    model = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)

    assert model.ensemble_size == ensemble_size


def test_deep_ensemble_raises_for_incorrect_model() -> None:

    example_data = empty_dataset([1], [1])
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(example_data)

    model = GaussianNetwork(input_tensor_spec, output_tensor_spec)

    with pytest.raises(ValueError):
        DeepEnsemble(model)


def test_deep_ensemble_raises_for_non_keras_optimizer() -> None:
    example_data = empty_dataset([1], [1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE)

    with pytest.raises(ValueError):
        DeepEnsemble(keras_ensemble, KerasOptimizer(gpflow.optimizers.Scipy()))


@pytest.mark.parametrize("ensemble_size", [-1, 1])
def test_deep_ensemble_raises_for_incorrect_ensemble_size(ensemble_size: int) -> None:

    example_data = empty_dataset([1], [1])

    with pytest.raises(ValueError):
        trieste_deep_ensemble_model(example_data, ensemble_size, False, False)


def test_deep_ensemble_default_optimizer_is_correct() -> None:
    example_data = empty_dataset([1], [1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, False)
    model = DeepEnsemble(keras_ensemble)
    default_loss = negative_log_likelihood
    default_fit_args = {
        "verbose": 0,
        "epochs": 100,
        "batch_size": 100,
    }

    assert isinstance(model.optimizer, KerasOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.optimizers.Optimizer)
    assert model.optimizer.fit_args == default_fit_args
    assert model.optimizer.loss == default_loss


def test_deep_ensemble_optimizer_changed_correctly() -> None:
    example_data = empty_dataset([1], [1])

    custom_fit_args = {
        "verbose": 1,
        "epochs": 10,
        "batch_size": 10,
    }
    custom_optimizer = tf.optimizers.RMSprop()
    custom_loss = tf.keras.losses.MeanSquaredError()
    optimizer_wrapper = KerasOptimizer(custom_optimizer, custom_loss, custom_fit_args)

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE)
    model = DeepEnsemble(keras_ensemble, optimizer_wrapper)

    assert model.optimizer == optimizer_wrapper
    assert model.optimizer.optimizer == custom_optimizer
    assert model.optimizer.fit_args == custom_fit_args


def test_deep_ensemble_is_compiled() -> None:
    example_data = empty_dataset([1], [1])
    model = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE)

    assert model.model.compiled_loss is not None
    assert model.model.compiled_metrics is not None
    assert model.model.optimizer is not None


@pytest.mark.skip
def test_config_builds_deep_ensemble_and_default_optimizer_is_correct() -> None:
    example_data = empty_dataset([1], [1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE)

    model_config = {"model": keras_ensemble}
    model = create_model(model_config)
    default_fit_args = {
        "verbose": 0,
        "epochs": 100,
        "batch_size": 100,
    }

    assert isinstance(model, DeepEnsemble)
    assert isinstance(model.optimizer, KerasOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.keras.optimizers.Optimizer)
    assert model.optimizer.fit_args == default_fit_args


@pytest.mark.parametrize("dataset_size", [10, 100])
def test_deep_ensemble_predict_call_shape(
    ensemble_size: int,
    dataset_size: int,
):
    example_data = _get_example_data([dataset_size, 1])
    model = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)

    predicted_means, predicted_vars = model.predict(example_data.query_points)

    assert tf.is_tensor(predicted_vars)
    assert predicted_vars.shape == example_data.observations.shape
    assert tf.is_tensor(predicted_means)
    assert predicted_means.shape == example_data.observations.shape


@pytest.mark.parametrize("dataset_size", [10, 100])
def test_deep_ensemble_predict_ensemble_call_shape(
    ensemble_size: int,
    dataset_size: int,
):
    example_data = _get_example_data([dataset_size, 1])
    model = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)

    predictions = model.predict_ensemble(example_data.query_points)

    assert isinstance(predictions, list)
    assert len(predictions) == ensemble_size
    for pred in predictions:
        assert isinstance(pred, tuple)
        predicted_means, predicted_vars = pred
        assert tf.is_tensor(predicted_vars)
        assert predicted_vars.shape == example_data.observations.shape
        assert tf.is_tensor(predicted_means)
        assert predicted_means.shape == example_data.observations.shape


@pytest.mark.parametrize("num_samples", [10, 100])
@pytest.mark.parametrize("dataset_size", [50, 150])
def test_deep_ensemble_sample_call_shape(
    ensemble_size: int,
    num_samples: int,
    dataset_size: int,
) -> None:
    example_data = _get_example_data([dataset_size, 1])
    model = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)

    samples = model.sample(example_data.query_points, num_samples)

    assert tf.is_tensor(samples)
    assert samples.shape == [num_samples, dataset_size, 1]


@random_seed
def test_deep_ensemble_optimize_with_defaults(independent_normal: bool) -> None:
    example_data = _get_example_data([100, 1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, independent_normal)

    model = DeepEnsemble(keras_ensemble)

    model.optimize(example_data)
    loss = model.model.history.history["loss"]

    assert loss[-1] < loss[0]


@random_seed
@pytest.mark.parametrize("epochs", [5, 15])
def test_deep_ensemble_optimize(
    independent_normal: bool,
    ensemble_size: int,
    bootstrap_data: bool,
    epochs: int,
) -> None:
    example_data = _get_example_data([100, 1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, ensemble_size, independent_normal)

    custom_optimizer = tf.optimizers.RMSprop()
    custom_fit_args = {
        "verbose": 0,
        "epochs": epochs,
        "batch_size": 10,
    }
    custom_loss = tf.keras.losses.MeanSquaredError()
    optimizer_wrapper = KerasOptimizer(custom_optimizer, custom_loss, custom_fit_args)

    model = DeepEnsemble(keras_ensemble, optimizer_wrapper, bootstrap_data)

    model.optimize(example_data)
    loss = model.model.history.history["loss"]
    ensemble_losses = ["output_loss" in elt for elt in model.model.history.history.keys()]

    assert loss[-1] < loss[0]
    assert len(loss) == epochs
    assert sum(ensemble_losses) == ensemble_size


@random_seed
def test_deep_ensemble_loss(
    independent_normal: bool,
    bootstrap_data: bool,
) -> None:
    example_data = _get_example_data([100, 1])

    loss = negative_log_likelihood
    optimizer = tf.optimizers.Adam()

    model = DeepEnsemble(
        trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, independent_normal),
        KerasOptimizer(optimizer, loss),
        bootstrap_data,
    )

    reference_model = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, independent_normal)
    reference_model.model.compile(optimizer=optimizer, loss=loss)
    reference_model.model.set_weights(model.model.get_weights())

    tranformed_x, tranformed_y = ensemblise_data(
        reference_model, example_data, _ENSEMBLE_SIZE, bootstrap_data
    )
    loss = model.model.evaluate(tranformed_x, tranformed_y)
    reference_loss = reference_model.model.evaluate(tranformed_x, tranformed_y)

    npt.assert_allclose(loss, reference_loss, rtol=1e-6)


@random_seed
def test_deep_ensemble_predict_ensemble(
    independent_normal: bool,
) -> None:
    example_data = _get_example_data([100, 1])

    loss = negative_log_likelihood
    optimizer = tf.optimizers.Adam()

    model = DeepEnsemble(
        trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, independent_normal),
        KerasOptimizer(optimizer, loss),
    )

    reference_model = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, independent_normal)
    reference_model.model.compile(optimizer=optimizer, loss=loss)
    reference_model.model.set_weights(model.model.get_weights())

    means_vars = model.predict_ensemble(example_data.query_points)
    tranformed_x, tranformed_y = ensemblise_data(
        reference_model, example_data, _ENSEMBLE_SIZE, False
    )
    ensemble_distributions = reference_model.model(tranformed_x)
    reference_means_vars = [(dist.mean(), dist.variance()) for dist in ensemble_distributions]

    assert len(means_vars) == len(reference_means_vars)
    for i in range(len(means_vars)):
        npt.assert_allclose(means_vars[i][0], reference_means_vars[i][0])  # means
        npt.assert_allclose(means_vars[i][1], reference_means_vars[i][1])  # vars


@random_seed
def test_deep_ensemble_sample(
    independent_normal: bool,
    ensemble_size: int,
    bootstrap_data: bool,
) -> None:
    example_data = _get_example_data([100, 1])
    model = trieste_deep_ensemble_model(
        example_data, ensemble_size, bootstrap_data, independent_normal
    )
    num_samples = 100_000

    samples = model.sample(example_data.query_points, num_samples)
    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2, axis=0)

    ref_mean, ref_variance = model.predict(example_data.query_points)

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, ref_mean, atol=4 * error)
    npt.assert_allclose(sample_variance, ref_variance, atol=8 * error)


@random_seed
def test_deep_ensemble_prepare_data_call(
    independent_normal: bool,
    ensemble_size: int,
    bootstrap_data: bool,
) -> None:
    n_rows = 100
    x = tf.constant(np.arange(0, n_rows, 1), shape=[n_rows, 1])
    y = tf.constant(np.arange(0, n_rows, 1), shape=[n_rows, 1])
    example_data = Dataset(x, y)

    model = trieste_deep_ensemble_model(
        example_data, ensemble_size, bootstrap_data, independent_normal
    )

    # call with whole dataset
    data = model.prepare_data(example_data)
    assert isinstance(data, tuple)
    for ensemble_data in data:
        assert isinstance(ensemble_data, dict)
        assert len(ensemble_data.keys()) == ensemble_size
        for member_data in ensemble_data:
            if bootstrap_data:
                assert tf.reduce_any(ensemble_data[member_data] != x)
            else:
                assert tf.reduce_all(ensemble_data[member_data] == x)
    for inp, out in zip(data[0], data[1]):
        assert "".join(filter(str.isdigit, inp)) == "".join(filter(str.isdigit, out))

    # call with query points alone
    inputs = model.prepare_data(example_data.query_points)
    assert isinstance(inputs, dict)
    assert len(inputs.keys()) == ensemble_size
    for member_data in inputs:
        assert tf.reduce_all(inputs[member_data] == x)

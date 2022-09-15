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
from __future__ import annotations

import copy
import operator
import unittest.mock
from typing import Any, Optional

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.callbacks import Callback

from tests.util.misc import ShapeLike, empty_dataset, random_seed
from tests.util.models.keras.models import trieste_deep_ensemble_model, trieste_keras_ensemble_model
from trieste.data import Dataset
from trieste.logging import step_number, tensorboard_writer
from trieste.models.keras import (
    DeepEnsemble,
    KerasEnsemble,
    negative_log_likelihood,
    sample_with_replacement,
)
from trieste.models.optimizer import KerasOptimizer, TrainingData

_ENSEMBLE_SIZE = 3


@pytest.fixture(name="ensemble_size", params=[2, 5])
def _ensemble_size_fixture(request: Any) -> int:
    return request.param


@pytest.fixture(name="num_outputs", params=[1, 3])
def _num_outputs_fixture(request: Any) -> int:
    return request.param


@pytest.fixture(name="dataset_size", params=[10, 100])
def _dataset_size_fixture(request: Any) -> int:
    return request.param


@pytest.fixture(name="independent_normal", params=[False, True])
def _independent_normal_fixture(request: Any) -> bool:
    return request.param


@pytest.fixture(name="bootstrap_data", params=[False, True])
def _bootstrap_data_fixture(request: Any) -> bool:
    return request.param


def _get_example_data(
    query_point_shape: ShapeLike, observation_shape: Optional[ShapeLike] = None
) -> Dataset:
    qp = tf.random.uniform(tf.TensorShape(query_point_shape), dtype=tf.float64)

    if observation_shape is None:
        observation_shape = query_point_shape[:-1] + [1]  # type: ignore
    obs = tf.random.uniform(tf.TensorShape(observation_shape), dtype=tf.float64)

    return Dataset(qp, obs)


def _ensemblise_data(
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


@pytest.mark.parametrize("optimizer", [tf.optimizers.Adam(), tf.optimizers.RMSprop()])
@pytest.mark.parametrize("diversify", [False, True])
def test_deep_ensemble_repr(
    optimizer: tf.optimizers.Optimizer, bootstrap_data: bool, diversify: bool
) -> None:
    example_data = empty_dataset([1], [1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE)
    keras_ensemble.model.compile(optimizer, loss=negative_log_likelihood)
    optimizer_wrapper = KerasOptimizer(optimizer, loss=negative_log_likelihood)
    model = DeepEnsemble(keras_ensemble, optimizer_wrapper, bootstrap_data, diversify)

    expected_repr = (
        f"DeepEnsemble({keras_ensemble.model!r}, {optimizer_wrapper!r}, "
        f"{bootstrap_data!r}, {diversify!r})"
    )

    assert type(model).__name__ in repr(model)
    assert repr(model) == expected_repr


def test_deep_ensemble_model_attributes() -> None:
    example_data = empty_dataset([1], [1])
    model, keras_ensemble, optimizer = trieste_deep_ensemble_model(
        example_data, _ENSEMBLE_SIZE, False, False
    )

    keras_ensemble.model.compile(optimizer=optimizer.optimizer, loss=optimizer.loss)

    assert model.model is keras_ensemble.model


def test_deep_ensemble_ensemble_size_attributes(ensemble_size: int) -> None:
    example_data = empty_dataset([1], [1])
    model, _, _ = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)

    assert model.ensemble_size == ensemble_size


def test_deep_ensemble_raises_for_incorrect_ensemble_size() -> None:
    with pytest.raises(ValueError):
        trieste_deep_ensemble_model(empty_dataset([1], [1]), 1)


def test_deep_ensemble_default_optimizer_is_correct() -> None:
    example_data = empty_dataset([1], [1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, False)
    model = DeepEnsemble(keras_ensemble)
    default_loss = negative_log_likelihood
    default_fit_args = {
        "verbose": 0,
        "epochs": 3000,
        "batch_size": 16,
    }
    del model.optimizer.fit_args["callbacks"]

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
    optimizer_wrapper = KerasOptimizer(custom_optimizer, custom_fit_args, custom_loss)

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE)
    model = DeepEnsemble(keras_ensemble, optimizer_wrapper)

    assert model.optimizer == optimizer_wrapper
    assert model.optimizer.optimizer == custom_optimizer
    assert model.optimizer.fit_args == custom_fit_args


def test_deep_ensemble_is_compiled() -> None:
    example_data = empty_dataset([1], [1])
    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE)

    assert model.model.compiled_loss is not None
    assert model.model.compiled_metrics is not None
    assert model.model.optimizer is not None


def test_deep_ensemble_resets_lr_with_lr_schedule() -> None:
    example_data = _get_example_data([100, 1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE)

    epochs = 10
    init_lr = 0.01

    def scheduler(epoch: int, lr: float) -> float:
        if epoch == epoch // 2:
            return lr * 0.1
        else:
            return lr

    fit_args = {
        "epochs": epochs,
        "batch_size": 100,
        "verbose": 0,
        "callbacks": tf.keras.callbacks.LearningRateScheduler(scheduler),
    }
    optimizer = KerasOptimizer(tf.optimizers.Adam(init_lr), fit_args)
    model = DeepEnsemble(keras_ensemble, optimizer)

    npt.assert_allclose(model.model.optimizer.lr.numpy(), init_lr, rtol=1e-6)

    model.optimize(example_data)

    npt.assert_allclose(model.model.optimizer.lr.numpy(), init_lr, rtol=1e-6)


def test_deep_ensemble_ensemble_distributions(ensemble_size: int, dataset_size: int) -> None:
    example_data = _get_example_data([dataset_size, 1])
    model, _, _ = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)

    distributions = model.ensemble_distributions(example_data.query_points)
    # breakpoint()
    assert len(distributions) == ensemble_size
    for dist in distributions:
        assert isinstance(dist, tfp.distributions.Distribution)
        try:
            predicted_means = dist.mean()
        except Exception as exc:
            assert False, f"calling 'mean' raised an exception {exc}"
        try:
            predicted_vars = dist.variance()
        except Exception as exc:
            assert False, f"calling 'variance' raised an exception {exc}"
        assert tf.is_tensor(predicted_means)
        assert tf.is_tensor(predicted_vars)
        assert predicted_means.shape[-2:] == example_data.observations.shape
        assert predicted_vars.shape[-2:] == example_data.observations.shape


def test_deep_ensemble_predict_call_shape(
    ensemble_size: int, dataset_size: int, num_outputs: int
) -> None:
    example_data = _get_example_data([dataset_size, num_outputs], [dataset_size, num_outputs])
    model, _, _ = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)

    predicted_means, predicted_vars = model.predict(example_data.query_points)

    assert tf.is_tensor(predicted_vars)
    assert predicted_vars.shape == example_data.observations.shape
    assert tf.is_tensor(predicted_means)
    assert predicted_means.shape == example_data.observations.shape


def test_deep_ensemble_predict_ensemble_call_shape(
    ensemble_size: int, dataset_size: int, num_outputs: int
) -> None:
    example_data = _get_example_data([dataset_size, num_outputs], [dataset_size, num_outputs])
    model, _, _ = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)

    predicted_means, predicted_vars = model.predict_ensemble(example_data.query_points)

    assert predicted_means.shape[-3] == ensemble_size
    assert predicted_vars.shape[-3] == ensemble_size
    assert tf.is_tensor(predicted_means)
    assert tf.is_tensor(predicted_vars)
    assert predicted_means.shape[-2:] == example_data.observations.shape
    assert predicted_vars.shape[-2:] == example_data.observations.shape


@pytest.mark.parametrize("num_samples", [6, 12])
@pytest.mark.parametrize("dataset_size", [4, 8])
def test_deep_ensemble_sample_call_shape(
    num_samples: int, dataset_size: int, num_outputs: int
) -> None:
    example_data = _get_example_data([dataset_size, num_outputs], [dataset_size, num_outputs])
    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE, False, False)

    samples = model.sample(example_data.query_points, num_samples)

    assert tf.is_tensor(samples)
    assert samples.shape == [num_samples, dataset_size, num_outputs]


@pytest.mark.parametrize("num_samples", [6, 12])
@pytest.mark.parametrize("dataset_size", [4, 8])
def test_deep_ensemble_sample_ensemble_call_shape(
    num_samples: int, dataset_size: int, num_outputs: int
) -> None:
    example_data = _get_example_data([dataset_size, num_outputs], [dataset_size, num_outputs])
    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE, False, False)

    samples = model.sample_ensemble(example_data.query_points, num_samples)

    assert tf.is_tensor(samples)
    assert samples.shape == [num_samples, dataset_size, num_outputs]


@random_seed
def test_deep_ensemble_optimize_with_defaults() -> None:
    example_data = _get_example_data([100, 1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, False)

    model = DeepEnsemble(keras_ensemble)

    model.optimize(example_data)
    loss = model.model.history.history["loss"]

    assert loss[-1] < loss[0]


@random_seed
@pytest.mark.parametrize("epochs", [5, 15])
def test_deep_ensemble_optimize(ensemble_size: int, bootstrap_data: bool, epochs: int) -> None:
    example_data = _get_example_data([100, 1])

    keras_ensemble = trieste_keras_ensemble_model(example_data, ensemble_size, False)

    custom_optimizer = tf.optimizers.RMSprop()
    custom_fit_args = {
        "verbose": 0,
        "epochs": epochs,
        "batch_size": 10,
    }
    custom_loss = tf.keras.losses.MeanSquaredError()
    optimizer_wrapper = KerasOptimizer(custom_optimizer, custom_fit_args, custom_loss)

    model = DeepEnsemble(keras_ensemble, optimizer_wrapper, bootstrap_data)

    model.optimize(example_data)
    loss = model.model.history.history["loss"]
    ensemble_losses = ["output_loss" in elt for elt in model.model.history.history.keys()]

    assert loss[-1] < loss[0]
    assert len(loss) == epochs
    assert sum(ensemble_losses) == ensemble_size


@random_seed
def test_deep_ensemble_loss(bootstrap_data: bool) -> None:
    example_data = _get_example_data([100, 1])

    loss = negative_log_likelihood
    optimizer = tf.optimizers.Adam()

    model = DeepEnsemble(
        trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, False),
        KerasOptimizer(optimizer, loss=loss),
        bootstrap_data,
    )

    reference_model = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, False)
    reference_model.model.compile(optimizer=optimizer, loss=loss)
    reference_model.model.set_weights(model.model.get_weights())

    tranformed_x, tranformed_y = _ensemblise_data(
        reference_model, example_data, _ENSEMBLE_SIZE, bootstrap_data
    )
    loss = model.model.evaluate(tranformed_x, tranformed_y)[: _ENSEMBLE_SIZE + 1]
    reference_loss = reference_model.model.evaluate(tranformed_x, tranformed_y)

    npt.assert_allclose(loss, reference_loss, rtol=1e-6)


@random_seed
def test_deep_ensemble_predict_ensemble() -> None:
    example_data = _get_example_data([100, 1])

    loss = negative_log_likelihood
    optimizer = tf.optimizers.Adam()

    model = DeepEnsemble(
        trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, False),
        KerasOptimizer(optimizer, loss=loss),
    )

    reference_model = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, False)
    reference_model.model.compile(optimizer=optimizer, loss=loss)
    reference_model.model.set_weights(model.model.get_weights())

    predicted_means, predicted_vars = model.predict_ensemble(example_data.query_points)
    tranformed_x, tranformed_y = _ensemblise_data(
        reference_model, example_data, _ENSEMBLE_SIZE, False
    )
    ensemble_distributions = reference_model.model(tranformed_x)
    reference_means = tf.convert_to_tensor([dist.mean() for dist in ensemble_distributions])
    reference_vars = tf.convert_to_tensor([dist.variance() for dist in ensemble_distributions])

    npt.assert_allclose(predicted_means, reference_means)
    npt.assert_allclose(predicted_vars, reference_vars)


@random_seed
def test_deep_ensemble_sample() -> None:
    example_data = _get_example_data([100, 1])
    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE, False, False)
    num_samples = 100_000

    samples = model.sample(example_data.query_points, num_samples)
    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2, axis=0)

    ref_mean, ref_variance = model.predict(example_data.query_points)

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, ref_mean, atol=4 * error)
    npt.assert_allclose(sample_variance, ref_variance, atol=8 * error)


@random_seed
def test_deep_ensemble_sample_ensemble(ensemble_size: int) -> None:
    example_data = _get_example_data([20, 1])
    model, _, _ = trieste_deep_ensemble_model(example_data, ensemble_size, False, False)
    num_samples = 2000

    samples = model.sample_ensemble(example_data.query_points, num_samples)
    sample_mean = tf.reduce_mean(samples, axis=0)

    ref_mean, _ = model.predict(example_data.query_points)

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, ref_mean, atol=2.5 * error)


@random_seed
def test_deep_ensemble_prepare_data_call(
    ensemble_size: int,
    bootstrap_data: bool,
) -> None:
    n_rows = 100
    x = tf.constant(np.arange(0, n_rows, 1), shape=[n_rows, 1])
    y = tf.constant(np.arange(0, n_rows, 1), shape=[n_rows, 1])
    example_data = Dataset(x, y)

    model, _, _ = trieste_deep_ensemble_model(example_data, ensemble_size, bootstrap_data, False)

    # call with whole dataset
    data = model.prepare_dataset(example_data)
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
    inputs = model.prepare_query_points(example_data.query_points)
    assert isinstance(inputs, dict)
    assert len(inputs.keys()) == ensemble_size
    for member_data in inputs:
        assert tf.reduce_all(inputs[member_data] == x)


def test_deep_ensemble_deep_copyable() -> None:
    example_data = _get_example_data([10, 3], [10, 3])
    model, _, _ = trieste_deep_ensemble_model(example_data, 2, False, False)
    model_copy = copy.deepcopy(model)

    mean_f, variance_f = model.predict(example_data.query_points)
    mean_f_copy, variance_f_copy = model_copy.predict(example_data.query_points)
    npt.assert_allclose(mean_f, mean_f_copy)
    npt.assert_allclose(variance_f, variance_f_copy)

    # check that updating the original doesn't break or change the deepcopy
    new_example_data = _get_example_data([20, 3], [20, 3])
    model.update(new_example_data)
    model.optimize(new_example_data)

    mean_f_updated, variance_f_updated = model.predict(example_data.query_points)
    mean_f_copy_updated, variance_f_copy_updated = model_copy.predict(example_data.query_points)
    npt.assert_allclose(mean_f_copy_updated, mean_f_copy)
    npt.assert_allclose(variance_f_copy_updated, variance_f_copy)
    npt.assert_array_compare(operator.__ne__, mean_f_updated, mean_f)
    npt.assert_array_compare(operator.__ne__, variance_f_updated, variance_f)

    # check that we can also update the copy
    newer_example_data = _get_example_data([30, 3], [30, 3])
    model_copy.update(newer_example_data)
    model_copy.optimize(newer_example_data)

    mean_f_updated_2, variance_f_updated_2 = model.predict(example_data.query_points)
    mean_f_copy_updated_2, variance_f_copy_updated_2 = model_copy.predict(example_data.query_points)
    npt.assert_allclose(mean_f_updated_2, mean_f_updated)
    npt.assert_allclose(variance_f_updated_2, variance_f_updated)
    npt.assert_array_compare(operator.__ne__, mean_f_copy_updated_2, mean_f_copy_updated)
    npt.assert_array_compare(operator.__ne__, variance_f_copy_updated_2, variance_f_copy_updated)


def test_deep_ensemble_deep_copies_optimizer_state() -> None:
    example_data = _get_example_data([10, 3], [10, 3])
    model, _, _ = trieste_deep_ensemble_model(example_data, 2, False, False)
    new_example_data = _get_example_data([20, 3], [20, 3])
    model.update(new_example_data)
    assert not model.model.optimizer.get_weights()
    model.optimize(new_example_data)
    assert model.model.optimizer.get_weights()

    model_copy = copy.deepcopy(model)
    assert model.model.optimizer is not model_copy.model.optimizer
    npt.assert_allclose(model_copy.model.optimizer.iterations, 1)
    npt.assert_equal(model.model.optimizer.get_weights(), model_copy.model.optimizer.get_weights())


@pytest.mark.parametrize(
    "callbacks",
    [
        [
            tf.keras.callbacks.CSVLogger("csv"),
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=100),
            tf.keras.callbacks.History(),
            tf.keras.callbacks.LambdaCallback(lambda epoch, lr: lr),
            tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr),
            tf.keras.callbacks.ProgbarLogger(),
            tf.keras.callbacks.ReduceLROnPlateau(),
            tf.keras.callbacks.RemoteMonitor(),
            tf.keras.callbacks.TensorBoard(),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
        pytest.param(
            [
                tf.keras.callbacks.experimental.BackupAndRestore("backup"),
                tf.keras.callbacks.BaseLogger(),
                tf.keras.callbacks.ModelCheckpoint("weights"),
            ],
            marks=pytest.mark.skip(reason="callbacks currently causing optimize to fail"),
        ),
    ],
)
def test_deep_ensemble_deep_copies_different_callback_types(callbacks: list[Callback]) -> None:
    example_data = _get_example_data([10, 3], [10, 3])
    model, _, _ = trieste_deep_ensemble_model(example_data, 2, False, False)
    model.optimizer.fit_args["callbacks"] = callbacks

    new_example_data = _get_example_data([20, 3], [20, 3])
    model.update(new_example_data)
    model.optimize(new_example_data)

    model_copy = copy.deepcopy(model)
    assert model.model.optimizer is not model_copy.model.optimizer
    assert tuple(type(callback) for callback in model.optimizer.fit_args["callbacks"]) == tuple(
        type(callback) for callback in model_copy.optimizer.fit_args["callbacks"]
    )


def test_deep_ensemble_deep_copies_optimizer_callback_models() -> None:
    example_data = _get_example_data([10, 3], [10, 3])
    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, False)
    model = DeepEnsemble(keras_ensemble)
    new_example_data = _get_example_data([20, 3], [20, 3])
    model.update(new_example_data)
    model.optimize(new_example_data)

    callback = model.optimizer.fit_args["callbacks"][0]
    assert isinstance(callback, tf.keras.callbacks.EarlyStopping)
    assert callback.model is model.model

    model_copy = copy.deepcopy(model)
    callback_copy = model_copy.optimizer.fit_args["callbacks"][0]
    assert isinstance(callback_copy, tf.keras.callbacks.EarlyStopping)
    assert callback_copy.model is model_copy.model is not callback.model
    npt.assert_equal(callback_copy.model.get_weights(), callback.model.get_weights())


def test_deep_ensemble_deep_copies_optimizer_without_callbacks() -> None:
    example_data = _get_example_data([10, 3], [10, 3])
    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, False)
    model = DeepEnsemble(keras_ensemble)
    del model.optimizer.fit_args["callbacks"]

    model_copy = copy.deepcopy(model)
    assert model_copy.optimizer is not model.optimizer
    assert model_copy.optimizer.fit_args == model.optimizer.fit_args


def test_deep_ensemble_deep_copies_optimization_history() -> None:
    example_data = _get_example_data([10, 3], [10, 3])
    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, False)
    model = DeepEnsemble(keras_ensemble)
    model.optimize(example_data)

    assert model.model.history.history
    expected_history = model.model.history.history

    model_copy = copy.deepcopy(model)
    assert model_copy.model.history.history
    history = model_copy.model.history.history

    assert history.keys() == expected_history.keys()
    for k, v in expected_history.items():
        assert history[k] == v


@unittest.mock.patch("trieste.logging.tf.summary.histogram")
@unittest.mock.patch("trieste.logging.tf.summary.scalar")
@pytest.mark.parametrize("use_dataset", [True, False])
def test_deep_ensemble_log(
    mocked_summary_scalar: unittest.mock.MagicMock,
    mocked_summary_histogram: unittest.mock.MagicMock,
    use_dataset: bool,
) -> None:
    example_data = _get_example_data([10, 3], [10, 3])
    keras_ensemble = trieste_keras_ensemble_model(example_data, _ENSEMBLE_SIZE, False)
    model = DeepEnsemble(keras_ensemble)
    model.optimize(example_data)

    mocked_summary_writer = unittest.mock.MagicMock()
    with tensorboard_writer(mocked_summary_writer):
        with step_number(42):
            if use_dataset:
                model.log(example_data)
            else:
                model.log(None)

    assert len(mocked_summary_writer.method_calls) == 1
    assert mocked_summary_writer.method_calls[0][0] == "as_default"
    assert mocked_summary_writer.method_calls[0][-1]["step"] == 42

    loss_names = ["loss/diff", "loss/final", "loss/min", "loss/max"]
    metrics_names = ["mse/diff", "mse/final", "mse/min", "mse/max"]
    accuracy_names = [
        "accuracy/root_mean_square_error",
        "accuracy/mean_absolute_error",
        "accuracy/z_residuals_std",
    ]
    variance_stats = [
        "variance/predict_variance_mean",
        "variance/empirical",
        "variance/root_mean_variance_error",
    ]

    names_scalars = ["epochs/num_epochs"] + loss_names + metrics_names
    num_scalars = 1 + len(loss_names)
    names_histogram = ["loss/epoch"] + ["mse/epoch"]
    num_histogram = 1
    if use_dataset:
        names_scalars = names_scalars + accuracy_names + variance_stats
        num_scalars = num_scalars + len(accuracy_names) + len(variance_stats)
        names_histogram = (
            names_histogram
            + ["accuracy/absolute_errors", "accuracy/z_residuals"]
            + ["variance/predict_variance", "variance/variance_error"]
        )
        num_histogram += 4

    assert mocked_summary_scalar.call_count == num_scalars
    for i in range(len(mocked_summary_scalar.call_args_list)):
        assert any([j in mocked_summary_scalar.call_args_list[i][0][0] for j in names_scalars])

    assert mocked_summary_histogram.call_count == num_histogram
    for i in range(len(mocked_summary_histogram.call_args_list)):
        assert any([j in mocked_summary_histogram.call_args_list[i][0][0] for j in names_histogram])

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
In this module, we test the *behaviour* of Trieste models against reference GPflux models (thus
implicitly assuming the latter are correct).
*NOTE:* Where GPflux models are used as the underlying model in an Trieste model, we should
*not* test that the underlying model is used in any particular way. To do so would break
encapsulation. For example, we should *not* test that methods on the GPflux models are called
(except in the rare case that such behaviour is an explicitly documented behaviour of the
Trieste model).
"""

from __future__ import annotations

import copy
import operator
import unittest.mock
from functools import partial
from typing import Callable

import gpflow
import gpflux.encoders
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
from gpflux.models import DeepGP
from gpflux.models.deep_gp import sample_dgp
from tensorflow.python.keras.callbacks import Callback

from tests.util.misc import random_seed
from tests.util.models.gpflux.models import single_layer_dgp_model
from tests.util.models.models import fnc_2sin_x_over_3, fnc_3x_plus_10
from trieste.data import Dataset
from trieste.logging import step_number, tensorboard_writer
from trieste.models.gpflux import DeepGaussianProcess
from trieste.models.optimizer import KerasOptimizer
from trieste.types import TensorType


def test_deep_gaussian_process_raises_for_non_tf_optimizer(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    dgp = two_layer_model(x)
    optimizer = KerasOptimizer(gpflow.optimizers.Scipy())

    with pytest.raises(ValueError):
        DeepGaussianProcess(dgp, optimizer)


def test_deep_gaussian_process_raises_for_keras_layer() -> None:
    keras_layer_1 = tf.keras.layers.Dense(50, activation="relu")
    keras_layer_2 = tf.keras.layers.Dense(2, activation="relu")

    kernel = gpflow.kernels.SquaredExponential()
    num_inducing = 5
    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.concatenate(
            [
                np.random.randn(num_inducing, 2),
            ],
            axis=1,
        )
    )
    gp_layer = gpflux.layers.GPLayer(
        kernel,
        inducing_variable,
        num_data=5,
        num_latent_gps=1,
        mean_function=gpflow.mean_functions.Zero(),
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.01))

    dgp = DeepGP([keras_layer_1, keras_layer_2, gp_layer], likelihood_layer)

    with pytest.raises(ValueError):
        DeepGaussianProcess(dgp)


def test_deep_gaussian_process_model_attribute(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    dgp = two_layer_model(x)
    model = DeepGaussianProcess(dgp)

    assert model.model_gpflux is dgp


def test_deep_gaussian_process_update(two_layer_model: Callable[[TensorType], DeepGP]) -> None:
    x = tf.zeros([1, 4], dtype=tf.float64)
    dgp = two_layer_model(x)
    model = DeepGaussianProcess(dgp)

    assert model.model_gpflux.num_data == 1

    for layer in model.model_gpflux.f_layers:
        assert layer.num_data == 1

    model.update(Dataset(tf.zeros([5, 4]), tf.zeros([5, 1])))

    assert model.model_gpflux.num_data == 5

    for layer in model.model_gpflux.f_layers:
        assert layer.num_data == 5


@pytest.mark.parametrize(
    "new_data",
    [Dataset(tf.zeros([3, 5]), tf.zeros([3, 1])), Dataset(tf.zeros([3, 4]), tf.zeros([3, 2]))],
)
def test_deep_gaussian_process_update_raises_for_invalid_shapes(
    two_layer_model: Callable[[TensorType], DeepGP], new_data: Dataset
) -> None:
    x = tf.zeros([1, 4], dtype=tf.float64)
    dgp = two_layer_model(x)
    model = DeepGaussianProcess(dgp)

    with pytest.raises(ValueError):
        model.update(new_data)


def test_deep_gaussian_process_optimize_with_defaults(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = fnc_2sin_x_over_3(x_observed)
    data = x_observed, y_observed
    dataset = Dataset(*data)
    model = DeepGaussianProcess(two_layer_model(x_observed))
    elbo = model.model_gpflux.elbo(data)
    model.optimize(dataset)
    assert model.model_gpflux.elbo(data) > elbo


@pytest.mark.parametrize("batch_size", [10, 100])
def test_deep_gaussian_process_optimize(
    two_layer_model: Callable[[TensorType], DeepGP], batch_size: int
) -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = fnc_2sin_x_over_3(x_observed)
    data = x_observed, y_observed
    dataset = Dataset(*data)

    fit_args = {"batch_size": batch_size, "epochs": 10, "verbose": 0}
    optimizer = KerasOptimizer(tf.optimizers.Adam(), fit_args)

    model = DeepGaussianProcess(two_layer_model(x_observed), optimizer)
    elbo = model.model_gpflux.elbo(data)
    model.optimize(dataset)
    assert model.model_gpflux.elbo(data) > elbo


def test_deep_gaussian_process_loss(two_layer_model: Callable[[TensorType], DeepGP]) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    y = fnc_3x_plus_10(x)

    reference_model = two_layer_model(x)
    model = DeepGaussianProcess(two_layer_model(x))
    internal_model = model.model_gpflux

    npt.assert_allclose(internal_model.elbo((x, y)), reference_model.elbo((x, y)), rtol=1e-6)


def test_deep_gaussian_process_predict() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())

    reference_model = single_layer_dgp_model(x)
    model = DeepGaussianProcess(single_layer_dgp_model(x))

    test_x = tf.constant([[2.5]], dtype=gpflow.default_float())

    ref_mean, ref_var = reference_model.predict_f(test_x)
    f_mean, f_var = model.predict(test_x)

    npt.assert_allclose(f_mean, ref_mean)
    npt.assert_allclose(f_var, ref_var)


@random_seed
def test_deep_gaussian_process_sample(two_layer_model: Callable[[TensorType], DeepGP]) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model = DeepGaussianProcess(two_layer_model(x))
    num_samples = 100
    test_x = tf.constant([[2.5]], dtype=gpflow.default_float())
    samples = model.sample(test_x, num_samples)

    assert samples.shape == [num_samples, 1, 1]

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2)

    reference_model = two_layer_model(x)

    def get_samples(query_points: TensorType, num_samples: int) -> TensorType:
        samples = []
        for _ in range(num_samples):
            samples.append(sample_dgp(reference_model)(query_points))
        return tf.stack(samples)

    ref_samples = get_samples(test_x, num_samples)

    ref_mean = tf.reduce_mean(ref_samples, axis=0)
    ref_variance = tf.reduce_mean((ref_samples - ref_mean) ** 2)

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, ref_mean, atol=2 * error)
    npt.assert_allclose(sample_mean, 0, atol=error)
    npt.assert_allclose(sample_variance, ref_variance, atol=4 * error)


def test_deep_gaussian_process_resets_lr_with_lr_schedule(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    y = fnc_3x_plus_10(x)

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

    model = DeepGaussianProcess(two_layer_model(x), optimizer)

    npt.assert_allclose(model.model_keras.optimizer.lr.numpy(), init_lr, rtol=1e-6)

    dataset = Dataset(x, y)

    model.optimize(dataset)

    npt.assert_allclose(model.model_keras.optimizer.lr.numpy(), init_lr, rtol=1e-6)


def test_deep_gaussian_process_default_optimizer_is_correct(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())

    model = DeepGaussianProcess(two_layer_model(x))
    model_fit_args = dict(model.optimizer.fit_args)
    model_fit_args.pop("callbacks")
    fit_args = {
        "verbose": 0,
        "epochs": 400,
        "batch_size": 1000,
    }

    assert isinstance(model.optimizer, KerasOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.optimizers.Optimizer)
    assert model_fit_args == fit_args


def test_deep_gaussian_process_subclass_default_optimizer_is_correct(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    class DummySubClass(DeepGaussianProcess):
        """Dummy subclass"""

    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())

    model = DummySubClass(two_layer_model(x))
    model_fit_args = dict(model.optimizer.fit_args)
    model_fit_args.pop("callbacks")
    fit_args = {
        "verbose": 0,
        "epochs": 400,
        "batch_size": 1000,
    }

    assert isinstance(model.optimizer, KerasOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.optimizers.Optimizer)
    assert model_fit_args == fit_args


def test_deepgp_deep_copyable() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model = DeepGaussianProcess(partial(single_layer_dgp_model, x))
    model_copy = copy.deepcopy(model)

    test_x = tf.constant([[2.5]], dtype=gpflow.default_float())

    assert model.model_gpflux.inputs.dtype == model_copy.model_gpflux.inputs.dtype
    assert model.model_gpflux.targets.dtype == model_copy.model_gpflux.targets.dtype

    mean_f, variance_f = model.predict(test_x)
    mean_f_copy, variance_f_copy = model_copy.predict(test_x)
    npt.assert_allclose(mean_f, mean_f_copy)
    npt.assert_allclose(variance_f, variance_f_copy)

    # check that updating the original doesn't break or change the deepcopy
    dataset = Dataset(x, fnc_3x_plus_10(x))
    model.update(dataset)
    model.optimize(dataset)

    mean_f_updated, variance_f_updated = model.predict(test_x)
    mean_f_copy_updated, variance_f_copy_updated = model_copy.predict(test_x)
    npt.assert_allclose(mean_f_copy_updated, mean_f_copy)
    npt.assert_allclose(variance_f_copy_updated, variance_f_copy)
    npt.assert_array_compare(operator.__ne__, mean_f_updated, mean_f)
    npt.assert_array_compare(operator.__ne__, variance_f_updated, variance_f)

    # # check that we can also update the copy
    dataset2 = Dataset(x, fnc_2sin_x_over_3(x))
    model_copy.update(dataset2)
    model_copy.optimize(dataset2)

    mean_f_updated_2, variance_f_updated_2 = model.predict(test_x)
    mean_f_copy_updated_2, variance_f_copy_updated_2 = model_copy.predict(test_x)
    npt.assert_allclose(mean_f_updated_2, mean_f_updated)
    npt.assert_allclose(variance_f_updated_2, variance_f_updated)
    npt.assert_array_compare(operator.__ne__, mean_f_copy_updated_2, mean_f_copy_updated)
    npt.assert_array_compare(operator.__ne__, variance_f_copy_updated_2, variance_f_copy_updated)


def test_deepgp_deep_copies_optimizer_state() -> None:

    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model = DeepGaussianProcess(partial(single_layer_dgp_model, x))
    dataset = Dataset(x, fnc_3x_plus_10(x))
    model.update(dataset)
    assert not model.optimizer.optimizer.get_weights()
    model.optimize(dataset)
    assert model.optimizer.optimizer.get_weights()
    npt.assert_allclose(model.optimizer.optimizer.iterations, 400)
    assert model.optimizer.fit_args["callbacks"][0].model is model.model_keras

    model_copy = copy.deepcopy(model)
    assert model.optimizer.optimizer is not model_copy.optimizer.optimizer
    npt.assert_allclose(model_copy.optimizer.optimizer.iterations, 400)
    npt.assert_equal(
        model.optimizer.optimizer.get_weights(), model_copy.optimizer.optimizer.get_weights()
    )
    assert model_copy.optimizer.fit_args["callbacks"][0].model is model_copy.model_keras


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
def test_deepgp_deep_copies_different_callback_types(callbacks: list[Callback]) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model = DeepGaussianProcess(partial(single_layer_dgp_model, x))
    model.optimizer.fit_args["callbacks"] = callbacks

    dataset = Dataset(x, fnc_3x_plus_10(x))
    model.update(dataset)
    model.optimize(dataset)

    model_copy = copy.deepcopy(model)
    assert model.optimizer is not model_copy.optimizer
    assert tuple(type(callback) for callback in model.optimizer.fit_args["callbacks"]) == tuple(
        type(callback) for callback in model_copy.optimizer.fit_args["callbacks"]
    )


def test_deepgp_deep_copies_optimization_history() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model = DeepGaussianProcess(partial(single_layer_dgp_model, x))
    dataset = Dataset(x, fnc_3x_plus_10(x))
    model.update(dataset)
    model.optimize(dataset)

    assert model.model_keras.history.history
    expected_history = model.model_keras.history.history

    model_copy = copy.deepcopy(model)
    assert model_copy.model_keras.history.history
    history = model_copy.model_keras.history.history

    assert history.keys() == expected_history.keys()
    for k, v in expected_history.items():
        assert history[k] == v


@unittest.mock.patch("trieste.logging.tf.summary.histogram")
@unittest.mock.patch("trieste.logging.tf.summary.scalar")
@pytest.mark.parametrize("use_dataset", [False, True])
def test_deepgp_log(
    mocked_summary_scalar: unittest.mock.MagicMock,
    mocked_summary_histogram: unittest.mock.MagicMock,
    use_dataset: bool,
) -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = fnc_2sin_x_over_3(x_observed)
    dataset = Dataset(x_observed, y_observed)

    model = DeepGaussianProcess(
        single_layer_dgp_model(x_observed),
        KerasOptimizer(tf.optimizers.Adam(), {"batch_size": 200, "epochs": 3, "verbose": 0}),
    )
    model.optimize(dataset)

    mocked_summary_writer = unittest.mock.MagicMock()
    with tensorboard_writer(mocked_summary_writer):
        with step_number(42):
            if use_dataset:
                model.log(dataset)
            else:
                model.log(None)

    assert len(mocked_summary_writer.method_calls) == 1
    assert mocked_summary_writer.method_calls[0][0] == "as_default"
    assert mocked_summary_writer.method_calls[0][-1]["step"] == 42

    loss_names = ["loss/diff", "loss/final", "prior_kl/diff", "prior_kl/final"]
    metrics_names = ["mse/diff", "mse/final"]
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
    par_names = ["kernel.variance", "kernel.lengthscales", "likelihood"]

    names_scalars = ["epochs/num_epochs"] + loss_names + metrics_names + par_names
    num_scalars = len(names_scalars)
    names_histogram = ["loss/epoch", "mse/epoch", "prior_kl/epoch"]
    num_histogram = len(names_histogram)
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

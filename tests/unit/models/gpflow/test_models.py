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
In this module, we test the *behaviour* of trieste models against reference GPflow models (thus
implicitly assuming the latter are correct).

*NOTE:* Where GPflow models are used as the underlying model in an trieste model, we should
*not* test that the underlying model is used in any particular way. To do so would break
encapsulation. For example, we should *not* test that methods on the GPflow models are called
(except in the rare case that such behaviour is an explicitly documented behaviour of the
trieste model).
"""

from __future__ import annotations

import unittest.mock
from typing import Any, cast

import gpflow
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import SVGP

from tests.util.misc import random_seed
from tests.util.models.gpflow.models import (
    ModelFactoryType,
    gpr_model,
    mock_data,
    sgpr_model,
    svgp_model,
    vgp_matern_model,
    vgp_model,
)
from tests.util.models.models import fnc_2sin_x_over_3, fnc_3x_plus_10
from trieste.data import Dataset
from trieste.logging import step_number, tensorboard_writer
from trieste.models import TrainableProbabilisticModel
from trieste.models.config import create_model
from trieste.models.gpflow import (
    GaussianProcessRegression,
    SparseVariational,
    VariationalGaussianProcess,
)
from trieste.models.gpflow.models import NumDataPropertyMixin
from trieste.models.gpflow.sampler import RandomFourierFeatureTrajectorySampler
from trieste.models.optimizer import BatchOptimizer, DatasetTransformer, Optimizer


def _3x_plus_gaussian_noise(x: tf.Tensor) -> tf.Tensor:
    return 3.0 * x + np.random.normal(scale=0.01, size=x.shape)


def test_gpflow_wrappers_loss(gpflow_interface_factory: ModelFactoryType) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    y = fnc_3x_plus_10(x)

    model, _reference_model = gpflow_interface_factory(x, y)
    internal_model = model.model
    reference_model = _reference_model(x, y)

    if isinstance(internal_model, SVGP):
        args = {"data": (x, y)}
    else:
        args = {}
    npt.assert_allclose(
        internal_model.training_loss(**args), reference_model.training_loss(**args), rtol=1e-6
    )


def test_gpflow_wrappers_update(gpflow_interface_factory: ModelFactoryType) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    y = fnc_3x_plus_10(x)

    model, _reference_model = gpflow_interface_factory(x, y)

    x_new = tf.concat([x, tf.constant([[10.0], [11.0]], dtype=gpflow.default_float())], 0)
    new_data = Dataset(x_new, fnc_3x_plus_10(x_new))
    # Would be nice if ModelFactoryType could return an intersection type of
    # GPflowPredictor and TrainableProbabilisticModel but this isn't possible
    cast(TrainableProbabilisticModel, model).update(new_data)

    reference_model = _reference_model(x_new, fnc_3x_plus_10(x_new))
    internal_model = model.model

    if isinstance(internal_model, SVGP):
        args = {"data": (new_data.query_points, new_data.observations)}
    else:
        args = {}

    npt.assert_allclose(
        internal_model.training_loss(**args), reference_model.training_loss(**args), rtol=1e-6
    )


def test_gpflow_wrappers_ref_optimize(gpflow_interface_factory: ModelFactoryType) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    y = fnc_2sin_x_over_3(x)
    data = Dataset(x, y)

    model, _reference_model = gpflow_interface_factory(x, y)

    reference_model = _reference_model(x, y)
    model.optimize(data)
    internal_model = model.model

    if isinstance(internal_model, SVGP):
        data_iter = iter(
            tf.data.Dataset.from_tensor_slices(data.astuple())
            .shuffle(len(data))
            .batch(100)
            .prefetch(tf.data.experimental.AUTOTUNE)
            .repeat()
        )
        tf.optimizers.Adam().minimize(
            reference_model.training_loss_closure(data=data_iter, compile=False),
            reference_model.trainable_variables,
        )
        # there is a difference here and the code is pretty much the same
        # not sure where it comes from
        npt.assert_allclose(
            internal_model.training_loss(next(data_iter)),
            reference_model.training_loss(next(data_iter)),
            rtol=1e-1,
        )
    else:
        reference_model.data = (
            tf.Variable(
                reference_model.data[0],
                trainable=False,
                shape=[None, *reference_model.data[0].shape[1:]],
            ),
            tf.Variable(
                reference_model.data[1],
                trainable=False,
                shape=[None, *reference_model.data[1].shape[1:]],
            ),
        )
        gpflow.optimizers.Scipy().minimize(
            reference_model.training_loss_closure(compile=False),
            reference_model.trainable_variables,
        )

        npt.assert_allclose(
            internal_model.training_loss(), reference_model.training_loss(), rtol=1e-6
        )


@pytest.mark.parametrize("num_outputs", [1, 2])
def test_gaussian_process_regression_pairwise_covariance(num_outputs: int) -> None:
    x = tf.constant(np.arange(1, 5).reshape(-1, 1), dtype=gpflow.default_float())  # shape: [4, 1]
    y = fnc_3x_plus_10(x)
    model = GaussianProcessRegression(gpr_model(x, tf.repeat(y, num_outputs, axis=1)))

    query_points_1 = tf.concat([0.5 * x, 0.5 * x], 0)  # shape: [8, 1]
    query_points_2 = tf.concat([2 * x, 2 * x, 2 * x], 0)  # shape: [12, 1]

    all_query_points = tf.concat([query_points_1, query_points_2], 0)
    _, predictive_covariance = model.predict_joint(all_query_points)
    expected_covariance = predictive_covariance[:, :8, 8:]

    actual_covariance = model.covariance_between_points(query_points_1, query_points_2)

    np.testing.assert_allclose(expected_covariance, actual_covariance, atol=1e-5)


def test_gaussian_process_regression_sgpr_raises_for_covariance_between_points() -> None:
    data = mock_data()
    model = GaussianProcessRegression(sgpr_model(*data))

    with pytest.raises(NotImplementedError):
        model.covariance_between_points(data[0], data[0])


def test_gaussian_process_regression_raises_for_invalid_init() -> None:
    x_np = np.arange(5, dtype=np.float64).reshape(-1, 1)
    x = tf.convert_to_tensor(x_np, x_np.dtype)
    y = fnc_3x_plus_10(x)

    with pytest.raises(ValueError):
        GaussianProcessRegression(gpr_model(x, y), num_kernel_samples=-1)

    with pytest.raises(ValueError):
        optimizer1 = BatchOptimizer(gpflow.optimizers.Scipy())
        GaussianProcessRegression(gpr_model(x, y), optimizer=optimizer1)

    with pytest.raises(ValueError):
        optimizer2 = Optimizer(tf.optimizers.Adam())
        GaussianProcessRegression(gpr_model(x, y), optimizer=optimizer2)


def test_gaussian_process_regression_raises_for_covariance_between_invalid_query_points_2() -> None:
    data = mock_data()
    model = GaussianProcessRegression(gpr_model(*data))

    with pytest.raises(ValueError):
        model.covariance_between_points(data[0], tf.expand_dims(data[0], axis=0))


def test_gaussian_process_regression_raises_for_conditionals_with_sgpr() -> None:
    data = mock_data()
    model = GaussianProcessRegression(sgpr_model(*data))

    with pytest.raises(NotImplementedError):
        model.conditional_predict_f(data[0], additional_data=Dataset(data[0], data[1]))

    with pytest.raises(NotImplementedError):
        model.conditional_predict_joint(data[0], additional_data=Dataset(data[0], data[1]))

    with pytest.raises(NotImplementedError):
        model.conditional_predict_y(data[0], additional_data=Dataset(data[0], data[1]))

    with pytest.raises(NotImplementedError):
        model.conditional_predict_f_sample(
            data[0], additional_data=Dataset(data[0], data[1]), num_samples=1
        )


def test_gaussian_process_regression_correctly_returns_internal_data() -> None:
    data = mock_data()
    model = GaussianProcessRegression(gpr_model(*data))
    returned_data = model.get_internal_data()
    npt.assert_array_equal(returned_data.query_points, data[0])
    npt.assert_array_equal(returned_data.observations, data[1])


@random_seed
@unittest.mock.patch(
    "trieste.models.gpflow.models.GaussianProcessRegression.find_best_model_initialization"
)
@pytest.mark.parametrize("prior_for_lengthscale", [True, False])
def test_gaussian_process_regression_correctly_counts_params_that_can_be_sampled(
    mocked_model_initializer: Any,
    dim: int,
    prior_for_lengthscale: bool,
) -> None:
    x = tf.constant(np.arange(1, 5 * dim + 1).reshape(-1, dim), dtype=tf.float64)  # shape: [5, d]
    optimizer = Optimizer(
        optimizer=gpflow.optimizers.Scipy(),
        minimize_args={"options": dict(maxiter=10)},
    )
    model = GaussianProcessRegression(gpr_model(x, fnc_3x_plus_10(x)), optimizer=optimizer)
    model.model.kernel = gpflow.kernels.RBF(lengthscales=tf.ones([dim], dtype=tf.float64))
    model.model.likelihood.variance.assign(1.0)
    gpflow.set_trainable(model.model.likelihood, True)

    if prior_for_lengthscale:
        model.model.kernel.lengthscales.prior = tfp.distributions.LogNormal(
            loc=tf.math.log(model.model.kernel.lengthscales), scale=1.0
        )

    else:
        upper = tf.cast([10.0] * dim, dtype=tf.float64)
        lower = upper / 100
        model.model.kernel.lengthscales = gpflow.Parameter(
            model.model.kernel.lengthscales, transform=tfp.bijectors.Sigmoid(low=lower, high=upper)
        )

    model.model.likelihood.variance.prior = tfp.distributions.LogNormal(
        loc=tf.cast(-2.0, dtype=tf.float64), scale=tf.cast(5.0, dtype=tf.float64)
    )

    dataset = Dataset(x, tf.cast(fnc_3x_plus_10(x), dtype=tf.float64))
    model.optimize(dataset)

    mocked_model_initializer.assert_called_once()
    num_samples = mocked_model_initializer.call_args[0][0]
    npt.assert_array_equal(num_samples, 10 * (dim + 1))


def test_gaussian_process_regression_best_initialization_changes_params_with_priors(
    dim: int,
) -> None:
    x = tf.constant(
        np.arange(1, 1 + 10 * dim).reshape(-1, dim), dtype=gpflow.default_float()
    )  # shape: [10, dim]
    model = GaussianProcessRegression(gpr_model(x, fnc_3x_plus_10(x)[:, 0:1]))
    model.model.kernel = gpflow.kernels.RBF(lengthscales=[0.2] * dim)

    model.model.kernel.lengthscales.prior = tfp.distributions.LogNormal(
        loc=tf.math.log(model.model.kernel.lengthscales), scale=1.0
    )

    model.find_best_model_initialization(2)

    npt.assert_allclose(1.0, model.model.kernel.variance)
    npt.assert_array_equal(dim, model.model.kernel.lengthscales.shape)
    npt.assert_raises(
        AssertionError, npt.assert_allclose, [0.2, 0.2], model.model.kernel.lengthscales
    )


def test_gaussian_process_regression_best_initialization_changes_params_with_sigmoid_bijectors(
    dim: int,
) -> None:
    x = tf.constant(
        np.arange(1, 1 + 10 * dim).reshape(-1, dim), dtype=gpflow.default_float()
    )  # shape: [10, dim]
    model = GaussianProcessRegression(gpr_model(x, fnc_3x_plus_10(x)[:, 0:1]))
    model.model.kernel = gpflow.kernels.RBF(lengthscales=[0.2] * dim)

    upper = tf.cast([10.0] * dim, dtype=tf.float64)
    lower = upper / 100
    model.model.kernel.lengthscales = gpflow.Parameter(
        model.model.kernel.lengthscales, transform=tfp.bijectors.Sigmoid(low=lower, high=upper)
    )

    model.find_best_model_initialization(2)

    npt.assert_allclose(1.0, model.model.kernel.variance)
    npt.assert_array_equal(dim, model.model.kernel.lengthscales.shape)
    npt.assert_raises(
        AssertionError, npt.assert_allclose, [0.2, 0.2], model.model.kernel.lengthscales
    )


@random_seed
def test_gaussian_process_regression_best_initialization_improves_training_loss(dim: int) -> None:
    x = tf.constant(
        np.arange(1, 1 + 10 * dim).reshape(-1, dim), dtype=gpflow.default_float()
    )  # shape: [10, dim]
    model = GaussianProcessRegression(gpr_model(x, fnc_3x_plus_10(x)[:, 0:1]))
    model.model.kernel = gpflow.kernels.RBF(variance=0.01, lengthscales=[0.011] * dim)

    upper = tf.cast([100.0] * dim, dtype=tf.float64)
    lower = upper / 10000
    model.model.kernel.lengthscales = gpflow.Parameter(
        model.model.kernel.lengthscales, transform=tfp.bijectors.Sigmoid(low=lower, high=upper)
    )

    pre_init_likelihood = -model.model.training_loss()
    model.find_best_model_initialization(10)
    post_init_likelihood = -model.model.training_loss()

    npt.assert_array_less(pre_init_likelihood, post_init_likelihood)


@random_seed
def test_gaussian_process_regression_best_initialization_improves_likelihood(dim: int) -> None:
    x = tf.constant(
        np.arange(1, 1 + 10 * dim).reshape(-1, dim), dtype=gpflow.default_float()
    )  # shape: [10, dim]
    model = GaussianProcessRegression(gpr_model(x, fnc_3x_plus_10(x)[:, 0:1]))
    model.model.kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=[0.2] * dim)

    model.model.kernel.variance.prior = tfp.distributions.LogNormal(
        loc=np.float64(-2.0), scale=np.float64(1.0)
    )
    upper = tf.cast([10.0] * dim, dtype=tf.float64)
    lower = upper / 100
    model.model.kernel.lengthscales = gpflow.Parameter(
        model.model.kernel.lengthscales, transform=tfp.bijectors.Sigmoid(low=lower, high=upper)
    )

    pre_init_loss = model.model.training_loss()
    model.find_best_model_initialization(100)
    post_init_loss = model.model.training_loss()

    npt.assert_array_less(post_init_loss, pre_init_loss)


def test_gaussian_process_regression_default_optimizer_is_correct() -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = _3x_plus_gaussian_noise(x_observed)

    model = GaussianProcessRegression(gpr_model(x_observed[:10], y_observed[:10]))

    assert isinstance(model.optimizer, Optimizer)
    assert isinstance(model.optimizer.optimizer, gpflow.optimizers.Scipy)


def test_gpr_config_builds_and_default_optimizer_is_correct() -> None:
    data = mock_data()

    model_config = {"model": gpr_model(*data)}
    model = create_model(model_config)

    assert isinstance(model, GaussianProcessRegression)
    assert isinstance(model.optimizer, Optimizer)
    assert isinstance(model.optimizer.optimizer, gpflow.optimizers.Scipy)


def test_sgpr_config_builds_and_default_optimizer_is_correct() -> None:
    data = mock_data()

    model_config = {"model": sgpr_model(*data)}
    model = create_model(model_config)

    assert isinstance(model, GaussianProcessRegression)
    assert isinstance(model.optimizer, Optimizer)
    assert isinstance(model.optimizer.optimizer, gpflow.optimizers.Scipy)


@random_seed
def test_gaussian_process_regression_trajectory_sampler_returns_correct_trajectory_sampler(
    dim: int,
) -> None:

    x = tf.constant(
        np.arange(1, 1 + 10 * dim).reshape(-1, dim), dtype=gpflow.default_float()
    )  # shape: [10, dim]
    model = GaussianProcessRegression(gpr_model(x, fnc_3x_plus_10(x)[:, 0:1]))
    model.model.kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=[0.2] * dim)
    trajectory_sampler = model.trajectory_sampler()

    assert isinstance(trajectory_sampler, RandomFourierFeatureTrajectorySampler)


@random_seed
def test_gaussian_process_regression_trajectory_sampler_has_correct_samples() -> None:

    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model = GaussianProcessRegression(gpr_model(x, _3x_plus_gaussian_noise(x)))
    x_predict = tf.constant([[50.5]], gpflow.default_float())

    samples = []
    num_samples = 10
    trajectory_sampler = model.trajectory_sampler()
    trajectory = trajectory_sampler.get_trajectory()
    samples.append(-1.0 * trajectory(tf.expand_dims(x_predict, -2)))
    for _ in range(num_samples - 1):
        trajectory.resample()  # type: ignore
        samples.append(trajectory(tf.expand_dims(x_predict, -2)))

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2)

    true_mean, true_variance = model.predict(x_predict)

    linear_error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean + 1.0, true_mean + 1.0, rtol=linear_error)
    npt.assert_allclose(sample_variance, true_variance, rtol=2 * linear_error)


def test_gpflow_wrappers_predict_y(gpflow_interface_factory: ModelFactoryType) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model, _ = gpflow_interface_factory(x, _3x_plus_gaussian_noise(x))
    x_predict = tf.constant([[50.5]], gpflow.default_float())
    mean_f, variance_f = model.predict(x_predict)
    mean_y, variance_y = model.predict_y(x_predict)

    npt.assert_allclose(mean_f, mean_y)
    npt.assert_array_less(variance_f, variance_y)


@unittest.mock.patch("trieste.models.gpflow.interface.tf.summary.scalar")
def test_gpflow_wrappers_log(
    mocked_summary_scalar: unittest.mock.MagicMock, gpflow_interface_factory: ModelFactoryType
) -> None:
    x = tf.constant(np.arange(1, 5).reshape(-1, 1), dtype=gpflow.default_float())  # shape: [4, 1]
    model, _ = gpflow_interface_factory(x, fnc_3x_plus_10(x))
    mocked_summary_writer = unittest.mock.MagicMock()
    with tensorboard_writer(mocked_summary_writer):
        with step_number(42):
            model.log()

    assert len(mocked_summary_writer.method_calls) == 1
    assert mocked_summary_writer.method_calls[0][0] == "as_default"
    assert mocked_summary_writer.method_calls[0][-1]["step"] == 42

    assert mocked_summary_scalar.call_count == 2
    assert mocked_summary_scalar.call_args_list[0][0][0] == "kernel.variance"
    assert mocked_summary_scalar.call_args_list[0][0][1].numpy() == 1
    assert mocked_summary_scalar.call_args_list[1][0][0] == "kernel.lengthscale"
    assert mocked_summary_scalar.call_args_list[1][0][1].numpy() == 1


def test_variational_gaussian_process_raises_for_invalid_init() -> None:
    x_np = np.arange(5, dtype=np.float64).reshape(-1, 1)
    x = tf.convert_to_tensor(x_np, x_np.dtype)
    y = fnc_3x_plus_10(x)

    with pytest.raises(ValueError):
        VariationalGaussianProcess(vgp_model(x, y), natgrad_gamma=1)

    with pytest.raises(ValueError):
        optimizer = Optimizer(gpflow.optimizers.Scipy())
        VariationalGaussianProcess(vgp_model(x, y), optimizer=optimizer, use_natgrads=True)

    with pytest.raises(ValueError):
        optimizer = BatchOptimizer(gpflow.optimizers.Scipy())
        VariationalGaussianProcess(vgp_model(x, y), optimizer=optimizer, use_natgrads=True)

    with pytest.raises(ValueError):
        optimizer = Optimizer(tf.optimizers.Adam())
        VariationalGaussianProcess(vgp_model(x, y), optimizer=optimizer, use_natgrads=False)


def test_variational_gaussian_process_correctly_returns_internal_data() -> None:
    data = mock_data()
    model = VariationalGaussianProcess(vgp_model(*data))
    returned_data = model.get_internal_data()
    npt.assert_array_equal(returned_data.query_points, data[0])
    npt.assert_array_equal(returned_data.observations, data[1])


def test_variational_gaussian_process_update_updates_num_data() -> None:
    x_np = np.arange(5, dtype=np.float64).reshape(-1, 1)
    x = tf.convert_to_tensor(x_np, x_np.dtype)
    y = fnc_3x_plus_10(x)
    m = VariationalGaussianProcess(vgp_model(x, y))
    num_data = m.model.num_data

    x_new = tf.concat([x, [[10.0], [11.0]]], 0)
    y_new = fnc_3x_plus_10(x_new)
    m.update(Dataset(x_new, y_new))
    new_num_data = m.model.num_data
    assert new_num_data - num_data == 2


def test_variational_gaussian_process_update() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())

    data = Dataset(x, fnc_3x_plus_10(x))
    m = VariationalGaussianProcess(vgp_model(data.query_points, data.observations))

    reference_model = vgp_model(data.query_points, data.observations)

    npt.assert_allclose(m.model.q_mu, reference_model.q_mu, atol=1e-5)
    npt.assert_allclose(m.model.q_sqrt, reference_model.q_sqrt, atol=1e-5)

    x_new = tf.concat([x, tf.constant([[10.0], [11.0]], dtype=gpflow.default_float())], 0)
    new_data = Dataset(x_new, fnc_3x_plus_10(x_new))

    m.update(new_data)
    reference_model_new = vgp_model(new_data.query_points, new_data.observations)

    npt.assert_allclose(m.model.q_mu, reference_model_new.q_mu, atol=1e-5)
    npt.assert_allclose(m.model.q_sqrt, reference_model_new.q_sqrt, atol=1e-5)


@random_seed
def test_variational_gaussian_process_update_q_mu_sqrt_unchanged() -> None:
    x_observed = tf.constant(np.arange(10).reshape((-1, 1)), dtype=gpflow.default_float())
    y_observed = fnc_2sin_x_over_3(x_observed)
    model = VariationalGaussianProcess(vgp_matern_model(x_observed, y_observed))

    old_q_mu = model.model.q_mu.numpy()
    old_q_sqrt = model.model.q_sqrt.numpy()
    data = Dataset(x_observed, y_observed)
    model.update(data)

    new_q_mu = model.model.q_mu.numpy()
    new_q_sqrt = model.model.q_sqrt.numpy()

    npt.assert_allclose(old_q_mu, new_q_mu, atol=1e-5)
    npt.assert_allclose(old_q_sqrt, new_q_sqrt, atol=1e-5)


@random_seed
def test_gpflow_wrappers_default_optimize(
    gpflow_interface_factory: ModelFactoryType,
) -> None:
    data = mock_data()
    model, _ = gpflow_interface_factory(*data)
    internal_model = model.model
    if isinstance(internal_model, SVGP):
        args = {"data": data}
    else:
        args = {}
    loss = internal_model.training_loss(**args)
    model.optimize(Dataset(*data))
    assert internal_model.training_loss(**args) < loss


def test_gaussian_process_regression_optimize(compile: bool) -> None:

    data = mock_data()

    optimizer = Optimizer(gpflow.optimizers.Scipy(), compile=compile)
    model = GaussianProcessRegression(gpr_model(*data), optimizer)

    loss = model.model.training_loss()
    model.optimize(Dataset(*data))

    assert model.model.training_loss() < loss


@random_seed
def test_variational_gaussian_process_predict() -> None:
    x_observed = tf.constant(np.arange(3).reshape((-1, 1)), dtype=gpflow.default_float())
    y_observed = _3x_plus_gaussian_noise(x_observed)
    model = VariationalGaussianProcess(vgp_model(x_observed, y_observed))
    internal_model = model.model

    gpflow.optimizers.Scipy().minimize(
        internal_model.training_loss_closure(),
        internal_model.trainable_variables,
    )
    x_predict = tf.constant([[1.5]], gpflow.default_float())
    mean, variance = model.predict(x_predict)
    mean_y, variance_y = model.predict_y(x_predict)

    reference_model = vgp_model(x_observed, y_observed)

    reference_model.data = (
        tf.Variable(
            reference_model.data[0],
            trainable=False,
            shape=[None, *reference_model.data[0].shape[1:]],
        ),
        tf.Variable(
            reference_model.data[1],
            trainable=False,
            shape=[None, *reference_model.data[1].shape[1:]],
        ),
    )

    gpflow.optimizers.Scipy().minimize(
        reference_model.training_loss_closure(),
        reference_model.trainable_variables,
    )
    reference_mean, reference_variance = reference_model.predict_f(x_predict)

    npt.assert_allclose(mean, reference_mean)
    npt.assert_allclose(variance, reference_variance, atol=1e-3)
    npt.assert_allclose(variance_y - model.get_observation_noise(), variance, atol=5e-5)


def test_sparse_variational_model_attribute() -> None:
    model = svgp_model(*mock_data())
    sv = SparseVariational(model)
    assert sv.model is model
    assert isinstance(sv.model, SVGP)
    assert isinstance(sv.model, NumDataPropertyMixin)


def test_sparse_variational_model_num_data_mixin_supports_subclasses() -> None:
    class SVGPSubclass(SVGP):  # type: ignore[misc]
        @property
        def mol(self) -> int:
            return 42

    x = mock_data()[0]
    model = SVGPSubclass(
        gpflow.kernels.Matern32(), gpflow.likelihoods.Gaussian(), x[:2], num_data=len(x)
    )
    sv = SparseVariational(model)
    assert sv.model is model
    assert isinstance(sv.model, NumDataPropertyMixin)
    assert isinstance(sv.model, SVGPSubclass)
    assert sv.model.mol == 42


def test_sparse_variational_update_updates_num_data() -> None:
    model = SparseVariational(
        svgp_model(tf.zeros([1, 4]), tf.zeros([1, 1])),
    )
    model.update(Dataset(tf.zeros([5, 4]), tf.zeros([5, 1])))
    assert model.model.num_data == 5


@pytest.mark.parametrize(
    "new_data",
    [Dataset(tf.zeros([3, 5]), tf.zeros([3, 1])), Dataset(tf.zeros([3, 4]), tf.zeros([3, 2]))],
)
def test_sparse_variational_update_raises_for_invalid_shapes(new_data: Dataset) -> None:
    model = SparseVariational(
        svgp_model(tf.zeros([1, 4]), tf.zeros([1, 1])),
    )
    with pytest.raises(ValueError):
        model.update(new_data)


def test_sparse_variational_optimize_with_defaults() -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = _3x_plus_gaussian_noise(x_observed)
    data = x_observed, y_observed
    dataset = Dataset(*data)
    optimizer = BatchOptimizer(tf.optimizers.Adam(), max_iter=20)
    model = SparseVariational(svgp_model(x_observed, y_observed), optimizer=optimizer)
    loss = model.model.training_loss(data)
    model.optimize(dataset)
    assert model.model.training_loss(data) < loss


def test_sparse_variational_optimize(batcher: DatasetTransformer, compile: bool) -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = _3x_plus_gaussian_noise(x_observed)
    data = x_observed, y_observed
    dataset = Dataset(*data)

    optimizer = BatchOptimizer(
        tf.optimizers.Adam(),
        max_iter=10,
        batch_size=10,
        dataset_builder=batcher,
        compile=compile,
    )
    model = SparseVariational(svgp_model(x_observed, y_observed), optimizer=optimizer)
    loss = model.model.training_loss(data)
    model.optimize(dataset)
    assert model.model.training_loss(data) < loss


def test_sparse_variational_default_optimizer_is_correct() -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = _3x_plus_gaussian_noise(x_observed)

    model = SparseVariational(svgp_model(x_observed, y_observed))

    assert isinstance(model.optimizer, BatchOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.optimizers.Optimizer)


def test_svgp_config_builds_and_default_optimizer_is_correct() -> None:
    data = mock_data()

    model_config = {"model": svgp_model(*data)}
    model = create_model(model_config)

    assert isinstance(model, SparseVariational)
    assert isinstance(model.optimizer, BatchOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.optimizers.Optimizer)


def test_sparse_variational_raises_for_invalid_init() -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = _3x_plus_gaussian_noise(x_observed)

    with pytest.raises(ValueError):
        optimizer1 = BatchOptimizer(gpflow.optimizers.Scipy())
        SparseVariational(svgp_model(x_observed, y_observed), optimizer=optimizer1)

    with pytest.raises(ValueError):
        optimizer2 = Optimizer(tf.optimizers.Adam())
        SparseVariational(svgp_model(x_observed, y_observed), optimizer=optimizer2)


@pytest.mark.parametrize("use_natgrads", [True, False])
def test_variational_gaussian_process_optimize_with_and_without_natgrads(
    batcher: DatasetTransformer, compile: bool, use_natgrads: bool
) -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = _3x_plus_gaussian_noise(x_observed)
    data = x_observed, y_observed
    dataset = Dataset(*data)

    if use_natgrads:
        optimizer = BatchOptimizer(
            tf.optimizers.Adam(),
            max_iter=10,
            batch_size=10,
            dataset_builder=batcher,
            compile=compile,
        )
    else:
        optimizer = Optimizer(gpflow.optimizers.Scipy(), compile=compile)  # type:ignore

    model = VariationalGaussianProcess(
        vgp_model(x_observed[:10], y_observed[:10]), optimizer=optimizer, use_natgrads=use_natgrads
    )
    loss = model.model.training_loss()
    model.optimize(dataset)
    assert model.model.training_loss() < loss


def test_variational_gaussian_process_optimize_natgrads_only_updates_variational_params(
    compile: bool,
) -> None:
    x_observed = np.linspace(0, 100, 10).reshape((-1, 1))
    y_observed = _3x_plus_gaussian_noise(x_observed)
    data = x_observed, y_observed
    dataset = Dataset(*data)

    class DummyBatchOptimizer(BatchOptimizer):
        def optimize(self, model: tf.Module, dataset: Dataset) -> None:
            pass

    optimizer = DummyBatchOptimizer(tf.optimizers.Adam(), compile=compile, max_iter=10)

    model = VariationalGaussianProcess(
        vgp_matern_model(x_observed[:10], y_observed[:10]), optimizer=optimizer, use_natgrads=True
    )

    old_num_trainable_params = len(model.model.trainable_variables)
    old_kernel_params = model.get_kernel().parameters[0].numpy()
    old_q_mu = model.model.q_mu.numpy()
    old_q_sqrt = model.model.q_sqrt.numpy()

    model.optimize(dataset)

    new_num_trainable_params = len(model.model.trainable_variables)
    new_kernel_params = model.get_kernel().parameters[0].numpy()
    new_q_mu = model.model.q_mu.numpy()
    new_q_sqrt = model.model.q_sqrt.numpy()

    npt.assert_allclose(old_kernel_params, new_kernel_params, atol=1e-3)
    npt.assert_equal(old_num_trainable_params, new_num_trainable_params)
    npt.assert_raises(AssertionError, npt.assert_allclose, old_q_mu, new_q_mu)
    npt.assert_raises(AssertionError, npt.assert_allclose, old_q_sqrt, new_q_sqrt)


@pytest.mark.parametrize("use_natgrads", [True, False])
def test_variational_gaussian_process_default_optimizer_is_correct(use_natgrads: bool) -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = _3x_plus_gaussian_noise(x_observed)

    model = VariationalGaussianProcess(
        vgp_model(x_observed[:10], y_observed[:10]), use_natgrads=use_natgrads
    )

    if use_natgrads:
        assert isinstance(model.optimizer, BatchOptimizer)
        assert isinstance(model.optimizer.optimizer, tf.optimizers.Optimizer)
    else:
        assert isinstance(model.optimizer, Optimizer)
        assert isinstance(model.optimizer.optimizer, gpflow.optimizers.Scipy)


@pytest.mark.parametrize("use_natgrads", [True, False])
def test_vgp_config_builds_and_default_optimizer_is_correct(use_natgrads: bool) -> None:
    data = mock_data()

    model_config = {"model": vgp_model(*data), "model_args": {"use_natgrads": use_natgrads}}
    model = create_model(model_config)

    assert isinstance(model, VariationalGaussianProcess)
    if use_natgrads:
        assert isinstance(model.optimizer, BatchOptimizer)
        assert isinstance(model.optimizer.optimizer, tf.optimizers.Optimizer)
    else:
        assert isinstance(model.optimizer, Optimizer)
        assert isinstance(model.optimizer.optimizer, gpflow.optimizers.Scipy)


@random_seed
def test_gpflow_predictor_get_observation_noise_raises_for_likelihood_with_variance(
    gpflow_interface_factory: ModelFactoryType,
) -> None:
    data = mock_data()
    model, _ = gpflow_interface_factory(*data)
    model.model.likelihood = gpflow.likelihoods.Gaussian()  # has variance attribute
    model.get_observation_noise()

    model.model.likelihood = gpflow.likelihoods.Bernoulli()  # does not have variance attribute
    with pytest.raises(NotImplementedError):
        model.get_observation_noise()


def test_gaussian_process_regression_conditional_predict_equations() -> None:
    x = gpflow.utilities.to_default_float(
        tf.constant(np.arange(1, 8).reshape(-1, 1) / 8.0)
    )  # shape: [7, 1]
    y = fnc_2sin_x_over_3(x)

    model7 = GaussianProcessRegression(gpr_model(x, y))
    model5 = GaussianProcessRegression(gpr_model(x[:5, :], y[:5, :]))

    additional_data = Dataset(x[5:, :], y[5:, :])

    query_points = tf.concat([0.5 * x, 2.0 * x], 0)  # shape: [14, 1]

    predj_mean7, predj_cov7 = model7.predict_joint(query_points)
    predj_mean5, predj_cov5 = model5.conditional_predict_joint(query_points, additional_data)

    pred_mean7, pred_var7 = model7.predict(query_points)
    pred_mean5, pred_var5 = model5.conditional_predict_f(query_points, additional_data)

    predy_mean7, predy_var7 = model7.predict_y(query_points)
    predy_mean5, predy_var5 = model5.conditional_predict_y(query_points, additional_data)

    np.testing.assert_allclose(tf.transpose(tf.linalg.diag_part(predj_cov5)), pred_var5, atol=1e-5)
    np.testing.assert_allclose(predj_mean5, pred_mean5, atol=1e-5)
    np.testing.assert_allclose(predj_mean5, predj_mean7, atol=1e-5)
    np.testing.assert_allclose(pred_mean7, pred_mean5, atol=1e-5)
    np.testing.assert_allclose(pred_var7, pred_var5, atol=1e-5)
    np.testing.assert_allclose(predj_cov7, predj_cov5, atol=1e-5)
    np.testing.assert_allclose(predy_mean7, predy_mean5, atol=1e-5)
    np.testing.assert_allclose(predy_var7, predy_var5, atol=1e-5)


def test_gaussian_process_regression_conditional_predict_equations_broadcast() -> None:
    x = gpflow.utilities.to_default_float(
        tf.constant(np.arange(1, 24).reshape(-1, 1) / 8.0)
    )  # shape: [23, 1]
    y = fnc_2sin_x_over_3(x)

    model5 = GaussianProcessRegression(gpr_model(x[:5, :], y[:5, :]))

    additional_data = Dataset(tf.reshape(x[5:, :], [3, 6, -1]), tf.reshape(y[5:, :], [3, 6, -1]))

    query_points = tf.concat([0.5 * x, 2.0 * x], 0)  # shape: [46, 1]

    predj_mean5, predj_cov5 = model5.conditional_predict_joint(query_points, additional_data)
    pred_mean5, pred_var5 = model5.conditional_predict_f(query_points, additional_data)
    predy_mean5, predy_var5 = model5.conditional_predict_y(query_points, additional_data)

    for i in range(3):
        xi = tf.concat([x[:5, :], additional_data.query_points[i, ...]], axis=0)
        yi = tf.concat([y[:5, :], additional_data.observations[i, ...]], axis=0)

        modeli = GaussianProcessRegression(gpr_model(xi, yi))
        predj_meani, predj_covi = modeli.predict_joint(query_points)
        pred_meani, pred_vari = modeli.predict(query_points)
        predy_meani, predy_vari = modeli.predict_y(query_points)

        np.testing.assert_allclose(predj_mean5[i, ...], predj_meani, atol=1e-5)
        np.testing.assert_allclose(pred_meani, pred_mean5[i, ...], atol=1e-5)
        np.testing.assert_allclose(pred_vari, pred_var5[i, ...], atol=1e-5)
        np.testing.assert_allclose(predj_covi, predj_cov5[i, ...], atol=1e-5)
        np.testing.assert_allclose(predy_vari, predy_var5[i, ...], atol=1e-5)
        np.testing.assert_allclose(predy_vari, predy_var5[i, ...], atol=1e-5)


def test_gaussian_process_regression_conditional_predict_f_sample() -> None:
    x = gpflow.utilities.to_default_float(
        tf.constant(np.arange(1, 24).reshape(-1, 1) / 8.0)
    )  # shape: [23, 1]
    y = fnc_2sin_x_over_3(x)

    model5 = GaussianProcessRegression(gpr_model(x[:5, :], y[:5, :]))
    additional_data = Dataset(tf.reshape(x[5:, :], [3, 6, -1]), tf.reshape(y[5:, :], [3, 6, -1]))
    query_points = tf.concat([0.5 * x, 2.0 * x], 0)  # shape: [46, 1]
    samples = model5.conditional_predict_f_sample(query_points, additional_data, num_samples=100000)
    npt.assert_array_equal([3, 100000, 46, 1], samples.shape)

    for i in range(3):
        xi = tf.concat([x[:5, :], additional_data.query_points[i, ...]], axis=0)
        yi = tf.concat([y[:5, :], additional_data.observations[i, ...]], axis=0)

        modeli = GaussianProcessRegression(gpr_model(xi, yi))
        predj_meani, predj_covi = modeli.predict_joint(query_points)
        sample_mean = tf.reduce_mean(samples[i], axis=0)
        sample_cov = tfp.stats.covariance(samples[i, :, :, 0], sample_axis=0)
        np.testing.assert_allclose(sample_mean, predj_meani, atol=1e-2, rtol=1e-2)
        np.testing.assert_allclose(sample_cov, predj_covi[0], atol=1e-2, rtol=1e-2)

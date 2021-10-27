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
from typing import Any

import gpflow
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import SGPR, SVGP, VGP

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
from trieste.models.gpflow import (
    GaussianProcessRegression,
    SparseVariational,
    VariationalGaussianProcess,
)
from trieste.models.optimizer import BatchOptimizer, DatasetTransformer, Optimizer


def _3x_plus_gaussian_noise(x: tf.Tensor) -> tf.Tensor:
    return 3.0 * x + np.random.normal(scale=0.01, size=x.shape)


def test_gaussian_process_regression_loss(gpflow_interface_factory: ModelFactoryType) -> None:
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


def test_gaussian_process_regression_update(gpflow_interface_factory: ModelFactoryType) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    y = fnc_3x_plus_10(x)

    model, _reference_model = gpflow_interface_factory(x, y)

    x_new = tf.concat([x, tf.constant([[10.0], [11.0]], dtype=gpflow.default_float())], 0)
    new_data = Dataset(x_new, fnc_3x_plus_10(x_new))
    model.update(new_data)

    reference_model = _reference_model(x_new, fnc_3x_plus_10(x_new))
    internal_model = model.model

    if isinstance(internal_model, SVGP):
        args = {"data": (new_data.query_points, new_data.observations)}
    else:
        args = {}

    npt.assert_allclose(
        internal_model.training_loss(**args), reference_model.training_loss(**args), rtol=1e-6
    )


def test_gaussian_process_regression_ref_optimize(
    gpflow_interface_factory: ModelFactoryType,
) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    y = fnc_2sin_x_over_3(x)

    model, _reference_model = gpflow_interface_factory(
        x, y, optimizer=Optimizer(gpflow.optimizers.Scipy())
    )

    reference_model = _reference_model(x, y)
    model.optimize(Dataset(x, y))
    internal_model = model.model

    if isinstance(internal_model, SVGP):
        args = {"data": (x, y)}
    else:
        args = {}
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
        reference_model.training_loss_closure(**args, compile=False),
        reference_model.trainable_variables,
    )

    npt.assert_allclose(
        internal_model.training_loss(**args), reference_model.training_loss(**args), rtol=1e-6
    )


def test_gaussian_process_regression_pairwise_covariance(
    gpflow_interface_factory: ModelFactoryType,
) -> None:
    x = tf.constant(np.arange(1, 5).reshape(-1, 1), dtype=gpflow.default_float())  # shape: [4, 1]
    y = fnc_3x_plus_10(x)
    model, _ = gpflow_interface_factory(x, y)

    if isinstance(model.model, (SGPR, VGP, SVGP)):
        pytest.skip("covariance_between_points is only implemented for the GPR model.")

    query_points_1 = tf.concat([0.5 * x, 0.5 * x], 0)  # shape: [8, 1]
    query_points_2 = tf.concat([2 * x, 2 * x, 2 * x], 0)  # shape: [12, 1]

    all_query_points = tf.concat([query_points_1, query_points_2], 0)
    _, predictive_covariance = model.predict_joint(all_query_points)
    expected_covariance = predictive_covariance[0, :8, 8:]

    actual_covariance = model.covariance_between_points(query_points_1, query_points_2)

    np.testing.assert_allclose(expected_covariance, actual_covariance, atol=1e-5)


def test_sgpr_raises_for_covariance_between_points() -> None:
    data = mock_data()
    model = GaussianProcessRegression(sgpr_model(*data))

    with pytest.raises(NotImplementedError):
        model.covariance_between_points(data[0], data[0])


def test_gpr_raises_for_invalid_num_kernel_samples() -> None:
    x_np = np.arange(5, dtype=np.float64).reshape(-1, 1)
    x = tf.convert_to_tensor(x_np, x_np.dtype)
    y = fnc_3x_plus_10(x)

    with pytest.raises(ValueError):
        GaussianProcessRegression(gpr_model(x, y), num_kernel_samples=-1)


@random_seed
@unittest.mock.patch(
    "trieste.models.gpflow.models.GaussianProcessRegression.find_best_model_initialization"
)
@pytest.mark.parametrize("prior_for_lengthscale", [True, False])
def test_gaussian_process_regression_correctly_counts_params_that_can_be_sampled(
    mocked_model_initializer: Any,
    dim: int,
    prior_for_lengthscale: bool,
    gpflow_interface_factory: ModelFactoryType,
) -> None:
    x = tf.constant(np.arange(1, 5 * dim + 1).reshape(-1, dim), dtype=tf.float64)  # shape: [5, d]
    model, _ = gpflow_interface_factory(x, fnc_3x_plus_10(x))
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

    if isinstance(model, (VariationalGaussianProcess, SparseVariational)):
        pytest.skip("find_best_model_initialization is only implemented for the GPR models.")

    dataset = Dataset(x, tf.cast(fnc_3x_plus_10(x), dtype=tf.float64))
    model.optimize(dataset)

    mocked_model_initializer.assert_called_once()
    num_samples = mocked_model_initializer.call_args[0][0]
    npt.assert_array_equal(num_samples, 10 * (dim + 1))


def test_find_best_model_initialization_changes_params_with_priors(
    gpflow_interface_factory: ModelFactoryType, dim: int
) -> None:
    x = tf.constant(
        np.arange(1, 1 + 10 * dim).reshape(-1, dim), dtype=gpflow.default_float()
    )  # shape: [10, dim]
    model, _ = gpflow_interface_factory(x, fnc_3x_plus_10(x)[:, 0:1])
    model.model.kernel = gpflow.kernels.RBF(lengthscales=[0.2] * dim)

    if isinstance(model, (VariationalGaussianProcess, SparseVariational)):
        pytest.skip("find_best_model_initialization is only implemented for the GPR models.")

    model.model.kernel.lengthscales.prior = tfp.distributions.LogNormal(
        loc=tf.math.log(model.model.kernel.lengthscales), scale=1.0
    )

    model.find_best_model_initialization(2)

    npt.assert_allclose(1.0, model.model.kernel.variance)
    npt.assert_array_equal(dim, model.model.kernel.lengthscales.shape)
    npt.assert_raises(
        AssertionError, npt.assert_allclose, [0.2, 0.2], model.model.kernel.lengthscales
    )


def test_find_best_model_initialization_changes_params_with_sigmoid_bijectors(
    gpflow_interface_factory: ModelFactoryType, dim: int
) -> None:
    x = tf.constant(
        np.arange(1, 1 + 10 * dim).reshape(-1, dim), dtype=gpflow.default_float()
    )  # shape: [10, dim]
    model, _ = gpflow_interface_factory(x, fnc_3x_plus_10(x)[:, 0:1])
    model.model.kernel = gpflow.kernels.RBF(lengthscales=[0.2] * dim)

    if isinstance(model, (VariationalGaussianProcess, SparseVariational)):
        pytest.skip("find_best_model_initialization is only implemented for the GPR models.")

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
def test_find_best_model_initialization_without_priors_improves_training_loss(
    gpflow_interface_factory: ModelFactoryType, dim: int
) -> None:
    x = tf.constant(
        np.arange(1, 1 + 10 * dim).reshape(-1, dim), dtype=gpflow.default_float()
    )  # shape: [10, dim]
    model, _ = gpflow_interface_factory(x, fnc_3x_plus_10(x)[:, 0:1])
    model.model.kernel = gpflow.kernels.RBF(variance=0.01, lengthscales=[0.011] * dim)

    if isinstance(model, (VariationalGaussianProcess, SparseVariational)):
        pytest.skip("find_best_model_initialization is only implemented for the GPR models.")

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
def test_find_best_model_initialization_improves_likelihood(
    gpflow_interface_factory: ModelFactoryType, dim: int
) -> None:
    x = tf.constant(
        np.arange(1, 1 + 10 * dim).reshape(-1, dim), dtype=gpflow.default_float()
    )  # shape: [10, dim]
    model, _ = gpflow_interface_factory(x, fnc_3x_plus_10(x)[:, 0:1])
    model.model.kernel = gpflow.kernels.RBF(variance=1.0, lengthscales=[0.2] * dim)

    if isinstance(model, (VariationalGaussianProcess, SparseVariational)):
        pytest.skip("find_best_model_initialization is only implemented for the GPR models.")

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


def test_gaussian_process_regression_predict_y(gpflow_interface_factory: ModelFactoryType) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model, _ = gpflow_interface_factory(x, _3x_plus_gaussian_noise(x))
    x_predict = tf.constant([[50.5]], gpflow.default_float())
    mean_f, variance_f = model.predict(x_predict)
    mean_y, variance_y = model.predict_y(x_predict)

    npt.assert_allclose(mean_f, mean_y)
    npt.assert_array_less(variance_f, variance_y)


def test_vgp_raises_for_invalid_init() -> None:
    x_np = np.arange(5, dtype=np.float64).reshape(-1, 1)
    x = tf.convert_to_tensor(x_np, x_np.dtype)
    y = fnc_3x_plus_10(x)

    with pytest.raises(ValueError):
        VariationalGaussianProcess(vgp_model(x, y), natgrad_gamma=1)

    with pytest.raises(ValueError):
        optimizer = Optimizer(gpflow.optimizers.Scipy())
        VariationalGaussianProcess(vgp_model(x, y), optimizer=optimizer, use_natgrads=True)


def test_vgp_update_updates_num_data() -> None:
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


def test_vgp_update() -> None:
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
def test_vgp_update_q_mu_sqrt_unchanged() -> None:
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
def test_gaussian_process_regression_default_optimize(
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


@random_seed
@pytest.mark.parametrize("optimizer", [gpflow.optimizers.Scipy(), tf.optimizers.Adam()])
def test_gaussian_process_regression_optimize(
    optimizer: gpflow.optimizers.Scipy | tf.optimizers.Optimizer,
    gpflow_interface_factory: ModelFactoryType,
    compile: bool,
) -> None:
    data = mock_data()
    if isinstance(optimizer, gpflow.optimizers.Scipy):
        create_optimizer = Optimizer
    elif isinstance(optimizer, tf.optimizers.Optimizer):
        create_optimizer = BatchOptimizer
    optimizer_wrapper = create_optimizer(optimizer, compile=compile)
    model, _ = gpflow_interface_factory(*data, optimizer=optimizer_wrapper)
    internal_model = model.model
    if isinstance(internal_model, SVGP):
        args = {"data": data}
    else:
        args = {}
    loss = internal_model.training_loss(**args)
    model.optimize(Dataset(*data))
    assert internal_model.training_loss(**args) < loss


@random_seed
def test_variational_gaussian_process_predict() -> None:
    x_observed = tf.constant(np.arange(100).reshape((-1, 1)), dtype=gpflow.default_float())
    y_observed = _3x_plus_gaussian_noise(x_observed)
    model = VariationalGaussianProcess(vgp_model(x_observed, y_observed))
    internal_model = model.model

    gpflow.optimizers.Scipy().minimize(
        internal_model.training_loss_closure(),
        internal_model.trainable_variables,
    )
    x_predict = tf.constant([[50.5]], gpflow.default_float())
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


@pytest.mark.parametrize("use_natgrads", [True, False])
def test_vgp_optimize_with_and_without_natgrads(
    batcher: DatasetTransformer, compile: bool, use_natgrads: bool
) -> None:
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
    model = VariationalGaussianProcess(
        vgp_model(x_observed[:10], y_observed[:10]), optimizer=optimizer, use_natgrads=use_natgrads
    )
    loss = model.model.training_loss()
    model.optimize(dataset)
    assert model.model.training_loss() < loss


def test_vgp_optimize_natgrads_only_updates_variational_params(compile: bool) -> None:
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

    old_num_trainable_params = len(model.trainable_variables)
    old_kernel_params = model.get_kernel().parameters[0].numpy()
    old_q_mu = model.model.q_mu.numpy()
    old_q_sqrt = model.model.q_sqrt.numpy()

    model.optimize(dataset)

    new_num_trainable_params = len(model.trainable_variables)
    new_kernel_params = model.get_kernel().parameters[0].numpy()
    new_q_mu = model.model.q_mu.numpy()
    new_q_sqrt = model.model.q_sqrt.numpy()

    npt.assert_allclose(old_kernel_params, new_kernel_params, atol=1e-3)
    npt.assert_equal(old_num_trainable_params, new_num_trainable_params)
    npt.assert_raises(AssertionError, npt.assert_allclose, old_q_mu, new_q_mu)
    npt.assert_raises(AssertionError, npt.assert_allclose, old_q_sqrt, new_q_sqrt)


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

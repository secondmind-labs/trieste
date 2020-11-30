# Copyright 2020 The Trieste Contributors
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
from typing import Tuple, Callable, Union

import gpflow
from gpflow.models import GPModel, GPR, SGPR, VGP, SVGP
import pytest
import tensorflow as tf
import numpy as np
import numpy.testing as npt

from trieste.data import Dataset
from trieste.models.model_interfaces import (
    CustomTrainable,
    Batcher,
    GaussianProcessRegression,
    GPflowPredictor,
    SparseVariational,
    VariationalGaussianProcess,
)
from trieste.type import ObserverEvaluations, TensorType, QueryPoints

from tests.util.misc import random_seed


class _MinimalTrainable(CustomTrainable):
    def loss(self) -> tf.Tensor:
        raise NotImplementedError

    def update(self, dataset: Dataset) -> None:
        raise NotImplementedError

    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        raise NotImplementedError

    def sample(self, query_points: QueryPoints, num_samples: int) -> ObserverEvaluations:
        raise NotImplementedError


def test_trainable_model_interface_default_optimizer() -> None:
    # gpflow.optimizers.Scipy.__init__ is that of object, so it's sufficient to test the type
    assert isinstance(_MinimalTrainable().optimizer, gpflow.optimizers.Scipy)


def test_trainable_model_interface_set_optimizer() -> None:
    model = _MinimalTrainable()
    optimizer = tf.optimizers.Adam()
    model.set_optimizer(optimizer)
    assert model.optimizer is optimizer


def test_trainable_model_interface_default_optimizer_args() -> None:
    assert _MinimalTrainable().optimizer_args == {}


def test_trainable_model_interface_set_optimizer_args() -> None:
    model = _MinimalTrainable()
    optimizer_args = {"a": 1, "b": 2}
    model.set_optimizer_args(optimizer_args)
    assert model.optimizer_args == optimizer_args


def test_trainable_model_interface_set_optimize() -> None:
    class _OptimizeCallable:
        call_count = 0

        def __call__(self) -> None:
            self.call_count += 1

    optimize_callable = _OptimizeCallable()
    model = _MinimalTrainable()
    model.set_optimize(optimize_callable)
    model.optimize()
    assert optimize_callable.call_count == 1


def _mock_data() -> Tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.constant([[1.1], [2.2], [3.3], [4.4]], gpflow.default_float()),
        tf.constant([[1.2], [3.4], [5.6], [7.8]], gpflow.default_float()),
    )


def _gpr(x: tf.Tensor, y: tf.Tensor) -> GPR:
    return GPR((x, y), gpflow.kernels.Linear())


def _sgpr(x: tf.Tensor, y: tf.Tensor) -> SGPR:
    return SGPR((x, y), gpflow.kernels.Linear(), x[: len(x) // 2])


def _svgp(inducing_variable: tf.Tensor) -> SVGP:
    return SVGP(gpflow.kernels.Linear(), gpflow.likelihoods.Gaussian(), inducing_variable)


def _vgp(x: tf.Tensor, y: tf.Tensor) -> VGP:
    likelihood = gpflow.likelihoods.Gaussian()
    kernel = gpflow.kernels.Linear()
    m = VGP((x, y), kernel, likelihood)
    variational_variables = [m.q_mu.unconstrained_variable, m.q_sqrt.unconstrained_variable]
    gpflow.optimizers.Scipy().minimize(m.training_loss_closure(), variational_variables)
    return m


def _vgp_matern(x: tf.Tensor, y: tf.Tensor) -> VGP:
    likelihood = gpflow.likelihoods.Gaussian()
    kernel = gpflow.kernels.Matern32(lengthscales=0.2)
    m = VGP((x, y), kernel, likelihood)
    variational_variables = [m.q_mu.unconstrained_variable, m.q_sqrt.unconstrained_variable]
    gpflow.optimizers.Scipy().minimize(m.training_loss_closure(), variational_variables)
    return m


@pytest.fixture(
    name="gpr_interface_factory",
    params=[
        (GaussianProcessRegression, _gpr),
        (GaussianProcessRegression, _sgpr),
        (VariationalGaussianProcess, _vgp),
    ],
)
def _(request) -> Callable[[tf.Tensor, tf.Tensor], GaussianProcessRegression]:
    return lambda x, y: request.param[0](request.param[1](x, y))


def _reference_gpr(x: tf.Tensor, y: tf.Tensor) -> gpflow.models.GPR:
    return _gpr(x, y)


def _3x_plus_10(x: tf.Tensor) -> tf.Tensor:
    return 3.0 * x + 10


def _2sin_x_over_3(x: tf.Tensor) -> tf.Tensor:
    return 2.0 * tf.math.sin(x / 3.0)


def test_gaussian_process_regression_loss(gpr_interface_factory) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model = gpr_interface_factory(x, _3x_plus_10(x))
    reference_model = _reference_gpr(x, _3x_plus_10(x))
    npt.assert_allclose(model.loss(), -reference_model.log_marginal_likelihood(), rtol=1e-6)


def test_gaussian_process_regression_update(gpr_interface_factory) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model = gpr_interface_factory(x, _3x_plus_10(x))

    x_new = tf.concat([x, tf.constant([[10.0], [11.0]], dtype=gpflow.default_float())], 0)
    model.update(Dataset(x_new, _3x_plus_10(x_new)))
    model.optimize()
    reference_model = _reference_gpr(x_new, _3x_plus_10(x_new))
    gpflow.optimizers.Scipy().minimize(
        reference_model.training_loss_closure(), reference_model.trainable_variables
    )
    npt.assert_allclose(model.loss(), reference_model.training_loss(), rtol=1e-6)


def test_vgp_update_updates_num_data() -> None:
    x_np = np.arange(5, dtype=np.float64).reshape(-1, 1)
    x = tf.convert_to_tensor(x_np, x_np.dtype)
    y = _3x_plus_10(x)
    m = VariationalGaussianProcess(_vgp(x, y))
    num_data = m.model.num_data

    x_new = tf.concat([x, [[10.0], [11.0]]], 0)
    y_new = _3x_plus_10(x_new)
    m.update(Dataset(x_new, y_new))
    new_num_data = m.model.num_data
    assert new_num_data - num_data == 2


@random_seed
def test_vgp_update_q_mu_sqrt_unchanged() -> None:
    x_observed = tf.constant(np.arange(10).reshape((-1, 1)), dtype=gpflow.default_float())
    y_observed = _2sin_x_over_3(x_observed)
    model = VariationalGaussianProcess(_vgp_matern(x_observed, y_observed))

    old_q_mu = model.model.q_mu.numpy()
    old_q_sqrt = model.model.q_sqrt.numpy()
    data = Dataset(x_observed, y_observed)
    model.update(data)

    new_q_mu = model.model.q_mu.numpy()
    new_q_sqrt = model.model.q_sqrt.numpy()

    npt.assert_allclose(old_q_mu, new_q_mu, atol=1e-5)
    npt.assert_allclose(old_q_sqrt, new_q_sqrt, atol=1e-5)


@random_seed
def test_gaussian_process_regression_default_optimize(gpr_interface_factory) -> None:
    model = gpr_interface_factory(*_mock_data())
    loss = model.loss()
    model.optimize()
    assert model.loss() < loss


@random_seed
@pytest.mark.parametrize("optimizer", [gpflow.optimizers.Scipy(), tf.optimizers.Adam(), None])
def test_gaussian_process_regression_optimize(
    optimizer: Union[gpflow.optimizers.Scipy, tf.optimizers.Optimizer, None], gpr_interface_factory
) -> None:
    model = gpr_interface_factory(*_mock_data())
    model.set_optimizer(optimizer)
    model.set_optimize()
    loss = model.loss()
    model.optimize()
    assert model.loss() < loss


def _3x_plus_gaussian_noise(x: tf.Tensor) -> tf.Tensor:
    return 3.0 * x + np.random.normal(scale=0.01, size=x.shape)


@random_seed
def test_variational_gaussian_process_predict() -> None:
    x_observed = tf.constant(np.arange(100).reshape((-1, 1)), dtype=gpflow.default_float())
    y_observed = _3x_plus_gaussian_noise(x_observed)
    model = VariationalGaussianProcess(_vgp(x_observed, y_observed))

    gpflow.optimizers.Scipy().minimize(
        model.loss,
        model.trainable_variables,
    )
    x_predict = tf.constant([[50.5]], gpflow.default_float())
    mean, variance = model.predict(x_predict)

    reference_model = _reference_gpr(x_observed, y_observed)
    gpflow.optimizers.Scipy().minimize(
        reference_model.training_loss_closure(),
        reference_model.trainable_variables,
    )
    reference_mean, reference_variance = reference_model.predict_f(x_predict)

    npt.assert_allclose(mean, reference_mean)
    npt.assert_allclose(variance, reference_variance, atol=1e-3)


class _QuadraticPredictor(GPflowPredictor):
    @property
    def model(self) -> GPModel:
        return _QuadraticGPModel()


class _QuadraticGPModel(GPModel):
    def __init__(self):
        super().__init__(
            gpflow.kernels.Polynomial(2),  # not actually used
            gpflow.likelihoods.Gaussian(),
            num_latent_gps=1,
        )

    def predict_f(
        self, Xnew: tf.Tensor, full_cov: bool = False, full_output_cov: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        assert not full_output_cov, "Test utility not implemented for full output covariance"
        mean = tf.reduce_sum(Xnew ** 2, axis=1, keepdims=True)
        *leading, x_samples, y_dims = mean.shape
        var_shape = [*leading, y_dims, x_samples, x_samples] if full_cov else mean.shape
        return mean, tf.ones(var_shape, dtype=mean.dtype)

    def maximum_log_likelihood_objective(self, *args, **kwargs) -> tf.Tensor:
        raise NotImplementedError


def test_gpflow_predictor_predict() -> None:
    model = _QuadraticPredictor()
    mean, variance = model.predict(tf.constant([[2.5]], gpflow.default_float()))
    assert mean.shape == [1, 1]
    assert variance.shape == [1, 1]
    npt.assert_allclose(mean, [[6.25]], rtol=0.01)
    npt.assert_allclose(variance, [[1.0]], rtol=0.01)


@random_seed
def test_gpflow_predictor_sample() -> None:
    model = _QuadraticPredictor()
    num_samples = 20_000
    samples = model.sample(tf.constant([[2.5]], gpflow.default_float()), num_samples)

    assert samples.shape == [num_samples, 1, 1]

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2)

    linear_error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, [[6.25]], rtol=linear_error)
    npt.assert_allclose(sample_variance, 1.0, rtol=2 * linear_error)


def test_gpflow_predictor_sample_no_samples() -> None:
    samples = _QuadraticPredictor().sample(tf.constant([[50.0]], gpflow.default_float()), 0)
    assert samples.shape == (0, 1, 1)


def test_sparse_variational_model_attribute() -> None:
    model = _svgp(_mock_data()[0])
    sv = SparseVariational(model, Dataset(*_mock_data()), tf.optimizers.Adam(), iterations=10)
    assert sv.model is model


@pytest.mark.parametrize(
    "new_data",
    [Dataset(tf.zeros([3, 5]), tf.zeros([3, 1])), Dataset(tf.zeros([3, 4]), tf.zeros([3, 2]))],
)
def test_sparse_variational_update_raises_for_invalid_shapes(new_data: Dataset) -> None:
    model = SparseVariational(
        _svgp(tf.zeros([1, 4])),
        Dataset(tf.zeros([3, 4]), tf.zeros([3, 1])),
        tf.optimizers.Adam(),
        iterations=10,
    )
    with pytest.raises(ValueError):
        model.update(new_data)


def test_sparse_variational_optimize_with_defaults() -> None:
    x_observed = tf.constant(np.arange(100).reshape((-1, 1)), dtype=gpflow.default_float())
    y_observed = _3x_plus_gaussian_noise(x_observed)
    model = SparseVariational(
        _svgp(x_observed[:10]),
        Dataset(tf.constant(x_observed), tf.constant(y_observed)),
        tf.optimizers.Adam(),
        iterations=20,
    )
    loss = model.model.training_loss((x_observed, y_observed))
    model.optimize()
    assert model.model.training_loss((x_observed, y_observed)) < loss


@pytest.mark.parametrize("apply_jit", [True, False])
@pytest.mark.parametrize(
    "batcher",
    [
        lambda ds: tf.data.Dataset.from_tensors((ds.query_points, ds.observations))
        .shuffle(100)
        .batch(10),
        lambda ds: [(ds.query_points, ds.observations)],
    ],
)
def test_sparse_variational_optimize(batcher: Batcher, apply_jit: bool) -> None:
    x_observed = tf.constant(np.arange(100).reshape((-1, 1)), dtype=gpflow.default_float())
    y_observed = _3x_plus_gaussian_noise(x_observed)

    model = SparseVariational(
        _svgp(x_observed[:10]),
        Dataset(tf.constant(x_observed), tf.constant(y_observed)),
        tf.optimizers.Adam(),
        iterations=20,
        batcher=batcher,
        apply_jit=apply_jit,
    )
    loss = model.model.training_loss((x_observed, y_observed))
    model.optimize()
    assert model.model.training_loss((x_observed, y_observed)) < loss

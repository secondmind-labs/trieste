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
from typing import Tuple, Callable, Union, Iterable

import gpflow
from gpflow.models import GPModel, GPR, SGPR, VGP, SVGP
import pytest
import tensorflow as tf
import numpy as np
import numpy.testing as npt

from dataclasses import astuple
from trieste.datasets import Dataset
from trieste.models.optimizer import Optimizer, create_optimizer
from trieste.models.model_interfaces import (
    ModelInterface,
    GaussianProcessRegression,
    GPflowPredictor,
    SparseVariational,
    VariationalGaussianProcess,
)
from trieste.type import ObserverEvaluations, TensorType, QueryPoints

from tests.util.misc import random_seed
from tests.util.model import StaticModelInterface


class _MinimalTrainable(ModelInterface):
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def optimize(self, dataset: Dataset):
        self.optimizer.optimize(None, None)

    def update(self, dataset: Dataset) -> None:
        raise NotImplementedError

    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        raise NotImplementedError

    def sample(self, query_points: QueryPoints, num_samples: int) -> ObserverEvaluations:
        raise NotImplementedError


def test_trainable_model_interface_set_optimize() -> None:
    class _OptimizerMock(Optimizer):
        def __init__(self):
            self.call_count = 0

        def optimize(self, model: None, dataset: None):
            self.call_count += 1

    optimizer = _OptimizerMock()
    model = _MinimalTrainable(optimizer)
    model.optimize(None)
    assert optimizer.call_count == 1


def _mock_data() -> Tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.constant([[1.2]], gpflow.default_float()),
        tf.constant([[3.4]], gpflow.default_float()),
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


@pytest.fixture(
    name="gpr_interface_factory",
    params=[
        (GaussianProcessRegression, _gpr),
        (GaussianProcessRegression, _sgpr),
        (VariationalGaussianProcess, _vgp),
    ],
)
def _gpr_interface_factory(request) -> Callable[[tf.Tensor, tf.Tensor], GaussianProcessRegression]:
    return lambda x, y, optimizer=None: request.param[0](
        request.param[1](x, y), optimizer=optimizer
    )


def _reference_gpr(x: tf.Tensor, y: tf.Tensor) -> gpflow.models.GPR:
    return _gpr(x, y)


def _3x_plus_10(x: tf.Tensor) -> tf.Tensor:
    return 3.0 * x + 10


def test_gaussian_process_regression_loss(gpr_interface_factory) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model = gpr_interface_factory(x, _3x_plus_10(x))
    reference_model = _reference_gpr(x, _3x_plus_10(x))
    internal_model = model.model
    npt.assert_allclose(
        internal_model.training_loss(), -reference_model.log_marginal_likelihood(), rtol=1e-6
    )


def test_gaussian_process_regression_update(gpr_interface_factory) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model = gpr_interface_factory(x, _3x_plus_10(x))

    x_new = tf.concat([x, tf.constant([[10.0], [11.0]], dtype=gpflow.default_float())], 0)
    new_data = Dataset(x_new, _3x_plus_10(x_new))
    model.update(new_data)
    model.optimize(new_data)
    reference_model = _reference_gpr(x_new, _3x_plus_10(x_new))
    gpflow.optimizers.Scipy().minimize(
        reference_model.training_loss_closure(), reference_model.trainable_variables
    )
    internal_model = model.model
    npt.assert_allclose(internal_model.training_loss(), reference_model.training_loss(), rtol=1e-6)


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


@random_seed(1357)
def test_gaussian_process_regression_default_optimize(gpr_interface_factory) -> None:
    data = _mock_data()
    model = gpr_interface_factory(*data)
    internal_model = model.model
    loss = internal_model.training_loss()
    model.optimize(Dataset(*data))
    assert internal_model.training_loss() < loss


@random_seed(1357)
@pytest.mark.parametrize("optimizer", [gpflow.optimizers.Scipy(), tf.optimizers.Adam(), None])
def test_gaussian_process_regression_optimize(
    optimizer: Union[gpflow.optimizers.Scipy, tf.optimizers.Optimizer, None], gpr_interface_factory
) -> None:
    data = _mock_data()
    optimizer_wrapper = create_optimizer(optimizer, {})
    model = gpr_interface_factory(*data, optimizer=optimizer_wrapper)
    internal_model = model.model
    loss = internal_model.training_loss()
    model.optimize(Dataset(*data))
    assert internal_model.training_loss() < loss


def _3x_plus_gaussian_noise(x: tf.Tensor) -> tf.Tensor:
    return 3.0 * x + np.random.normal(scale=0.01, size=x.shape)


@random_seed(1357)
def test_variational_gaussian_process_predict() -> None:
    x_observed = tf.constant(np.arange(100).reshape((-1, 1)), dtype=gpflow.default_float())
    y_observed = _3x_plus_gaussian_noise(x_observed)
    model = VariationalGaussianProcess(_vgp(x_observed, y_observed))
    internal_model = model.model

    gpflow.optimizers.Scipy().minimize(
        internal_model.training_loss_closure(),
        internal_model.trainable_variables,
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


class _QuadraticStaticPredictor(GPflowPredictor, StaticModelInterface):
    @property
    def model(self) -> GPModel:
        return _QuadraticStaticGPModel()


class _QuadraticStaticGPModel(GPModel):
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
    model = _QuadraticStaticPredictor()
    mean, variance = model.predict(tf.constant([[2.5]], gpflow.default_float()))
    assert mean.shape == [1, 1]
    assert variance.shape == [1, 1]
    npt.assert_allclose(mean, [[6.25]], rtol=0.01)
    npt.assert_allclose(variance, [[1.0]], rtol=0.01)


@random_seed(1357)
def test_gpflow_predictor_sample() -> None:
    model = _QuadraticStaticPredictor()
    samples = model.sample(tf.constant([[2.5]], gpflow.default_float()), 10_000)

    assert samples.shape == [10_000, 1, 1]

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2)

    npt.assert_allclose(sample_mean, [[6.25]], rtol=0.01)
    npt.assert_allclose(sample_variance, 1.0, rtol=0.01)


def test_gpflow_predictor_sample_no_samples() -> None:
    samples = _QuadraticStaticPredictor().sample(tf.constant([[50.0]], gpflow.default_float()), 0)
    assert samples.shape == (0, 1, 1)


def test_sparse_variational_model_attribute() -> None:
    model = _svgp(_mock_data()[0])
    sv = SparseVariational(model, Dataset(*_mock_data()))
    assert sv.model is model


@pytest.mark.parametrize(
    "new_data",
    [Dataset(tf.zeros([3, 5]), tf.zeros([3, 1])), Dataset(tf.zeros([3, 4]), tf.zeros([3, 2]))],
)
def test_sparse_variational_update_raises_for_invalid_shapes(new_data: Dataset) -> None:
    model = SparseVariational(
        _svgp(tf.zeros([1, 4])),
        Dataset(tf.zeros([3, 4]), tf.zeros([3, 1])),
    )
    with pytest.raises(ValueError):
        model.update(new_data)


def test_sparse_variational_optimize_with_defaults() -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = _3x_plus_gaussian_noise(x_observed)
    data = x_observed, y_observed
    dataset = Dataset(*data)
    optimizer = create_optimizer(tf.optimizers.Adam(), dict(max_iter=20))
    model = SparseVariational(_svgp(x_observed[:10]), dataset, optimizer=optimizer)
    loss = model.model.training_loss(data)
    model.optimize(dataset)
    assert model.model.training_loss(data) < loss


def _batcher_1(dataset: Dataset, batch_size: int) -> Iterable:
    ds = tf.data.Dataset.from_tensor_slices(astuple(dataset))
    ds = ds.shuffle(100)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    return iter(ds)


def _batcher_2(dataset: Dataset, batch_size: int):
    return astuple(dataset)


@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("batcher", [_batcher_1, _batcher_2])
def test_sparse_variational_optimize(batcher, compile: bool) -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = _3x_plus_gaussian_noise(x_observed)
    data = x_observed, y_observed
    dataset = Dataset(*data)

    optimizer = create_optimizer(
        tf.optimizers.Adam(),
        dict(max_iter=20, batch_size=10, dataset_builder=batcher, compile=compile),
    )
    model = SparseVariational(_svgp(x_observed[:10]), dataset, optimizer=optimizer)
    loss = model.model.training_loss(data)
    model.optimize(dataset)
    assert model.model.training_loss(data) < loss

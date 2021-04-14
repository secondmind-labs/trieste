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
from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Sequence

import gpflow
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import GPR, SGPR, SVGP, VGP, GPModel
from packaging.version import parse

from tests.util.misc import assert_datasets_allclose, quadratic, random_seed
from tests.util.model import GaussianProcess, PseudoTrainableProbModel
from trieste.data import Dataset
from trieste.models.model_interfaces import (
    GaussianProcessRegression,
    GPflowPredictor,
    ModelStack,
    SparseVariational,
    TrainableProbabilisticModel,
    VariationalGaussianProcess,
    module_deepcopy,
)
from trieste.models.optimizer import Optimizer, create_optimizer
from trieste.type import TensorType


def _model_stack() -> tuple[ModelStack, tuple[TrainableProbabilisticModel, ...]]:
    class Model(GaussianProcess, PseudoTrainableProbModel):
        def __init__(self, mean_shifts: list[float], kernel_amplitudes: list[float]):
            super().__init__(
                [(lambda y: lambda x: quadratic(x) + y)(shift) for shift in mean_shifts],
                [tfp.math.psd_kernels.ExponentiatedQuadratic(x) for x in kernel_amplitudes],
            )

    model01 = Model([0.0, 0.5], [1.0, 0.3])
    model2 = Model([2.0], [2.0])
    model3 = Model([-1.0], [0.1])

    return ModelStack((model01, 2), (model2, 1), (model3, 1)), (model01, model2, model3)


def test_model_stack_predict() -> None:
    stack, (model01, model2, model3) = _model_stack()
    query_points = tf.random.uniform([5, 7, 3])
    mean, var = stack.predict(query_points)

    assert mean.shape == [5, 7, 4]
    assert var.shape == [5, 7, 4]

    mean01, var01 = model01.predict(query_points)
    mean2, var2 = model2.predict(query_points)
    mean3, var3 = model3.predict(query_points)

    npt.assert_allclose(mean[..., :2], mean01)
    npt.assert_allclose(mean[..., 2:3], mean2)
    npt.assert_allclose(mean[..., 3:], mean3)
    npt.assert_allclose(var[..., :2], var01)
    npt.assert_allclose(var[..., 2:3], var2)
    npt.assert_allclose(var[..., 3:], var3)


def test_model_stack_predict_joint() -> None:
    stack, (model01, model2, model3) = _model_stack()
    query_points = tf.random.uniform([5, 7, 3])
    mean, cov = stack.predict_joint(query_points)

    assert mean.shape == [5, 7, 4]
    assert cov.shape == [5, 4, 7, 7]

    mean01, cov01 = model01.predict_joint(query_points)
    mean2, cov2 = model2.predict_joint(query_points)
    mean3, cov3 = model3.predict_joint(query_points)

    npt.assert_allclose(mean[..., :2], mean01)
    npt.assert_allclose(mean[..., 2:3], mean2)
    npt.assert_allclose(mean[..., 3:], mean3)
    npt.assert_allclose(cov[..., :2, :, :], cov01)
    npt.assert_allclose(cov[..., 2:3, :, :], cov2)
    npt.assert_allclose(cov[..., 3:, :, :], cov3)


@random_seed
def test_model_stack_sample() -> None:
    query_points = tf.random.uniform([5, 7, 3], maxval=10.0)
    stack, (model01, model2, model3) = _model_stack()
    samples = stack.sample(query_points, 10_000)

    assert samples.shape == [5, 7, 10_000, 4]

    mean = tf.reduce_mean(samples, axis=-2)
    var = tf.math.reduce_variance(samples, axis=-2)

    mean01, var01 = model01.predict(query_points)
    mean2, var2 = model2.predict(query_points)
    mean3, var3 = model3.predict(query_points)

    npt.assert_allclose(mean[..., :2], mean01, rtol=0.01)
    npt.assert_allclose(mean[..., 2:3], mean2, rtol=0.01)
    npt.assert_allclose(mean[..., 3:], mean3, rtol=0.01)
    npt.assert_allclose(var[..., :2], var01, rtol=0.04)
    npt.assert_allclose(var[..., 2:3], var2, rtol=0.04)
    npt.assert_allclose(var[..., 3:], var3, rtol=0.04)


def test_model_stack_training() -> None:
    class Model(GaussianProcess, TrainableProbabilisticModel):
        def __init__(
            self,
            mean_functions: Sequence[Callable[[TensorType], TensorType]],
            kernels: Sequence[tfp.math.psd_kernels.PositiveSemidefiniteKernel],
            output_dims: slice,
        ):
            super().__init__(mean_functions, kernels)
            self._output_dims = output_dims

        def _assert_data(self, dataset: Dataset) -> None:
            qp, obs = dataset.astuple()
            expected_obs = data.observations[..., self._output_dims]
            assert_datasets_allclose(dataset, Dataset(qp, expected_obs))

        optimize = _assert_data
        update = _assert_data

    rbf = tfp.math.psd_kernels.ExponentiatedQuadratic()
    model01 = Model([quadratic, quadratic], [rbf, rbf], slice(0, 2))
    model2 = Model([quadratic], [rbf], slice(2, 3))
    model3 = Model([quadratic], [rbf], slice(3, 4))

    stack = ModelStack((model01, 2), (model2, 1), (model3, 1))
    data = Dataset(tf.random.uniform([5, 7, 3]), tf.random.uniform([5, 7, 4]))
    stack.update(data)
    stack.optimize(data)


class _ModuleWithBijector(tf.Module):
    def __init__(self):
        super().__init__()
        self.const = tf.constant(1.2)
        self.var = tf.Variable(3.4)
        self.bijector = tfp.bijectors.Exp()


class _NestedModuleWithBijector(tf.Module):
    def __init__(self):
        super().__init__()
        self.inst = _ModuleWithBijector()
        self.bijector = tfp.bijectors.Sigmoid(0.0, 1.0)


def test_module_deepcopy_is_copyable() -> None:
    class Deepcopyable(_NestedModuleWithBijector):
        __deepcopy__ = module_deepcopy

    module = Deepcopyable()

    _ = module.bijector.forward(0.5)
    _ = module.bijector.inverse(0.5)
    _ = module.inst.bijector.forward(0.5)
    _ = module.inst.bijector.inverse(0.5)

    module_copy = copy.deepcopy(module)

    module.inst.var.assign(5.6)

    assert module_copy.inst.const == 1.2
    assert module_copy.inst.var == 3.4
    assert module_copy.inst.bijector.forward(7.8) == tf.exp(7.8)


if parse(tfp.__version__) < parse("0.12"):

    def test_cant_copy_tf_module_with_used_bijector() -> None:
        module = _NestedModuleWithBijector()

        _ = module.bijector.forward(0.5)

        with pytest.raises(TypeError, match="HashableWeakRef"):
            copy.deepcopy(module)


def _mock_data() -> tuple[tf.Tensor, tf.Tensor]:
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
def _gpr_interface_factory(
    request,
) -> Callable[[TensorType, TensorType, Optimizer | None], GaussianProcessRegression]:
    def model_interface_factory(
        x: TensorType, y: TensorType, optimizer: Optimizer | None = None
    ) -> GaussianProcessRegression:
        model_interface: type[GaussianProcessRegression] = request.param[0]
        base_model: GaussianProcessRegression = request.param[1](x, y)
        return model_interface(base_model, optimizer=optimizer)  # type: ignore

    return model_interface_factory


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
    data = _mock_data()
    model = gpr_interface_factory(*data)
    internal_model = model.model
    loss = internal_model.training_loss()
    model.optimize(Dataset(*data))
    assert internal_model.training_loss() < loss


@random_seed
@pytest.mark.parametrize("optimizer", [gpflow.optimizers.Scipy(), tf.optimizers.Adam(), None])
def test_gaussian_process_regression_optimize(
    optimizer: gpflow.optimizers.Scipy | tf.optimizers.Optimizer | None, gpr_interface_factory
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


@random_seed
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


class _QuadraticPredictor(GPflowPredictor):
    @property
    def model(self) -> GPModel:
        return _QuadraticGPModel()

    def update(self, dataset: Dataset) -> None:
        pass


class _QuadraticGPModel(GPModel):
    def __init__(self):
        super().__init__(
            gpflow.kernels.Polynomial(2),  # not actually used
            gpflow.likelihoods.Gaussian(),
            num_latent_gps=1,
        )

    def predict_f(
        self, Xnew: tf.Tensor, full_cov: bool = False, full_output_cov: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
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
    sv = SparseVariational(model, Dataset(*_mock_data()))
    assert sv.model is model


def test_sparse_variational_update_updates_num_data() -> None:
    model = SparseVariational(
        _svgp(tf.zeros([1, 4])),
        Dataset(tf.zeros([3, 4]), tf.zeros([3, 1])),
    )
    model.update(Dataset(tf.zeros([5, 4]), tf.zeros([5, 1])))
    assert model.model.num_data == 5


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
    ds = tf.data.Dataset.from_tensor_slices(dataset.astuple())
    ds = ds.shuffle(100)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    return iter(ds)


def _batcher_2(dataset: Dataset, batch_size: int):
    return dataset.astuple()


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

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
from time import time
from typing import Any, cast

import gpflow
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.inducing_variables import (
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.models import SGPR, SVGP, VGP

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, random_seed
from tests.util.models.gpflow.models import (
    ModelFactoryType,
    gpr_model,
    mock_data,
    sgpr_model,
    svgp_model,
    two_output_svgp_model,
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
    SparseGaussianProcessRegression,
    SparseVariational,
    VariationalGaussianProcess,
)
from trieste.models.gpflow.inducing_point_selectors import (
    InducingPointSelector,
    KMeansInducingPointSelector,
    RandomSubSampleInducingPointSelector,
    UniformInducingPointSelector,
)
from trieste.models.gpflow.sampler import (
    DecoupledTrajectorySampler,
    RandomFourierFeatureTrajectorySampler,
)
from trieste.models.optimizer import BatchOptimizer, DatasetTransformer, Optimizer
from trieste.space import Box
from trieste.types import TensorType


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
            reference_model.training_loss_closure(compile=True),
            reference_model.trainable_variables,
        )

        npt.assert_allclose(
            internal_model.training_loss(), reference_model.training_loss(), rtol=1e-6
        )


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


@random_seed
def test_gpflow_models_pairwise_covariance(gpflow_interface_factory: ModelFactoryType) -> None:
    x = tf.constant(np.arange(1, 5).reshape(-1, 1), dtype=gpflow.default_float())  # shape: [4, 1]
    y = fnc_3x_plus_10(x)
    model, _ = gpflow_interface_factory(x, y)

    if isinstance(model.model, (VGP, SVGP)):  # for speed just update q_sqrt rather than optimize
        num_inducing_points = tf.shape(model.model.q_sqrt)[1]
        sampled_q_sqrt = tfp.distributions.WishartTriL(5, tf.eye(num_inducing_points)).sample(1)
        model.model.q_sqrt.assign(sampled_q_sqrt)
        model.update_posterior_cache()

    query_points_1 = tf.concat([0.5 * x, 0.5 * x], 0)  # shape: [8, 1]
    query_points_2 = tf.concat([2 * x, 2 * x, 2 * x], 0)  # shape: [12, 1]

    all_query_points = tf.concat([query_points_1, query_points_2], 0)
    _, predictive_covariance = model.predict_joint(all_query_points)
    expected_covariance = predictive_covariance[0, :8, 8:]

    actual_covariance = model.covariance_between_points(  # type: ignore
        query_points_1, query_points_2
    )

    np.testing.assert_allclose(expected_covariance, actual_covariance[0], atol=1e-4)


@random_seed
def test_gpflow_models_raise_for_pairwise_covariance_for_invalid_query_points(
    gpflow_interface_factory: ModelFactoryType,
) -> None:
    data = mock_data()
    model, _ = gpflow_interface_factory(*data)

    with pytest.raises(ValueError):
        model.covariance_between_points(data[0], tf.expand_dims(data[0], axis=0))  # type: ignore


@random_seed
@pytest.mark.parametrize("after_model_optimize", [True, False])
@pytest.mark.parametrize("after_model_update", [True, False])
def test_gpflow_models_cached_predictions_correct(
    after_model_optimize: bool,
    after_model_update: bool,
    gpflow_interface_factory: ModelFactoryType,
) -> None:
    x = np.linspace(0, 5, 10).reshape((-1, 1))
    y = fnc_2sin_x_over_3(x)
    data = x, y
    dataset = Dataset(*data)
    model, _ = gpflow_interface_factory(x, y)

    if after_model_optimize:
        model._optimizer = BatchOptimizer(tf.optimizers.SGD(), max_iter=1)
        model.optimize(dataset)

    if after_model_update:
        new_x = np.linspace(0, 5, 3).reshape((-1, 1))
        new_y = fnc_2sin_x_over_3(new_x)
        new_dataset = Dataset(new_x, new_y)
        cast(TrainableProbabilisticModel, model).update(new_dataset)

    x_predict = np.linspace(0, 5, 2).reshape((-1, 1))

    # get cached predictions
    cached_fmean, cached_fvar = model.predict(x_predict)
    cached_joint_mean, cached_joint_var = model.predict_joint(x_predict)
    cached_ymean, cached_yvar = model.predict_y(x_predict)

    # get reference (slow) predictions from underlying model
    reference_fmean, reference_fvar = model.model.predict_f(x_predict)
    reference_joint_mean, reference_joint_var = model.model.predict_f(x_predict, full_cov=True)
    reference_ymean, reference_yvar = model.model.predict_y(x_predict)

    npt.assert_allclose(cached_fmean, reference_fmean)
    npt.assert_allclose(cached_ymean, reference_ymean)
    npt.assert_allclose(cached_joint_mean, reference_joint_mean)
    npt.assert_allclose(cached_fvar, reference_fvar, atol=1e-5)
    npt.assert_allclose(cached_yvar, reference_yvar, atol=1e-5)
    npt.assert_allclose(cached_joint_var, reference_joint_var, atol=1e-5)
    npt.assert_allclose(cached_yvar - model.get_observation_noise(), cached_fvar, atol=5e-5)


@random_seed
def test_gpflow_models_cached_predictions_faster(
    gpflow_interface_factory: ModelFactoryType,
) -> None:
    x = np.linspace(0, 10, 10).reshape((-1, 1))
    y = fnc_2sin_x_over_3(x)
    model, _ = gpflow_interface_factory(x, y)
    n_calls = 100

    x_predict = np.linspace(0, 5, 2).reshape((-1, 1))
    t_0 = time()
    [model.predict(x_predict) for _ in range(n_calls)]
    time_with_cache = time() - t_0
    t_0 = time()
    [model.model.predict_f(x_predict) for _ in range(n_calls)]
    time_without_cache = time() - t_0

    npt.assert_array_less(time_with_cache, time_without_cache)


def test_gaussian_process_regression_raises_for_invalid_init() -> None:
    x_np = np.arange(5, dtype=np.float64).reshape(-1, 1)
    x = tf.convert_to_tensor(x_np, x_np.dtype)
    y = fnc_3x_plus_10(x)

    with pytest.raises(ValueError):
        GaussianProcessRegression(gpr_model(x, y), num_kernel_samples=-1)

    with pytest.raises(ValueError):
        GaussianProcessRegression(gpr_model(x, y), num_rff_features=-1)

    with pytest.raises(ValueError):
        GaussianProcessRegression(gpr_model(x, y), num_rff_features=0)

    with pytest.raises(ValueError):
        optimizer1 = BatchOptimizer(gpflow.optimizers.Scipy())
        GaussianProcessRegression(gpr_model(x, y), optimizer=optimizer1)

    with pytest.raises(ValueError):
        optimizer2 = Optimizer(tf.optimizers.Adam())
        GaussianProcessRegression(gpr_model(x, y), optimizer=optimizer2)


def test_gaussian_process_regression_optimize_with_defaults(compile: bool) -> None:
    data = mock_data()

    model = GaussianProcessRegression(gpr_model(*data))

    loss = model.model.training_loss()
    model.optimize(Dataset(*data))

    assert model.model.training_loss() < loss


def test_gaussian_process_regression_optimize(compile: bool) -> None:
    data = mock_data()

    optimizer = Optimizer(gpflow.optimizers.Scipy(), compile=compile)
    model = GaussianProcessRegression(gpr_model(*data), optimizer)

    loss = model.model.training_loss()
    model.optimize(Dataset(*data))

    assert model.model.training_loss() < loss


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


def test_gaussian_process_regression_trajectory_sampler_raises_multi_latent_gp() -> None:
    x = tf.constant(np.arange(1, 5).reshape(-1, 1), dtype=gpflow.default_float())  # shape: [4, 1]
    y = fnc_3x_plus_10(x)
    model = GaussianProcessRegression(gpr_model(x, tf.repeat(y, 2, axis=1)))

    with pytest.raises(NotImplementedError):
        model.trajectory_sampler()


@random_seed
@pytest.mark.parametrize("use_decoupled_sampler", [True, False])
def test_gaussian_process_regression_trajectory_sampler_has_correct_samples(
    use_decoupled_sampler: bool,
) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model = GaussianProcessRegression(
        gpr_model(x, _3x_plus_gaussian_noise(x)), use_decoupled_sampler=use_decoupled_sampler
    )
    model.model.likelihood.variance.assign(1e-3)
    model.update_posterior_cache()

    num_samples = 100
    trajectory_sampler = model.trajectory_sampler()

    if use_decoupled_sampler:
        assert isinstance(trajectory_sampler, DecoupledTrajectorySampler)
    else:
        assert isinstance(trajectory_sampler, RandomFourierFeatureTrajectorySampler)

    trajectory = trajectory_sampler.get_trajectory()
    x_predict = tf.constant([[1.0], [2.0], [3.0], [1.5], [2.5], [3.5]], gpflow.default_float())
    x_predict_parallel = tf.expand_dims(x_predict, -2)  # [N, 1, D]
    x_predict_parallel = tf.tile(x_predict_parallel, [1, num_samples, 1])  # [N, B, D]
    samples = trajectory(x_predict_parallel)  # [N, B]
    sample_mean = tf.reduce_mean(samples, axis=1, keepdims=True)
    sample_variance = tf.math.reduce_variance(samples, axis=1, keepdims=True)

    true_mean, true_variance = model.predict(x_predict)

    # test predictions approx correct away from data
    npt.assert_allclose(sample_mean[3:] + 1.0, true_mean[3:] + 1.0, rtol=0.1)
    npt.assert_allclose(sample_variance[3:], true_variance[3:], rtol=0.5)

    # test predictions correct at data
    npt.assert_allclose(sample_mean[:3] + 1.0, true_mean[:3] + 1.0, rtol=0.001)
    npt.assert_allclose(sample_variance[:3], true_variance[:3], rtol=0.15)


def test_gpr_config_builds_and_default_optimizer_is_correct() -> None:
    data = mock_data()

    model_config = {"model": gpr_model(*data)}
    model = create_model(model_config)

    assert isinstance(model, GaussianProcessRegression)
    assert isinstance(model.optimizer, Optimizer)
    assert isinstance(model.optimizer.optimizer, gpflow.optimizers.Scipy)


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


def test_sparse_gaussian_process_regression_raises_for_invalid_init() -> None:
    x_np = np.arange(5, dtype=np.float64).reshape(-1, 1)
    x = tf.convert_to_tensor(x_np, x_np.dtype)
    y = fnc_3x_plus_10(x)

    with pytest.raises(ValueError):
        SparseGaussianProcessRegression(sgpr_model(x, y), num_rff_features=-1)

    with pytest.raises(ValueError):
        SparseGaussianProcessRegression(sgpr_model(x, y), num_rff_features=0)

    with pytest.raises(ValueError):
        optimizer1 = BatchOptimizer(gpflow.optimizers.Scipy())
        SparseGaussianProcessRegression(sgpr_model(x, y), optimizer=optimizer1)

    with pytest.raises(ValueError):
        optimizer2 = Optimizer(tf.optimizers.Adam())
        SparseGaussianProcessRegression(sgpr_model(x, y), optimizer=optimizer2)


def test_sparse_gaussian_process_regression_default_optimizer_is_correct() -> None:
    data = mock_data()

    model = SparseGaussianProcessRegression(sgpr_model(*data))

    assert isinstance(model.optimizer, Optimizer)
    assert isinstance(model.optimizer.optimizer, gpflow.optimizers.Scipy)


def test_sparse_gaussian_process_regression_model_attribute() -> None:
    sgpr = sgpr_model(*mock_data())
    model = SparseGaussianProcessRegression(sgpr)

    assert model.model is sgpr
    assert isinstance(model.model, SGPR)
    assert model.inducing_point_selector is None


def test_sgpr_config_builds_and_default_optimizer_is_correct() -> None:
    data = mock_data()

    model_config = {"model": sgpr_model(*data)}
    model = create_model(model_config)

    assert isinstance(model, SparseGaussianProcessRegression)
    assert isinstance(model.optimizer, Optimizer)
    assert isinstance(model.optimizer.optimizer, gpflow.optimizers.Scipy)


def test_sparse_gaussian_process_regression_correctly_returns_internal_data() -> None:
    data = mock_data()
    model = SparseGaussianProcessRegression(sgpr_model(*data))
    returned_data = model.get_internal_data()

    npt.assert_array_equal(returned_data.query_points, data[0])
    npt.assert_array_equal(returned_data.observations, data[1])


def test_sparse_gaussian_process_regression_update_updates_num_data() -> None:
    x_np = np.arange(5, dtype=np.float64).reshape(-1, 1)
    x = tf.convert_to_tensor(x_np, x_np.dtype)
    y = fnc_3x_plus_10(x)
    m = SparseGaussianProcessRegression(sgpr_model(x, y))
    num_data = m.model.num_data.numpy()

    x_new = tf.concat([x, [[10.0], [11.0]]], 0)
    y_new = fnc_3x_plus_10(x_new)
    m.update(Dataset(x_new, y_new))
    new_num_data = m.model.num_data.numpy()

    assert new_num_data - num_data == 2


def test_sparse_gaussian_process_regression_optimize_with_defaults() -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = _3x_plus_gaussian_noise(x_observed)
    data = x_observed, y_observed
    dataset = Dataset(*data)

    model = SparseGaussianProcessRegression(sgpr_model(x_observed, y_observed))
    loss = model.model.training_loss()
    model.optimize(dataset)

    assert model.model.training_loss() < loss


def test_sparse_gaussian_process_regression_optimize(compile: bool) -> None:
    x_observed = np.linspace(0, 100, 100).reshape((-1, 1))
    y_observed = _3x_plus_gaussian_noise(x_observed)
    data = x_observed, y_observed
    dataset = Dataset(*data)

    optimizer = Optimizer(gpflow.optimizers.Scipy(), compile=compile)
    model = SparseGaussianProcessRegression(sgpr_model(x_observed, y_observed), optimizer=optimizer)
    loss = model.model.training_loss()
    model.optimize(dataset)

    assert model.model.training_loss() < loss


def test_sparse_gaussian_process_regression_trajectory_sampler_raises_multi_latent_gp() -> None:
    data = mock_data()
    model = SparseGaussianProcessRegression(sgpr_model(*data, num_latent_gps=2))

    with pytest.raises(NotImplementedError):
        model.trajectory_sampler()


@random_seed
def test_sparse_gaussian_process_regression_trajectory_sampler_has_correct_samples() -> None:
    x = tf.constant(np.arange(10).reshape((-1, 1)), dtype=gpflow.default_float())
    sgpr = SGPR((x, _3x_plus_gaussian_noise(x)), gpflow.kernels.Matern32(), x, noise_variance=1e-5)
    model = SparseGaussianProcessRegression(sgpr)

    num_samples = 100
    trajectory_sampler = model.trajectory_sampler()

    assert isinstance(trajectory_sampler, DecoupledTrajectorySampler)

    trajectory = trajectory_sampler.get_trajectory()
    x_predict = tf.constant([[1.0], [2.0], [3.0], [1.5], [2.5], [3.5]], gpflow.default_float())
    x_predict_parallel = tf.expand_dims(x_predict, -2)  # [N, 1, D]
    x_predict_parallel = tf.tile(x_predict_parallel, [1, num_samples, 1])  # [N, B, D]
    samples = trajectory(x_predict_parallel)  # [N, B]
    sample_mean = tf.reduce_mean(samples, axis=1, keepdims=True)
    sample_variance = tf.math.reduce_variance(samples, axis=1, keepdims=True)

    true_mean, true_variance = model.predict(x_predict)

    # test predictions approx correct away from data
    npt.assert_allclose(sample_mean[3:] + 1.0, true_mean[3:] + 1.0, rtol=0.2)
    npt.assert_allclose(sample_variance[3:], true_variance[3:], rtol=0.5)

    # test predictions almost correct at data
    npt.assert_allclose(sample_mean[:3] + 1.0, true_mean[:3] + 1.0, rtol=0.01)
    npt.assert_allclose(sample_variance[:3], true_variance[:3], rtol=0.3)


def test_sparse_gaussian_process_regression_get_inducing_raises_multi_latent_gp() -> None:
    data = mock_data()
    model = SparseGaussianProcessRegression(sgpr_model(*data, num_latent_gps=2))

    with pytest.raises(NotImplementedError):
        model.get_inducing_variables()


def test_sparse_gaussian_process_regression_correctly_returns_inducing_points() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    data = Dataset(x, fnc_3x_plus_10(x))
    model = SparseGaussianProcessRegression(sgpr_model(data.query_points, data.observations))
    model.update(data)

    inducing_points, q_mu, q_sqrt, w = model.get_inducing_variables()

    ref_q_mu, ref_q_var = model.model.compute_qu()
    ref_q_sqrt = tf.linalg.cholesky(ref_q_var)
    ref_q_sqrt = tf.expand_dims(ref_q_sqrt, 0)

    npt.assert_allclose(inducing_points, model.model.inducing_variable.Z, atol=1e-5)
    npt.assert_allclose(q_mu, ref_q_mu, atol=1e-5)
    npt.assert_allclose(q_sqrt, ref_q_sqrt, atol=1e-5)
    assert not w


@pytest.mark.parametrize(
    "selector",
    [
        UniformInducingPointSelector(Box([0.0], [1.0])),
        RandomSubSampleInducingPointSelector(Box([0.0], [1.0])),
        KMeansInducingPointSelector(Box([0.0], [1.0])),
    ],
)
def test_sparse_gaussian_process_regression_assigns_correct_inducing_point_selector(
    selector: InducingPointSelector[SparseGaussianProcessRegression],
) -> None:
    model = sgpr_model(*mock_data())
    sv = SparseGaussianProcessRegression(model, inducing_point_selector=selector)
    assert isinstance(sv.inducing_point_selector, type(selector))


@pytest.mark.parametrize("recalc_every_model_update", [True, False])
def test_sparse_gaussian_process_regression_chooses_new_inducing_points_correct_number_of_times(
    recalc_every_model_update: bool,
) -> None:
    model = sgpr_model(*mock_data())
    selector = UniformInducingPointSelector(
        Box([0.0], [1.0]), recalc_every_model_update=recalc_every_model_update
    )
    sv = SparseGaussianProcessRegression(model, inducing_point_selector=selector)
    old_inducing_points = sv.model.inducing_variable.Z.numpy()
    sv.update(Dataset(*mock_data()))
    first_inducing_points = sv.model.inducing_variable.Z.numpy()
    npt.assert_raises(
        AssertionError, npt.assert_array_equal, old_inducing_points, first_inducing_points
    )
    sv.update(Dataset(*mock_data()))
    second_inducing_points = sv.model.inducing_variable.Z.numpy()
    if recalc_every_model_update:
        npt.assert_raises(
            AssertionError, npt.assert_array_equal, old_inducing_points, second_inducing_points
        )
        npt.assert_raises(
            AssertionError, npt.assert_array_equal, first_inducing_points, second_inducing_points
        )
    else:
        npt.assert_raises(
            AssertionError, npt.assert_array_equal, old_inducing_points, second_inducing_points
        )
        npt.assert_array_equal(first_inducing_points, second_inducing_points)


@random_seed
def test_sparse_gaussian_process_regression_update_inducing_points_raises_changed_shape() -> None:
    model = SparseGaussianProcessRegression(
        sgpr_model(
            tf.zeros([5, 2], gpflow.default_float()), tf.zeros([5, 1], gpflow.default_float())
        ),
    )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):  # current inducing point has 2 elements
        model._update_inducing_variables(tf.zeros([3, 2], gpflow.default_float()))


@pytest.mark.parametrize(
    "new_data",
    [Dataset(tf.zeros([3, 5]), tf.zeros([3, 1])), Dataset(tf.zeros([3, 4]), tf.zeros([3, 2]))],
)
def test_sparse_gaussian_process_regression_update_raises_for_invalid_shapes(
    new_data: Dataset,
) -> None:
    model = SparseGaussianProcessRegression(
        sgpr_model(
            tf.zeros([1, 4], gpflow.default_float()), tf.zeros([1, 1], gpflow.default_float())
        ),
    )
    with pytest.raises(ValueError):
        model.update(new_data)


def test_variational_gaussian_process_raises_for_invalid_init() -> None:
    x_np = np.arange(5, dtype=np.float64).reshape(-1, 1)
    x = tf.convert_to_tensor(x_np, x_np.dtype)
    y = fnc_3x_plus_10(x)

    with pytest.raises(ValueError):
        VariationalGaussianProcess(vgp_model(x, y), natgrad_gamma=1)

    with pytest.raises(ValueError):
        VariationalGaussianProcess(vgp_model(x, y), num_rff_features=-1)

    with pytest.raises(ValueError):
        VariationalGaussianProcess(vgp_model(x, y), num_rff_features=0)

    with pytest.raises(ValueError):
        optimizer = Optimizer(gpflow.optimizers.Scipy())
        VariationalGaussianProcess(vgp_model(x, y), optimizer=optimizer, use_natgrads=True)

    with pytest.raises(ValueError):
        optimizer = BatchOptimizer(gpflow.optimizers.Scipy())
        VariationalGaussianProcess(vgp_model(x, y), optimizer=optimizer, use_natgrads=True)

    with pytest.raises(ValueError):
        optimizer = Optimizer(tf.optimizers.Adam())
        VariationalGaussianProcess(vgp_model(x, y), optimizer=optimizer, use_natgrads=False)


def test_variational_gaussian_process_update_updates_num_data() -> None:
    x_np = np.arange(5, dtype=np.float64).reshape(-1, 1)
    x = tf.convert_to_tensor(x_np, x_np.dtype)
    y = fnc_3x_plus_10(x)
    m = VariationalGaussianProcess(vgp_model(x, y))
    num_data = m.model.num_data.numpy()

    x_new = tf.concat([x, [[10.0], [11.0]]], 0)
    y_new = fnc_3x_plus_10(x_new)
    m.update(Dataset(x_new, y_new))
    new_num_data = m.model.num_data.numpy()
    assert new_num_data - num_data == 2


def test_variational_gaussian_process_correctly_returns_inducing_points() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    data = Dataset(x, fnc_3x_plus_10(x))
    model = VariationalGaussianProcess(vgp_model(data.query_points, data.observations))
    model.update(data)

    inducing_points, q_mu, q_sqrt, whiten = model.get_inducing_variables()

    npt.assert_allclose(inducing_points, x, atol=1e-5)
    npt.assert_allclose(q_mu, model.model.q_mu, atol=1e-5)
    npt.assert_allclose(q_sqrt, model.model.q_sqrt, atol=1e-5)
    assert whiten


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


def test_variational_gaussian_process_trajectory_sampler_raises_multi_latent_gp() -> None:
    data = mock_data()
    model = VariationalGaussianProcess(vgp_model(*data, num_latent_gps=2))

    with pytest.raises(NotImplementedError):
        model.trajectory_sampler()


@random_seed
def test_variational_gaussian_process_trajectory_sampler_has_correct_samples() -> None:
    x_observed = tf.constant(np.arange(10).reshape((-1, 1)), dtype=gpflow.default_float())
    y_observed = _3x_plus_gaussian_noise(x_observed)
    optimizer = BatchOptimizer(tf.optimizers.Adam(), max_iter=20)
    likelihood = gpflow.likelihoods.Gaussian(1e-5)
    kernel = gpflow.kernels.Matern32(lengthscales=0.2)
    vgp = VGP((x_observed, y_observed), kernel, likelihood)
    model = VariationalGaussianProcess(vgp, optimizer=optimizer, use_natgrads=True)
    model.update(Dataset(x_observed, y_observed))
    model.optimize(Dataset(x_observed, y_observed))

    num_samples = 100
    trajectory_sampler = model.trajectory_sampler()
    assert isinstance(trajectory_sampler, DecoupledTrajectorySampler)

    trajectory = trajectory_sampler.get_trajectory()
    x_predict = tf.constant([[1.0], [2.0], [3.0], [1.5], [2.5], [3.5]], gpflow.default_float())
    x_predict_parallel = tf.expand_dims(x_predict, -2)  # [N, 1, D]
    x_predict_parallel = tf.tile(x_predict_parallel, [1, num_samples, 1])  # [N, B, D]
    samples = trajectory(x_predict_parallel)  # [N, B]
    sample_mean = tf.reduce_mean(samples, axis=1, keepdims=True)
    sample_variance = tf.math.reduce_variance(samples, axis=1, keepdims=True)

    true_mean, true_variance = model.predict(x_predict)

    # test predictions approx correct away from data
    npt.assert_allclose(sample_mean[3:] + 1.0, true_mean[3:] + 1.0, rtol=0.1)
    npt.assert_allclose(sample_variance[3:], true_variance[3:], rtol=0.5)

    # test predictions correct at data
    npt.assert_allclose(sample_mean[:3] + 1.0, true_mean[:3] + 1.0, rtol=0.001)
    npt.assert_allclose(sample_variance[:3], true_variance[:3], rtol=0.5)


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
    model.update_posterior_cache()
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


def test_sparse_variational_raises_for_model_with_q_diag_true() -> None:
    x = mock_data()[0]
    model = SVGP(
        gpflow.kernels.Matern32(),
        gpflow.likelihoods.Gaussian(),
        x[:2],
        num_data=len(x),
        q_diag=True,
    )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        SparseVariational(model)


def test_sparse_variational_model_attribute() -> None:
    model = svgp_model(*mock_data())
    sv = SparseVariational(model)

    assert sv.model is model
    assert isinstance(sv.model, SVGP)
    assert sv.inducing_point_selector is None


@pytest.mark.parametrize(
    "selector",
    [
        UniformInducingPointSelector(Box([0.0], [1.0])),
        RandomSubSampleInducingPointSelector(Box([0.0], [1.0])),
        KMeansInducingPointSelector(Box([0.0], [1.0])),
    ],
)
def test_sparse_variational_assigns_correct_inducing_point_selector(
    selector: InducingPointSelector[SparseVariational],
) -> None:
    model = svgp_model(*mock_data())
    sv = SparseVariational(model, inducing_point_selector=selector)
    assert isinstance(sv.inducing_point_selector, type(selector))


@pytest.mark.parametrize("recalc_every_model_update", [True, False])
def test_sparse_variational_chooses_new_inducing_points_correct_number_of_times(
    recalc_every_model_update: bool,
) -> None:
    model = svgp_model(*mock_data())
    selector = UniformInducingPointSelector(
        Box([0.0], [1.0]), recalc_every_model_update=recalc_every_model_update
    )
    sv = SparseVariational(model, inducing_point_selector=selector)
    old_inducing_points = sv.model.inducing_variable.Z.numpy()
    sv.update(Dataset(*mock_data()))
    first_inducing_points = sv.model.inducing_variable.Z.numpy()
    npt.assert_raises(
        AssertionError, npt.assert_array_equal, old_inducing_points, first_inducing_points
    )
    sv.update(Dataset(*mock_data()))
    second_inducing_points = sv.model.inducing_variable.Z.numpy()
    if recalc_every_model_update:
        npt.assert_raises(
            AssertionError, npt.assert_array_equal, old_inducing_points, second_inducing_points
        )
        npt.assert_raises(
            AssertionError, npt.assert_array_equal, first_inducing_points, second_inducing_points
        )
    else:
        npt.assert_raises(
            AssertionError, npt.assert_array_equal, old_inducing_points, second_inducing_points
        )
        npt.assert_array_equal(first_inducing_points, second_inducing_points)


@random_seed
@pytest.mark.parametrize(
    "mo_type", ["shared+shared", "separate+shared", "separate+separate", "auto"]
)
def test_sparse_variational_update_updates_num_data(mo_type: str) -> None:
    x = tf.constant(np.arange(1, 7).reshape(-1, 1), dtype=gpflow.default_float())  # shape: [6, 1]
    svgp = two_output_svgp_model(x, mo_type, True)
    model = SparseVariational(svgp)
    model.update(Dataset(tf.zeros([5, 1]), tf.zeros([5, 2])))
    assert model.model.num_data == 5


@pytest.mark.parametrize("whiten", [True, False])
def test_sparse_variational_correctly_returns_inducing_points(whiten: bool) -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    data = Dataset(x, fnc_3x_plus_10(x))
    model = SparseVariational(svgp_model(data.query_points, data.observations))
    model.model.whiten = whiten
    model.update(data)

    inducing_points, q_mu, q_sqrt, w = model.get_inducing_variables()

    npt.assert_allclose(inducing_points, model.model.inducing_variable.Z, atol=1e-5)
    npt.assert_allclose(q_mu, model.model.q_mu, atol=1e-5)
    npt.assert_allclose(q_sqrt, model.model.q_sqrt, atol=1e-5)
    assert whiten == w


@random_seed
@pytest.mark.parametrize(
    "mo_type", ["shared+shared", "separate+shared", "separate+separate", "auto"]
)
@pytest.mark.parametrize("whiten", [True, False])
def test_sparse_variational_correctly_returns_inducing_points_for_multi_output(
    whiten: bool, mo_type: str
) -> None:
    x = tf.constant(np.arange(6).reshape(-1, 1), dtype=gpflow.default_float())
    svgp = two_output_svgp_model(x, mo_type, whiten)
    model = SparseVariational(svgp)
    model.model.whiten = whiten
    model.update(Dataset(tf.zeros([5, 1]), tf.zeros([5, 2])))

    inducing_points, q_mu, q_sqrt, w = model.get_inducing_variables()

    if isinstance(model.model.inducing_variable, SharedIndependentInducingVariables):
        npt.assert_allclose(
            inducing_points,
            cast(TensorType, model.model.inducing_variable.inducing_variable).Z,
            atol=1e-5,
        )
    elif isinstance(model.model.inducing_variable, SeparateIndependentInducingVariables):
        for i, points in enumerate(model.model.inducing_variable.inducing_variables):
            npt.assert_allclose(inducing_points[i], cast(TensorType, points).Z, atol=1e-5)
    else:
        npt.assert_allclose(inducing_points, model.model.inducing_variable.Z, atol=1e-5)

    npt.assert_allclose(q_mu, model.model.q_mu, atol=1e-5)
    npt.assert_allclose(q_sqrt, model.model.q_sqrt, atol=1e-5)
    assert whiten == w


@random_seed
def test_sparse_variational_updates_inducing_points_raises_if_you_change_shape() -> None:
    model = SparseVariational(
        svgp_model(tf.zeros([5, 2]), tf.zeros([5, 1])),
    )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):  # current inducing point has 2 elements
        model._update_inducing_variables(tf.zeros([3, 2]))


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


def test_sparse_variational_gaussian_process_trajectory_sampler_raises_multi_latent_gp() -> None:
    data = mock_data()
    model = SparseVariational(svgp_model(*data, num_latent_gps=2))

    with pytest.raises(NotImplementedError):
        model.trajectory_sampler()


@random_seed
def test_sparse_variational_trajectory_sampler_has_correct_samples() -> None:
    x = tf.constant(np.arange(10).reshape((-1, 1)), dtype=gpflow.default_float())
    y = _3x_plus_gaussian_noise(x)
    svgp = SVGP(gpflow.kernels.Matern32(), gpflow.likelihoods.Gaussian(), x, num_data=len(x))
    model = SparseVariational(svgp)
    # for speed just pretend update rather than optimize
    model.model.q_sqrt.assign(tf.expand_dims(tf.eye(len(x)) * tf.math.sqrt(1e-5), 0))
    model.model.q_mu.assign(y)
    model.update_posterior_cache()

    num_samples = 100
    trajectory_sampler = model.trajectory_sampler()

    assert isinstance(trajectory_sampler, DecoupledTrajectorySampler)

    trajectory = trajectory_sampler.get_trajectory()
    x_predict = tf.constant([[1.0], [2.0], [3.0], [1.5], [2.5], [3.5]], gpflow.default_float())
    x_predict_parallel = tf.expand_dims(x_predict, -2)  # [N, 1, D]
    x_predict_parallel = tf.tile(x_predict_parallel, [1, num_samples, 1])  # [N, B, D]
    samples = trajectory(x_predict_parallel)  # [N, B]
    sample_mean = tf.reduce_mean(samples, axis=1, keepdims=True)
    sample_variance = tf.math.reduce_variance(samples, axis=1, keepdims=True)

    true_mean, true_variance = model.predict(x_predict)

    # test predictions approx correct away from data
    npt.assert_allclose(sample_mean[3:] + 1.0, true_mean[3:] + 1.0, rtol=0.2)
    npt.assert_allclose(sample_variance[3:], true_variance[3:], rtol=0.5)

    # test predictions almost correct at data
    npt.assert_allclose(sample_mean[:3] + 1.0, true_mean[:3] + 1.0, rtol=0.01)
    npt.assert_allclose(sample_variance[:3], true_variance[:3], rtol=0.3)


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
        SparseVariational(svgp_model(x_observed, y_observed), num_rff_features=0)

    with pytest.raises(ValueError):
        SparseVariational(svgp_model(x_observed, y_observed), num_rff_features=-1)

    with pytest.raises(ValueError):
        optimizer1 = BatchOptimizer(gpflow.optimizers.Scipy())
        SparseVariational(svgp_model(x_observed, y_observed), optimizer=optimizer1)

    with pytest.raises(ValueError):
        optimizer2 = Optimizer(tf.optimizers.Adam())
        SparseVariational(svgp_model(x_observed, y_observed), optimizer=optimizer2)


@random_seed
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize(
    "mo_type", ["shared+shared", "separate+shared", "separate+separate", "auto"]
)
def test_sparse_variational_pairwise_covariance_for_non_whitened(
    whiten: bool, mo_type: str
) -> None:
    x = tf.constant(np.arange(1, 7).reshape(-1, 1), dtype=gpflow.default_float())  # shape: [6, 1]
    y1 = fnc_3x_plus_10(x)
    y2 = y1 * 0.5

    svgp = two_output_svgp_model(x, mo_type, whiten)

    model = SparseVariational(svgp, BatchOptimizer(tf.optimizers.Adam(), max_iter=3, batch_size=10))
    model.model.whiten = whiten

    model.optimize(Dataset(x, tf.concat([y1, y2], axis=-1)))

    query_points_1 = tf.concat([0.5 * x, 0.5 * x], 0)  # shape: [12, 1]
    query_points_2 = tf.concat([2 * x, 2 * x, 2 * x], 0)  # shape: [18, 1]

    all_query_points = tf.concat([query_points_1, query_points_2], 0)
    _, predictive_covariance = model.predict_joint(all_query_points)
    expected_covariance = predictive_covariance[:, :12, 12:]

    actual_covariance = model.covariance_between_points(query_points_1, query_points_2)

    np.testing.assert_allclose(expected_covariance, actual_covariance, atol=1e-4)

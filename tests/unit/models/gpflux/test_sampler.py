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

from typing import Callable, Tuple
from unittest.mock import patch

import gpflow.kernels
import gpflux.layers
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from gpflux.helpers import construct_basic_inducing_variables, construct_basic_kernel
from gpflux.models import DeepGP

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, ShapeLike, mk_dataset, quadratic, random_seed
from tests.util.models.gpflow.models import QuadraticMeanAndRBFKernel
from tests.util.models.gpflux.models import simple_two_layer_dgp_model, two_layer_trieste_dgp
from trieste.data import Dataset
from trieste.models.gpflux import DeepGaussianProcess
from trieste.models.gpflux.sampler import (
    DeepGaussianProcessDecoupledLayer,
    DeepGaussianProcessDecoupledTrajectorySampler,
    DeepGaussianProcessReparamSampler,
    dgp_feature_decomposition_trajectory,
)
from trieste.space import Box
from trieste.types import TensorType


@pytest.mark.parametrize("sample_size", [0, -2])
def test_dgp_reparam_sampler_raises_for_invalid_sample_size(sample_size: int) -> None:
    search_space = Box([0.0], [1.0]) ** 4
    x = search_space.sample(10)
    data = mk_dataset(x, quadratic(x))

    dgp = two_layer_trieste_dgp(data, search_space)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        DeepGaussianProcessReparamSampler(sample_size, dgp)


def test_dgp_reparam_sampler_raises_for_invalid_model() -> None:
    with pytest.raises(ValueError, match="Model must be .*"):
        DeepGaussianProcessReparamSampler(10, QuadraticMeanAndRBFKernel())  # type: ignore


@pytest.mark.parametrize("shape", [[], [1], [2], [2, 3, 4]])
def test_dgp_reparam_sampler_sample_raises_for_invalid_at_shape(shape: ShapeLike) -> None:
    search_space = Box([0.0], [1.0])
    x = search_space.sample(10)
    data = mk_dataset(x, quadratic(x))

    dgp = two_layer_trieste_dgp(data, search_space)
    sampler = DeepGaussianProcessReparamSampler(1, dgp)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.zeros(shape))


def _build_dataset_and_train_deep_gp(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> Tuple[Dataset, DeepGaussianProcess]:
    x = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    y = tf.random.normal([100, 1], dtype=tf.float64)
    dataset = Dataset(x, y)

    dgp = two_layer_model(x)
    model = DeepGaussianProcess(dgp)
    model.optimizer.fit_args = {
        "verbose": 0,
        "epochs": 200,
        "batch_size": 1000,
    }

    model.optimize(dataset)

    return dataset, model


@random_seed
def test_dgp_reparam_sampler_samples_approximate_expected_distribution(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    sample_size = 250
    dataset, model = _build_dataset_and_train_deep_gp(two_layer_model)

    samples = DeepGaussianProcessReparamSampler(sample_size, model).sample(
        dataset.query_points[:, None, :]
    )  # [N, S, 1, L]

    assert samples.shape == [len(dataset.query_points), sample_size, 1, 1]

    sample_mean = tf.reduce_mean(samples, axis=1, keepdims=True)
    sample_variance = tf.squeeze(tf.reduce_mean((samples - sample_mean) ** 2, axis=1), -2)
    sample_mean = tf.squeeze(sample_mean, [1, 2])

    num_samples = 50
    means = []
    vars = []
    for _ in range(num_samples):
        Fmean_sample, Fvar_sample = model.predict(dataset.query_points)
        means.append(Fmean_sample)
        vars.append(Fvar_sample)
    ref_mean = tf.reduce_mean(tf.stack(means), axis=0)
    ref_variance = tf.reduce_mean(tf.stack(vars) + tf.stack(means) ** 2, axis=0) - ref_mean**2

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, ref_mean, atol=2 * error)
    npt.assert_allclose(sample_variance, ref_variance, atol=4 * error)


@random_seed
def test_dgp_reparam_sampler_sample_is_continuous(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    sampler = DeepGaussianProcessReparamSampler(100, model)
    xs = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)[:, None, :]
    npt.assert_array_less(tf.abs(sampler.sample(xs + 1e-20) - sampler.sample(xs)), 1e-20)


def test_dgp_reparam_sampler_sample_is_repeatable(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    sampler = DeepGaussianProcessReparamSampler(100, model)
    xs = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)[:, None, :]

    # also check there are no unnecessary casts from float64 to float32 or vice versa
    original_tf_cast = tf.cast

    def patched_tf_cast(x: TensorType, dtype: tf.DType) -> TensorType:
        if isinstance(x, tf.Tensor) and x.dtype in (tf.float32, tf.float64) and x.dtype != dtype:
            raise ValueError(f"unexpected cast: {x} to {dtype}")
        return original_tf_cast(x, dtype)

    with patch("tensorflow.cast", side_effect=patched_tf_cast):
        npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))


@random_seed
def test_dgp_reparam_sampler_samples_are_distinct_for_new_instances(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    sampler1 = DeepGaussianProcessReparamSampler(100, model)
    sampler2 = DeepGaussianProcessReparamSampler(100, model)

    xs = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)[:, None, :]
    npt.assert_array_less(1e-9, tf.abs(sampler2.sample(xs) - sampler1.sample(xs)))


@pytest.mark.parametrize("num_features", [0, -2])
def test_dgp_decoupled_trajectory_sampler_raises_for_invalid_number_of_features(
    num_features: int, two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        DeepGaussianProcessDecoupledTrajectorySampler(model, num_features=num_features)


def test_dgp_decoupled_trajectory_sampler_raises_for_invalid_model() -> None:
    with pytest.raises(ValueError, match="Model must be .*"):
        DeepGaussianProcessDecoupledTrajectorySampler(
            QuadraticMeanAndRBFKernel(), 10  # type: ignore
        )


def _generate_xs_for_decoupled_trajectory(num_evals: int, batch_size: int) -> TensorType:
    xs = tf.random.uniform([num_evals, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    xs = tf.expand_dims(xs, -2)
    return tf.tile(xs, [1, batch_size, 1])


@pytest.mark.parametrize("num_evals", [10, 100])
def test_dgp_decoupled_trajectory_sampler_returns_trajectory_function_with_correct_shapes(
    num_evals: int, two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    batch_size = 5

    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    sampler = DeepGaussianProcessDecoupledTrajectorySampler(model)

    trajectory = sampler.get_trajectory()
    xs = _generate_xs_for_decoupled_trajectory(num_evals, batch_size)

    tf.debugging.assert_shapes([(trajectory(xs), [num_evals, batch_size, 1])])

    assert isinstance(trajectory, dgp_feature_decomposition_trajectory)


@random_seed
def test_dgp_decoupled_trajectory_sampler_returns_deterministic_trajectory(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    sampler = DeepGaussianProcessDecoupledTrajectorySampler(model)
    xs = _generate_xs_for_decoupled_trajectory(10, 5)

    trajectory = sampler.get_trajectory()
    trajectory_eval_1 = trajectory(xs)
    trajectory_eval_2 = trajectory(xs)
    npt.assert_allclose(trajectory_eval_1, trajectory_eval_2)


@random_seed
def test_dgp_decoupled_trajectory_sampler_sample_is_continuous(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    sampler = DeepGaussianProcessDecoupledTrajectorySampler(model)
    xs = _generate_xs_for_decoupled_trajectory(10, 5)

    trajectory = sampler.get_trajectory()
    npt.assert_array_less(tf.abs(trajectory(xs + 1e-20) - trajectory(xs)), 1e-20)


@random_seed
def test_dgp_decoupled_trajectory_sampler_samples_approximate_expected_distribution(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    sample_size = 100
    dataset, model = _build_dataset_and_train_deep_gp(two_layer_model)

    sampler = DeepGaussianProcessDecoupledTrajectorySampler(model)
    trajectory = sampler.get_trajectory()

    xs = tf.expand_dims(dataset.query_points, -2)  # [N, 1, D]
    xs = tf.tile(xs, [1, sample_size, 1])  # [N, B, D]

    samples = trajectory(xs)

    assert samples.shape == [len(dataset.query_points), sample_size, 1]

    sample_mean = tf.reduce_mean(samples, axis=1)
    sample_variance = tf.math.reduce_variance(samples, axis=1)

    num_samples = 50
    means = []
    vars = []
    for _ in range(num_samples):
        Fmean_sample, Fvar_sample = model.predict(dataset.query_points)
        means.append(Fmean_sample)
        vars.append(Fvar_sample)
    ref_mean = tf.reduce_mean(tf.stack(means), axis=0)
    ref_variance = tf.reduce_mean(tf.stack(vars) + tf.stack(means) ** 2, axis=0) - ref_mean**2

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, ref_mean, atol=2 * error)
    npt.assert_allclose(sample_variance, ref_variance, atol=4 * error)


@random_seed
def test_dgp_decoupled_trajectory_sampler_samples_are_distinct_for_new_instances(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    sampler_1 = DeepGaussianProcessDecoupledTrajectorySampler(model)
    trajectory_1 = sampler_1.get_trajectory()
    sampler_2 = DeepGaussianProcessDecoupledTrajectorySampler(model)
    trajectory_2 = sampler_2.get_trajectory()

    xs = _generate_xs_for_decoupled_trajectory(10, 2)

    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(trajectory_1(xs) - trajectory_2(xs)))
    )  # distinct between sample draws
    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(trajectory_1(xs)[:, 0] - trajectory_1(xs)[:, 1]))
    )  # distinct between samples within draws
    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(trajectory_2(xs)[:, 0] - trajectory_2(xs)[:, 1]))
    )  # distinct between samples within draws


@random_seed
def test_dgp_decoupled_trajectory_resample_trajectory_provides_new_samples_without_retracing(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    xs = _generate_xs_for_decoupled_trajectory(10, 5)

    sampler = DeepGaussianProcessDecoupledTrajectorySampler(model)
    trajectory = sampler.get_trajectory()
    evals_1 = trajectory(xs)
    for _ in range(5):
        trajectory = sampler.resample_trajectory(trajectory)
        evals_new = trajectory(xs)
        npt.assert_array_less(1e-1, tf.reduce_max(tf.abs(evals_1 - evals_new)))

    assert trajectory.__call__._get_tracing_count() == 1  # type: ignore


@random_seed
def test_dgp_decoupled_trajectory_update_trajectory_updates_and_doesnt_retrace(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    xs = _generate_xs_for_decoupled_trajectory(10, 5)

    sampler = DeepGaussianProcessDecoupledTrajectorySampler(model)
    trajectory = sampler.get_trajectory()
    eval_before = trajectory(xs)

    for _ in range(3):  # do three updates on new data and see if samples are new
        x_train = tf.random.uniform([20, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
        y_train = tf.random.normal([20, 1], dtype=tf.float64)
        new_dataset = Dataset(x_train, y_train)
        model.update(new_dataset)
        model.optimize(new_dataset)

        trajectory_updated = sampler.update_trajectory(trajectory)
        eval_after = trajectory(xs)

        assert trajectory_updated is trajectory  # check update was in place

        npt.assert_array_less(
            0.1, tf.reduce_max(tf.abs(eval_before - eval_after))
        )  # two samples should be different

    assert trajectory.__call__._get_tracing_count() == 1  # type: ignore


def test_dgp_decoupled_layer_raises_for_invalid_layer() -> None:
    w_dim = 1
    prior_means = np.zeros(w_dim)
    prior_std = np.ones(w_dim)
    encoder = gpflux.encoders.DirectlyParameterizedNormalDiag(10, w_dim)
    prior = tfp.distributions.MultivariateNormalDiag(prior_means, prior_std)
    lv = gpflux.layers.LatentVariableLayer(prior, encoder)

    x = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    model = DeepGaussianProcess(simple_two_layer_dgp_model(x))
    model.model_gpflux.f_layers[0] = lv

    with pytest.raises(ValueError, match="Layers other than .*"):
        DeepGaussianProcessDecoupledLayer(model, 0)


@pytest.mark.parametrize("num_features", [0, -2])
def test_dgp_decoupled_layer_raises_for_invalid_number_of_features(num_features: int) -> None:
    kernel = construct_basic_kernel(
        gpflow.kernels.SquaredExponential(), output_dim=1, share_hyperparams=True
    )
    inducing_var = construct_basic_inducing_variables(
        num_inducing=5,
        input_dim=1,
        share_variables=True,
        z_init=tf.random.normal([5, 1], dtype=gpflow.default_float()),
    )
    layer = gpflux.layers.GPLayer(
        kernel,
        inducing_var,
        num_data=10,
        num_latent_gps=2,
    )
    x = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    model = DeepGaussianProcess(simple_two_layer_dgp_model(x))
    model.model_gpflux.f_layers[0] = layer

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        DeepGaussianProcessDecoupledLayer(model, 0, num_features)


def test_dgp_decoupled_layer_raises_for_invalid_inducing_variables() -> None:
    ip1 = gpflow.inducing_variables.InducingPoints(
        tf.random.normal([5, 1], dtype=gpflow.default_float())
    )
    ip2 = gpflow.inducing_variables.InducingPoints(
        tf.random.normal([5, 1], dtype=gpflow.default_float())
    )
    inducing_var = gpflow.inducing_variables.SeparateIndependentInducingVariables([ip1, ip2])

    layer = gpflux.layers.GPLayer(
        gpflow.kernels.SeparateIndependent(
            [gpflow.kernels.SquaredExponential(), gpflow.kernels.SquaredExponential()]
        ),
        inducing_var,
        num_data=10,
        num_latent_gps=2,
    )

    x = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    model = DeepGaussianProcess(simple_two_layer_dgp_model(x))
    model.model_gpflux.f_layers[0] = layer

    with pytest.raises(ValueError, match="SeparateIndependentInducingVariables .*"):
        DeepGaussianProcessDecoupledLayer(model, 0)


def test_dgp_decoupled_layer_returns_trajectory_with_correct_shapes(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    num_evals = 20
    batch_size = 5

    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    layer = model.model_gpflux.f_layers[0]
    P = layer.num_latent_gps

    decoupled_layer = DeepGaussianProcessDecoupledLayer(model, 0)

    xs = _generate_xs_for_decoupled_trajectory(num_evals, batch_size)

    tf.debugging.assert_shapes([(decoupled_layer(xs), [num_evals, batch_size, P])])


@random_seed
def test_dgp_decoupled_layer_returns_deterministic_trajectory(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    decoupled_layer = DeepGaussianProcessDecoupledLayer(model, 0)

    xs = _generate_xs_for_decoupled_trajectory(10, 5)

    eval_1 = decoupled_layer(xs)
    eval_2 = decoupled_layer(xs)

    npt.assert_allclose(eval_1, eval_2)


@random_seed
def test_dgp_decoupled_layer_samples_are_distinct_for_new_instances(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    decoupled_layer_1 = DeepGaussianProcessDecoupledLayer(model, 0)
    decoupled_layer_2 = DeepGaussianProcessDecoupledLayer(model, 0)

    xs = _generate_xs_for_decoupled_trajectory(100, 5)

    npt.assert_array_less(
        1e-2, tf.reduce_sum(tf.abs(decoupled_layer_1(xs) - decoupled_layer_2(xs)))
    )  # distinct between sample draws
    npt.assert_array_less(
        1e-2, tf.reduce_sum(tf.abs(decoupled_layer_1(xs)[:, 0] - decoupled_layer_1(xs)[:, 1]))
    )  # distinct between samples within draws
    npt.assert_array_less(
        1e-2, tf.reduce_sum(tf.abs(decoupled_layer_2(xs)[:, 0] - decoupled_layer_2(xs)[:, 1]))
    )  # distinct between samples within draws


@random_seed
def test_dgp_decoupled_layer_resample_provides_new_samples(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    decoupled_layer = DeepGaussianProcessDecoupledLayer(model, 0)

    xs = _generate_xs_for_decoupled_trajectory(10, 5)

    evals_1 = decoupled_layer(xs)
    for _ in range(5):
        decoupled_layer.resample()
        evals_new = decoupled_layer(xs)
        npt.assert_array_less(1e-2, tf.reduce_sum(tf.abs(evals_1 - evals_new)))


@random_seed
def test_dgp_decoupled_layer_update_updates(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    decoupled_layer = DeepGaussianProcessDecoupledLayer(model, 0)

    xs = _generate_xs_for_decoupled_trajectory(10, 5)

    evals_1 = decoupled_layer(xs)

    original_W = decoupled_layer._feature_functions.W.value().numpy()
    original_b = decoupled_layer._feature_functions.b.value().numpy()
    for _ in range(5):
        x_train = tf.random.uniform([20, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
        y_train = tf.random.normal([20, 1], dtype=tf.float64)
        new_dataset = Dataset(x_train, y_train)
        model.update(new_dataset)
        model.optimize(new_dataset)

        decoupled_layer.update()
        evals_new = decoupled_layer(xs)
        npt.assert_array_less(1e-2, tf.reduce_sum(tf.abs(evals_1 - evals_new)))

        # Check that RFF weights change
        npt.assert_array_less(
            1e-2, tf.reduce_sum(tf.abs(original_b - decoupled_layer._feature_functions.b))
        )
        npt.assert_array_less(
            1e-2, tf.reduce_sum(tf.abs(original_W - decoupled_layer._feature_functions.W))
        )

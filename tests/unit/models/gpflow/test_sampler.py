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
from __future__ import annotations

import math

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, ShapeLike, quadratic, random_seed
from tests.util.models.gpflow.models import GaussianProcess, QuadraticMeanAndRBFKernel, rbf
from trieste.data import Dataset
from trieste.models.gpflow import (
    BatchReparametrizationSampler,
    IndependentReparametrizationSampler,
    RandomFourierFeatureTrajectorySampler,
)
from trieste.objectives.single_objectives import branin


@pytest.mark.parametrize(
    "sampler",
    [
        BatchReparametrizationSampler,
        IndependentReparametrizationSampler,
    ],
)
def test_reparametrization_sampler_reprs(
    sampler: type[BatchReparametrizationSampler | IndependentReparametrizationSampler],
) -> None:
    assert (
        repr(sampler(20, QuadraticMeanAndRBFKernel()))
        == f"{sampler.__name__}(20, QuadraticMeanAndRBFKernel())"
    )


def test_independent_reparametrization_sampler_sample_raises_for_negative_jitter() -> None:
    sampler = IndependentReparametrizationSampler(100, QuadraticMeanAndRBFKernel())
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.constant([[0.0]]), jitter=-1e-6)


@pytest.mark.parametrize("sample_size", [0, -2])
def test_independent_reparametrization_sampler_raises_for_invalid_sample_size(
    sample_size: int,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        IndependentReparametrizationSampler(sample_size, QuadraticMeanAndRBFKernel())


@pytest.mark.parametrize("shape", [[], [1], [2], [2, 3]])
def test_independent_reparametrization_sampler_sample_raises_for_invalid_at_shape(
    shape: ShapeLike,
) -> None:
    sampler = IndependentReparametrizationSampler(1, QuadraticMeanAndRBFKernel())

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.zeros(shape))


def _assert_kolmogorov_smirnov_95(
    # fmt: off
    samples: tf.Tensor,  # [..., S]
    distribution: tfp.distributions.Distribution
    # fmt: on
) -> None:
    assert distribution.event_shape == ()
    tf.debugging.assert_shapes([(samples, [..., "S"])])

    sample_size = samples.shape[-1]
    samples_sorted = tf.sort(samples, axis=-1)  # [..., S]
    edf = tf.range(1.0, sample_size + 1, dtype=samples.dtype) / sample_size  # [S]
    expected_cdf = distribution.cdf(samples_sorted)  # [..., S]

    _95_percent_bound = 1.36 / math.sqrt(sample_size)
    assert tf.reduce_max(tf.abs(edf - expected_cdf)) < _95_percent_bound


def _dim_two_gp(mean_shift: tuple[float, float] = (0.0, 0.0)) -> GaussianProcess:
    matern52 = tfp.math.psd_kernels.MaternFiveHalves(
        amplitude=tf.cast(2.3, tf.float64), length_scale=tf.cast(0.5, tf.float64)
    )
    return GaussianProcess(
        [lambda x: mean_shift[0] + branin(x), lambda x: mean_shift[1] + quadratic(x)],
        [matern52, rbf()],
    )


@random_seed
def test_independent_reparametrization_sampler_samples_approximate_expected_distribution() -> None:
    sample_size = 1000
    x = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)

    model = _dim_two_gp()
    samples = IndependentReparametrizationSampler(sample_size, model).sample(x)  # [N, S, 1, L]

    assert samples.shape == [len(x), sample_size, 1, 2]

    mean, var = model.predict(tf.squeeze(x, -2))  # [N, L], [N, L]
    _assert_kolmogorov_smirnov_95(
        tf.linalg.matrix_transpose(tf.squeeze(samples, -2)),
        tfp.distributions.Normal(mean[..., None], tf.sqrt(var[..., None])),
    )


@random_seed
def test_independent_reparametrization_sampler_sample_is_continuous() -> None:
    sampler = IndependentReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_array_less(tf.abs(sampler.sample(xs + 1e-20) - sampler.sample(xs)), 1e-20)


def test_independent_reparametrization_sampler_sample_is_repeatable() -> None:
    sampler = IndependentReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))


@random_seed
def test_independent_reparametrization_sampler_samples_are_distinct_for_new_instances() -> None:
    sampler1 = IndependentReparametrizationSampler(100, _dim_two_gp())
    sampler2 = IndependentReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_array_less(1e-9, tf.abs(sampler2.sample(xs) - sampler1.sample(xs)))


def test_independent_reparametrization_sampler_reset_sampler() -> None:
    sampler = IndependentReparametrizationSampler(100, _dim_two_gp())
    assert not sampler._initialized
    xs = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    sampler.sample(xs)
    assert sampler._initialized
    sampler.reset_sampler()
    assert not sampler._initialized
    sampler.sample(xs)
    assert sampler._initialized


@pytest.mark.parametrize("sample_size", [0, -2])
def test_batch_reparametrization_sampler_raises_for_invalid_sample_size(sample_size: int) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchReparametrizationSampler(sample_size, _dim_two_gp())


@random_seed
def test_batch_reparametrization_sampler_samples_approximate_mean_and_covariance() -> None:
    model = _dim_two_gp()
    sample_size = 10_000
    leading_dims = [3]
    batch_size = 4
    xs = tf.random.uniform(leading_dims + [batch_size, 2], maxval=1.0, dtype=tf.float64)
    samples = BatchReparametrizationSampler(sample_size, model).sample(xs)

    assert samples.shape == leading_dims + [sample_size, batch_size, 2]

    samples_mean = tf.reduce_mean(samples, axis=-3)
    samples_covariance = tf.transpose(
        tfp.stats.covariance(samples, sample_axis=-3, event_axis=-2), [0, 3, 1, 2]
    )

    model_mean, model_cov = model.predict_joint(xs)

    npt.assert_allclose(samples_mean, model_mean, rtol=0.02)
    npt.assert_allclose(samples_covariance, model_cov, rtol=0.04)


def test_batch_reparametrization_sampler_samples_are_continuous() -> None:
    sampler = BatchReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    npt.assert_array_less(tf.abs(sampler.sample(xs + 1e-20) - sampler.sample(xs)), 1e-20)


def test_batch_reparametrization_sampler_samples_are_repeatable() -> None:
    sampler = BatchReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))


@random_seed
def test_batch_reparametrization_sampler_samples_are_distinct_for_new_instances() -> None:
    model = _dim_two_gp()
    sampler1 = BatchReparametrizationSampler(100, model)
    sampler2 = BatchReparametrizationSampler(100, model)
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    npt.assert_array_less(1e-9, tf.abs(sampler2.sample(xs) - sampler1.sample(xs)))


@pytest.mark.parametrize("at", [tf.constant([0.0]), tf.constant(0.0), tf.ones([0, 1])])
def test_batch_reparametrization_sampler_sample_raises_for_invalid_at_shape(at: tf.Tensor) -> None:
    sampler = BatchReparametrizationSampler(100, QuadraticMeanAndRBFKernel())

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(at)


def test_batch_reparametrization_sampler_sample_raises_for_negative_jitter() -> None:
    sampler = BatchReparametrizationSampler(100, QuadraticMeanAndRBFKernel())

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.constant([[0.0]]), jitter=-1e-6)


def test_batch_reparametrization_sampler_sample_raises_for_inconsistent_batch_size() -> None:
    sampler = BatchReparametrizationSampler(100, QuadraticMeanAndRBFKernel())
    sampler.sample(tf.constant([[0.0], [1.0], [2.0]]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.constant([[0.0], [1.0]]))


def test_batch_reparametrization_sampler_reset_sampler() -> None:
    sampler = BatchReparametrizationSampler(100, QuadraticMeanAndRBFKernel())
    assert not sampler._initialized
    xs = tf.constant([[0.0], [1.0], [2.0]])
    sampler.sample(xs)
    assert sampler._initialized
    sampler.reset_sampler()
    assert not sampler._initialized
    sampler.sample(xs)
    assert sampler._initialized


@pytest.mark.parametrize("num_features", [0, -2])
def test_rff_trajectory_sampler_raises_for_invalid_number_of_features(
    num_features: int,
) -> None:
    model = QuadraticMeanAndRBFKernel(noise_variance=tf.constant(1.0, dtype=tf.float64))
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    dataset = Dataset(
        tf.constant([[-2.0]], dtype=tf.float64), tf.constant([[4.1]], dtype=tf.float64)
    )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        RandomFourierFeatureTrajectorySampler(model, dataset, num_features=num_features)


def test_rff_trajectory_sampler_raises_for_a_non_gpflow_kernel() -> None:
    model = QuadraticMeanAndRBFKernel()
    dataset = Dataset(tf.constant([[-2.0]]), tf.constant([[4.1]]))
    with pytest.raises(AssertionError):
        RandomFourierFeatureTrajectorySampler(model, dataset, num_features=100)


@pytest.mark.parametrize("num_evals", [10, 100])
def test_rff_trajectory_sampler_returns_trajectory_function_with_correct_shaped_output(
    num_evals: int,
) -> None:
    model = QuadraticMeanAndRBFKernel(noise_variance=tf.constant(1.0, dtype=tf.float64))
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    dataset = Dataset(
        tf.constant([[-2.0]], dtype=tf.float64), tf.constant([[4.1]], dtype=tf.float64)
    )
    sampler = RandomFourierFeatureTrajectorySampler(model, dataset, num_features=100)

    trajectory = sampler.get_trajectory()
    xs = tf.linspace([-10.0], [10.0], num_evals)

    tf.debugging.assert_shapes([(trajectory(xs), [num_evals, 1])])


def test_rff_trajectory_sampler_returns_deterministic_trajectory() -> None:
    model = QuadraticMeanAndRBFKernel(noise_variance=tf.constant(1.0, dtype=tf.float64))
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)

    sampler = RandomFourierFeatureTrajectorySampler(model, dataset, num_features=100)
    trajectory = sampler.get_trajectory()

    trajectory_eval_1 = trajectory(xs)
    trajectory_eval_2 = trajectory(xs)

    npt.assert_allclose(trajectory_eval_1, trajectory_eval_2)


def test_rff_trajectory_sampler_returns_same_posterior_from_each_calculation_method() -> None:
    model = QuadraticMeanAndRBFKernel(noise_variance=tf.constant(1.0, dtype=tf.float64))
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)

    sampler = RandomFourierFeatureTrajectorySampler(model, dataset, num_features=100)
    sampler.get_trajectory()

    posterior_1 = sampler._prepare_theta_posterior_in_design_space()
    posterior_2 = sampler._prepare_theta_posterior_in_gram_space()

    npt.assert_allclose(posterior_1.loc, posterior_2.loc, rtol=0.02)
    npt.assert_allclose(posterior_1.scale_tril, posterior_2.scale_tril, rtol=0.02)


@random_seed
def test_rff_trajectory_sampler_samples_are_distinct_for_new_instances() -> None:
    model = QuadraticMeanAndRBFKernel(noise_variance=tf.constant(1.0, dtype=tf.float64))
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)

    sampler1 = RandomFourierFeatureTrajectorySampler(model, dataset, num_features=100)
    trajectory1 = sampler1.get_trajectory()

    sampler2 = RandomFourierFeatureTrajectorySampler(model, dataset, num_features=100)
    trajectory2 = sampler2.get_trajectory()

    npt.assert_array_less(1e-9, tf.abs(trajectory1(xs) - trajectory2(xs)))

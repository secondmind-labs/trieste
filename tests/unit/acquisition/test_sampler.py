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
from tests.util.models.gpflow.models import GaussianProcess, QuadraticMeanAndRBFKernel, rbf, QuadraticMeanAndRBFKernel_with_trajectories
from trieste.acquisition.sampler import (
    ExactThompsonSampler,
    GumbelSampler,
    ThompsonSamplerFromTrajectory,
)
from trieste.data import Dataset
from trieste.models.gpflow.sampler import RandomFourierFeatureTrajectorySampler
from trieste.objectives.single_objectives import branin
from trieste.space import Box


@pytest.mark.parametrize("sample_size", [0, -2])
def test_gumbel_sampler_raises_for_invalid_sample_size(
    sample_size: int,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        GumbelSampler(sample_size, QuadraticMeanAndRBFKernel())


@pytest.mark.parametrize("shape", [[], [1], [2], [1, 2, 3]])
def test_gumbel_sampler_sample_raises_for_invalid_at_shape(
    shape: ShapeLike,
) -> None:
    sampler = GumbelSampler(1, QuadraticMeanAndRBFKernel())

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.zeros(shape))


@pytest.mark.parametrize("sample_size", [10, 100])
def test_gumbel_sampler_returns_correctly_shaped_samples(sample_size: int) -> None:
    search_space = Box([0, 0], [1, 1])
    gumbel_sampler = GumbelSampler(sample_size, QuadraticMeanAndRBFKernel())
    query_points = search_space.sample(5)
    gumbel_samples = gumbel_sampler.sample(query_points)
    tf.debugging.assert_shapes([(gumbel_samples, [sample_size, 1])])


def test_gumbel_samples_are_minima() -> None:
    search_space = Box([0, 0], [1, 1])

    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)

    model = QuadraticMeanAndRBFKernel()
    gumbel_sampler = GumbelSampler(5, model)

    query_points = search_space.sample(100)
    query_points = tf.concat([dataset.query_points, query_points], 0)
    gumbel_samples = gumbel_sampler.sample(query_points)

    fmean, _ = model.predict(dataset.query_points)
    assert max(gumbel_samples) < min(fmean)


@pytest.mark.parametrize("sample_size", [0, -2])
def test_exact_thompson_sampler_raises_for_invalid_sample_size(
    sample_size: int,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ExactThompsonSampler(sample_size, QuadraticMeanAndRBFKernel())


@pytest.mark.parametrize("shape", [[], [1], [2], [1, 2, 3]])
def test_exact_thompson_sampler_sample_raises_for_invalid_at_shape(
    shape: ShapeLike,
) -> None:
    sampler = ExactThompsonSampler(1, QuadraticMeanAndRBFKernel())

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.zeros(shape))


@pytest.mark.parametrize("sample_min_value", [True, False])
@pytest.mark.parametrize("sample_size", [10, 100])
def test_exact_thompson_sampler_returns_correctly_shaped_samples(
    sample_min_value: bool, sample_size: int
) -> None:
    search_space = Box([0, 0], [1, 1])
    thompson_sampler = ExactThompsonSampler(
        sample_size, QuadraticMeanAndRBFKernel(), sample_min_value=sample_min_value
    )
    query_points = search_space.sample(500)
    thompson_samples = thompson_sampler.sample(query_points)
    if sample_min_value:
        tf.debugging.assert_shapes([(thompson_samples, [sample_size, 1])])
    else:
        tf.debugging.assert_shapes([(thompson_samples, [sample_size, 2])])


def test_exact_thompson_samples_are_minima() -> None:
    search_space = Box([0, 0], [1, 1])

    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)

    model = QuadraticMeanAndRBFKernel()
    gumbel_sampler = ExactThompsonSampler(5, model, sample_min_value=True)

    query_points = search_space.sample(100)
    query_points = tf.concat([dataset.query_points, query_points], 0)
    gumbel_samples = gumbel_sampler.sample(query_points)

    fmean, _ = model.predict(dataset.query_points)
    assert max(gumbel_samples) < min(fmean)


@pytest.mark.parametrize("sample_size", [0, -2])
def test_thompson_trajectory_sampler_raises_for_invalid_sample_size(
    sample_size: int,
) -> None:
    model = QuadraticMeanAndRBFKernel()
    dataset = Dataset(tf.constant([[-2.0]]), tf.constant([[4.1]]))
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ThompsonSamplerFromTrajectory(sample_size, model)




@pytest.mark.parametrize("shape", [[], [1], [2], [1, 2, 3]])
def test_thompson_trajectory_sampler_sample_raises_for_invalid_at_shape(
    shape: ShapeLike,
) -> None:
    dataset = Dataset(
        tf.constant([[-2.0]], dtype=tf.float64), tf.constant([[4.1]], dtype=tf.float64)
    )
    model = QuadraticMeanAndRBFKernel_with_trajectories(
        dataset=dataset, noise_variance=tf.constant(1.0, dtype=tf.float64)
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions

    sampler = ThompsonSamplerFromTrajectory(1, model)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.zeros(shape))


@pytest.mark.parametrize("sample_min_value", [True, False])
@pytest.mark.parametrize("sample_size", [10, 100])
def test_thompson_trajectory_sampler_returns_correctly_shaped_samples(
    sample_min_value: bool, sample_size: int
) -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)

    model = QuadraticMeanAndRBFKernel_with_trajectories(
        dataset=dataset, noise_variance=tf.constant(1.0, dtype=tf.float64)
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions

    sampler = ThompsonSamplerFromTrajectory(sample_size, model, sample_min_value=sample_min_value)

    query_points = search_space.sample(100)
    thompson_samples = sampler.sample(query_points)
    if sample_min_value:
        tf.debugging.assert_shapes([(thompson_samples, [sample_size, 1])])
    else:
        tf.debugging.assert_shapes([(thompson_samples, [sample_size, 2])])


@random_seed
def test_rff_thompson_samples_are_minima() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)
    model = QuadraticMeanAndRBFKernel_with_trajectories(
        dataset=dataset, noise_variance=tf.constant(1e-5, dtype=tf.float64)
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions

    sampler = ThompsonSamplerFromTrajectory(1, model, sample_min_value=True)

    query_points = search_space.sample(100)
    query_points = tf.concat([dataset.query_points, query_points], 0)
    thompson_samples = sampler.sample(query_points)

    fmean, _ = model.predict(dataset.query_points)
    assert max(thompson_samples) < min(fmean)

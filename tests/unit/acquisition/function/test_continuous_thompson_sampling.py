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

from __future__ import annotations

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import quadratic
from tests.util.models.gpflow.models import (
    QuadraticMeanAndRBFKernel,
    QuadraticMeanAndRBFKernelWithSamplers,
)
from trieste.acquisition.function.continuous_thompson_sampling import (
    GreedyContinuousThompsonSampling,
    ParallelContinuousThompsonSampling,
    negate_trajectory_function,
)
from trieste.acquisition.function.function import lower_confidence_bound
from trieste.data import Dataset
from trieste.models import TrajectoryFunction, TrajectoryFunctionClass, TrajectorySampler
from trieste.models.gpflow import (
    RandomFourierFeatureTrajectorySampler,
    feature_decomposition_trajectory,
)


class DumbTrajectorySampler(RandomFourierFeatureTrajectorySampler):
    """A RandomFourierFeatureTrajectorySampler that doesn't update trajectories in place."""

    def update_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        tf.debugging.Assert(
            isinstance(trajectory, feature_decomposition_trajectory), [tf.constant([])]
        )
        return self.get_trajectory()


class ModelWithDumbSamplers(QuadraticMeanAndRBFKernelWithSamplers):
    """A model that uses DumbTrajectorySampler."""

    def trajectory_sampler(self) -> TrajectorySampler[QuadraticMeanAndRBFKernelWithSamplers]:
        return DumbTrajectorySampler(self, 100)


def test_greedy_thompson_sampling_raises_for_model_without_trajectory_sampler() -> None:
    model = QuadraticMeanAndRBFKernel()
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    with pytest.raises(ValueError):
        GreedyContinuousThompsonSampling().prepare_acquisition_function(model)  # type: ignore


@pytest.mark.parametrize("dumb_samplers", [True, False])
def test_greedy_thompson_sampling_builder_builds_trajectory(dumb_samplers: bool) -> None:
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)
    model_type = ModelWithDumbSamplers if dumb_samplers else QuadraticMeanAndRBFKernelWithSamplers
    model = model_type(dataset, noise_variance=tf.constant(1.0, dtype=tf.float64))
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    builder = GreedyContinuousThompsonSampling()
    acq_fn = builder.prepare_acquisition_function(model)
    assert isinstance(acq_fn, TrajectoryFunctionClass)
    new_acq_fn = builder.update_acquisition_function(acq_fn, model)
    assert isinstance(new_acq_fn, TrajectoryFunctionClass)


@pytest.mark.parametrize("dumb_samplers", [True, False])
def test_greedy_thompson_sampling_builder_raises_when_update_with_wrong_function(
    dumb_samplers: bool,
) -> None:
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)
    model_type = ModelWithDumbSamplers if dumb_samplers else QuadraticMeanAndRBFKernelWithSamplers
    model = model_type(dataset, noise_variance=tf.constant(1.0, dtype=tf.float64))
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    builder = GreedyContinuousThompsonSampling()
    builder.prepare_acquisition_function(model)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.update_acquisition_function(lower_confidence_bound(model, 0.1), model)


def test_parallel_thompson_sampling_raises_for_model_without_trajectory_sampler() -> None:
    model = QuadraticMeanAndRBFKernel()
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    with pytest.raises(ValueError):
        ParallelContinuousThompsonSampling().prepare_acquisition_function(model)  # type: ignore


@pytest.mark.parametrize("dumb_samplers", [True, False])
def test_parallel_thompson_sampling_builder_builds_trajectory(dumb_samplers: bool) -> None:
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)
    model_type = ModelWithDumbSamplers if dumb_samplers else QuadraticMeanAndRBFKernelWithSamplers
    model = model_type(dataset, noise_variance=tf.constant(1.0, dtype=tf.float64))
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    builder = ParallelContinuousThompsonSampling()
    acq_fn = builder.prepare_acquisition_function(model)
    assert isinstance(acq_fn, TrajectoryFunctionClass)
    assert acq_fn.__class__.__name__ == "NegatedTrajectory"
    new_acq_fn = builder.update_acquisition_function(acq_fn, model)
    assert isinstance(new_acq_fn, TrajectoryFunctionClass)
    assert new_acq_fn.__class__.__name__ == "NegatedTrajectory"


@pytest.mark.parametrize("dumb_samplers", [True, False])
def test_parallel_thompson_sampling_builder_raises_when_update_with_wrong_function(
    dumb_samplers: bool,
) -> None:
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)
    model_type = ModelWithDumbSamplers if dumb_samplers else QuadraticMeanAndRBFKernelWithSamplers
    model = model_type(dataset, noise_variance=tf.constant(1.0, dtype=tf.float64))
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    builder = ParallelContinuousThompsonSampling()
    builder.prepare_acquisition_function(model)
    with pytest.raises(ValueError):
        builder.update_acquisition_function(lower_confidence_bound(model, 0.1), model)


def test_parallel_thompson_sampling_raises_for_changing_batch_size() -> None:
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)
    model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset, noise_variance=tf.constant(1.0, dtype=tf.float64)
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    builder = ParallelContinuousThompsonSampling()
    acq_fn = builder.prepare_acquisition_function(model)
    query_at = tf.reshape(tf.linspace([[-10]], [[10]], 100), [10, 5, 2])
    acq_fn(query_at)
    with pytest.raises(tf.errors.InvalidArgumentError):
        query_at = tf.reshape(tf.linspace([[-10]], [[10]], 100), [5, 10, 2])
        acq_fn(query_at)


def test_negate_trajectory_function_negates_and_keeps_methods() -> None:
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)
    model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset, noise_variance=tf.constant(1.0, dtype=tf.float64)
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    builder = ParallelContinuousThompsonSampling()

    acq_fn = builder.prepare_acquisition_function(model)
    query_at = tf.reshape(tf.linspace([[-10]], [[10]], 100), [10, 5, 2])
    evals = acq_fn(query_at)

    neg_acq_fn = negate_trajectory_function(acq_fn)
    neg_evals = acq_fn(query_at)
    npt.assert_array_equal(evals, -1.0 * neg_evals)

    assert hasattr(neg_acq_fn, "update")
    assert hasattr(neg_acq_fn, "resample")

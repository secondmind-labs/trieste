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

from collections.abc import Mapping
from typing import Callable, Optional, Sequence
from unittest.mock import MagicMock

import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.autograph.impl.api import StagingError

from tests.util.misc import (
    TF_DEBUGGING_ERROR_TYPES,
    ShapeLike,
    mk_dataset,
    quadratic,
    raise_exc,
    random_seed,
    various_shapes,
)
from tests.util.models.gpflow.models import (
    GaussianProcess,
    GaussianProcessWithBatchSamplers,
    GaussianProcessWithoutNoise,
    GaussianProcessWithSamplers,
    QuadraticMeanAndRBFKernel,
    QuadraticMeanAndRBFKernelWithBatchSamplers,
    QuadraticMeanAndRBFKernelWithSamplers,
    rbf,
)
from trieste.acquisition.function.function import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    AugmentedExpectedImprovement,
    BatchExpectedImprovement,
    BatchMonteCarloExpectedImprovement,
    ExpectedConstrainedImprovement,
    ExpectedImprovement,
    FastConstraintsFeasibility,
    MakePositive,
    MonteCarloAugmentedExpectedImprovement,
    MonteCarloExpectedImprovement,
    MultipleOptimismNegativeLowerConfidenceBound,
    NegativeLowerConfidenceBound,
    ProbabilityOfFeasibility,
    ProbabilityOfImprovement,
    augmented_expected_improvement,
    expected_improvement,
    fast_constraints_feasibility,
    lower_confidence_bound,
    multiple_optimism_lower_confidence_bound,
    probability_below_threshold,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.objectives import Branin
from trieste.space import Box, LinearConstraint, SearchSpace
from trieste.types import Tag, TensorType

# tags
FOO: Tag = "foo"
NA: Tag = ""


def test_probability_of_improvement_builder_builds_pi_using_best_from_model() -> None:
    dataset = Dataset(
        tf.constant([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]]),
    )
    model = QuadraticMeanAndRBFKernel()
    acq_fn = ProbabilityOfImprovement().prepare_acquisition_function(model, dataset=dataset)
    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    expected = probability_below_threshold(model, tf.constant(0.0))(xs)
    npt.assert_allclose(acq_fn(xs), expected)


def test_probability_of_improvement_builder_updates_pi_using_best_from_model() -> None:
    dataset = Dataset(
        tf.constant([[-2.0], [-1.0]]),
        tf.constant([[4.1], [0.9]]),
    )
    model = QuadraticMeanAndRBFKernel()
    acq_fn = ProbabilityOfImprovement().prepare_acquisition_function(model, dataset=dataset)
    assert acq_fn.__call__._get_tracing_count() == 0  # type: ignore
    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    expected = probability_below_threshold(model, tf.constant(1.0))(xs)
    npt.assert_allclose(acq_fn(xs), expected)
    assert acq_fn.__call__._get_tracing_count() == 1  # type: ignore

    new_dataset = Dataset(
        tf.concat([dataset.query_points, tf.constant([[0.0], [1.0], [2.0]])], 0),
        tf.concat([dataset.observations, tf.constant([[0.1], [1.1], [3.9]])], 0),
    )
    updated_acq_fn = ProbabilityOfImprovement().update_acquisition_function(
        acq_fn, model, dataset=new_dataset
    )
    # assert updated_acq_fn == acq_fn
    expected = probability_below_threshold(model, tf.constant(0.0))(xs)
    npt.assert_allclose(updated_acq_fn(xs), expected)
    assert acq_fn.__call__._get_tracing_count() == 1  # type: ignore


def test_probability_of_improvement_builder_raises_for_empty_data() -> None:
    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))

    with pytest.raises(tf.errors.InvalidArgumentError):
        ProbabilityOfImprovement().prepare_acquisition_function(
            QuadraticMeanAndRBFKernel(), dataset=data
        )
    with pytest.raises(tf.errors.InvalidArgumentError):
        ProbabilityOfImprovement().prepare_acquisition_function(QuadraticMeanAndRBFKernel())


@random_seed
@pytest.mark.parametrize("best", [tf.constant([50.0]), Branin.minimum, Branin.minimum * 1.01])
@pytest.mark.parametrize(
    "variance_scale, num_samples_per_point, rtol, atol",
    [
        (0.1, 1000, 0.01, 1e-9),
        (1.0, 50_000, 0.01, 1e-3),
        (10.0, 100_000, 0.01, 1e-2),
        (100.0, 150_000, 0.01, 1e-1),
    ],
)
def test_probability_below_threshold_as_probability_of_improvement(
    variance_scale: float, num_samples_per_point: int, best: tf.Tensor, rtol: float, atol: float
) -> None:
    variance_scale = tf.constant(variance_scale, tf.float64)
    best = tf.cast(best, dtype=tf.float64)[0]

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    kernel = tfp.math.psd_kernels.MaternFiveHalves(variance_scale, length_scale=0.25)
    model = GaussianProcess([Branin.objective], [kernel])

    mean, variance = model.predict(xs)
    samples = tfp.distributions.Normal(mean, tf.sqrt(variance)).sample(num_samples_per_point)
    samples_improvement = tf.where(samples < best, 1, 0)
    pi_approx = tf.reduce_sum(samples_improvement, axis=0) / num_samples_per_point

    pif = probability_below_threshold(model, best)
    pi = pif(xs[..., None, :])

    npt.assert_allclose(pi, pi_approx, rtol=rtol, atol=atol)


def test_expected_improvement_builder_builds_expected_improvement_using_best_from_model() -> None:
    dataset = Dataset(
        tf.constant([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]]),
    )
    model = QuadraticMeanAndRBFKernel()
    acq_fn = ExpectedImprovement().prepare_acquisition_function(model, dataset=dataset)
    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    expected = expected_improvement(model, tf.constant([0.0]))(xs)
    npt.assert_allclose(acq_fn(xs), expected)


def test_expected_improvement_builder_updates_expected_improvement_using_best_from_model() -> None:
    dataset = Dataset(
        tf.constant([[-2.0], [-1.0]]),
        tf.constant([[4.1], [0.9]]),
    )
    model = QuadraticMeanAndRBFKernel()
    acq_fn = ExpectedImprovement().prepare_acquisition_function(model, dataset=dataset)
    assert acq_fn.__call__._get_tracing_count() == 0  # type: ignore
    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    expected = expected_improvement(model, tf.constant([1.0]))(xs)
    npt.assert_allclose(acq_fn(xs), expected)
    assert acq_fn.__call__._get_tracing_count() == 1  # type: ignore

    new_dataset = Dataset(
        tf.concat([dataset.query_points, tf.constant([[0.0], [1.0], [2.0]])], 0),
        tf.concat([dataset.observations, tf.constant([[0.1], [1.1], [3.9]])], 0),
    )
    updated_acq_fn = ExpectedImprovement().update_acquisition_function(
        acq_fn, model, dataset=new_dataset
    )
    assert updated_acq_fn == acq_fn
    expected = expected_improvement(model, tf.constant([0.0]))(xs)
    npt.assert_allclose(acq_fn(xs), expected)
    assert acq_fn.__call__._get_tracing_count() == 1  # type: ignore


def test_expected_improvement_builder_raises_for_empty_data() -> None:
    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))

    with pytest.raises(tf.errors.InvalidArgumentError):
        ExpectedImprovement().prepare_acquisition_function(
            QuadraticMeanAndRBFKernel(), dataset=data
        )
    with pytest.raises(tf.errors.InvalidArgumentError):
        ExpectedImprovement().prepare_acquisition_function(QuadraticMeanAndRBFKernel())


def test_expected_improvement_is_relative_to_feasible_point() -> None:
    search_space = Box([-1.0], [1.0], [LinearConstraint(A=tf.constant([[1.0]]), lb=0.0, ub=1.0)])

    full_data = Dataset(tf.constant([[-0.2], [0.3]]), tf.constant([[0.04], [0.09]]))
    full_ei = ExpectedImprovement(search_space).prepare_acquisition_function(
        QuadraticMeanAndRBFKernel(),
        dataset=full_data,
    )

    filtered_data = Dataset(tf.constant([[0.3]]), tf.constant([[0.09]]))
    filtered_ei = ExpectedImprovement().prepare_acquisition_function(
        QuadraticMeanAndRBFKernel(), dataset=filtered_data
    )

    npt.assert_allclose(full_ei(tf.constant([[0.1]])), filtered_ei(tf.constant([[0.1]])))


def test_expected_improvement_uses_max_when_no_feasible_points() -> None:
    search_space = Box([-2.5], [2.5], [LinearConstraint(A=tf.constant([[1.0]]), lb=0.5, ub=0.9)])
    data = Dataset(
        tf.constant([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]]),
    )
    builder = ExpectedImprovement(search_space)
    ei = builder.prepare_acquisition_function(
        QuadraticMeanAndRBFKernel(),
        dataset=data,
    )

    filtered_data = Dataset(tf.constant([[-2.0]]), tf.constant([[4.1]]))
    filtered_ei = ExpectedImprovement().prepare_acquisition_function(
        QuadraticMeanAndRBFKernel(), dataset=filtered_data
    )

    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    npt.assert_allclose(ei(xs), filtered_ei(xs))

    ei = builder.update_acquisition_function(
        ei,
        QuadraticMeanAndRBFKernel(),
        dataset=data,
    )
    npt.assert_allclose(ei(xs), filtered_ei(xs))


def test_expected_improvement_switches_to_improvement_on_feasible_points() -> None:
    search_space = Box([0.0], [1.0], [LinearConstraint(A=tf.constant([[1.0]]), lb=0.5, ub=0.9)])
    data = Dataset(tf.constant([[0.2], [1.0]]), tf.constant([[4.0], [1.0]]))
    builder = ExpectedImprovement(search_space)
    ei = builder.prepare_acquisition_function(
        QuadraticMeanAndRBFKernel(),
        dataset=data,
    )

    data = Dataset(tf.constant([[0.2], [0.7]]), tf.constant([[4.0], [1.0]]))
    ei = builder.update_acquisition_function(
        ei,
        QuadraticMeanAndRBFKernel(),
        dataset=data,
    )

    filtered_data = Dataset(tf.constant([[0.7]]), tf.constant([[1.0]]))
    filtered_ei = ExpectedImprovement().prepare_acquisition_function(
        QuadraticMeanAndRBFKernel(), dataset=filtered_data
    )

    npt.assert_allclose(ei(tf.constant([[0.1]])), filtered_ei(tf.constant([[0.1]])))


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_expected_improvement_raises_for_invalid_batch_size(at: TensorType) -> None:
    ei = expected_improvement(QuadraticMeanAndRBFKernel(), tf.constant([1.0]))

    with pytest.raises(StagingError):
        ei(at)


@random_seed
@pytest.mark.parametrize("best", [tf.constant([50.0]), Branin.minimum, Branin.minimum * 1.01])
@pytest.mark.parametrize("test_update", [False, True])
@pytest.mark.parametrize(
    "variance_scale, num_samples_per_point, rtol, atol",
    [
        (0.1, 1000, 0.01, 1e-9),
        (1.0, 50_000, 0.01, 1e-3),
        (10.0, 100_000, 0.01, 1e-2),
        (100.0, 150_000, 0.01, 1e-1),
    ],
)
def test_expected_improvement(
    variance_scale: float,
    num_samples_per_point: int,
    best: tf.Tensor,
    rtol: float,
    atol: float,
    test_update: bool,
) -> None:
    variance_scale = tf.constant(variance_scale, tf.float64)
    best = tf.cast(best, dtype=tf.float64)

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    kernel = tfp.math.psd_kernels.MaternFiveHalves(variance_scale, length_scale=0.25)
    model = GaussianProcess([Branin.objective], [kernel])

    mean, variance = model.predict(xs)
    samples = tfp.distributions.Normal(mean, tf.sqrt(variance)).sample(num_samples_per_point)
    samples_improvement = tf.where(samples < best, best - samples, 0)
    ei_approx = tf.reduce_mean(samples_improvement, axis=0)

    if test_update:
        eif = expected_improvement(model, tf.constant([100.0], dtype=tf.float64))
        eif.update(best)
    else:
        eif = expected_improvement(model, best)
    ei = eif(xs[..., None, :])

    npt.assert_allclose(ei, ei_approx, rtol=rtol, atol=atol)


def test_augmented_expected_improvement_builder_raises_for_empty_data() -> None:
    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))

    with pytest.raises(tf.errors.InvalidArgumentError):
        AugmentedExpectedImprovement().prepare_acquisition_function(
            QuadraticMeanAndRBFKernel(),
            dataset=data,
        )
    with pytest.raises(tf.errors.InvalidArgumentError):
        AugmentedExpectedImprovement().prepare_acquisition_function(QuadraticMeanAndRBFKernel())


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_augmented_expected_improvement_raises_for_invalid_batch_size(at: TensorType) -> None:
    aei = augmented_expected_improvement(QuadraticMeanAndRBFKernel(), tf.constant([1.0]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        aei(at)


@pytest.mark.parametrize("observation_noise", [1e-8, 1.0, 10.0])
def test_augmented_expected_improvement_builder_builds_expected_improvement_times_augmentation(
    observation_noise: float,
) -> None:
    dataset = Dataset(
        tf.constant([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]]),
    )

    model = QuadraticMeanAndRBFKernel(noise_variance=observation_noise)
    acq_fn = AugmentedExpectedImprovement().prepare_acquisition_function(model, dataset=dataset)

    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    ei = ExpectedImprovement().prepare_acquisition_function(model, dataset=dataset)(xs)

    @tf.function
    def augmentation() -> TensorType:
        _, variance = model.predict(tf.squeeze(xs, -2))
        return 1.0 - (tf.math.sqrt(observation_noise)) / (
            tf.math.sqrt(observation_noise + variance)
        )

    npt.assert_allclose(acq_fn(xs), ei * augmentation(), rtol=1e-6)


@pytest.mark.parametrize("observation_noise", [1e-8, 1.0, 10.0])
def test_augmented_expected_improvement_builder_updates_acquisition_function(
    observation_noise: float,
) -> None:
    partial_dataset = Dataset(
        tf.constant([[-2.0], [-1.0]]),
        tf.constant([[4.1], [0.9]]),
    )
    full_dataset = Dataset(
        tf.constant([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]]),
    )
    model = QuadraticMeanAndRBFKernel(noise_variance=observation_noise)

    partial_data_acq_fn = AugmentedExpectedImprovement().prepare_acquisition_function(
        model,
        dataset=partial_dataset,
    )
    updated_acq_fn = AugmentedExpectedImprovement().update_acquisition_function(
        partial_data_acq_fn,
        model,
        dataset=full_dataset,
    )
    assert updated_acq_fn == partial_data_acq_fn
    full_data_acq_fn = AugmentedExpectedImprovement().prepare_acquisition_function(
        model, dataset=full_dataset
    )

    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    npt.assert_allclose(updated_acq_fn(xs), full_data_acq_fn(xs))


@pytest.mark.parametrize("sample_size", [-2, 0])
def test_mc_expected_improvement_raises_for_invalid_sample_size(sample_size: int) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        MonteCarloExpectedImprovement(sample_size)


def test_mc_expected_improvement_raises_for_invalid_jitter() -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        MonteCarloExpectedImprovement(100, jitter=-1.0)


@random_seed
def test_mc_expected_improvement_builds_expected_improvement_using_best_from_model() -> None:
    dataset = Dataset(
        tf.constant([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]]),
    )
    model = QuadraticMeanAndRBFKernelWithSamplers(dataset)
    acq_fn = MonteCarloExpectedImprovement(int(1e6)).prepare_acquisition_function(model, dataset)
    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    expected = expected_improvement(model, tf.constant([0.0]))(xs)

    npt.assert_allclose(acq_fn(xs), expected, rtol=1e-4, atol=2e-3)


def test_mc_expected_improvement_builder_raises_for_model_without_reparam_sampler() -> None:
    data = Dataset(tf.zeros([1, 1]), tf.ones([1, 1]))
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(1.0)
    noise_variance = 1.0

    with pytest.raises(ValueError, match="MonteCarloExpectedImprovement only supports models *."):
        (
            MonteCarloExpectedImprovement(100).prepare_acquisition_function(
                GaussianProcess([lambda x: quadratic(x)], [kernel], noise_variance),  # type: ignore
                data,
            )
        )


def test_mc_expected_improvement_builder_raises_for_model_with_wrong_event_shape() -> None:
    data = mk_dataset([(0.0, 0.0)], [(0.0, 0.0)])
    matern52 = tfp.math.psd_kernels.MaternFiveHalves(
        amplitude=tf.cast(2.3, tf.float64), length_scale=tf.cast(0.5, tf.float64)
    )
    model = GaussianProcessWithSamplers(
        [lambda x: Branin.objective(x), lambda x: quadratic(x)], [matern52, rbf()]
    )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES, match="Expected model with output *."):
        MonteCarloExpectedImprovement(100).prepare_acquisition_function(model, dataset=data)


def test_mc_expected_improvement_builder_raises_for_empty_data() -> None:
    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES, match="Dataset must be populated."):
        (
            MonteCarloExpectedImprovement(100).prepare_acquisition_function(
                QuadraticMeanAndRBFKernelWithSamplers(data), dataset=data
            )
        )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        (
            MonteCarloExpectedImprovement(100).prepare_acquisition_function(
                QuadraticMeanAndRBFKernelWithSamplers(data)
            )
        )


def test_mc_expected_improvement_updater_raises_for_empty_data() -> None:
    dataset = Dataset(
        tf.constant([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]]),
    )
    model = QuadraticMeanAndRBFKernelWithSamplers(dataset)
    builder = MonteCarloExpectedImprovement(10)
    acq_fn = builder.prepare_acquisition_function(model, dataset)

    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES, match="Dataset must be populated."):
        builder.update_acquisition_function(acq_fn, model, dataset=data)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.update_acquisition_function(acq_fn, model)


@random_seed
@pytest.mark.parametrize("test_update", [False, True])
@pytest.mark.parametrize(
    "variance_scale, num_samples_per_point, rtol, atol",
    [
        (0.1, 25_000, 0.01, 1e-3),
        (1.0, 50_000, 0.01, 2e-3),
        (10.0, 100_000, 0.01, 1e-2),
        (100.0, 150_000, 0.01, 1e-1),
    ],
)
def test_mc_expected_improvement_close_to_expected_improvement(
    variance_scale: float,
    num_samples_per_point: int,
    rtol: float,
    atol: float,
    test_update: bool,
) -> None:
    variance_scale = tf.constant(variance_scale, tf.float64)

    dataset = Dataset(
        tf.constant(
            [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=tf.float64
        ),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]], dtype=tf.float64),
    )

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    kernel = tfp.math.psd_kernels.MaternFiveHalves(variance_scale, length_scale=0.25)
    model = GaussianProcessWithSamplers([Branin.objective], [kernel])

    if test_update:
        builder = MonteCarloExpectedImprovement(num_samples_per_point)

        init_data = Dataset(
            tf.constant([[0.1, 0.1]], dtype=tf.float64), tf.constant([[100.0]], dtype=tf.float64)
        )
        eif = builder.prepare_acquisition_function(model, init_data)
        eif = builder.update_acquisition_function(eif, model, dataset)
    else:
        eif = MonteCarloExpectedImprovement(num_samples_per_point).prepare_acquisition_function(
            model, dataset
        )
    ei_approx = eif(xs[..., None, :])

    best = tf.reduce_min(Branin.objective(dataset.query_points))
    eif = expected_improvement(model, best)
    ei = eif(xs[..., None, :])  # type: ignore

    npt.assert_allclose(ei, ei_approx, rtol=rtol, atol=atol)


@random_seed
def test_mc_expected_improvement_updates_without_retracing() -> None:
    known_query_points = tf.random.uniform([10, 2], dtype=tf.float64)
    data = Dataset(known_query_points[8:], quadratic(known_query_points[8:]))
    model = QuadraticMeanAndRBFKernelWithSamplers(dataset=data)
    builder = MonteCarloExpectedImprovement(10_000)
    ei = ExpectedImprovement().prepare_acquisition_function(model, dataset=data)
    xs = tf.random.uniform([5, 1, 2], dtype=tf.float64)

    mcei = builder.prepare_acquisition_function(model, dataset=data)
    assert mcei.__call__._get_tracing_count() == 0  # type: ignore
    npt.assert_allclose(mcei(xs), ei(xs), rtol=0.06)
    assert mcei.__call__._get_tracing_count() == 1  # type: ignore

    data = Dataset(known_query_points, quadratic(known_query_points))
    up_mcei = builder.update_acquisition_function(mcei, model, dataset=data)
    ei = ExpectedImprovement().prepare_acquisition_function(model, dataset=data)
    assert up_mcei == mcei
    assert mcei.__call__._get_tracing_count() == 1  # type: ignore
    npt.assert_allclose(mcei(xs), ei(xs), rtol=0.06)
    assert mcei.__call__._get_tracing_count() == 1  # type: ignore


@pytest.mark.parametrize("sample_size", [-2, 0])
def test_mc_augmented_expected_improvement_raises_for_invalid_sample_size(sample_size: int) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        MonteCarloAugmentedExpectedImprovement(sample_size)


def test_mc_augmented_expected_improvement_raises_for_invalid_jitter() -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        MonteCarloAugmentedExpectedImprovement(100, jitter=-1.0)


@random_seed
def test_mc_augmented_expected_improvement_builds_aei_using_best_from_model() -> None:
    dataset = Dataset(
        tf.constant([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]]),
    )
    model = QuadraticMeanAndRBFKernelWithSamplers(dataset)
    acq_fn = MonteCarloAugmentedExpectedImprovement(int(1e6)).prepare_acquisition_function(
        model, dataset
    )
    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    expected = augmented_expected_improvement(model, tf.constant([0.0]))(xs)
    npt.assert_allclose(acq_fn(xs), expected, rtol=1e-4, atol=2e-3)


def test_mc_augmented_expected_improvement_raises_for_invalid_models() -> None:
    data = Dataset(tf.zeros([1, 1]), tf.ones([1, 1]))
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(1.0)
    noise_variance = 1.0

    with pytest.raises(
        ValueError, match="MonteCarloAugmentedExpectedImprovement only supports models .*"
    ):
        (
            MonteCarloAugmentedExpectedImprovement(100).prepare_acquisition_function(
                GaussianProcess([lambda x: quadratic(x)], [kernel], noise_variance),  # type: ignore
                data,
            )
        )
    with pytest.raises(
        ValueError, match="MonteCarloAugmentedExpectedImprovement only supports models .*"
    ):
        (
            MonteCarloAugmentedExpectedImprovement(100).prepare_acquisition_function(
                GaussianProcessWithoutNoise([lambda x: quadratic(x)], [kernel]),  # type: ignore
                data,
            )
        )


def test_mc_augmented_expected_improvement_builder_raises_for_empty_data() -> None:
    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES, match="Dataset must be populated."):
        (
            MonteCarloAugmentedExpectedImprovement(100).prepare_acquisition_function(
                QuadraticMeanAndRBFKernelWithSamplers(data), dataset=data
            )
        )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        (
            MonteCarloAugmentedExpectedImprovement(100).prepare_acquisition_function(
                QuadraticMeanAndRBFKernelWithSamplers(data)
            )
        )


def test_mc_augmented_expected_improvement_updater_raises_for_empty_data() -> None:
    dataset = Dataset(
        tf.constant([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]]),
    )
    model = QuadraticMeanAndRBFKernelWithSamplers(dataset)
    builder = MonteCarloAugmentedExpectedImprovement(10)
    acq_fn = builder.prepare_acquisition_function(model, dataset)

    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES, match="Dataset must be populated."):
        builder.update_acquisition_function(acq_fn, model, dataset=data)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.update_acquisition_function(acq_fn, model)


@random_seed
@pytest.mark.parametrize("test_update", [False, True])
@pytest.mark.parametrize(
    "variance_scale, noise_variance, num_samples_per_point, rtol, atol",
    [
        (0.1, 1e-4, 150_000, 0.01, 1e-3),
        (1.0, 1e-4, 150_000, 0.01, 1e-3),
        (10.0, 1e-4, 150_000, 0.01, 2e-3),
        (100.0, 1e-4, 150_000, 0.01, 2e-2),
        (0.1, 1e-3, 150_000, 0.01, 1e-3),
        (1.0, 1e-3, 150_000, 0.01, 1e-3),
        (10.0, 1e-3, 150_000, 0.01, 2e-3),
        (100.0, 1e-3, 150_000, 0.01, 2e-2),
    ],
)
def test_mc_augmented_expected_improvement_close_to_augmented_expected_improvement(
    variance_scale: float,
    noise_variance: float,
    num_samples_per_point: int,
    rtol: float,
    atol: float,
    test_update: bool,
) -> None:
    variance_scale = tf.constant(variance_scale, tf.float64)

    dataset = Dataset(
        tf.constant(
            [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=tf.float64
        ),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]], dtype=tf.float64),
    )

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    kernel = tfp.math.psd_kernels.MaternFiveHalves(variance_scale, length_scale=0.25)
    model = GaussianProcessWithSamplers(
        [Branin.objective], [kernel], noise_variance=tf.constant(noise_variance, tf.float64)
    )

    if test_update:
        init_data = Dataset(
            tf.constant([[0.1, 0.1]], dtype=tf.float64), tf.constant([[100.0]], dtype=tf.float64)
        )
        builder = MonteCarloAugmentedExpectedImprovement(num_samples_per_point)
        aeif = builder.prepare_acquisition_function(model, init_data)
        model._noise_variance = tf.constant(noise_variance, tf.float64)
        aeif = builder.update_acquisition_function(aeif, model, dataset)
    else:
        aeif = MonteCarloAugmentedExpectedImprovement(
            num_samples_per_point
        ).prepare_acquisition_function(model, dataset)
    aei_approx = aeif(xs[..., None, :])

    best = tf.reduce_min(Branin.objective(dataset.query_points))
    aeif = augmented_expected_improvement(model, best)
    aei = aeif(xs[..., None, :])  # type: ignore

    npt.assert_allclose(aei, aei_approx, rtol=rtol, atol=atol)


@random_seed
def test_mc_augmented_expected_improvement_updates_without_retracing() -> None:
    known_query_points = tf.random.uniform([10, 2], dtype=tf.float64)
    data = Dataset(known_query_points[8:], quadratic(known_query_points[8:]))
    model = QuadraticMeanAndRBFKernelWithSamplers(dataset=data)
    model._noise_variance = tf.cast(model.get_observation_noise(), tf.float64)
    builder = MonteCarloAugmentedExpectedImprovement(10_000)
    aei = AugmentedExpectedImprovement().prepare_acquisition_function(model, dataset=data)
    xs = tf.random.uniform([5, 1, 2], dtype=tf.float64)

    mcaei = builder.prepare_acquisition_function(model, dataset=data)
    assert mcaei.__call__._get_tracing_count() == 0  # type: ignore
    npt.assert_allclose(mcaei(xs), aei(xs), rtol=0.06)
    assert mcaei.__call__._get_tracing_count() == 1  # type: ignore

    data = Dataset(known_query_points, quadratic(known_query_points))
    up_mcaei = builder.update_acquisition_function(mcaei, model, dataset=data)
    aei = AugmentedExpectedImprovement().prepare_acquisition_function(model, dataset=data)
    assert up_mcaei == mcaei
    assert mcaei.__call__._get_tracing_count() == 1  # type: ignore
    npt.assert_allclose(mcaei(xs), aei(xs), rtol=0.06)
    assert mcaei.__call__._get_tracing_count() == 1  # type: ignore


def test_negative_lower_confidence_bound_builder_builds_negative_lower_confidence_bound() -> None:
    model = QuadraticMeanAndRBFKernel()
    beta = 1.96
    acq_fn = NegativeLowerConfidenceBound(beta).prepare_acquisition_function(model)
    query_at = tf.linspace([[-10]], [[10]], 100)
    expected = -lower_confidence_bound(model, beta)(query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)


def test_negative_lower_confidence_bound_builder_updates_without_retracing() -> None:
    model = QuadraticMeanAndRBFKernel()
    beta = 1.96
    builder = NegativeLowerConfidenceBound(beta)
    acq_fn = builder.prepare_acquisition_function(model)
    assert acq_fn._get_tracing_count() == 0  # type: ignore
    query_at = tf.linspace([[-10]], [[10]], 100)
    expected = -lower_confidence_bound(model, beta)(query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)
    assert acq_fn._get_tracing_count() == 1  # type: ignore

    up_acq_fn = builder.update_acquisition_function(acq_fn, model)
    assert up_acq_fn == acq_fn
    npt.assert_array_almost_equal(acq_fn(query_at), expected)
    assert acq_fn._get_tracing_count() == 1  # type: ignore


@pytest.mark.parametrize("beta", [-0.1, -2.0])
def test_lower_confidence_bound_raises_for_negative_beta(beta: float) -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        lower_confidence_bound(QuadraticMeanAndRBFKernel(), beta)


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_lower_confidence_bound_raises_for_invalid_batch_size(at: TensorType) -> None:
    lcb = lower_confidence_bound(QuadraticMeanAndRBFKernel(), tf.constant(1.0))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        lcb(at)


@pytest.mark.parametrize("beta", [0.0, 0.1, 7.8])
def test_lower_confidence_bound(beta: float) -> None:
    query_at = tf.linspace([[-3]], [[3]], 10)
    actual = lower_confidence_bound(QuadraticMeanAndRBFKernel(), beta)(query_at)
    npt.assert_array_almost_equal(actual, tf.squeeze(query_at, -2) ** 2 - beta)


@pytest.mark.parametrize(
    "threshold, at, expected",
    [
        (0.0, tf.constant([[0.0]]), 0.5),
        # values looked up on a standard normal table
        (2.0, tf.constant([[1.0]]), 0.5 + 0.34134),
        (-0.25, tf.constant([[-0.5]]), 0.5 - 0.19146),
    ],
)
def test_probability_below_threshold_as_probability_of_feasibility(
    threshold: float, at: tf.Tensor, expected: float
) -> None:
    actual = probability_below_threshold(QuadraticMeanAndRBFKernel(), threshold)(at)
    npt.assert_allclose(actual, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "at",
    [
        tf.constant([[0.0]], tf.float64),
        tf.constant([[-3.4]], tf.float64),
        tf.constant([[0.2]], tf.float64),
    ],
)
@pytest.mark.parametrize("threshold", [-2.3, 0.2])
def test_probability_of_feasibility_builder_builds_pof(threshold: float, at: tf.Tensor) -> None:
    builder = ProbabilityOfFeasibility(threshold)
    acq = builder.prepare_acquisition_function(QuadraticMeanAndRBFKernel())
    expected = probability_below_threshold(QuadraticMeanAndRBFKernel(), threshold)(at)

    npt.assert_allclose(acq(at), expected)


@pytest.mark.parametrize("shape", various_shapes() - {()})
def test_probability_below_threshold_raises_on_non_scalar_threshold(shape: ShapeLike) -> None:
    threshold = tf.ones(shape)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        probability_below_threshold(QuadraticMeanAndRBFKernel(), threshold)


@pytest.mark.parametrize("shape", [[], [0], [2], [2, 1], [1, 2, 1]])
def test_probability_below_threshold_raises_on_invalid_at_shape(shape: ShapeLike) -> None:
    at = tf.ones(shape)
    pof = probability_below_threshold(QuadraticMeanAndRBFKernel(), 0.0)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        pof(at)


@pytest.mark.parametrize("shape", various_shapes() - {()})
def test_probability_of_feasibility_builder_raises_on_non_scalar_threshold(
    shape: ShapeLike,
) -> None:
    threshold = tf.ones(shape)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ProbabilityOfFeasibility(threshold)


@pytest.mark.parametrize("at", [tf.constant([[0.0]], tf.float64)])
@pytest.mark.parametrize("threshold", [-2.3, 0.2])
def test_probability_of_feasibility_builder_updates_without_retracing(
    threshold: float, at: tf.Tensor
) -> None:
    builder = ProbabilityOfFeasibility(threshold)
    model = QuadraticMeanAndRBFKernel()
    expected = probability_below_threshold(QuadraticMeanAndRBFKernel(), threshold)(at)
    acq = builder.prepare_acquisition_function(model)
    assert acq.__call__._get_tracing_count() == 0  # type: ignore
    npt.assert_allclose(acq(at), expected)
    assert acq.__call__._get_tracing_count() == 1  # type: ignore
    up_acq = builder.update_acquisition_function(acq, model)
    assert up_acq == acq
    npt.assert_allclose(acq(at), expected)
    assert acq.__call__._get_tracing_count() == 1  # type: ignore


def _box_feasibility_constraints() -> Sequence[LinearConstraint]:
    return [LinearConstraint(A=tf.eye(3), lb=tf.zeros((3)) + 0.3, ub=tf.ones((3)) - 0.3)]


@pytest.mark.parametrize(
    "smoother, expected",
    [
        (None, tf.constant([1.0, 0.0, 0.0, 1.0])),
        (tfp.distributions.Normal(0.0, 0.1).cdf, tf.constant([0.871, 0.029, 0.029, 0.462])),
        (tf.math.sigmoid, tf.constant([0.028, 0.026, 0.026, 0.027])),
    ],
)
def test_fast_constraints_feasibility_smoothing_values(
    smoother: Optional[Callable[[TensorType], TensorType]],
    expected: TensorType,
) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), _box_feasibility_constraints())
    points = box.sample_sobol(4, skip=0)
    acq = fast_constraints_feasibility(box, smoother)
    got = tf.squeeze(acq(points))

    npt.assert_allclose(got, expected, atol=0.005)


def test_fast_constraints_feasibility_builder_builds_same_func() -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), _box_feasibility_constraints())
    points = box.sample_sobol(4)
    builder = FastConstraintsFeasibility(box)
    acq = builder.prepare_acquisition_function(QuadraticMeanAndRBFKernel())
    expected = fast_constraints_feasibility(box)(points)

    npt.assert_allclose(acq(points), expected)


def test_fast_constraints_feasibility_builder_updates_without_retracing() -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), _box_feasibility_constraints())
    points = box.sample_sobol(4)
    builder = FastConstraintsFeasibility(box)
    expected = fast_constraints_feasibility(box)(points)
    acq = builder.prepare_acquisition_function(QuadraticMeanAndRBFKernel())
    assert acq._get_tracing_count() == 0  # type: ignore
    npt.assert_allclose(acq(points), expected)
    assert acq._get_tracing_count() == 1  # type: ignore
    up_acq = builder.update_acquisition_function(acq, QuadraticMeanAndRBFKernel())
    assert up_acq == acq

    points = box.sample_sobol(4)
    expected = fast_constraints_feasibility(box)(points)
    npt.assert_allclose(acq(points), expected)
    assert acq._get_tracing_count() == 1  # type: ignore


def test_fast_constraints_feasibility_raises_without_constraints() -> None:
    box = Box(tf.zeros((2)), tf.ones((2)))
    with pytest.raises(NotImplementedError):
        _ = FastConstraintsFeasibility(box)
    with pytest.raises(NotImplementedError):
        _ = fast_constraints_feasibility(box)


def test_expected_constrained_improvement_raises_for_non_scalar_min_pof() -> None:
    pof = ProbabilityOfFeasibility(0.0).using(NA)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ExpectedConstrainedImprovement(NA, pof, tf.constant([0.0]))


def test_expected_constrained_improvement_raises_for_out_of_range_min_pof() -> None:
    pof = ProbabilityOfFeasibility(0.0).using(NA)
    with pytest.raises(tf.errors.InvalidArgumentError):
        ExpectedConstrainedImprovement(NA, pof, 1.5)


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_expected_constrained_improvement_raises_for_invalid_batch_size(at: TensorType) -> None:
    pof = ProbabilityOfFeasibility(0.0).using(NA)
    builder = ExpectedConstrainedImprovement(NA, pof, tf.constant(0.0))
    initial_query_points = tf.constant([[-1.0]])
    initial_objective_function_values = tf.constant([[1.0]])
    data = {NA: Dataset(initial_query_points, initial_objective_function_values)}

    eci = builder.prepare_acquisition_function({NA: QuadraticMeanAndRBFKernel()}, datasets=data)

    with pytest.raises(StagingError):
        eci(at)


def test_expected_constrained_improvement_can_reproduce_expected_improvement() -> None:
    class _Certainty(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[Tag, ProbabilisticModel],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
        ) -> AcquisitionFunction:
            return lambda x: tf.ones_like(tf.squeeze(x, -2))

    data = {FOO: Dataset(tf.constant([[0.5]]), tf.constant([[0.25]]))}
    models_ = {FOO: QuadraticMeanAndRBFKernel()}

    builder = ExpectedConstrainedImprovement(FOO, _Certainty(), 0)
    eci = builder.prepare_acquisition_function(models_, datasets=data)

    ei = ExpectedImprovement().using(FOO).prepare_acquisition_function(models_, datasets=data)

    at = tf.constant([[[-0.1]], [[1.23]], [[-6.78]]])
    npt.assert_allclose(eci(at), ei(at))

    new_data = {FOO: Dataset(tf.constant([[0.5], [1.0]]), tf.constant([[0.25], [0.5]]))}
    up_eci = builder.update_acquisition_function(eci, models_, datasets=new_data)
    assert up_eci == eci
    up_ei = (
        ExpectedImprovement().using(FOO).prepare_acquisition_function(models_, datasets=new_data)
    )

    npt.assert_allclose(eci(at), up_ei(at))
    assert eci._get_tracing_count() == 1  # type: ignore


@pytest.mark.parametrize(
    "search_space, dataset",
    [
        (None, Dataset(tf.constant([[-0.2], [0.3]]), tf.constant([[0.04], [0.09]]))),
        (
            Box([-1.0], [1.0], [LinearConstraint(A=tf.constant([[1.0]]), lb=0.25, ub=1.0)]),
            Dataset(tf.constant([[-0.2], [0.2], [0.3]]), tf.constant([[0.04], [0.04], [0.09]])),
        ),
    ],
)
def test_expected_constrained_improvement_is_relative_to_feasible_point(
    search_space: SearchSpace, dataset: Dataset
) -> None:
    class _Constraint(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[Tag, ProbabilisticModel],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
        ) -> AcquisitionFunction:
            return lambda x: tf.cast(tf.squeeze(x, -2) >= 0, x.dtype)

    models_ = {FOO: QuadraticMeanAndRBFKernel()}

    eci_data = {FOO: Dataset(tf.constant([[-0.2], [0.3]]), tf.constant([[0.04], [0.09]]))}
    eci = ExpectedConstrainedImprovement(
        FOO, _Constraint(), search_space=search_space
    ).prepare_acquisition_function(
        models_,
        datasets=eci_data,
    )

    ei_data = {FOO: Dataset(tf.constant([[0.3]]), tf.constant([[0.09]]))}
    ei = ExpectedImprovement().using(FOO).prepare_acquisition_function(models_, datasets=ei_data)

    npt.assert_allclose(eci(tf.constant([[0.1]])), ei(tf.constant([[0.1]])))


def test_expected_constrained_improvement_is_less_for_constrained_points() -> None:
    class _Constraint(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[Tag, ProbabilisticModel],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
        ) -> AcquisitionFunction:
            return lambda x: tf.cast(tf.squeeze(x, -2) >= 0, x.dtype)

    def two_global_minima(x: tf.Tensor) -> tf.Tensor:
        return x**4 / 4 - x**2 / 2

    initial_query_points = tf.constant([[-2.0], [0.0], [1.2]])
    data = {FOO: Dataset(initial_query_points, two_global_minima(initial_query_points))}
    models_ = {FOO: GaussianProcess([two_global_minima], [rbf()])}

    eci = ExpectedConstrainedImprovement(FOO, _Constraint()).prepare_acquisition_function(
        models_,
        datasets=data,
    )

    npt.assert_array_less(eci(tf.constant([[-1.0]])), eci(tf.constant([[1.0]])))


def test_expected_constrained_improvement_raises_for_empty_data() -> None:
    class _Constraint(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[Tag, ProbabilisticModel],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
        ) -> AcquisitionFunction:
            return raise_exc

    data = {FOO: Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))}
    models_ = {FOO: QuadraticMeanAndRBFKernel()}
    builder = ExpectedConstrainedImprovement(FOO, _Constraint())

    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(models_, datasets=data)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(models_)


@pytest.mark.parametrize(
    "search_space, observations",
    [
        (None, tf.constant([[-2.0], [1.0]])),
        (Box([-2.0], [1.0]), tf.constant([[-2.0], [1.0]])),
        (
            Box([-2.0], [1.0], [LinearConstraint(A=tf.constant([[1.0]]), lb=0.5, ub=1.0)]),
            tf.constant([[0.2], [1.0]]),
        ),
    ],
)
def test_expected_constrained_improvement_is_constraint_when_no_feasible_points(
    search_space: SearchSpace, observations: TensorType
) -> None:
    class _Constraint(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[Tag, ProbabilisticModel],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
        ) -> AcquisitionFunction:
            def acquisition(x: TensorType) -> TensorType:
                x_ = tf.squeeze(x, -2)
                return tf.cast(tf.logical_and(0.0 <= x_, x_ < 1.0), x.dtype)

            return acquisition

    data = {FOO: Dataset(tf.constant([[-2.0], [1.0]]), tf.constant([[4.0], [1.0]]))}
    models_ = {FOO: QuadraticMeanAndRBFKernel()}
    eci = ExpectedConstrainedImprovement(
        FOO, _Constraint(), search_space=search_space
    ).prepare_acquisition_function(
        models_,
        datasets=data,
    )

    constraint_fn = _Constraint().prepare_acquisition_function(models_, datasets=data)

    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    npt.assert_allclose(eci(xs), constraint_fn(xs))


def test_expected_constrained_improvement_min_feasibility_probability_bound_is_inclusive() -> None:
    def pof(x_: TensorType) -> TensorType:
        return tfp.bijectors.Sigmoid().forward(tf.squeeze(x_, -2))

    class _Constraint(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[Tag, ProbabilisticModel],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
        ) -> AcquisitionFunction:
            return pof

    models_ = {FOO: QuadraticMeanAndRBFKernel()}

    data = {FOO: Dataset(tf.constant([[1.1], [2.0]]), tf.constant([[1.21], [4.0]]))}
    eci = ExpectedConstrainedImprovement(
        FOO, _Constraint(), min_feasibility_probability=tfp.bijectors.Sigmoid().forward(1.0)
    ).prepare_acquisition_function(
        models_,
        datasets=data,
    )

    ei = ExpectedImprovement().using(FOO).prepare_acquisition_function(models_, datasets=data)
    x = tf.constant([[1.5]])
    npt.assert_allclose(eci(x), ei(x) * pof(x))


@pytest.mark.parametrize("sample_size", [-2, 0])
def test_batch_expected_improvement_raises_for_invalid_sample_size(
    sample_size: int,
) -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        BatchExpectedImprovement(sample_size=sample_size)


@pytest.mark.parametrize("sample_size", [2])
@pytest.mark.parametrize("jitter", [-1e0])
def test_batch_expected_improvement_raises_for_invalid_jitter(
    sample_size: int,
    jitter: float,
) -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        BatchExpectedImprovement(sample_size=sample_size, jitter=jitter)


@pytest.mark.parametrize("sample_size", [100])
@pytest.mark.parametrize("jitter", [1e-6])
def test_batch_expected_improvement_raises_for_empty_data(
    sample_size: int,
    jitter: float,
) -> None:
    builder = BatchExpectedImprovement(
        sample_size=sample_size,
        jitter=jitter,
    )
    data = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))
    matern52 = tfp.math.psd_kernels.MaternFiveHalves(
        amplitude=tf.cast(2.3, tf.float64), length_scale=tf.cast(0.5, tf.float64)
    )
    model = GaussianProcessWithBatchSamplers(
        [lambda x: Branin.objective(x), lambda x: quadratic(x)], [matern52, rbf()]
    )
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(model, dataset=data)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(model)


@pytest.mark.parametrize("num_data", [4, 8])
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("dimension", [2, 4])
@random_seed
def test_batch_expected_improvement_can_reproduce_mc_excpected_improvement_handcrafted_problem(
    num_data: int,
    batch_size: int,
    dimension: int,
    jitter: float = 1e-6,
    sample_size: int = 200,
    mc_sample_size: int = 100000,
) -> None:
    xs = tf.random.uniform([num_data, dimension], dtype=tf.float64)

    data = Dataset(xs, quadratic(xs))
    model = QuadraticMeanAndRBFKernelWithBatchSamplers(dataset=data)
    mean, cov = model.predict_joint(xs)

    mvn = tfp.distributions.MultivariateNormalFullCovariance(tf.linalg.matrix_transpose(mean), cov)
    mvn_samples = mvn.sample(10000)

    dummy_inputs = [dimension * [0.1]]
    dummy_outputs = [dimension * [0.1**2.0]]

    min_predictive_mean_at_known_points = dimension * 0.1**2.0

    # fmt: off
    expected = tf.reduce_mean(tf.reduce_max(tf.maximum(
        min_predictive_mean_at_known_points - mvn_samples, 0.0
    ), axis=-1), axis=0)
    # fmt: on

    builder = BatchMonteCarloExpectedImprovement(10_000)
    acq = builder.prepare_acquisition_function(
        model, dataset=mk_dataset(dummy_inputs, dummy_outputs)
    )

    npt.assert_allclose(acq(xs), expected, rtol=0.05)


@pytest.mark.parametrize("num_data", [4, 8, 16])
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("dimension", [2, 4, 6])
@random_seed
def test_batch_expected_improvement_can_reproduce_mc_excpected_improvement_random_problems(
    num_data: int,
    batch_size: int,
    dimension: int,
    jitter: float = 1e-6,
    sample_size: int = 200,
    mc_sample_size: int = 100000,
    num_parallel: int = 4,
) -> None:
    known_query_points = tf.random.uniform([num_data, dimension], dtype=tf.float64)

    data = Dataset(known_query_points, quadratic(known_query_points))
    model = QuadraticMeanAndRBFKernelWithBatchSamplers(dataset=data)

    batch_ei = BatchExpectedImprovement(
        sample_size=sample_size,
        jitter=jitter,
    ).prepare_acquisition_function(
        model=model,
        dataset=data,
    )

    batch_mcei = BatchMonteCarloExpectedImprovement(
        sample_size=mc_sample_size,
        jitter=jitter,
    ).prepare_acquisition_function(
        model=model,
        dataset=data,
    )

    xs = tf.random.uniform([num_parallel, batch_size, dimension], dtype=tf.float64)

    npt.assert_allclose(batch_mcei(xs), batch_ei(xs), rtol=2e-2)
    # and again, since the sampler uses cacheing
    npt.assert_allclose(batch_mcei(xs), batch_ei(xs), rtol=2e-2)


@pytest.mark.parametrize("num_data", [10])
@pytest.mark.parametrize("num_parallel", [3])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("sample_size", [100])
@pytest.mark.parametrize("dimension", [2])
@pytest.mark.parametrize("jitter", [1e-6])
@pytest.mark.parametrize("mc_sample_size", [int(4e5)])
@random_seed
def test_batch_expected_improvement_updates_without_retracing(
    num_data: int,
    num_parallel: int,
    batch_size: int,
    sample_size: int,
    dimension: int,
    jitter: float,
    mc_sample_size: int,
) -> None:
    known_query_points = tf.random.uniform([num_data, dimension], dtype=tf.float64)
    data = Dataset(
        known_query_points[num_data - 2 :], quadratic(known_query_points[num_data - 2 :])
    )
    model = QuadraticMeanAndRBFKernelWithBatchSamplers(dataset=data)

    batch_ei_builder = BatchExpectedImprovement(
        sample_size=sample_size,
        jitter=jitter,
    )

    batch_mcei_builder = BatchMonteCarloExpectedImprovement(
        sample_size=mc_sample_size,
        jitter=jitter,
    )

    xs = tf.random.uniform([num_parallel, batch_size, dimension], dtype=tf.float64)

    batch_ei = batch_ei_builder.prepare_acquisition_function(model=model, dataset=data)
    batch_mcei = batch_mcei_builder.prepare_acquisition_function(model=model, dataset=data)
    assert batch_ei.__call__._get_tracing_count() == 0  # type: ignore
    npt.assert_allclose(batch_mcei(xs), batch_ei(xs), rtol=2e-2)
    assert batch_ei.__call__._get_tracing_count() == 1  # type: ignore

    data = Dataset(known_query_points, quadratic(known_query_points))
    up_batch_ei = batch_ei_builder.update_acquisition_function(batch_ei, model, dataset=data)
    batch_mcei = batch_mcei_builder.update_acquisition_function(batch_mcei, model, dataset=data)
    assert up_batch_ei == batch_ei
    assert batch_ei.__call__._get_tracing_count() == 1  # type: ignore
    npt.assert_allclose(batch_mcei(xs), batch_ei(xs), rtol=2e-2)
    assert batch_ei.__call__._get_tracing_count() == 1  # type: ignore


@pytest.mark.parametrize("sample_size", [-2, 0])
def test_batch_monte_carlo_expected_improvement_raises_for_invalid_sample_size(
    sample_size: int,
) -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        BatchMonteCarloExpectedImprovement(sample_size)


def test_batch_monte_carlo_expected_improvement_raises_for_invalid_jitter() -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        BatchMonteCarloExpectedImprovement(100, jitter=-1.0)


def test_batch_monte_carlo_expected_improvement_raises_for_empty_data() -> None:
    builder = BatchMonteCarloExpectedImprovement(100)
    data = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))
    matern52 = tfp.math.psd_kernels.MaternFiveHalves(
        amplitude=tf.cast(2.3, tf.float64), length_scale=tf.cast(0.5, tf.float64)
    )
    model = GaussianProcessWithBatchSamplers(
        [lambda x: Branin.objective(x), lambda x: quadratic(x)], [matern52, rbf()]
    )
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(model, dataset=data)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(model)


def test_batch_monte_carlo_expected_improvement_raises_for_model_with_wrong_event_shape() -> None:
    builder = BatchMonteCarloExpectedImprovement(100)
    data = mk_dataset([(0.0, 0.0)], [(0.0, 0.0)])
    matern52 = tfp.math.psd_kernels.MaternFiveHalves(
        amplitude=tf.cast(2.3, tf.float64), length_scale=tf.cast(0.5, tf.float64)
    )
    model = GaussianProcessWithBatchSamplers(
        [lambda x: Branin.objective(x), lambda x: quadratic(x)], [matern52, rbf()]
    )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(model, dataset=data)


@random_seed
def test_batch_monte_carlo_expected_improvement_raises_for_model_without_reparam_sampler() -> None:
    known_query_points = tf.random.uniform([5, 2], dtype=tf.float64)
    data = Dataset(known_query_points, quadratic(known_query_points))
    model = QuadraticMeanAndRBFKernel()
    with pytest.raises(ValueError):
        (
            BatchMonteCarloExpectedImprovement(10_000).prepare_acquisition_function(
                model, dataset=data  # type: ignore
            )
        )


@random_seed
def test_batch_monte_carlo_expected_improvement_can_reproduce_ei() -> None:
    known_query_points = tf.random.uniform([5, 2], dtype=tf.float64)
    data = Dataset(known_query_points, quadratic(known_query_points))
    model = QuadraticMeanAndRBFKernelWithBatchSamplers(dataset=data)
    batch_ei = BatchMonteCarloExpectedImprovement(10_000).prepare_acquisition_function(
        model, dataset=data
    )
    ei = ExpectedImprovement().prepare_acquisition_function(model, dataset=data)
    xs = tf.random.uniform([3, 5, 1, 2], dtype=tf.float64)
    npt.assert_allclose(batch_ei(xs), ei(xs), rtol=0.06)
    # and again, since the sampler uses cacheing
    npt.assert_allclose(batch_ei(xs), ei(xs), rtol=0.06)


@random_seed
def test_batch_monte_carlo_expected_improvement() -> None:
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    data = Dataset(xs, quadratic(xs))
    model = QuadraticMeanAndRBFKernelWithBatchSamplers(dataset=data)
    mean, cov = model.predict_joint(xs)
    mvn = tfp.distributions.MultivariateNormalFullCovariance(tf.linalg.matrix_transpose(mean), cov)
    mvn_samples = mvn.sample(10_000)
    min_predictive_mean_at_known_points = 0.09
    # fmt: off
    expected = tf.reduce_mean(tf.reduce_max(tf.maximum(
        min_predictive_mean_at_known_points - mvn_samples, 0.0
    ), axis=-1), axis=0)
    # fmt: on

    builder = BatchMonteCarloExpectedImprovement(10_000)
    acq = builder.prepare_acquisition_function(
        model, dataset=mk_dataset([[0.3], [0.5]], [[0.09], [0.25]])
    )

    npt.assert_allclose(acq(xs), expected, rtol=0.05)


@random_seed
def test_batch_monte_carlo_expected_improvement_updates_without_retracing() -> None:
    known_query_points = tf.random.uniform([10, 2], dtype=tf.float64)
    data = Dataset(known_query_points[8:], quadratic(known_query_points[8:]))
    model = QuadraticMeanAndRBFKernelWithBatchSamplers(dataset=data)
    builder = BatchMonteCarloExpectedImprovement(10_000)
    ei = ExpectedImprovement().prepare_acquisition_function(model, dataset=data)
    xs = tf.random.uniform([3, 5, 1, 2], dtype=tf.float64)

    batch_ei = builder.prepare_acquisition_function(model, dataset=data)
    assert batch_ei.__call__._get_tracing_count() == 0  # type: ignore
    npt.assert_allclose(batch_ei(xs), ei(xs), rtol=0.06)
    assert batch_ei.__call__._get_tracing_count() == 2  # type: ignore

    data = Dataset(known_query_points, quadratic(known_query_points))
    up_batch_ei = builder.update_acquisition_function(batch_ei, model, dataset=data)
    ei = ExpectedImprovement().update_acquisition_function(ei, model, dataset=data)
    assert up_batch_ei == batch_ei
    assert batch_ei.__call__._get_tracing_count() == 2  # type: ignore
    npt.assert_allclose(batch_ei(xs), ei(xs), rtol=0.06)
    assert batch_ei.__call__._get_tracing_count() == 2  # type: ignore


def test_multiple_optimism_builder_builds_negative_lower_confidence_bound() -> None:
    model = QuadraticMeanAndRBFKernel()
    search_space = Box([0, 0], [1, 1])
    acq_fn = MultipleOptimismNegativeLowerConfidenceBound(
        search_space
    ).prepare_acquisition_function(model)
    query_at = tf.reshape(tf.linspace([[-10]], [[10]], 100), [10, 5, 2])
    expected = multiple_optimism_lower_confidence_bound(model, search_space.dimension)(query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)


def test_multiple_optimism_builder_updates_without_retracing() -> None:
    model = QuadraticMeanAndRBFKernel()
    search_space = Box([0, 0], [1, 1])
    builder = MultipleOptimismNegativeLowerConfidenceBound(search_space)
    acq_fn = builder.prepare_acquisition_function(model)
    assert acq_fn.__call__._get_tracing_count() == 0  # type: ignore
    query_at = tf.reshape(tf.linspace([[-10]], [[10]], 100), [10, 5, 2])
    expected = multiple_optimism_lower_confidence_bound(model, search_space.dimension)(query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)
    assert acq_fn.__call__._get_tracing_count() == 1  # type: ignore

    up_acq_fn = builder.update_acquisition_function(acq_fn, model)
    assert up_acq_fn == acq_fn
    npt.assert_array_almost_equal(acq_fn(query_at), expected)
    assert acq_fn.__call__._get_tracing_count() == 1  # type: ignore


def test_multiple_optimism_builder_raises_when_update_with_wrong_function() -> None:
    model = QuadraticMeanAndRBFKernel()
    search_space = Box([0, 0], [1, 1])
    builder = MultipleOptimismNegativeLowerConfidenceBound(search_space)
    builder.prepare_acquisition_function(model)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.update_acquisition_function(lower_confidence_bound(model, 0.1), model)


@pytest.mark.parametrize("d", [0, -5])
def test_multiple_optimism_negative_confidence_bound_raises_for_negative_search_space_dim(
    d: int,
) -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        multiple_optimism_lower_confidence_bound(QuadraticMeanAndRBFKernel(), d)


def test_multiple_optimism_negative_confidence_bound_raises_for_changing_batch_size() -> None:
    model = QuadraticMeanAndRBFKernel()
    search_space = Box([0, 0], [1, 1])
    acq_fn = MultipleOptimismNegativeLowerConfidenceBound(
        search_space
    ).prepare_acquisition_function(model)
    query_at = tf.reshape(tf.linspace([[-10]], [[10]], 100), [10, 5, 2])
    acq_fn(query_at)
    with pytest.raises(tf.errors.InvalidArgumentError):
        query_at = tf.reshape(tf.linspace([[-10]], [[10]], 100), [5, 10, 2])
        acq_fn(query_at)


@pytest.mark.parametrize("in_place_update", [False, True])
def test_make_positive(in_place_update: bool) -> None:
    base = MagicMock()
    base.prepare_acquisition_function.side_effect = lambda *args: lambda x: x
    if in_place_update:
        base.update_acquisition_function.side_effect = lambda f, *args: f
    else:
        base.update_acquisition_function.side_effect = lambda *args: lambda x: 3.0
    builder: MakePositive[ProbabilisticModel] = MakePositive(base)

    model = QuadraticMeanAndRBFKernel()
    acq_fn = builder.prepare_acquisition_function(model)
    xs = tf.linspace([-1], [1], 10)
    npt.assert_allclose(acq_fn(xs), tf.math.log(1 + tf.math.exp(xs)))
    assert base.prepare_acquisition_function.call_count == 1
    assert base.update_acquisition_function.call_count == 0

    up_acq_fn = builder.update_acquisition_function(acq_fn, model)
    assert base.prepare_acquisition_function.call_count == 1
    assert base.update_acquisition_function.call_count == 1
    if in_place_update:
        assert up_acq_fn is acq_fn
    else:
        npt.assert_allclose(up_acq_fn(xs), tf.math.log(1 + tf.math.exp(3.0)))

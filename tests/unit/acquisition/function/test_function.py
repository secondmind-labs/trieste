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
from typing import Optional

import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

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
    GaussianProcessWithSamplers,
    QuadraticMeanAndRBFKernel,
    QuadraticMeanAndRBFKernelWithSamplers,
    rbf,
)
from trieste.acquisition.function.function import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    AugmentedExpectedImprovement,
    BatchMonteCarloExpectedImprovement,
    ExpectedConstrainedImprovement,
    ExpectedImprovement,
    MultipleOptimismNegativeLowerConfidenceBound,
    NegativeLowerConfidenceBound,
    ProbabilityOfFeasibility,
    augmented_expected_improvement,
    expected_improvement,
    lower_confidence_bound,
    multiple_optimism_lower_confidence_bound,
    probability_of_feasibility,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.objectives import BRANIN_MINIMUM, branin
from trieste.space import Box
from trieste.types import TensorType


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


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_expected_improvement_raises_for_invalid_batch_size(at: TensorType) -> None:
    ei = expected_improvement(QuadraticMeanAndRBFKernel(), tf.constant([1.0]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ei(at)


@random_seed
@pytest.mark.parametrize("best", [tf.constant([50.0]), BRANIN_MINIMUM, BRANIN_MINIMUM * 1.01])
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
    model = GaussianProcess([branin], [kernel])

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
def test_probability_of_feasibility(threshold: float, at: tf.Tensor, expected: float) -> None:
    actual = probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold)(at)
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
    expected = probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold)(at)

    npt.assert_allclose(acq(at), expected)


@pytest.mark.parametrize("shape", various_shapes() - {()})
def test_probability_of_feasibility_raises_on_non_scalar_threshold(shape: ShapeLike) -> None:
    threshold = tf.ones(shape)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold)


@pytest.mark.parametrize("shape", [[], [0], [2], [2, 1], [1, 2, 1]])
def test_probability_of_feasibility_raises_on_invalid_at_shape(shape: ShapeLike) -> None:
    at = tf.ones(shape)
    pof = probability_of_feasibility(QuadraticMeanAndRBFKernel(), 0.0)
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
    expected = probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold)(at)
    acq = builder.prepare_acquisition_function(model)
    assert acq._get_tracing_count() == 0  # type: ignore
    npt.assert_allclose(acq(at), expected)
    assert acq._get_tracing_count() == 1  # type: ignore
    up_acq = builder.update_acquisition_function(acq, model)
    assert up_acq == acq
    npt.assert_allclose(acq(at), expected)
    assert acq._get_tracing_count() == 1  # type: ignore


def test_expected_constrained_improvement_raises_for_non_scalar_min_pof() -> None:
    pof = ProbabilityOfFeasibility(0.0).using("")
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ExpectedConstrainedImprovement("", pof, tf.constant([0.0]))


def test_expected_constrained_improvement_raises_for_out_of_range_min_pof() -> None:
    pof = ProbabilityOfFeasibility(0.0).using("")
    with pytest.raises(tf.errors.InvalidArgumentError):
        ExpectedConstrainedImprovement("", pof, 1.5)


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_expected_constrained_improvement_raises_for_invalid_batch_size(at: TensorType) -> None:
    pof = ProbabilityOfFeasibility(0.0).using("")
    builder = ExpectedConstrainedImprovement("", pof, tf.constant(0.0))
    initial_query_points = tf.constant([[-1.0]])
    initial_objective_function_values = tf.constant([[1.0]])
    data = {"": Dataset(initial_query_points, initial_objective_function_values)}

    eci = builder.prepare_acquisition_function({"": QuadraticMeanAndRBFKernel()}, datasets=data)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        eci(at)


def test_expected_constrained_improvement_can_reproduce_expected_improvement() -> None:
    class _Certainty(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> AcquisitionFunction:
            return lambda x: tf.ones_like(tf.squeeze(x, -2))

    data = {"foo": Dataset(tf.constant([[0.5]]), tf.constant([[0.25]]))}
    models_ = {"foo": QuadraticMeanAndRBFKernel()}

    builder = ExpectedConstrainedImprovement("foo", _Certainty(), 0)
    eci = builder.prepare_acquisition_function(models_, datasets=data)

    ei = ExpectedImprovement().using("foo").prepare_acquisition_function(models_, datasets=data)

    at = tf.constant([[[-0.1]], [[1.23]], [[-6.78]]])
    npt.assert_allclose(eci(at), ei(at))

    new_data = {"foo": Dataset(tf.constant([[0.5], [1.0]]), tf.constant([[0.25], [0.5]]))}
    up_eci = builder.update_acquisition_function(eci, models_, datasets=new_data)
    assert up_eci == eci
    up_ei = (
        ExpectedImprovement().using("foo").prepare_acquisition_function(models_, datasets=new_data)
    )

    npt.assert_allclose(eci(at), up_ei(at))
    assert eci._get_tracing_count() == 1  # type: ignore


def test_expected_constrained_improvement_is_relative_to_feasible_point() -> None:
    class _Constraint(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> AcquisitionFunction:
            return lambda x: tf.cast(tf.squeeze(x, -2) >= 0, x.dtype)

    models_ = {"foo": QuadraticMeanAndRBFKernel()}

    eci_data = {"foo": Dataset(tf.constant([[-0.2], [0.3]]), tf.constant([[0.04], [0.09]]))}
    eci = ExpectedConstrainedImprovement("foo", _Constraint()).prepare_acquisition_function(
        models_,
        datasets=eci_data,
    )

    ei_data = {"foo": Dataset(tf.constant([[0.3]]), tf.constant([[0.09]]))}
    ei = ExpectedImprovement().using("foo").prepare_acquisition_function(models_, datasets=ei_data)

    npt.assert_allclose(eci(tf.constant([[0.1]])), ei(tf.constant([[0.1]])))


def test_expected_constrained_improvement_is_less_for_constrained_points() -> None:
    class _Constraint(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> AcquisitionFunction:
            return lambda x: tf.cast(tf.squeeze(x, -2) >= 0, x.dtype)

    def two_global_minima(x: tf.Tensor) -> tf.Tensor:
        return x ** 4 / 4 - x ** 2 / 2

    initial_query_points = tf.constant([[-2.0], [0.0], [1.2]])
    data = {"foo": Dataset(initial_query_points, two_global_minima(initial_query_points))}
    models_ = {"foo": GaussianProcess([two_global_minima], [rbf()])}

    eci = ExpectedConstrainedImprovement("foo", _Constraint()).prepare_acquisition_function(
        models_,
        datasets=data,
    )

    npt.assert_array_less(eci(tf.constant([[-1.0]])), eci(tf.constant([[1.0]])))


def test_expected_constrained_improvement_raises_for_empty_data() -> None:
    class _Constraint(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> AcquisitionFunction:
            return raise_exc

    data = {"foo": Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))}
    models_ = {"foo": QuadraticMeanAndRBFKernel()}
    builder = ExpectedConstrainedImprovement("foo", _Constraint())

    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(models_, datasets=data)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(models_)


def test_expected_constrained_improvement_is_constraint_when_no_feasible_points() -> None:
    class _Constraint(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> AcquisitionFunction:
            def acquisition(x: TensorType) -> TensorType:
                x_ = tf.squeeze(x, -2)
                return tf.cast(tf.logical_and(0.0 <= x_, x_ < 1.0), x.dtype)

            return acquisition

    data = {"foo": Dataset(tf.constant([[-2.0], [1.0]]), tf.constant([[4.0], [1.0]]))}
    models_ = {"foo": QuadraticMeanAndRBFKernel()}
    eci = ExpectedConstrainedImprovement("foo", _Constraint()).prepare_acquisition_function(
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
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> AcquisitionFunction:
            return pof

    models_ = {"foo": QuadraticMeanAndRBFKernel()}

    data = {"foo": Dataset(tf.constant([[1.1], [2.0]]), tf.constant([[1.21], [4.0]]))}
    eci = ExpectedConstrainedImprovement(
        "foo", _Constraint(), min_feasibility_probability=tfp.bijectors.Sigmoid().forward(1.0)
    ).prepare_acquisition_function(
        models_,
        datasets=data,
    )

    ei = ExpectedImprovement().using("foo").prepare_acquisition_function(models_, datasets=data)
    x = tf.constant([[1.5]])
    npt.assert_allclose(eci(x), ei(x) * pof(x))


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
    model = GaussianProcessWithSamplers(
        [lambda x: branin(x), lambda x: quadratic(x)], [matern52, rbf()]
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
    model = GaussianProcessWithSamplers(
        [lambda x: branin(x), lambda x: quadratic(x)], [matern52, rbf()]
    )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(model, dataset=data)


@random_seed
def test_batch_monte_carlo_expected_improvement_raises_for_model_without_reparam_sampler() -> None:
    known_query_points = tf.random.uniform([5, 2], dtype=tf.float64)
    data = Dataset(known_query_points, quadratic(known_query_points))
    model = QuadraticMeanAndRBFKernel()
    with pytest.raises(ValueError):
        BatchMonteCarloExpectedImprovement(10_000).prepare_acquisition_function(
            model, dataset=data  # type: ignore
        )


@random_seed
def test_batch_monte_carlo_expected_improvement_can_reproduce_ei() -> None:
    known_query_points = tf.random.uniform([5, 2], dtype=tf.float64)
    data = Dataset(known_query_points, quadratic(known_query_points))
    model = QuadraticMeanAndRBFKernelWithSamplers(dataset=data)
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
    model = QuadraticMeanAndRBFKernelWithSamplers(dataset=data)
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
    data = Dataset(known_query_points[:5], quadratic(known_query_points[:5]))
    model = QuadraticMeanAndRBFKernelWithSamplers(dataset=data)
    builder = BatchMonteCarloExpectedImprovement(10_000)
    ei = ExpectedImprovement().prepare_acquisition_function(model, dataset=data)
    xs = tf.random.uniform([3, 5, 1, 2], dtype=tf.float64)

    batch_ei = builder.prepare_acquisition_function(model, dataset=data)
    assert batch_ei.__call__._get_tracing_count() == 0  # type: ignore
    npt.assert_allclose(batch_ei(xs), ei(xs), rtol=0.06)
    assert batch_ei.__call__._get_tracing_count() == 1  # type: ignore

    data = Dataset(known_query_points, quadratic(known_query_points))
    up_batch_ei = builder.update_acquisition_function(batch_ei, model, dataset=data)
    assert up_batch_ei == batch_ei
    assert batch_ei.__call__._get_tracing_count() == 1  # type: ignore
    npt.assert_allclose(batch_ei(xs), ei(xs), rtol=0.06)
    assert batch_ei.__call__._get_tracing_count() == 1  # type: ignore


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


# check builder update with different acq

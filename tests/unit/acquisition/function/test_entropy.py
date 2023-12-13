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

import unittest.mock
from unittest.mock import MagicMock

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, quadratic, random_seed
from tests.util.models.gpflow.models import (
    GaussianProcess,
    MultiFidelityQuadraticMeanAndRBFKernel,
    MultiFidelityQuadraticMeanAndRBFKernelWithSamplers,
    QuadraticMeanAndRBFKernel,
    QuadraticMeanAndRBFKernelWithSamplers,
)
from trieste.acquisition.function.entropy import (
    GIBBON,
    MUMBO,
    MinValueEntropySearch,
    MUMBOModelType,
    SupportsCovarianceObservationNoiseTrajectory,
    SupportsCovarianceWithTopFidelityPredictY,
    gibbon_quality_term,
    gibbon_repulsion_term,
    min_value_entropy_search,
    mumbo,
)
from trieste.acquisition.sampler import (
    ExactThompsonSampler,
    GumbelSampler,
    ThompsonSampler,
    ThompsonSamplerFromTrajectory,
)
from trieste.data import Dataset, add_fidelity_column
from trieste.objectives import Branin
from trieste.space import Box
from trieste.types import TensorType


def test_min_value_entropy_search_builder_raises_for_empty_data() -> None:
    empty_data = Dataset(tf.zeros([0, 2], dtype=tf.float64), tf.ones([0, 2], dtype=tf.float64))
    non_empty_data = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    builder = MinValueEntropySearch(search_space)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(QuadraticMeanAndRBFKernel(), dataset=empty_data)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(QuadraticMeanAndRBFKernel())
    acq = builder.prepare_acquisition_function(QuadraticMeanAndRBFKernel(), dataset=non_empty_data)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.update_acquisition_function(acq, QuadraticMeanAndRBFKernel(), dataset=empty_data)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.update_acquisition_function(acq, QuadraticMeanAndRBFKernel())


@pytest.mark.parametrize("param", [-2, 0])
def test_min_value_entropy_search_builder_raises_for_invalid_init_params(param: int) -> None:
    search_space = Box([0, 0], [1, 1])
    with pytest.raises(tf.errors.InvalidArgumentError):
        MinValueEntropySearch(search_space, num_samples=param)
    with pytest.raises(tf.errors.InvalidArgumentError):
        MinValueEntropySearch(search_space, grid_size=param)


@pytest.mark.parametrize(
    "sampler",
    [
        ExactThompsonSampler(sample_min_value=False),
        ThompsonSamplerFromTrajectory(sample_min_value=False),
    ],
)
def test_mes_raises_if_passed_sampler_with_sample_min_value_False(
    sampler: ThompsonSampler[GaussianProcess],
) -> None:
    search_space = Box([0, 0], [1, 1])
    with pytest.raises(ValueError):
        MinValueEntropySearch(search_space, min_value_sampler=sampler)


def test_mes_default_sampler_is_exact_thompson() -> None:
    search_space = Box([0, 0], [1, 1])
    builder = MinValueEntropySearch(search_space)
    assert isinstance(builder._min_value_sampler, ExactThompsonSampler)
    assert builder._min_value_sampler._sample_min_value


@pytest.mark.parametrize(
    "sampler",
    [
        ExactThompsonSampler(sample_min_value=True),
        GumbelSampler(sample_min_value=True),
        ThompsonSamplerFromTrajectory(sample_min_value=True),
    ],
)
def test_mes_initialized_with_passed_sampler(sampler: ThompsonSampler[GaussianProcess]) -> None:
    search_space = Box([0, 0], [1, 1])
    builder = MinValueEntropySearch(search_space, min_value_sampler=sampler)
    assert builder._min_value_sampler == sampler


def test_mes_raises_when_use_trajectory_sampler_and_model_without_trajectories() -> None:
    dataset = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    builder = MinValueEntropySearch(
        search_space, min_value_sampler=ThompsonSamplerFromTrajectory(sample_min_value=True)
    )
    model = QuadraticMeanAndRBFKernel()
    with pytest.raises(ValueError):
        builder.prepare_acquisition_function(model, dataset=dataset)  # type: ignore


@unittest.mock.patch("trieste.acquisition.function.entropy.min_value_entropy_search")
@pytest.mark.parametrize(
    "min_value_sampler",
    [ExactThompsonSampler(sample_min_value=True), GumbelSampler(sample_min_value=True)],
)
def test_min_value_entropy_search_builder_builds_min_value_samples(
    mocked_mves: MagicMock, min_value_sampler: ThompsonSampler[GaussianProcess]
) -> None:
    dataset = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    builder = MinValueEntropySearch(search_space, min_value_sampler=min_value_sampler)
    model = QuadraticMeanAndRBFKernelWithSamplers(dataset)
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    builder.prepare_acquisition_function(model, dataset=dataset)
    mocked_mves.assert_called_once()

    # check that the Gumbel samples look sensible
    min_value_samples = mocked_mves.call_args[0][1]
    query_points = builder._search_space.sample(num_samples=builder._grid_size)
    query_points = tf.concat([dataset.query_points, query_points], 0)
    fmean, _ = model.predict(query_points)
    assert max(min_value_samples) < min(fmean)


@pytest.mark.parametrize(
    "min_value_sampler",
    [ExactThompsonSampler(sample_min_value=True), GumbelSampler(sample_min_value=True)],
)
def test_min_value_entropy_search_builder_updates_acquisition_function(
    min_value_sampler: ThompsonSampler[GaussianProcess],
) -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    model = QuadraticMeanAndRBFKernel(noise_variance=tf.constant(1e-10, dtype=tf.float64))
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions

    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    partial_dataset = Dataset(xs[:10], ys[:10])
    full_dataset = Dataset(xs, ys)

    builder = MinValueEntropySearch(search_space, min_value_sampler=min_value_sampler)
    xs = tf.cast(tf.linspace([[0.0]], [[1.0]], 10), tf.float64)

    old_acq_fn = builder.prepare_acquisition_function(model, dataset=partial_dataset)
    tf.random.set_seed(0)  # to ensure consistent sampling
    updated_acq_fn = builder.update_acquisition_function(old_acq_fn, model, dataset=full_dataset)
    assert updated_acq_fn == old_acq_fn
    updated_values = updated_acq_fn(xs)

    tf.random.set_seed(0)  # to ensure consistent sampling
    new_acq_fn = builder.prepare_acquisition_function(model, dataset=full_dataset)
    new_values = new_acq_fn(xs)

    npt.assert_allclose(updated_values, new_values)


@random_seed
@unittest.mock.patch("trieste.acquisition.function.entropy.min_value_entropy_search")
def test_min_value_entropy_search_builder_builds_min_value_samples_trajectory_sampler(
    mocked_mves: MagicMock,
) -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)
    model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset=dataset, noise_variance=tf.constant(1e-10, dtype=tf.float64)
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions

    builder = MinValueEntropySearch(
        search_space, min_value_sampler=ThompsonSamplerFromTrajectory(sample_min_value=True)
    )
    builder.prepare_acquisition_function(model, dataset=dataset)
    mocked_mves.assert_called_once()

    # check that the Gumbel samples look sensible
    min_value_samples = mocked_mves.call_args[0][1]
    query_points = builder._search_space.sample(num_samples=builder._grid_size)
    query_points = tf.concat([dataset.query_points, query_points], 0)
    fmean, _ = model.predict(query_points)
    assert max(min_value_samples) < min(fmean) + 1e-4


@pytest.mark.parametrize("samples", [tf.constant([]), tf.constant([[[]]])])
def test_min_value_entropy_search_raises_for_min_values_samples_with_invalid_shape(
    samples: TensorType,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        min_value_entropy_search(QuadraticMeanAndRBFKernel(), samples)


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_min_value_entropy_search_raises_for_invalid_batch_size(at: TensorType) -> None:
    mes = min_value_entropy_search(QuadraticMeanAndRBFKernel(), tf.constant([[1.0], [2.0]]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        mes(at)


def test_min_value_entropy_search_returns_correct_shape() -> None:
    model = QuadraticMeanAndRBFKernel()
    min_value_samples = tf.constant([[1.0], [2.0]])
    query_at = tf.linspace([[-10.0]], [[10.0]], 5)
    evals = min_value_entropy_search(model, min_value_samples)(query_at)
    npt.assert_array_equal(evals.shape, tf.constant([5, 1]))


def test_min_value_entropy_search_chooses_same_as_probability_of_improvement() -> None:
    """
    When based on a single max-value sample, MES should choose the same point that probability of
    improvement would when calcualted with the max-value as its threshold (See :cite:`wang2017max`).
    """

    kernel = tfp.math.psd_kernels.MaternFiveHalves()
    model = GaussianProcess([Branin.objective], [kernel])

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    min_value_sample = tf.constant([[1.0]], dtype=tf.float64)
    mes_evals = min_value_entropy_search(model, min_value_sample)(xs[..., None, :])

    mean, variance = model.predict(xs)
    gamma = (tf.cast(min_value_sample, dtype=mean.dtype) - mean) / tf.sqrt(variance)
    norm = tfp.distributions.Normal(tf.cast(0, dtype=mean.dtype), tf.cast(1, dtype=mean.dtype))
    pi_evals = norm.cdf(gamma)

    npt.assert_array_equal(tf.argmax(mes_evals), tf.argmax(pi_evals))


def test_gibbon_builder_raises_for_empty_data() -> None:
    empty_data = Dataset(tf.zeros([0, 2], dtype=tf.float64), tf.ones([0, 2], dtype=tf.float64))
    non_empty_data = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    builder = GIBBON(search_space)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(QuadraticMeanAndRBFKernel(), empty_data)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(QuadraticMeanAndRBFKernel())
    acq = builder.prepare_acquisition_function(QuadraticMeanAndRBFKernel(), non_empty_data)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.update_acquisition_function(acq, QuadraticMeanAndRBFKernel(), empty_data)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.update_acquisition_function(acq, QuadraticMeanAndRBFKernel())


@pytest.mark.parametrize("param", [-2, 0])
def test_gibbon_builder_raises_for_invalid_init_params(param: int) -> None:
    search_space = Box([0, 0], [1, 1])
    with pytest.raises(tf.errors.InvalidArgumentError):
        GIBBON(search_space, num_samples=param)
    with pytest.raises(tf.errors.InvalidArgumentError):
        GIBBON(search_space, grid_size=param)


@pytest.mark.parametrize(
    "sampler",
    [
        ExactThompsonSampler(sample_min_value=False),
        ThompsonSamplerFromTrajectory(sample_min_value=False),
    ],
)
def test_gibbon_raises_if_passed_sampler_with_sample_min_value_False(
    sampler: ThompsonSampler[GaussianProcess],
) -> None:
    search_space = Box([0, 0], [1, 1])
    with pytest.raises(ValueError):
        GIBBON(search_space, min_value_sampler=sampler)


def test_gibbon_default_sampler_is_exact_thompson() -> None:
    search_space = Box([0, 0], [1, 1])
    builder = GIBBON(search_space)
    assert isinstance(builder._min_value_sampler, ExactThompsonSampler)
    assert builder._min_value_sampler._sample_min_value


@pytest.mark.parametrize(
    "sampler",
    [
        ExactThompsonSampler(sample_min_value=True),
        GumbelSampler(sample_min_value=True),
        ThompsonSamplerFromTrajectory(sample_min_value=True),
    ],
)
def test_gibbon_initialized_with_passed_sampler(sampler: ThompsonSampler[GaussianProcess]) -> None:
    search_space = Box([0, 0], [1, 1])
    builder = GIBBON(search_space, min_value_sampler=sampler)
    assert builder._min_value_sampler == sampler


def test_gibbon_raises_when_use_trajectory_sampler_and_model_without_trajectories() -> None:
    dataset = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    builder = GIBBON[SupportsCovarianceObservationNoiseTrajectory](
        search_space, min_value_sampler=ThompsonSamplerFromTrajectory(sample_min_value=True)
    )
    model = QuadraticMeanAndRBFKernel()
    with pytest.raises(ValueError):
        builder.prepare_acquisition_function(model, dataset=dataset)  # type: ignore


@pytest.mark.parametrize("samples", [tf.constant([]), tf.constant([[[]]])])
def test_gibbon_quality_term_raises_for_gumbel_samples_with_invalid_shape(
    samples: TensorType,
) -> None:
    with pytest.raises(ValueError):
        model = QuadraticMeanAndRBFKernel()
        gibbon_quality_term(model, samples)


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_gibbon_quality_term_raises_for_invalid_batch_size(at: TensorType) -> None:
    model = QuadraticMeanAndRBFKernel()
    gibbon_acq = gibbon_quality_term(model, tf.constant([[1.0], [2.0]]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        gibbon_acq(at)


def test_gibbon_quality_term_returns_correct_shape() -> None:
    model = QuadraticMeanAndRBFKernel()
    gumbel_samples = tf.constant([[1.0], [2.0]])
    query_at = tf.linspace([[-10.0]], [[10.0]], 5)
    evals = gibbon_quality_term(model, gumbel_samples)(query_at)
    npt.assert_array_equal(evals.shape, tf.constant([5, 1]))


@unittest.mock.patch("trieste.acquisition.function.entropy.gibbon_quality_term")
@pytest.mark.parametrize(
    "min_value_sampler",
    [ExactThompsonSampler(sample_min_value=True), GumbelSampler(sample_min_value=True)],
)
def test_gibbon_builder_builds_min_value_samples(
    mocked_mves: MagicMock,
    min_value_sampler: ThompsonSampler[GaussianProcess],
) -> None:
    dataset = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    builder = GIBBON(search_space, min_value_sampler=min_value_sampler)
    model = QuadraticMeanAndRBFKernel()
    builder.prepare_acquisition_function(model, dataset=dataset)
    mocked_mves.assert_called_once()

    # check that the Gumbel samples look sensible
    min_value_samples = builder._min_value_samples
    query_points = builder._search_space.sample(num_samples=builder._grid_size)
    query_points = tf.concat([dataset.query_points, query_points], 0)
    fmean, _ = model.predict(query_points)
    assert max(min_value_samples) < min(fmean)  # type: ignore


@pytest.mark.parametrize(
    "min_value_sampler",
    [ExactThompsonSampler(sample_min_value=True), GumbelSampler(sample_min_value=True)],
)
def test_gibbon_builder_updates_acquisition_function(
    min_value_sampler: ThompsonSampler[GaussianProcess],
) -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    x_range = tf.cast(tf.linspace(0.0, 1.0, 5), dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    partial_dataset = Dataset(xs[:10], ys[:10])
    full_dataset = Dataset(xs, ys)

    builder = GIBBON(search_space, min_value_sampler=min_value_sampler)
    xs = tf.cast(tf.linspace([[0.0]], [[1.0]], 10), tf.float64)
    model = QuadraticMeanAndRBFKernel()

    old_acq_fn = builder.prepare_acquisition_function(model, dataset=partial_dataset)
    tf.random.set_seed(0)  # to ensure consistent sampling
    updated_acq_fn = builder.update_acquisition_function(old_acq_fn, model, dataset=full_dataset)
    assert updated_acq_fn == old_acq_fn
    updated_values = updated_acq_fn(xs)

    tf.random.set_seed(0)  # to ensure consistent sampling
    new_acq_fn = builder.prepare_acquisition_function(model, dataset=full_dataset)
    new_values = new_acq_fn(xs)

    npt.assert_allclose(updated_values, new_values)


@pytest.mark.parametrize("pending_points", [tf.constant([0.0]), tf.constant([[[0.0], [1.0]]])])
def test_gibbon_builder_raises_for_invalid_pending_points_shape(
    pending_points: TensorType,
) -> None:
    data = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    space = Box([0, 0], [1, 1])
    builder = GIBBON(search_space=space)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(QuadraticMeanAndRBFKernel(), data, pending_points)


@random_seed
@unittest.mock.patch("trieste.acquisition.function.entropy.gibbon_quality_term")
def test_gibbon_builder_builds_min_value_samples_using_trajectories(mocked_mves: MagicMock) -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)

    model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset=dataset, noise_variance=tf.constant(1e-10, dtype=tf.float64)
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions

    builder = GIBBON[SupportsCovarianceObservationNoiseTrajectory](
        search_space, min_value_sampler=ThompsonSamplerFromTrajectory(sample_min_value=True)
    )
    builder.prepare_acquisition_function(model, dataset=dataset)
    mocked_mves.assert_called_once()

    # check that the Gumbel samples look sensible
    min_value_samples = mocked_mves.call_args[0][1]
    query_points = builder._search_space.sample(num_samples=builder._grid_size)
    query_points = tf.concat([dataset.query_points, query_points], 0)
    fmean, _ = model.predict(query_points)
    assert max(min_value_samples) < min(fmean) + 1e-4


def test_gibbon_chooses_same_as_min_value_entropy_search() -> None:
    """
    When based on a single max-value sample, GIBBON should choose the same point as
    MES (see :cite:`Moss:2021`).
    """
    model = QuadraticMeanAndRBFKernel(noise_variance=tf.constant(1e-10, dtype=tf.float64))

    x_range = tf.linspace(-1.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    min_value_sample = tf.constant([[1.0]], dtype=tf.float64)
    mes_evals = min_value_entropy_search(model, min_value_sample)(xs[..., None, :])
    gibbon_evals = gibbon_quality_term(model, min_value_sample)(xs[..., None, :])

    npt.assert_array_equal(tf.argmax(mes_evals), tf.argmax(gibbon_evals))


@pytest.mark.parametrize("rescaled_repulsion", [True, False])
@pytest.mark.parametrize("noise_variance", [0.1, 1e-10])
def test_batch_gibbon_is_sum_of_individual_gibbons_and_repulsion_term(
    rescaled_repulsion: bool, noise_variance: float
) -> None:
    """
    Check that batch GIBBON can be decomposed into the sum of sequential GIBBONs and a repulsion
    term (see :cite:`Moss:2021`).
    """
    noise_variance = tf.constant(noise_variance, dtype=tf.float64)
    model = QuadraticMeanAndRBFKernel(noise_variance=noise_variance)
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decomposition

    x_range = tf.linspace(0.0, 1.0, 4)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    pending_points = tf.constant([[0.11, 0.51], [0.21, 0.31], [0.41, 0.91]], dtype=tf.float64)
    min_value_sample = tf.constant([[-0.1, 0.1]], dtype=tf.float64)

    gibbon_of_new_points = gibbon_quality_term(model, min_value_sample)(xs[..., None, :])
    mean, var = model.predict(xs)
    _, pending_var = model.predict_joint(pending_points)
    pending_var += noise_variance * tf.eye(len(pending_points), dtype=pending_var.dtype)

    calculated_batch_gibbon = gibbon_of_new_points + gibbon_repulsion_term(
        model, pending_points, rescaled_repulsion=rescaled_repulsion
    )(xs[..., None, :])

    for i in tf.range(len(xs)):  # check across a set of candidate points
        candidate_and_pending = tf.concat([xs[i : i + 1], pending_points], axis=0)
        _, A = model.predict_joint(candidate_and_pending)
        A += noise_variance * tf.eye(len(pending_points) + 1, dtype=A.dtype)
        repulsion = tf.linalg.logdet(A) - tf.math.log(A[0, 0, 0]) - tf.linalg.logdet(pending_var)
        if rescaled_repulsion:  # down-weight repulsion term
            batch_size, search_space_dim = tf.cast(tf.shape(pending_points), dtype=mean.dtype)
            repulsion = repulsion * ((1 / batch_size) ** (2))

        reconstructed_batch_gibbon = 0.5 * repulsion + gibbon_of_new_points[i : i + 1]
        npt.assert_array_almost_equal(
            calculated_batch_gibbon[i : i + 1], reconstructed_batch_gibbon
        )


def test_mumbo_builder_raises_for_empty_data() -> None:
    empty_data = Dataset(tf.zeros([0, 2], dtype=tf.float64), tf.ones([0, 2], dtype=tf.float64))
    non_empty_data = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    builder = MUMBO(search_space)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(
            MultiFidelityQuadraticMeanAndRBFKernel(), dataset=empty_data
        )
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(MultiFidelityQuadraticMeanAndRBFKernel())
    acq = builder.prepare_acquisition_function(
        MultiFidelityQuadraticMeanAndRBFKernel(), dataset=non_empty_data
    )
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.update_acquisition_function(
            acq, MultiFidelityQuadraticMeanAndRBFKernel(), dataset=empty_data
        )
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.update_acquisition_function(acq, MultiFidelityQuadraticMeanAndRBFKernel())


@pytest.mark.parametrize("param", [-2, 0])
def test_mumbo_builder_raises_for_invalid_init_params(param: int) -> None:
    search_space = Box([0, 0], [1, 1])
    with pytest.raises(tf.errors.InvalidArgumentError):
        MUMBO(search_space, num_samples=param)
    with pytest.raises(tf.errors.InvalidArgumentError):
        MUMBO(search_space, grid_size=param)


@pytest.mark.parametrize(
    "sampler",
    [
        ExactThompsonSampler(sample_min_value=False),
        ThompsonSamplerFromTrajectory(sample_min_value=False),
    ],
)
def test_mumbo_raises_if_passed_sampler_with_sample_min_value_False(
    sampler: ThompsonSampler[MUMBOModelType],
) -> None:
    search_space = Box([0, 0], [1, 1])
    with pytest.raises(ValueError):
        MUMBO(search_space, min_value_sampler=sampler)


def test_mumbo_default_sampler_is_exact_thompson() -> None:
    search_space = Box([0, 0], [1, 1])
    builder = MUMBO(search_space)
    assert isinstance(builder._min_value_sampler, ExactThompsonSampler)
    assert builder._min_value_sampler._sample_min_value


@pytest.mark.parametrize(
    "sampler",
    [
        ExactThompsonSampler(sample_min_value=True),
        GumbelSampler(sample_min_value=True),
        ThompsonSamplerFromTrajectory(sample_min_value=True),
    ],
)
def test_mumbo_initialized_with_passed_sampler(sampler: ThompsonSampler[MUMBOModelType]) -> None:
    search_space = Box([0, 0], [1, 1])
    builder = MUMBO(search_space, min_value_sampler=sampler)
    assert builder._min_value_sampler == sampler


def test_mumbo_raises_when_use_trajectory_sampler_and_model_without_trajectories() -> None:
    dataset = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    builder = MUMBO(  # type: ignore
        search_space, min_value_sampler=ThompsonSamplerFromTrajectory(sample_min_value=True)
    )
    model = MultiFidelityQuadraticMeanAndRBFKernel()
    with pytest.raises(ValueError):
        builder.prepare_acquisition_function(model, dataset=dataset)  # type: ignore


@unittest.mock.patch("trieste.acquisition.function.entropy.mumbo")
@pytest.mark.parametrize(
    "min_value_sampler",
    [ExactThompsonSampler(sample_min_value=True), GumbelSampler(sample_min_value=True)],
)
def test_mumbo_builder_builds_min_value_samples(
    mocked_mves: MagicMock,
    min_value_sampler: ThompsonSampler[SupportsCovarianceWithTopFidelityPredictY],
) -> None:
    dataset = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    builder = MUMBO(search_space, min_value_sampler=min_value_sampler)
    model = MultiFidelityQuadraticMeanAndRBFKernelWithSamplers(dataset)
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    builder.prepare_acquisition_function(model, dataset=dataset)
    mocked_mves.assert_called_once()

    # check that the Gumbel samples look sensible
    min_value_samples = mocked_mves.call_args[0][1]
    query_points = builder._search_space.sample(num_samples=builder._grid_size)
    query_points = tf.concat([dataset.query_points, query_points], 0)
    query_points = add_fidelity_column(query_points[:, :-1], model.num_fidelities - 1)
    fmean, _ = model.predict(query_points)
    assert max(min_value_samples) < min(fmean)


@pytest.mark.parametrize(
    "min_value_sampler",
    [ExactThompsonSampler(sample_min_value=True), GumbelSampler(sample_min_value=True)],
)
def test_mumbo_builder_updates_acquisition_function(
    min_value_sampler: ThompsonSampler[SupportsCovarianceWithTopFidelityPredictY],
) -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    model = MultiFidelityQuadraticMeanAndRBFKernel(
        noise_variance=tf.constant(1e-10, dtype=tf.float64)
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions

    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    partial_dataset = Dataset(xs[:10], ys[:10])
    full_dataset = Dataset(xs, ys)

    builder = MUMBO(search_space, min_value_sampler=min_value_sampler)
    xs = tf.cast(tf.linspace([[0.0]], [[1.0]], 10), tf.float64)

    old_acq_fn = builder.prepare_acquisition_function(model, dataset=partial_dataset)
    tf.random.set_seed(0)  # to ensure consistent sampling
    updated_acq_fn = builder.update_acquisition_function(old_acq_fn, model, dataset=full_dataset)
    assert updated_acq_fn == old_acq_fn
    updated_values = updated_acq_fn(xs)

    tf.random.set_seed(0)  # to ensure consistent sampling
    new_acq_fn = builder.prepare_acquisition_function(model, dataset=full_dataset)
    new_values = new_acq_fn(xs)

    npt.assert_allclose(updated_values, new_values)


@random_seed
@unittest.mock.patch("trieste.acquisition.function.entropy.min_value_entropy_search")
def test_mumbo_builder_builds_min_value_samples_trajectory_sampler(
    mocked_mves: MagicMock,
) -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    x_range = tf.linspace(0.0, 1.0, 5)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    ys = quadratic(xs)
    dataset = Dataset(xs, ys)
    model = MultiFidelityQuadraticMeanAndRBFKernelWithSamplers(
        dataset=dataset, noise_variance=tf.constant(1e-10, dtype=tf.float64)
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions

    builder = MinValueEntropySearch(
        search_space, min_value_sampler=ThompsonSamplerFromTrajectory(sample_min_value=True)
    )
    builder.prepare_acquisition_function(model, dataset=dataset)
    mocked_mves.assert_called_once()

    # check that the Gumbel samples look sensible
    min_value_samples = mocked_mves.call_args[0][1]
    query_points = builder._search_space.sample(num_samples=builder._grid_size)
    query_points = tf.concat([dataset.query_points, query_points], 0)
    fmean, _ = model.predict(query_points)
    assert max(min_value_samples) < min(fmean) + 1e-4


@pytest.mark.parametrize("samples", [tf.constant([]), tf.constant([[[]]])])
def test_mumbo_raises_for_min_values_samples_with_invalid_shape(
    samples: TensorType,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        mumbo(MultiFidelityQuadraticMeanAndRBFKernel(), samples)


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_mumbo_raises_for_invalid_batch_size(at: TensorType) -> None:
    mes = mumbo(MultiFidelityQuadraticMeanAndRBFKernel(), tf.constant([[1.0], [2.0]]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        mes(at)


def test_mumbo_returns_correct_shape() -> None:
    model = MultiFidelityQuadraticMeanAndRBFKernel()
    min_value_samples = tf.constant([[1.0], [2.0]])
    query_at = tf.linspace([[-10.0]], [[10.0]], 5)
    evals = mumbo(model, min_value_samples)(query_at)
    npt.assert_array_equal(evals.shape, tf.constant([5, 1]))

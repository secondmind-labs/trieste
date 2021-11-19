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

import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, ShapeLike, random_seed, various_shapes
from tests.util.models.gpflow.models import GaussianProcess, QuadraticMeanAndRBFKernel
from tests.util.models.gpflux.models import trieste_deep_gaussian_process
from trieste.acquisition.function.active_learning import (
    ExpectedFeasibility,
    PredictiveVariance,
    bichon_ranjan_criterion,
    predictive_variance,
)
from trieste.objectives import branin
from trieste.types import TensorType
from trieste.utils import DEFAULTS


def test_predictive_variance_builder_builds_predictive_variance() -> None:
    model = QuadraticMeanAndRBFKernel()
    acq_fn = PredictiveVariance().prepare_acquisition_function(model)
    query_at = tf.linspace([[-10]], [[10]], 100)
    _, covariance = model.predict_joint(query_at)
    expected = tf.linalg.det(covariance)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)


@pytest.mark.parametrize(
    "at, acquisition_shape",
    [
        (tf.constant([[[1.0]]]), tf.constant([1, 1])),
        (tf.linspace([[-10.0]], [[10.0]], 5), tf.constant([5, 1])),
        (tf.constant([[[1.0, 1.0]]]), tf.constant([1, 1])),
        (tf.linspace([[-10.0, -10.0]], [[10.0, 10.0]], 5), tf.constant([5, 1])),
    ],
)
def test_predictive_variance_returns_correct_shape(
    at: TensorType, acquisition_shape: TensorType
) -> None:
    model = QuadraticMeanAndRBFKernel()
    acq_fn = PredictiveVariance().prepare_acquisition_function(model)
    npt.assert_array_equal(acq_fn(at).shape, acquisition_shape)


@random_seed
@pytest.mark.parametrize(
    "variance_scale, num_samples_per_point, rtol, atol",
    [
        (0.1, 10_000, 0.05, 1e-6),
        (1.0, 50_000, 0.05, 1e-3),
        (10.0, 100_000, 0.05, 1e-2),
        (100.0, 150_000, 0.05, 1e-1),
    ],
)
def test_predictive_variance(
    variance_scale: float,
    num_samples_per_point: int,
    rtol: float,
    atol: float,
) -> None:
    variance_scale = tf.constant(variance_scale, tf.float64)

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    kernel = tfp.math.psd_kernels.MaternFiveHalves(variance_scale, length_scale=0.25)
    model = GaussianProcess([branin], [kernel])

    mean, variance = model.predict(xs)
    samples = tfp.distributions.Normal(mean, tf.sqrt(variance)).sample(num_samples_per_point)
    predvar_approx = tf.math.reduce_variance(samples, axis=0)

    detcov = predictive_variance(model, DEFAULTS.JITTER)
    predvar = detcov(xs[..., None, :])

    npt.assert_allclose(predvar, predvar_approx, rtol=rtol, atol=atol)


def test_predictive_variance_builder_updates_without_retracing() -> None:
    model = QuadraticMeanAndRBFKernel()
    builder = PredictiveVariance()
    acq_fn = builder.prepare_acquisition_function(model)
    assert acq_fn._get_tracing_count() == 0  # type: ignore
    query_at = tf.linspace([[-10]], [[10]], 100)
    expected = predictive_variance(model, DEFAULTS.JITTER)(query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)
    assert acq_fn._get_tracing_count() == 1  # type: ignore

    up_acq_fn = builder.update_acquisition_function(acq_fn, model)
    assert up_acq_fn == acq_fn
    npt.assert_array_almost_equal(acq_fn(query_at), expected)
    assert acq_fn._get_tracing_count() == 1  # type: ignore


def test_predictive_variance_raises_for_void_predict_joint() -> None:
    model, _ = trieste_deep_gaussian_process(tf.zeros([0, 1]), 2, 20, 0.01, 100, 100)
    acq_fn = predictive_variance(model, DEFAULTS.JITTER)

    with pytest.raises(ValueError):
        acq_fn(tf.zeros([0, 1]))


@pytest.mark.parametrize("delta", [1, 2])
def test_expected_feasibility_builder_builds_acquisition_function(delta: int) -> None:
    threshold = 1
    alpha = 1
    query_at = tf.linspace([[-10]], [[10]], 100)

    model = QuadraticMeanAndRBFKernel()
    acq_fn = ExpectedFeasibility(threshold, alpha, delta).prepare_acquisition_function(model)
    expected = bichon_ranjan_criterion(model, threshold, alpha, delta)(query_at)

    npt.assert_array_almost_equal(acq_fn(query_at), expected)


@pytest.mark.parametrize(
    "at, acquisition_shape",
    [
        (tf.constant([[[1.0]]]), tf.constant([1, 1])),
        (tf.linspace([[-10.0]], [[10.0]], 5), tf.constant([5, 1])),
        (tf.constant([[[1.0, 1.0]]]), tf.constant([1, 1])),
        (tf.linspace([[-10.0, -10.0]], [[10.0, 10.0]], 5), tf.constant([5, 1])),
    ],
)
@pytest.mark.parametrize("delta", [1, 2])
def test_expected_feasibility_returns_correct_shape(
    at: TensorType, acquisition_shape: TensorType, delta: int
) -> None:
    threshold = 1
    alpha = 1
    model = QuadraticMeanAndRBFKernel()
    acq_fn = ExpectedFeasibility(threshold, alpha, delta).prepare_acquisition_function(model)
    npt.assert_array_equal(acq_fn(at).shape, acquisition_shape)


@pytest.mark.parametrize("delta", [1, 2])
def test_expected_feasibility_builder_updates_without_retracing(delta: int) -> None:
    threshold = 1
    alpha = 1

    model = QuadraticMeanAndRBFKernel()
    builder = ExpectedFeasibility(threshold, alpha, delta)
    acq_fn = builder.prepare_acquisition_function(model)
    assert acq_fn._get_tracing_count() == 0  # type: ignore

    query_at = tf.linspace([[-10]], [[10]], 100)
    expected = bichon_ranjan_criterion(model, threshold, alpha, delta)(query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)
    assert acq_fn._get_tracing_count() == 1  # type: ignore

    up_acq_fn = builder.update_acquisition_function(acq_fn, model)
    assert up_acq_fn == acq_fn

    npt.assert_array_almost_equal(acq_fn(query_at), expected)
    assert acq_fn._get_tracing_count() == 1  # type: ignore


@pytest.mark.parametrize("shape", various_shapes() - {()})
def test_expected_feasibility_builder_raises_on_non_scalar_threshold(
    shape: ShapeLike,
) -> None:
    threshold, alpha, delta = tf.ones(shape), tf.ones(shape), tf.ones(shape)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ExpectedFeasibility(threshold, 1, 1)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ExpectedFeasibility(1, alpha, 1)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ExpectedFeasibility(1, 1, delta)


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_expected_feasibility_builder_raises_on_non_positive_alpha(alpha: float) -> None:

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ExpectedFeasibility(1, alpha, 1)


@pytest.mark.parametrize("delta", [-1, 0, 1.5, 3])
def test_expected_feasibility_raises_for_invalid_delta(delta: int) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ExpectedFeasibility(1, 1, delta)


@pytest.mark.parametrize("delta", [1, 2])
@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_expected_feasibility_raises_for_invalid_batch_size(at: TensorType, delta: int) -> None:
    ef = bichon_ranjan_criterion(QuadraticMeanAndRBFKernel(), 1, 1, delta)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ef(at)


@pytest.mark.parametrize(
    "threshold, at",
    [
        (0.0, tf.constant([[0.0]])),
        (2.0, tf.constant([[1.0]])),
        (-0.25, tf.constant([[-0.5]])),
    ],
)
@pytest.mark.parametrize("delta", [1])
@pytest.mark.parametrize("alpha", [0.1, 1, 2])
def test_bichon_ranjan_criterion(threshold: float, at: tf.Tensor, alpha: float, delta: int) -> None:
    model = QuadraticMeanAndRBFKernel()
    actual = bichon_ranjan_criterion(model, threshold, alpha, delta)(at)

    # approach is to sample based on the model and compute the expectation eq directly
    mean, variance = model.predict(tf.squeeze(at, -2))
    stdev = tf.sqrt(variance)
    normal = tfp.distributions.Normal(tf.cast(0, at.dtype), tf.cast(1, at.dtype))
    samples = normal.sample(1000000)
    t = (threshold - mean) / stdev
    expected = tf.reduce_mean(tf.maximum(0, alpha ** delta - tf.abs(t + samples)))

    npt.assert_allclose(actual, expected, rtol=0.01)

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

from tests.util.misc import (
    TF_DEBUGGING_ERROR_TYPES,
    random_seed,
)
from tests.util.models.gpflow.models import GaussianProcess, QuadraticMeanAndRBFKernel
from tests.util.models.gpflux.models import trieste_deep_gaussian_process
from trieste.acquisition.function.function import (
    PredictiveVariance,
    predictive_variance,
    ExpectedFeasibility
    bichon_ranjan_criterion,
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


def test_expected_feasibility_builder_builds_acquisition_function() -> None:
    model = QuadraticMeanAndRBFKernel()
    acq_fn = ExpectedFeasibility().prepare_acquisition_function(model)
    query_at = tf.linspace([[-10]], [[10]], 100)
    _, covariance = model.predict_joint(query_at)
    expected = tf.linalg.det(covariance)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)


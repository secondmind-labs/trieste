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

from typing import Sequence

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import to_default_float

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, ShapeLike, random_seed, various_shapes
from tests.util.models.gpflow.models import (
    GaussianProcess,
    QuadraticMeanAndRBFKernel,
    gpr_model,
    vgp_model_bernoulli,
)
from tests.util.models.models import binary_line, fnc_2sin_x_over_3
from trieste.acquisition.function.active_learning import (
    BayesianActiveLearningByDisagreement,
    ExpectedFeasibility,
    IntegratedVarianceReduction,
    PredictiveVariance,
    bayesian_active_learning_by_disagreement,
    bichon_ranjan_criterion,
    integrated_variance_reduction,
    predictive_variance,
)
from trieste.data import Dataset
from trieste.models.gpflow import (
    GaussianProcessRegression,
    VariationalGaussianProcess,
    build_vgp_classifier,
)
from trieste.objectives import Branin
from trieste.space import Box
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
    model = GaussianProcess([Branin.objective], [kernel])

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
@pytest.mark.parametrize("delta", [1, 2])
@pytest.mark.parametrize("alpha", [0.1, 1, 2])
def test_bichon_ranjan_criterion(threshold: float, at: tf.Tensor, alpha: float, delta: int) -> None:
    model = QuadraticMeanAndRBFKernel()
    actual = bichon_ranjan_criterion(model, threshold, alpha, delta)(at)

    # approach is to sample based on the model and compute the expectation eq directly
    mean, variance = model.predict(tf.squeeze(at, -2))
    stdev = tf.sqrt(variance)
    normal = tfp.distributions.Normal(mean, stdev)
    samples = normal.sample(1000000)
    expected = tf.reduce_mean(
        tf.maximum(0, (alpha * stdev) ** delta - tf.abs(threshold - samples) ** delta)
    )

    npt.assert_allclose(actual, expected, rtol=0.01)


def test_integrated_variance_reduction() -> None:
    x = to_default_float(tf.constant(np.arange(1, 7).reshape(-1, 1) / 8.0))  # shape: [6, 1]
    y = fnc_2sin_x_over_3(x)

    model6 = GaussianProcessRegression(gpr_model(x, y))
    model5 = GaussianProcessRegression(gpr_model(x[:5, :], y[:5, :]))
    reduced_data = Dataset(x[:5, :], y[:5, :])
    query_points = x[5:, :]
    integration_points = tf.concat([0.37 * x, 1.7 * x], 0)  # shape: [14, 1]

    _, pred_var6 = model6.predict(integration_points)

    acq_noweight = IntegratedVarianceReduction(integration_points=integration_points)
    acq = IntegratedVarianceReduction(threshold=[0.5, 0.8], integration_points=integration_points)

    acq_function = acq.prepare_acquisition_function(model=model5, dataset=reduced_data)
    acq_function_noweight = acq_noweight.prepare_acquisition_function(
        model=model5, dataset=reduced_data
    )
    acq_values = -acq_function(tf.expand_dims(query_points, axis=-2))
    acq_values_noweight = -acq_function_noweight(tf.expand_dims(query_points, axis=-2))

    # Weighted criterion is always smaller than non-weighted
    np.testing.assert_array_less(acq_values, acq_values_noweight)

    # Non-weighted variance integral should match the one with fully updated model
    np.testing.assert_allclose(tf.reduce_mean(pred_var6), acq_values_noweight[0], atol=1e-5)


def test_integrated_variance_reduction_works_with_batch() -> None:
    x = to_default_float(tf.constant(np.arange(1, 8).reshape(-1, 1) / 8.0))  # shape: [7, 1]
    y = fnc_2sin_x_over_3(x)

    model7 = GaussianProcessRegression(gpr_model(x, y))
    model5 = GaussianProcessRegression(gpr_model(x[:5, :], y[:5, :]))
    reduced_data = Dataset(x[:5, :], y[:5, :])
    query_points = tf.expand_dims(x[5:, :], axis=0)  # one batch of 2

    integration_points = tf.concat([0.37 * x, 1.7 * x], 0)  # shape: [14, 1]

    _, pred_var7 = model7.predict(integration_points)

    acq = IntegratedVarianceReduction(integration_points=integration_points)
    acq_function = acq.prepare_acquisition_function(model=model5, dataset=reduced_data)
    acq_values = -acq_function(query_points)

    # Variance integral should match the one with fully updated model
    np.testing.assert_allclose(tf.reduce_mean(pred_var7), acq_values, atol=1e-5)


@pytest.mark.parametrize("integration_points", [tf.zeros([0, 2]), tf.zeros([1, 2, 3])])
def test_integrated_variance_reduction_raises_for_invalid_integration_points(
    integration_points: tf.Tensor,
) -> None:
    threshold = [1.0, 2.0]
    query_at = tf.zeros([1, 1, 1])

    x = to_default_float(tf.constant(np.arange(1, 8).reshape(-1, 1)))
    y = fnc_2sin_x_over_3(x)
    model = GaussianProcessRegression(gpr_model(x, y))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        integrated_variance_reduction(model, integration_points, threshold)(query_at)


@pytest.mark.parametrize("threshold", [[1.0, 2.0, 3.0], tf.zeros([2, 2]), [2.0, 1.0]])
def test_integrated_variance_reduction_raises_for_invalid_threshold(
    threshold: tf.Tensor | Sequence[float],
) -> None:
    integration_points = to_default_float(tf.zeros([5, 1]))
    query_at = tf.zeros([1, 1, 1])

    x = to_default_float(tf.constant(np.arange(1, 8).reshape(-1, 1)))
    y = fnc_2sin_x_over_3(x)
    model = GaussianProcessRegression(gpr_model(x, y))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        integrated_variance_reduction(model, integration_points, threshold)(query_at)


def test_integrated_variance_reduction_builds_acquisition_function() -> None:
    threshold = [1.0, 2.0]
    integration_points = to_default_float(tf.zeros([5, 1]))
    query_at = to_default_float(tf.linspace([[-10]], [[10]], 100))

    x = to_default_float(tf.constant(np.arange(1, 8).reshape(-1, 1) / 8.0))  # shape: [7, 1]
    y = fnc_2sin_x_over_3(x)
    model = GaussianProcessRegression(gpr_model(x, y))
    acq_fn = IntegratedVarianceReduction(
        integration_points, threshold
    ).prepare_acquisition_function(model)
    expected = integrated_variance_reduction(model, integration_points, threshold)(query_at)

    npt.assert_array_almost_equal(acq_fn(query_at), expected)


@pytest.mark.parametrize(
    "at",
    [
        tf.zeros([3, 2]),
        tf.zeros(
            [
                3,
            ]
        ),
    ],
)
def test_integrated_variance_reduction_raises_for_invalid_batch_size(at: TensorType) -> None:
    threshold = [1.0, 2.0]

    integration_points = to_default_float(tf.zeros([3, 1]))
    x = to_default_float(tf.zeros([1, 1]))
    y = to_default_float(tf.zeros([1, 1]))
    model = GaussianProcessRegression(gpr_model(x, y))
    acq_fn = IntegratedVarianceReduction(
        integration_points, threshold
    ).prepare_acquisition_function(model)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        acq_fn(to_default_float(at))


def test_integrated_variance_reduction_builder_updates_without_retracing() -> None:
    threshold = [1.0, 2.0]

    integration_points = to_default_float(tf.zeros([3, 1]))
    x = to_default_float(tf.zeros([1, 1]))
    y = to_default_float(tf.zeros([1, 1]))
    model = GaussianProcessRegression(gpr_model(x, y))

    builder = IntegratedVarianceReduction(integration_points, threshold)
    acq_fn = builder.prepare_acquisition_function(model)
    assert acq_fn.__call__._get_tracing_count() == 0  # type: ignore

    query_at = tf.linspace([[-10]], [[10]], 100)
    expected = integrated_variance_reduction(model, integration_points, threshold)(query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)
    assert acq_fn.__call__._get_tracing_count() == 1  # type: ignore

    up_acq_fn = builder.update_acquisition_function(acq_fn, model)
    assert up_acq_fn == acq_fn

    npt.assert_array_almost_equal(acq_fn(query_at), expected)
    assert acq_fn.__call__._get_tracing_count() == 1  # type: ignore


@pytest.mark.parametrize(
    "at",
    [
        (tf.constant([[[-1.0]]])),
        (tf.constant([[-0.5]])),
        (tf.constant([[0.0]])),
        (tf.constant([[0.5]])),
        (tf.constant([[1.0]])),
    ],
)
def test_bayesian_active_learning_by_disagreement_is_correct(at: tf.Tensor) -> None:
    """
    We perform an MC check as in Section 5 of Houlsby 2011 paper. We check only the
    2nd, more complicated term.
    """
    search_space = Box([-1], [1])
    x = to_default_float(tf.constant(np.linspace(-1, 1, 8).reshape(-1, 1)))
    y = to_default_float(tf.reshape(binary_line(x), [-1, 1]))
    model = VariationalGaussianProcess(
        build_vgp_classifier(Dataset(x, y), search_space, noise_free=True)
    )
    mean, var = model.predict(to_default_float(at))

    def entropy(p: TensorType) -> TensorType:
        return -p * tf.math.log(p + DEFAULTS.JITTER) - (1 - p) * tf.math.log(
            1 - p + DEFAULTS.JITTER
        )

    # we get the actual but substract term 1 which is computed here the same as in the method
    normal = tfp.distributions.Normal(to_default_float(0), to_default_float(1))
    actual_term1 = entropy(normal.cdf((mean / tf.sqrt(var + 1))))
    actual_term2 = actual_term1 - bayesian_active_learning_by_disagreement(model, DEFAULTS.JITTER)(
        [to_default_float(at)]
    )

    # MC based term 2, 1st and 2nd approximation
    samples = tfp.distributions.Normal(
        to_default_float(mean), to_default_float(tf.sqrt(var))
    ).sample(100000)
    MC_term21 = tf.reduce_mean(entropy(normal.cdf(samples)))
    MC_term22 = tf.reduce_mean(np.exp(-(samples**2) / np.pi * np.log(2)))

    npt.assert_allclose(actual_term2, MC_term21, rtol=0.05, atol=0.05)
    npt.assert_allclose(actual_term2, MC_term22, rtol=0.05, atol=0.05)


def test_bayesian_active_learning_by_disagreement_builder_builds_acquisition_function() -> None:
    x = to_default_float(tf.zeros([1, 1]))
    y = to_default_float(tf.zeros([1, 1]))
    model = VariationalGaussianProcess(vgp_model_bernoulli(x, y))
    acq_fn = BayesianActiveLearningByDisagreement().prepare_acquisition_function(model)
    query_at = tf.linspace([[-10]], [[10]], 100)
    expected = bayesian_active_learning_by_disagreement(model, DEFAULTS.JITTER)(query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)


@pytest.mark.parametrize("jitter", [0.0, -1.0])
def test_bayesian_active_learning_by_disagreement_raise_on_non_positive_jitter(
    jitter: float,
) -> None:
    x = to_default_float(tf.zeros([1, 1]))
    y = to_default_float(tf.zeros([1, 1]))
    model = VariationalGaussianProcess(vgp_model_bernoulli(x, y))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BayesianActiveLearningByDisagreement(jitter).prepare_acquisition_function(model)


@pytest.mark.parametrize(
    "x, at, acquisition_shape",
    [
        (tf.zeros([1, 1]), tf.constant([[[1.0]]]), tf.constant([1, 1])),
        (tf.zeros([1, 1]), tf.linspace([[-10.0]], [[10.0]], 5), tf.constant([5, 1])),
        (tf.zeros([1, 2]), tf.constant([[[1.0, 1.0]]]), tf.constant([1, 1])),
        (tf.zeros([1, 2]), tf.linspace([[-10.0, -10.0]], [[10.0, 10.0]], 5), tf.constant([5, 1])),
    ],
)
def test_bayesian_active_learning_by_disagreement_returns_correct_shape(
    x: TensorType, at: TensorType, acquisition_shape: TensorType
) -> None:
    x = to_default_float(x)
    y = to_default_float(tf.zeros([1, 1]))
    model = VariationalGaussianProcess(vgp_model_bernoulli(x, y))
    acq_fn = BayesianActiveLearningByDisagreement().prepare_acquisition_function(model)
    npt.assert_array_equal(acq_fn(to_default_float(at)).shape, acquisition_shape)


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_bayesian_active_learning_by_disagreement_raises_for_invalid_batch_size(
    at: TensorType,
) -> None:
    x = to_default_float(tf.zeros([1, 1]))
    y = to_default_float(tf.zeros([1, 1]))
    model = VariationalGaussianProcess(vgp_model_bernoulli(x, y))
    acq_fn = BayesianActiveLearningByDisagreement().prepare_acquisition_function(model)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        acq_fn(to_default_float(at))


def test_bayesian_active_learning_by_disagreement_builder_updates_without_retracing() -> None:
    x = to_default_float(tf.zeros([1, 1]))
    y = to_default_float(tf.zeros([1, 1]))
    model = VariationalGaussianProcess(vgp_model_bernoulli(x, y))
    builder = BayesianActiveLearningByDisagreement()
    acq_fn = builder.prepare_acquisition_function(model)

    assert acq_fn.__call__._get_tracing_count() == 0  # type: ignore

    query_at = tf.linspace([[-10]], [[10]], 100)
    expected = bayesian_active_learning_by_disagreement(model, DEFAULTS.JITTER)(query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)

    assert acq_fn.__call__._get_tracing_count() == 1  # type: ignore

    up_acq_fn = builder.update_acquisition_function(acq_fn, model)
    assert up_acq_fn == acq_fn

    npt.assert_array_almost_equal(acq_fn(query_at), expected)
    assert acq_fn.__call__._get_tracing_count() == 1  # type: ignore

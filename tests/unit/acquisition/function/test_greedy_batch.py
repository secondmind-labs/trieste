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

from typing import Callable, Mapping, Union

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import to_default_float
from gpflow.utilities.ops import leading_transpose

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, random_seed
from tests.util.models.gpflow.models import QuadraticMeanAndRBFKernel, gpr_model
from tests.util.models.models import fnc_2sin_x_over_3, fnc_3x_plus_10
from trieste.acquisition import (
    ExpectedImprovement,
    MinValueEntropySearch,
    PenalizationFunction,
    UpdatablePenalizationFunction,
)
from trieste.acquisition.function import NegativePredictiveMean, PredictiveVariance
from trieste.acquisition.function.greedy_batch import (
    Fantasizer,
    FantasizerModelOrStack,
    FantasizerModelStack,
    LocalPenalization,
    _generate_fantasized_model,
    hard_local_penalizer,
    soft_local_penalizer,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.models.gpflow import GaussianProcessRegression
from trieste.observer import OBJECTIVE
from trieste.space import Box
from trieste.types import Tag, TensorType


def test_locally_penalized_expected_improvement_builder_raises_for_empty_data() -> None:
    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))
    space = Box([0, 0], [1, 1])
    with pytest.raises(tf.errors.InvalidArgumentError):
        LocalPenalization(search_space=space).prepare_acquisition_function(
            QuadraticMeanAndRBFKernel(),
            dataset=data,
        )
    with pytest.raises(tf.errors.InvalidArgumentError):
        LocalPenalization(search_space=space).prepare_acquisition_function(
            QuadraticMeanAndRBFKernel(),
        )


def test_locally_penalized_expected_improvement_builder_raises_for_invalid_num_samples() -> None:
    search_space = Box([0, 0], [1, 1])
    with pytest.raises(tf.errors.InvalidArgumentError):
        LocalPenalization(search_space, num_samples=-5)


@pytest.mark.parametrize("pending_points", [tf.constant([0.0]), tf.constant([[[0.0], [1.0]]])])
def test_locally_penalized_expected_improvement_builder_raises_for_invalid_pending_points_shape(
    pending_points: TensorType,
) -> None:
    data = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    space = Box([0, 0], [1, 1])
    builder = LocalPenalization(search_space=space)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(QuadraticMeanAndRBFKernel(), data, pending_points)


@random_seed
@pytest.mark.parametrize(
    "base_builder",
    [
        ExpectedImprovement(),
        MinValueEntropySearch(Box([0, 0], [1, 1]), grid_size=10000, num_samples=10),
    ],
)
def test_locally_penalized_acquisitions_match_base_acquisition(
    base_builder: ExpectedImprovement | MinValueEntropySearch[ProbabilisticModel],
) -> None:
    data = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    model = QuadraticMeanAndRBFKernel()

    lp_acq_builder = LocalPenalization(search_space, base_acquisition_function_builder=base_builder)
    lp_acq = lp_acq_builder.prepare_acquisition_function(model, data, None)

    base_acq = base_builder.prepare_acquisition_function(model, dataset=data)

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    lp_acq_values = lp_acq(xs[..., None, :])
    base_acq_values = base_acq(xs[..., None, :])

    if isinstance(base_builder, ExpectedImprovement):
        npt.assert_array_equal(lp_acq_values, base_acq_values)
    else:  # check sampling-based acquisition functions are close
        npt.assert_allclose(lp_acq_values, base_acq_values, atol=0.001)


@random_seed
@pytest.mark.parametrize("penalizer", [soft_local_penalizer, hard_local_penalizer])
@pytest.mark.parametrize(
    "base_builder",
    [ExpectedImprovement(), MinValueEntropySearch(Box([0, 0], [1, 1]), grid_size=5000)],
)
def test_locally_penalized_acquisitions_combine_base_and_penalization_correctly(
    penalizer: Callable[..., Union[PenalizationFunction, UpdatablePenalizationFunction]],
    base_builder: ExpectedImprovement | MinValueEntropySearch[ProbabilisticModel],
) -> None:
    data = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    model = QuadraticMeanAndRBFKernel()
    pending_points = tf.zeros([2, 2], dtype=tf.float64)

    acq_builder = LocalPenalization(
        search_space, penalizer=penalizer, base_acquisition_function_builder=base_builder
    )
    lp_acq = acq_builder.prepare_acquisition_function(model, data, None)  # initialize
    lp_acq = acq_builder.update_acquisition_function(lp_acq, model, data, pending_points[:1], False)
    up_lp_acq = acq_builder.update_acquisition_function(lp_acq, model, data, pending_points, False)
    assert up_lp_acq == lp_acq  # in-place updates

    base_acq = base_builder.prepare_acquisition_function(model, dataset=data)

    best = acq_builder._eta
    lipschitz_constant = acq_builder._lipschitz_constant
    penalizer_value = penalizer(model, pending_points, lipschitz_constant, best)

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    lp_acq_values = lp_acq(xs[..., None, :])
    base_acq_values = base_acq(xs[..., None, :])
    penal_values = penalizer_value(xs[..., None, :])
    penalized_base_acq = tf.math.exp(tf.math.log(base_acq_values) + tf.math.log(penal_values))

    if isinstance(base_builder, ExpectedImprovement):
        npt.assert_array_equal(lp_acq_values, penalized_base_acq)
    else:  # check sampling-based acquisition functions are close
        npt.assert_allclose(lp_acq_values, penalized_base_acq, atol=0.001)


@pytest.mark.parametrize("penalizer", [soft_local_penalizer, hard_local_penalizer])
@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_lipschitz_penalizers_raises_for_invalid_batch_size(
    at: TensorType,
    penalizer: Callable[..., PenalizationFunction],
) -> None:
    pending_points = tf.zeros([1, 2], dtype=tf.float64)
    best = tf.constant([0], dtype=tf.float64)
    lipschitz_constant = tf.constant([1], dtype=tf.float64)
    lp = penalizer(QuadraticMeanAndRBFKernel(), pending_points, lipschitz_constant, best)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        lp(at)


@pytest.mark.parametrize("penalizer", [soft_local_penalizer, hard_local_penalizer])
@pytest.mark.parametrize("pending_points", [tf.constant([0.0]), tf.constant([[[0.0], [1.0]]])])
def test_lipschitz_penalizers_raises_for_invalid_pending_points_shape(
    pending_points: TensorType,
    penalizer: Callable[..., PenalizationFunction],
) -> None:
    best = tf.constant([0], dtype=tf.float64)
    lipschitz_constant = tf.constant([1], dtype=tf.float64)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        penalizer(QuadraticMeanAndRBFKernel(), pending_points, lipschitz_constant, best)


def test_fantasized_expected_improvement_builder_raises_for_invalid_fantasize_method() -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        Fantasizer(ExpectedImprovement().using(OBJECTIVE), "notKB")


def test_fantasized_expected_improvement_builder_raises_for_invalid_model() -> None:
    data = {
        OBJECTIVE: Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 1], dtype=tf.float64))
    }
    models = {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    pending_points = tf.zeros([3, 2], dtype=tf.float64)
    builder = Fantasizer()

    with pytest.raises(NotImplementedError):
        builder.prepare_acquisition_function(models, data, pending_points)  # type: ignore


def test_fantasized_expected_improvement_builder_raises_for_invalid_observation_shape() -> None:
    x = tf.zeros([3, 2], dtype=tf.float64)
    y1 = tf.ones([3, 1], dtype=tf.float64)
    y2 = tf.ones([3, 2], dtype=tf.float64)

    data = {OBJECTIVE: Dataset(x, y1)}
    models = {OBJECTIVE: GaussianProcessRegression(gpr_model(x, y2))}
    pending_points = tf.zeros([3, 2], dtype=tf.float64)
    builder = Fantasizer()

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(models, data, pending_points)


@pytest.mark.parametrize("pending_points", [tf.constant([0.0]), tf.constant([[[0.0], [1.0]]])])
def test_fantasized_expected_improvement_builder_raises_for_invalid_pending_points_shape(
    pending_points: TensorType,
) -> None:
    x = tf.zeros([3, 2], dtype=tf.float64)
    y = tf.ones([3, 1], dtype=tf.float64)

    data = {OBJECTIVE: Dataset(x, y)}
    models = {OBJECTIVE: GaussianProcessRegression(gpr_model(x, y))}

    builder = Fantasizer()
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(models, data, pending_points)


@pytest.mark.parametrize("model_type", ["gpr", "stack"])
def test_fantasize_with_kriging_believer_does_not_change_negative_predictive_mean(
    model_type: str,
) -> None:
    x = to_default_float(tf.constant(np.arange(1, 6).reshape(-1, 1) / 5.0))
    y = fnc_2sin_x_over_3(x)

    x_test = to_default_float(tf.constant(np.arange(1, 13).reshape(-1, 1) / 12.0))[..., None]
    pending_points = to_default_float(tf.constant([0.51, 0.81])[:, None])

    data = {OBJECTIVE: Dataset(x, y)}
    models: Mapping[Tag, FantasizerModelOrStack]
    if model_type == "stack":
        models = {OBJECTIVE: FantasizerModelStack((GaussianProcessRegression(gpr_model(x, y)), 1))}
    else:
        models = {OBJECTIVE: GaussianProcessRegression(gpr_model(x, y))}

    builder = Fantasizer(NegativePredictiveMean())
    acq0 = builder.prepare_acquisition_function(models, data)
    acq1 = builder.prepare_acquisition_function(models, data, pending_points)

    acq_val0 = acq0(x_test)
    acq_val1 = acq1(x_test)

    tf.assert_equal(acq_val1, acq_val0)


@pytest.mark.parametrize("model_type", ["gpr", "stack"])
@pytest.mark.parametrize("fantasize_method", ["KB", "sample"])
def test_fantasize_reduces_predictive_variance(model_type: str, fantasize_method: str) -> None:
    x = to_default_float(tf.constant(np.arange(1, 6).reshape(-1, 1) / 5.0))
    y = fnc_2sin_x_over_3(x)

    x_test = to_default_float(tf.constant(np.arange(1, 13).reshape(-1, 1) / 12.0))[..., None]
    pending_points = to_default_float(tf.constant([0.51, 0.81])[:, None])

    data = {OBJECTIVE: Dataset(x, y)}
    models: Mapping[Tag, FantasizerModelOrStack]
    if model_type == "stack":
        models = {OBJECTIVE: FantasizerModelStack((GaussianProcessRegression(gpr_model(x, y)), 1))}
    else:
        models = {OBJECTIVE: GaussianProcessRegression(gpr_model(x, y))}

    builder = Fantasizer(PredictiveVariance(), fantasize_method=fantasize_method)
    acq0 = builder.prepare_acquisition_function(models, data)
    acq1 = builder.update_acquisition_function(acq0, models, data, pending_points[:1])
    assert acq0._get_tracing_count() == 0  # type: ignore
    assert acq1._get_tracing_count() == 0  # type: ignore

    acq_val0 = acq0(x_test)
    acq_val1 = acq1(x_test)
    tf.assert_less(acq_val1, acq_val0)

    # check we avoid retracing, both for the fantasized functions...
    acq1_up = builder.update_acquisition_function(acq1, models, data, pending_points)
    assert acq1_up == acq1  # in-place updates
    acq1_up(x_test)
    assert acq1_up._get_tracing_count() == 1  # type: ignore

    # ...and the base functions
    acq0_up = builder.update_acquisition_function(acq1, models, data)
    assert acq0_up == acq0  # in-place updates
    acq0_up(x_test)
    assert acq0_up._get_tracing_count() == 1  # type: ignore


@pytest.mark.parametrize("model_type", ["gpr", "stack"])
def test_fantasize_allows_query_points_with_leading_dimensions(model_type: str) -> None:
    x = to_default_float(tf.constant(np.arange(1, 24).reshape(-1, 1) / 8.0))  # shape: [23, 1]
    y = fnc_2sin_x_over_3(x)

    model5 = GaussianProcessRegression(gpr_model(x[:5, :], y[:5, :]))

    additional_data = Dataset(tf.reshape(x[5:, :], [3, 6, -1]), tf.reshape(y[5:, :], [3, 6, -1]))

    query_points = to_default_float(tf.constant(np.arange(1, 21).reshape(-1, 1) / 20.0))[..., None]
    query_points = tf.reshape(query_points, [4, 5, 1])

    if model_type == "stack":
        fanta_model5 = _generate_fantasized_model(
            FantasizerModelStack((model5, 1)), additional_data
        )
    else:
        fanta_model5 = _generate_fantasized_model(model5, additional_data)

    num_samples = 100000
    samples_fm5 = fanta_model5.sample(query_points, num_samples)
    pred_f_mean_fm5, pred_f_var_fm5 = fanta_model5.predict(query_points)
    pred_y_mean_fm5, pred_y_var_fm5 = fanta_model5.predict_y(query_points)
    pred_j_mean_fm5, pred_j_cov_fm5 = fanta_model5.predict_joint(query_points)

    tf.assert_equal(samples_fm5.shape, [4, 3, num_samples, 5, 1])
    tf.assert_equal(pred_f_mean_fm5.shape, [4, 3, 5, 1])
    tf.assert_equal(pred_f_var_fm5.shape, [4, 3, 5, 1])
    tf.assert_equal(pred_j_cov_fm5.shape, [4, 3, 1, 5, 5])

    np.testing.assert_allclose(pred_f_mean_fm5, pred_j_mean_fm5, atol=1e-5)
    np.testing.assert_allclose(pred_f_mean_fm5, pred_y_mean_fm5, atol=1e-5)

    samples_fm5_mean = tf.reduce_mean(samples_fm5, axis=-3)
    samples_fm5_cov = tfp.stats.covariance(samples_fm5[..., 0], sample_axis=-2)

    for j in range(3):
        samples_m5 = model5.conditional_predict_f_sample(
            query_points[j], additional_data, num_samples
        )

        pred_f_mean_m5, pred_f_var_m5 = model5.conditional_predict_f(
            query_points[j], additional_data
        )
        pred_j_mean_m5, pred_j_cov_m5 = model5.conditional_predict_joint(
            query_points[j], additional_data
        )
        pred_y_mean_m5, pred_y_var_m5 = model5.conditional_predict_y(
            query_points[j], additional_data
        )

        sample_m5_mean = tf.reduce_mean(samples_m5, axis=1)
        sample_m5_cov = tfp.stats.covariance(samples_m5[..., 0], sample_axis=1)

        np.testing.assert_allclose(sample_m5_mean, samples_fm5_mean[j], atol=1e-2, rtol=1e-2)
        np.testing.assert_allclose(sample_m5_cov, samples_fm5_cov[j], atol=1e-2, rtol=1e-2)

        np.testing.assert_allclose(pred_f_mean_m5, pred_f_mean_fm5[j], atol=1e-5)
        np.testing.assert_allclose(pred_y_mean_m5, pred_y_mean_fm5[j], atol=1e-5)
        np.testing.assert_allclose(pred_j_mean_m5, pred_j_mean_fm5[j], atol=1e-5)

        np.testing.assert_allclose(pred_f_var_m5, pred_f_var_fm5[j], atol=1e-5)
        np.testing.assert_allclose(pred_y_var_m5, pred_y_var_fm5[j], atol=1e-5)
        np.testing.assert_allclose(pred_j_cov_m5, pred_j_cov_fm5[j], atol=1e-5)


def test_fantasized_stack_is_the_same_as_individually_fantasized() -> None:
    x = to_default_float(tf.constant(np.arange(1, 24).reshape(-1, 1) / 8.0))  # shape: [23, 1]
    y1 = fnc_2sin_x_over_3(x)
    y2 = fnc_3x_plus_10(x)
    model1 = GaussianProcessRegression(gpr_model(x[:5, :], y1[:5, :]))
    model2 = GaussianProcessRegression(gpr_model(x[:5, :], y2[:5, :]))
    stacked_models = FantasizerModelStack((model1, 1), (model2, 1))

    additional_data1 = Dataset(tf.reshape(x[5:, :], [3, 6, -1]), tf.reshape(y1[5:, :], [3, 6, -1]))
    additional_data2 = Dataset(tf.reshape(x[5:, :], [3, 6, -1]), tf.reshape(y2[5:, :], [3, 6, -1]))

    additional_data_stacked = Dataset(
        tf.reshape(x[5:, :], [3, 6, -1]),
        tf.reshape(tf.concat([y1[5:, :], y2[5:, :]], axis=-1), [3, 6, -1]),
    )

    query_points = to_default_float(tf.constant(np.arange(1, 21).reshape(-1, 1) / 20.0))[..., None]
    query_points = tf.reshape(query_points, [4, 5, 1])

    stack_fanta_model = _generate_fantasized_model(stacked_models, additional_data_stacked)

    fanta_model1 = _generate_fantasized_model(model1, additional_data1)
    fanta_model2 = _generate_fantasized_model(model2, additional_data2)

    num_samples = 100000
    samples_fm1 = fanta_model1.sample(query_points, num_samples)
    pred_f_mean_fm1, pred_f_var_fm1 = fanta_model1.predict(query_points)
    pred_y_mean_fm1, pred_y_var_fm1 = fanta_model1.predict_y(query_points)
    pred_j_mean_fm1, pred_j_cov_fm1 = fanta_model1.predict_joint(query_points)

    samples_fm2 = fanta_model2.sample(query_points, num_samples)
    pred_f_mean_fm2, pred_f_var_fm2 = fanta_model2.predict(query_points)
    pred_y_mean_fm2, pred_y_var_fm2 = fanta_model2.predict_y(query_points)
    pred_j_mean_fm2, pred_j_cov_fm2 = fanta_model2.predict_joint(query_points)

    samples_fms = stack_fanta_model.sample(query_points, num_samples)
    pred_f_mean_fms, pred_f_var_fms = stack_fanta_model.predict(query_points)
    pred_y_mean_fms, pred_y_var_fms = stack_fanta_model.predict_y(query_points)
    pred_j_mean_fms, pred_j_cov_fms = stack_fanta_model.predict_joint(query_points)

    np.testing.assert_equal(pred_f_mean_fms.shape, [4, 3, 5, 2])
    np.testing.assert_equal(pred_f_var_fms.shape, [4, 3, 5, 2])
    np.testing.assert_equal(pred_f_mean_fm1.shape, [4, 3, 5, 1])
    np.testing.assert_equal(pred_f_var_fm1.shape, [4, 3, 5, 1])
    np.testing.assert_equal(pred_j_cov_fms.shape, [4, 3, 2, 5, 5])
    np.testing.assert_equal(pred_j_cov_fm1.shape, [4, 3, 1, 5, 5])

    np.testing.assert_equal(samples_fms.shape, [4, 3, 100000, 5, 2])
    np.testing.assert_equal(samples_fm1.shape, [4, 3, 100000, 5, 1])

    np.testing.assert_allclose(
        pred_f_mean_fms, tf.concat([pred_f_mean_fm1, pred_f_mean_fm2], axis=-1), atol=1e-5
    )
    np.testing.assert_allclose(
        pred_y_mean_fms, tf.concat([pred_y_mean_fm1, pred_y_mean_fm2], axis=-1), atol=1e-5
    )
    np.testing.assert_allclose(
        pred_j_mean_fms, tf.concat([pred_j_mean_fm1, pred_j_mean_fm2], axis=-1), atol=1e-5
    )

    np.testing.assert_allclose(
        pred_f_var_fms, tf.concat([pred_f_var_fm1, pred_f_var_fm2], axis=-1), atol=1e-5
    )
    np.testing.assert_allclose(
        pred_y_var_fms, tf.concat([pred_y_var_fm1, pred_y_var_fm2], axis=-1), atol=1e-5
    )
    np.testing.assert_allclose(
        pred_j_cov_fms, tf.concat([pred_j_cov_fm1, pred_j_cov_fm2], axis=-3), atol=1e-5
    )

    sample_fms_mean = tf.reduce_mean(samples_fms, axis=2)
    sample_fms_cov = tfp.stats.covariance(
        leading_transpose(samples_fms, [..., -1, -2]), sample_axis=2
    )

    sample_fm1_mean = tf.reduce_mean(samples_fm1, axis=2)
    sample_fm1_cov = tfp.stats.covariance(samples_fm1[..., 0], sample_axis=2, keepdims=True)
    sample_fm2_mean = tf.reduce_mean(samples_fm2, axis=2)
    sample_fm2_cov = tfp.stats.covariance(samples_fm2[..., 0], sample_axis=2, keepdims=True)

    np.testing.assert_allclose(
        sample_fms_mean,
        tf.concat([sample_fm1_mean, sample_fm2_mean], axis=-1),
        atol=1e-2,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        sample_fms_cov, tf.concat([sample_fm1_cov, sample_fm2_cov], axis=2), atol=1e-2, rtol=1e-2
    )

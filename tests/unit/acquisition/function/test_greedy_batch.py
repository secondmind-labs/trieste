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

from typing import Callable, Union

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
from gpflow.utilities import to_default_float

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, random_seed
from tests.util.models.gpflow.models import QuadraticMeanAndRBFKernel, gpr_model
from tests.util.models.models import fnc_2sin_x_over_3
from trieste.acquisition import (
    ExpectedImprovement,
    MinValueEntropySearch,
    PenalizationFunction,
    UpdatablePenalizationFunction,
)
from trieste.acquisition.function import NegativePredictiveMean, PredictiveVariance
from trieste.acquisition.function.greedy_batch import (
    FantasizeAcquisitionFunction,
    LocalPenalizationAcquisitionFunction,
    hard_local_penalizer,
    soft_local_penalizer,
)
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import Box
from trieste.types import TensorType


def test_locally_penalized_expected_improvement_builder_raises_for_empty_data() -> None:
    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))
    space = Box([0, 0], [1, 1])
    with pytest.raises(tf.errors.InvalidArgumentError):
        LocalPenalizationAcquisitionFunction(search_space=space).prepare_acquisition_function(
            QuadraticMeanAndRBFKernel(),
            dataset=data,
        )
    with pytest.raises(tf.errors.InvalidArgumentError):
        LocalPenalizationAcquisitionFunction(search_space=space).prepare_acquisition_function(
            QuadraticMeanAndRBFKernel(),
        )


def test_locally_penalized_expected_improvement_builder_raises_for_invalid_num_samples() -> None:
    search_space = Box([0, 0], [1, 1])
    with pytest.raises(tf.errors.InvalidArgumentError):
        LocalPenalizationAcquisitionFunction(search_space, num_samples=-5)


@pytest.mark.parametrize("pending_points", [tf.constant([0.0]), tf.constant([[[0.0], [1.0]]])])
def test_locally_penalized_expected_improvement_builder_raises_for_invalid_pending_points_shape(
    pending_points: TensorType,
) -> None:
    data = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    space = Box([0, 0], [1, 1])
    builder = LocalPenalizationAcquisitionFunction(search_space=space)
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
    base_builder: ExpectedImprovement | MinValueEntropySearch,
) -> None:
    data = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    model = QuadraticMeanAndRBFKernel()

    lp_acq_builder = LocalPenalizationAcquisitionFunction(
        search_space, base_acquisition_function_builder=base_builder
    )
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
    base_builder: ExpectedImprovement | MinValueEntropySearch,
) -> None:
    data = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    model = QuadraticMeanAndRBFKernel()
    pending_points = tf.zeros([2, 2], dtype=tf.float64)

    acq_builder = LocalPenalizationAcquisitionFunction(
        search_space, penalizer=penalizer, base_acquisition_function_builder=base_builder
    )
    lp_acq = acq_builder.prepare_acquisition_function(model, data, None)  # initialize
    lp_acq = acq_builder.update_acquisition_function(lp_acq, model, data, pending_points[:1], False)
    up_lp_acq = acq_builder.update_acquisition_function(lp_acq, model, data, pending_points, False)
    assert up_lp_acq == lp_acq  # in-place updates

    base_acq = base_builder.prepare_acquisition_function(model, dataset=data)

    best = acq_builder._eta
    lipshitz_constant = acq_builder._lipschitz_constant
    penalizer = penalizer(model, pending_points, lipshitz_constant, best)

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    lp_acq_values = lp_acq(xs[..., None, :])
    base_acq_values = base_acq(xs[..., None, :])
    penal_values = penalizer(xs[..., None, :])
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
    lipshitz_constant = tf.constant([1], dtype=tf.float64)
    lp = penalizer(QuadraticMeanAndRBFKernel(), pending_points, lipshitz_constant, best)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        lp(at)


@pytest.mark.parametrize("penalizer", [soft_local_penalizer, hard_local_penalizer])
@pytest.mark.parametrize("pending_points", [tf.constant([0.0]), tf.constant([[[0.0], [1.0]]])])
def test_lipschitz_penalizers_raises_for_invalid_pending_points_shape(
    pending_points: TensorType,
    penalizer: Callable[..., PenalizationFunction],
) -> None:
    best = tf.constant([0], dtype=tf.float64)
    lipshitz_constant = tf.constant([1], dtype=tf.float64)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        soft_local_penalizer(QuadraticMeanAndRBFKernel(), pending_points, lipshitz_constant, best)


def test_fantasized_expected_improvement_builder_raises_for_invalid_num_samples() -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        FantasizeAcquisitionFunction(ExpectedImprovement().using("OBJECTIVE"), "notKB")


def test_fantasized_expected_improvement_builder_raises_for_invalid_model() -> None:
    data = {
        "OBJECTIVE": Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    }
    models = {"OBJECTIVE": QuadraticMeanAndRBFKernel()}
    pending_points = tf.zeros([3, 2], dtype=tf.float64)
    builder = FantasizeAcquisitionFunction()

    with pytest.raises(NotImplementedError):
        builder.prepare_acquisition_function(models, data, pending_points)


@pytest.mark.parametrize("pending_points", [tf.constant([0.0]), tf.constant([[[0.0], [1.0]]])])
def test_fantasized_expected_improvement_builder_raises_for_invalid_pending_points_shape(
    pending_points: TensorType,
) -> None:
    x = tf.zeros([3, 2], dtype=tf.float64)
    y = tf.ones([3, 1], dtype=tf.float64)

    data = {"OBJECTIVE": Dataset(x, y)}
    models = {"OBJECTIVE": GaussianProcessRegression(gpr_model(x, y))}

    builder = FantasizeAcquisitionFunction()
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(models, data, pending_points)


def test_fantasize_with_kriging_believer_does_not_change_negative_predictive_mean() -> None:
    x = to_default_float(tf.constant(np.arange(1, 6).reshape(-1, 1) / 5.0))
    y = fnc_2sin_x_over_3(x)

    x_test = to_default_float(tf.constant(np.arange(1, 13).reshape(-1, 1) / 12.0))[..., None]
    pending_points = to_default_float(tf.constant([0.51, 0.81])[:, None])

    data = {"OBJECTIVE": Dataset(x, y)}
    models = {"OBJECTIVE": GaussianProcessRegression(gpr_model(x, y))}

    builder = FantasizeAcquisitionFunction(NegativePredictiveMean())
    acq0 = builder.prepare_acquisition_function(models, data)
    acq1 = builder.prepare_acquisition_function(models, data, pending_points)

    acq_val0 = acq0(x_test)
    acq_val1 = acq1(x_test)

    tf.assert_equal(acq_val1, acq_val0)


def test_fantasize_reduces_predictive_variance() -> None:
    x = to_default_float(tf.constant(np.arange(1, 6).reshape(-1, 1) / 5.0))
    y = fnc_2sin_x_over_3(x)

    x_test = to_default_float(tf.constant(np.arange(1, 13).reshape(-1, 1) / 12.0))[..., None]
    pending_points = to_default_float(tf.constant([0.51, 0.81])[:, None])

    data = {"OBJECTIVE": Dataset(x, y)}
    models = {"OBJECTIVE": GaussianProcessRegression(gpr_model(x, y))}

    builder = FantasizeAcquisitionFunction(PredictiveVariance())
    acq0 = builder.prepare_acquisition_function(models, data)
    acq1 = builder.prepare_acquisition_function(models, data, pending_points)

    acq_val0 = acq0(x_test)
    acq_val1 = acq1(x_test)
    tf.assert_less(acq_val1, acq_val0)


test_fantasize_reduces_predictive_variance()


def test_fantasize_allows_query_points_with_leading_dimensions() -> None:
    x = to_default_float(tf.constant(np.arange(1, 6).reshape(-1, 1) / 5.0))
    y = fnc_2sin_x_over_3(x)

    x_test = to_default_float(tf.constant(np.arange(1, 13).reshape(-1, 1) / 12.0))[..., None]
    pending_points = to_default_float(tf.constant([0.51, 0.81])[:, None])

    data = {"OBJECTIVE": Dataset(x, y)}
    models = {"OBJECTIVE": GaussianProcessRegression(gpr_model(x, y))}

    builder = FantasizeAcquisitionFunction(PredictiveVariance())
    acq0 = builder.prepare_acquisition_function(models, data)
    acq1 = builder.prepare_acquisition_function(models, data, pending_points)

    acq_val0 = acq0(x_test)
    acq_val1 = acq1(x_test)
    tf.assert_less(acq_val1, acq_val0)

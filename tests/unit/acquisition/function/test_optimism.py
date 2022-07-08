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

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES
from tests.util.models.gpflow.models import QuadraticMeanAndRBFKernel
from trieste.acquisition import (
    MultipleOptimismNegativeLowerConfidenceBound,
    NegativeLowerConfidenceBound,
    lower_confidence_bound,
    multiple_optimism_lower_confidence_bound,
)
from trieste.space import Box
from trieste.types import TensorType


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

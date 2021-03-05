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

import copy
from collections.abc import Mapping

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import one_dimensional_range, quadratic, random_seed, zero_dataset
from tests.util.model import QuadraticMeanAndRBFKernel
from trieste.acquisition import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    BatchAcquisitionFunction,
    BatchAcquisitionFunctionBuilder,
    NegativeLowerConfidenceBound,
)
from trieste.acquisition.optimizer import AcquisitionOptimizer
from trieste.acquisition.rule import (
    OBJECTIVE,
    BatchAcquisitionRule,
    EfficientGlobalOptimization,
    ThompsonSampling,
    TrustRegion,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.space import Box
from trieste.type import TensorType


def _line_search_minimize(search_space: Box, f: AcquisitionFunction) -> TensorType:
    xs = tf.linspace(search_space.lower, search_space.upper, 10 ** 6)
    return xs[tf.squeeze(tf.argmax(f(xs))), None]


@pytest.mark.parametrize("optimizer", [_line_search_minimize, None])
def test_ego(optimizer: AcquisitionOptimizer[Box]) -> None:
    class NegQuadratic(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return lambda x: -quadratic(x - 1)

    search_space = Box([-10], [10])
    ego = EfficientGlobalOptimization(NegQuadratic(), optimizer)
    query_point, _ = ego.acquire(
        search_space, {OBJECTIVE: zero_dataset()}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    )
    npt.assert_allclose(query_point, [[1]], rtol=1e-4)


@pytest.mark.parametrize(
    "models",
    [
        {},
        {"foo": QuadraticMeanAndRBFKernel()},
        {"foo": QuadraticMeanAndRBFKernel(), OBJECTIVE: QuadraticMeanAndRBFKernel()},
    ],
)
@pytest.mark.parametrize("datasets", [{}, {OBJECTIVE: zero_dataset()}])
def test_thompson_sampling_raises_for_invalid_models_keys(
    datasets: dict[str, Dataset], models: dict[str, ProbabilisticModel]
) -> None:
    search_space = one_dimensional_range(-1, 1)
    rule = ThompsonSampling(100, 10)
    with pytest.raises(ValueError):
        rule.acquire(search_space, datasets, models)


@pytest.mark.parametrize("datasets", [{}, {"foo": zero_dataset()}])
@pytest.mark.parametrize(
    "models", [{}, {"foo": QuadraticMeanAndRBFKernel()}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}]
)
def test_trust_region_raises_for_missing_datasets_key(
    datasets: dict[str, Dataset], models: dict[str, ProbabilisticModel]
) -> None:
    search_space = one_dimensional_range(-1, 1)
    rule = TrustRegion()
    with pytest.raises(KeyError):
        rule.acquire(search_space, datasets, models, None)


def test_trust_region_for_default_state() -> None:
    tr = TrustRegion(NegativeLowerConfidenceBound(0).using(OBJECTIVE))
    dataset = Dataset(tf.constant([[0.1, 0.2]]), tf.constant([[0.012]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    query_point, state = tr.acquire(
        search_space, {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}, None
    )

    npt.assert_array_almost_equal(query_point, tf.constant([[0.0, 0.0]]), 5)
    npt.assert_array_almost_equal(state.acquisition_space.lower, lower_bound)
    npt.assert_array_almost_equal(state.acquisition_space.upper, upper_bound)
    npt.assert_array_almost_equal(state.y_min, [0.012])
    assert state.is_global


def test_trust_region_successful_global_to_global_trust_region_unchanged() -> None:
    tr = TrustRegion(NegativeLowerConfidenceBound(0).using(OBJECTIVE))
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.3]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = True
    previous_state = TrustRegion.State(search_space, eps, previous_y_min, is_global)

    query_point, current_state = tr.acquire(
        search_space, {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}, previous_state
    )

    npt.assert_array_almost_equal(current_state.eps, previous_state.eps)
    assert current_state.is_global
    npt.assert_array_almost_equal(query_point, tf.constant([[0.0, 0.0]]), 5)
    npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)
    npt.assert_array_almost_equal(current_state.acquisition_space.upper, upper_bound)


def test_trust_region_for_unsuccessful_global_to_local_trust_region_unchanged() -> None:
    tr = TrustRegion(NegativeLowerConfidenceBound(0).using(OBJECTIVE))
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.5]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = True
    acquisition_space = search_space
    previous_state = TrustRegion.State(acquisition_space, eps, previous_y_min, is_global)

    query_point, current_state = tr.acquire(
        search_space, {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}, previous_state
    )

    npt.assert_array_almost_equal(current_state.eps, previous_state.eps)
    assert not current_state.is_global
    npt.assert_array_less(lower_bound, current_state.acquisition_space.lower)
    npt.assert_array_less(current_state.acquisition_space.upper, upper_bound)
    assert query_point[0] in current_state.acquisition_space


def test_trust_region_for_successful_local_to_global_trust_region_increased() -> None:
    tr = TrustRegion(NegativeLowerConfidenceBound(0).using(OBJECTIVE))
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.3]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = False
    acquisition_space = Box(dataset.query_points[0] - eps, dataset.query_points[0] + eps)
    previous_state = TrustRegion.State(acquisition_space, eps, previous_y_min, is_global)

    _, current_state = tr.acquire(
        search_space, {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}, previous_state
    )

    npt.assert_array_less(previous_state.eps, current_state.eps)  # current TR larger than previous
    assert current_state.is_global
    npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)
    npt.assert_array_almost_equal(current_state.acquisition_space.upper, upper_bound)


def test_trust_region_for_unsuccessful_local_to_global_trust_region_reduced() -> None:
    tr = TrustRegion(NegativeLowerConfidenceBound(0).using(OBJECTIVE))
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.5]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = False
    acquisition_space = Box(dataset.query_points[0] - eps, dataset.query_points[0] + eps)
    previous_state = TrustRegion.State(acquisition_space, eps, previous_y_min, is_global)

    _, current_state = tr.acquire(
        search_space, {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}, previous_state
    )

    npt.assert_array_less(current_state.eps, previous_state.eps)  # current TR smaller than previous
    assert current_state.is_global
    npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)


def test_trust_region_state_deepcopy() -> None:
    tr_state = TrustRegion.State(
        Box(tf.constant([1.2]), tf.constant([3.4])), tf.constant(5.6), tf.constant(7.8), False
    )
    tr_state_copy = copy.deepcopy(tr_state)
    npt.assert_allclose(tr_state_copy.acquisition_space.lower, tr_state.acquisition_space.lower)
    npt.assert_allclose(tr_state_copy.acquisition_space.upper, tr_state.acquisition_space.upper)
    npt.assert_allclose(tr_state_copy.eps, tr_state.eps)
    npt.assert_allclose(tr_state_copy.y_min, tr_state.y_min)
    assert tr_state_copy.is_global == tr_state.is_global


class _BatchModelMinusMeanMaximumSingleBuilder(BatchAcquisitionFunctionBuilder):
    def prepare_acquisition_function(
        self, dataset: Mapping[str, Dataset], model: Mapping[str, ProbabilisticModel]
    ) -> BatchAcquisitionFunction:
        return lambda at: -tf.reduce_max(model[OBJECTIVE].predict(at)[0], axis=-2)


@random_seed
def test_batch_acquisition_rule_acquire() -> None:
    search_space = Box(tf.constant([-2.2, -1.0]), tf.constant([1.3, 3.3]))
    num_query_points = 4
    acq = _BatchModelMinusMeanMaximumSingleBuilder()
    ego: BatchAcquisitionRule[Box] = BatchAcquisitionRule(num_query_points, acq)
    dataset = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))
    query_point, _ = ego.acquire(
        search_space, {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    )

    npt.assert_allclose(query_point, [[0.0, 0.0]] * num_query_points, atol=1e-3)

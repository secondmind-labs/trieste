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
import copy
from typing import Dict, List, Mapping, Union

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.model import QuadraticMeanAndRBFKernel
from tests.util.misc import mk_dataset, random_seed, zero_dataset
from trieste.acquisition.function import (
    BatchAcquisitionFunction,
    BatchAcquisitionFunctionBuilder,
    NegativeLowerConfidenceBound,
)
from trieste.acquisition.rule import (
    OBJECTIVE,
    BatchAcquisitionRule,
    EfficientGlobalOptimization,
    ThompsonSampling,
    TrustRegion,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.space import Box, DiscreteSearchSpace


@pytest.mark.parametrize(
    "search_space, expected_minimum",
    [
        (
            DiscreteSearchSpace(tf.constant([[-2.2, -1.0], [0.1, -0.1], [1.3, 3.3]], tf.float64)),
            [[0.1, -0.1]],
        ),
        (Box([-2.2, -1.0], [1.3, 3.3]), [[0.0, 0.0]]),
    ],
)
def test_efficient_global_optimization(
    search_space: Union[Box, DiscreteSearchSpace], expected_minimum: List[List[float]]
) -> None:
    ego = EfficientGlobalOptimization(NegativeLowerConfidenceBound(0).using(""))
    dataset = Dataset(tf.zeros([0, 2], tf.float64), tf.zeros([0, 1], tf.float64))
    query_point, _ = ego.acquire(search_space, {"": dataset}, {"": QuadraticMeanAndRBFKernel()})
    npt.assert_allclose(query_point, expected_minimum, atol=1e-5)


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
    datasets: Dict[str, Dataset], models: Dict[str, ProbabilisticModel]
) -> None:
    search_space = Box([-1], [1])
    rule = ThompsonSampling(100, 10)
    with pytest.raises(ValueError):
        rule.acquire(search_space, datasets, models)


@pytest.mark.parametrize("datasets", [{}, {"foo": zero_dataset()}])
@pytest.mark.parametrize(
    "models", [{}, {"foo": QuadraticMeanAndRBFKernel()}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}]
)
def test_trust_region_raises_for_missing_datasets_key(
    datasets: Dict[str, Dataset], models: Dict[str, ProbabilisticModel]
) -> None:
    search_space = Box([-1], [1])
    rule = TrustRegion()
    with pytest.raises(KeyError):
        rule.acquire(search_space, datasets, models, None)


# todo test trust region __init__ parameters


@random_seed
def test_trust_region_for_initial_state() -> None:
    tr = TrustRegion(NegativeLowerConfidenceBound(0).using(OBJECTIVE))
    dataset = mk_dataset([[0.1, 0.2]], [[0.012]])
    search_space = Box([-2.2, -1.0], [1.3, 3.3])

    query_points, state = tr.acquire(
        search_space, {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    )

    npt.assert_allclose(query_points, [[0.0, 0.0]], atol=1e-5)
    npt.assert_allclose(state.acquisition_space.lower, search_space.lower)
    npt.assert_allclose(state.acquisition_space.upper, search_space.upper)
    npt.assert_allclose(state.eps, [1.75, 2.15] / tf.sqrt(5.0))
    npt.assert_allclose(state.y_min, [0.012])
    assert state.is_global


@random_seed
def test_trust_region_for_successful_global_step() -> None:
    tr = TrustRegion(NegativeLowerConfidenceBound(0).using(OBJECTIVE))
    dataset = mk_dataset([[0.1, 0.2], [-0.1, -0.2]], [[0.4], [0.3]])
    search_space = Box([-2.2, -1.0], [1.3, 3.3])

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_state = TrustRegion.State(search_space, eps, dataset.observations[0], True)

    query_points, current_state = tr.acquire(
        search_space, {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}, previous_state
    )

    npt.assert_allclose(query_points, [[0.0, 0.0]], atol=1e-6)
    npt.assert_allclose(current_state.acquisition_space.lower, search_space.lower)
    npt.assert_allclose(current_state.acquisition_space.upper, search_space.upper)
    npt.assert_allclose(current_state.eps, previous_state.eps)
    npt.assert_allclose(current_state.y_min, [0.3])
    assert current_state.is_global


@random_seed
def test_trust_region_for_unsuccessful_global_step() -> None:
    tr = TrustRegion(NegativeLowerConfidenceBound(0).using(OBJECTIVE))
    dataset = mk_dataset([[0.2, 0.3], [-0.2, -0.3]], [[1.6], [2.0]])
    search_space = Box([-2.2, -1.0], [1.3, 3.3])

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_state = TrustRegion.State(search_space, eps, dataset.observations[0], True)

    query_points, current_state = tr.acquire(
        search_space, {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}, previous_state
    )

    npt.assert_allclose(query_points, [[0.025, 0.085]], atol=2e-4)
    npt.assert_allclose(
        current_state.acquisition_space.lower + current_state.acquisition_space.upper, [0.4, 0.6]
    )
    npt.assert_allclose(
        current_state.acquisition_space.upper - current_state.acquisition_space.lower, [0.35, 0.43]
    )
    npt.assert_allclose(current_state.eps, previous_state.eps)
    npt.assert_allclose(current_state.y_min, [1.6])
    assert not current_state.is_global


@random_seed
def test_trust_region_for_successful_local_step() -> None:
    tr = TrustRegion(NegativeLowerConfidenceBound(0).using(OBJECTIVE))
    dataset = mk_dataset([[0.1, 0.2], [-0.1, -0.2]], [[0.4], [0.3]])
    search_space = Box([-2.2, -1.0], [1.3, 3.3])

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    acquisition_space = Box(dataset.query_points[0] - eps, dataset.query_points[0] + eps)
    previous_state = TrustRegion.State(acquisition_space, eps, dataset.observations[0], False)

    query_points, current_state = tr.acquire(
        search_space, {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}, previous_state
    )

    npt.assert_allclose(query_points, [[0.0, 0.0]], atol=1e-5)
    npt.assert_allclose(current_state.acquisition_space.lower, search_space.lower)
    npt.assert_allclose(current_state.acquisition_space.upper, search_space.upper)
    npt.assert_allclose(current_state.eps, previous_state.eps / 0.7)
    npt.assert_allclose(current_state.y_min, [0.3])
    assert current_state.is_global


@random_seed
def test_trust_region_for_unsuccessful_local_step() -> None:
    tr = TrustRegion(NegativeLowerConfidenceBound(0).using(OBJECTIVE))
    dataset = mk_dataset([[0.1, 0.2], [-0.1, -0.2]], [[0.4], [0.5]])
    search_space = Box([-2.2, -1.0], [1.3, 3.3])

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    acquisition_space = Box(dataset.query_points[0] - eps, dataset.query_points[0] + eps)
    previous_state = TrustRegion.State(acquisition_space, eps, dataset.observations[0], False)

    query_points, current_state = tr.acquire(
        search_space, {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}, previous_state
    )

    npt.assert_allclose(query_points, [[0.0, 0.0]], atol=1e-5)
    npt.assert_allclose(current_state.acquisition_space.lower, search_space.lower)
    npt.assert_allclose(current_state.acquisition_space.upper, search_space.upper)
    npt.assert_allclose(current_state.eps, previous_state.eps * 0.7)
    npt.assert_allclose(current_state.y_min, [0.4])
    assert current_state.is_global


def test_trust_region_state_deepcopy() -> None:
    tr_state = TrustRegion.State(
        Box([1.2], [3.4]), tf.cast(5.6, tf.float64), tf.constant(7.8, tf.float64), False
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
    search_space = Box([-2.2, -1.0], [1.3, 3.3])
    num_query_points = 4
    ego = BatchAcquisitionRule(num_query_points, _BatchModelMinusMeanMaximumSingleBuilder())
    dataset = Dataset(tf.zeros([0, 2], tf.float64), tf.zeros([0, 1], tf.float64))
    query_point, _ = ego.acquire(
        search_space, {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    )

    npt.assert_allclose(query_point, [[0.0, 0.0]] * num_query_points, atol=1e-3)

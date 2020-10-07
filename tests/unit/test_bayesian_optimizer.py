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

from typing import Dict, List, Optional, Tuple, Mapping

import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.acquisition.rule import AcquisitionRule, OBJECTIVE
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.datasets import Dataset
from trieste.models import ModelInterface
from trieste.space import Box
from trieste.type import ObserverEvaluations, QueryPoints, TensorType

from tests.util.misc import FixedAcquisitionRule, one_dimensional_range, zero_dataset
from tests.util.model import QuadraticWithUnitVariance, StaticWithUnitVariance


@pytest.mark.parametrize('steps', [0, 1, 2, 5])
def test_bayesian_optimizer_calls_observer_once_per_iteration(steps: int) -> None:
    class _CountingObserver:
        call_count = 0

        def __call__(self, x: tf.Tensor) -> Dict[str, Dataset]:
            self.call_count += 1
            return {OBJECTIVE: Dataset(x, tf.reduce_sum(x ** 2, axis=-1, keepdims=True))}

    observer = _CountingObserver()
    optimizer = BayesianOptimizer(observer, one_dimensional_range(-1, 1))
    data = Dataset(tf.constant([[0.5]]), tf.constant([[0.25]]))

    res, _ = optimizer.optimize(
        steps, {OBJECTIVE: data}, {OBJECTIVE: QuadraticWithUnitVariance()}
    )
    res.unwrap()

    assert observer.call_count == steps


@pytest.mark.parametrize('datasets, model_specs', [
    ({}, {}),
    ({'foo': zero_dataset()}, {}),
    ({'foo': zero_dataset()}, {'bar': QuadraticWithUnitVariance()}),
    ({'foo': zero_dataset()}, {
        'foo': QuadraticWithUnitVariance(),
        'bar': QuadraticWithUnitVariance()
    }),
])
def test_bayesian_optimizer_optimize_raises_for_invalid_rule_keys(
        datasets: Dict[str, Dataset],
        model_specs: Dict[str, ModelInterface]
) -> None:
    optimizer = BayesianOptimizer(lambda x: {'foo': Dataset(x, x[:1])}, one_dimensional_range(-1, 1))
    rule = FixedAcquisitionRule(tf.constant([[0.]]))
    with pytest.raises(ValueError):
        optimizer.optimize(10, datasets, model_specs, rule)


def test_bayesian_optimizer_optimize_raises_for_invalid_rule_keys_and_default_acquisition() -> None:
    optimizer = BayesianOptimizer(lambda x: x[:1], one_dimensional_range(-1, 1))
    with pytest.raises(ValueError):
        optimizer.optimize(3, {'foo': zero_dataset()}, {'foo': QuadraticWithUnitVariance()})


@pytest.mark.parametrize(
    'starting_state, expected_states_received, final_state',
    [(None, [None, 1, 2], 3), (3, [3, 4, 5], 6)]
)
def test_bayesian_optimizer_uses_specified_acquisition_state(
    starting_state: Optional[int],
    expected_states_received: List[Optional[int]],
    final_state: Optional[int]
) -> None:
    class Rule(AcquisitionRule[int, Box]):
        def __init__(self):
            self.states_received = []

        def acquire(
            self,
            search_space: Box,
            datasets: Mapping[str, Dataset],
            models: Mapping[str, ModelInterface],
            state: Optional[int],
        ) -> Tuple[QueryPoints, int]:
            self.states_received.append(state)

            if state is None:
                state = 0

            return tf.constant([[0.0]]), state + 1

    rule = Rule()

    res, history = BayesianOptimizer(
        lambda x: {"": Dataset(x, x ** 2)}, one_dimensional_range(-1, 1)
    ).optimize(
        3, {"": zero_dataset()}, {"": QuadraticWithUnitVariance()}, rule, starting_state
    )

    assert rule.states_received == expected_states_received
    assert res.unwrap().acquisition_state == final_state
    assert [state.acquisition_state for state in history] == expected_states_received


def test_bayesian_optimizer_optimize_returns_default_acquisition_state_of_correct_type() -> None:
    optimizer = BayesianOptimizer(
        lambda x: {OBJECTIVE: Dataset(x, x[:1])}, one_dimensional_range(-1, 1)
    )
    res, history = optimizer.optimize(
        3, {OBJECTIVE: zero_dataset()}, {OBJECTIVE: QuadraticWithUnitVariance()}
    )

    assert res.unwrap().acquisition_state is None
    assert all(logging_state.acquisition_state is None for logging_state in history)


def test_bayesian_optimizer_can_use_two_gprs_for_objective_defined_by_two_dimensions() -> None:
    class ExponentialWithUnitVariance(StaticWithUnitVariance):
        def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
            return tf.exp(- query_points), tf.ones_like(query_points)

    class LinearWithUnitVariance(StaticWithUnitVariance):
        def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
            return 2 * query_points, tf.ones_like(query_points)

    LINEAR = "linear"
    EXPONENTIAL = "exponential"

    class AdditionRule(AcquisitionRule[int, Box]):
        def acquire(
                self,
                search_space: Box,
                datasets: Mapping[str, Dataset],
                models: Mapping[str, ModelInterface],
                previous_state: Optional[int]
        ) -> Tuple[QueryPoints, int]:
            if previous_state is None:
                previous_state = 1

            candidate_query_points = search_space.sample(previous_state)
            linear_predictions, _ = models[LINEAR].predict(candidate_query_points)
            exponential_predictions, _ = models[EXPONENTIAL].predict(candidate_query_points)

            target = linear_predictions + exponential_predictions

            optimum_idx = tf.argmin(target, axis=0)[0]
            next_query_points = tf.expand_dims(candidate_query_points[optimum_idx, ...], axis=0)

            return next_query_points, previous_state * 2

    def linear_and_exponential(query_points: tf.Tensor) -> Dict[str, Dataset]:
        return {
            LINEAR: Dataset(query_points, 2 * query_points),
            EXPONENTIAL: Dataset(query_points, tf.exp(- query_points))
        }

    data = {
        LINEAR: Dataset(tf.constant([[0.0]]), tf.constant([[0.0]])),
        EXPONENTIAL: Dataset(tf.constant([[0.0]]), tf.constant([[1.0]]))
    }

    models: Dict[str, ModelInterface] = {  # mypy can't infer this type for some reason
        LINEAR: LinearWithUnitVariance(), EXPONENTIAL: ExponentialWithUnitVariance()
    }

    res, _ = BayesianOptimizer(
        linear_and_exponential,
        Box(tf.constant([-2.0]), tf.constant([2.0]))
    ).optimize(20, data, models, AdditionRule())
    record = res.unwrap()

    objective_values = record.datasets[LINEAR].observations + record.datasets[EXPONENTIAL].observations
    min_idx = tf.argmin(objective_values, axis=0)[0]
    npt.assert_allclose(record.datasets[LINEAR].query_points[min_idx], - tf.math.log(2.0), rtol=0.01)

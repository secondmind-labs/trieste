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

from typing import Dict, List, Mapping, NoReturn, Optional, Tuple

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import (
    FixedAcquisitionRule,
    assert_datasets_allclose,
    one_dimensional_range,
    quadratic,
    zero_dataset,
)
from tests.util.model import GaussianMarginal, PseudoTrainableProbModel, QuadraticWithUnitVariance
from trieste.acquisition.rule import OBJECTIVE, AcquisitionRule
from trieste.bayesian_optimizer import BayesianOptimizer, OptimizationResult, Record
from trieste.data import Dataset
from trieste.models import ProbabilisticModel, TrainableProbabilisticModel
from trieste.observer import Observer
from trieste.space import Box, SearchSpace
from trieste.type import TensorType
from trieste.utils import Err, Ok


class _PseudoTrainableQuadratic(QuadraticWithUnitVariance, PseudoTrainableProbModel):
    pass


class _Whoops(Exception):
    pass


def test_optimization_result_astuple() -> None:
    opt_result: OptimizationResult[None] = OptimizationResult(
        Err(_Whoops()), [Record({}, {}, None)]
    )
    final_result, history = opt_result.astuple()
    assert final_result is opt_result.final_result
    assert history is opt_result.history


def test_optimization_result_try_get_final_datasets_for_successful_optimization() -> None:
    data = {"foo": zero_dataset()}
    result: OptimizationResult[None] = OptimizationResult(
        Ok(Record(data, {"foo": _PseudoTrainableQuadratic()}, None)), []
    )
    assert result.try_get_final_datasets() is data


def test_optimization_result_try_get_final_datasets_for_failed_optimization() -> None:
    result: OptimizationResult[object] = OptimizationResult(Err(_Whoops()), [])
    with pytest.raises(_Whoops):
        result.try_get_final_datasets()


def test_optimization_result_try_get_final_models_for_successful_optimization() -> None:
    models = {"foo": _PseudoTrainableQuadratic()}
    result: OptimizationResult[None] = OptimizationResult(
        Ok(Record({"foo": zero_dataset()}, models, None)), []
    )
    assert result.try_get_final_models() is models


def test_optimization_result_try_get_final_models_for_failed_optimization() -> None:
    result: OptimizationResult[object] = OptimizationResult(Err(_Whoops()), [])
    with pytest.raises(_Whoops):
        result.try_get_final_models()


@pytest.mark.parametrize("steps", [0, 1, 2, 5])
def test_bayesian_optimizer_calls_observer_once_per_iteration(steps: int) -> None:
    class _CountingObserver:
        call_count = 0

        def __call__(self, x: tf.Tensor) -> Dict[str, Dataset]:
            self.call_count += 1
            return {OBJECTIVE: Dataset(x, tf.reduce_sum(x ** 2, axis=-1, keepdims=True))}

    observer = _CountingObserver()
    optimizer = BayesianOptimizer(observer, one_dimensional_range(-1, 1))
    data = Dataset(tf.constant([[0.5]]), tf.constant([[0.25]]))

    optimizer.optimize(
        steps, {OBJECTIVE: data}, {OBJECTIVE: _PseudoTrainableQuadratic()}
    ).final_result.unwrap()

    assert observer.call_count == steps


@pytest.mark.parametrize(
    "datasets, models",
    [
        ({}, {}),
        ({"foo": zero_dataset()}, {}),
        ({"foo": zero_dataset()}, {"bar": _PseudoTrainableQuadratic()}),
        (
            {"foo": zero_dataset()},
            {"foo": _PseudoTrainableQuadratic(), "bar": _PseudoTrainableQuadratic()},
        ),
    ],
)
def test_bayesian_optimizer_optimize_raises_for_invalid_keys(
    datasets: Dict[str, Dataset], models: Dict[str, TrainableProbabilisticModel]
) -> None:
    search_space = one_dimensional_range(-1, 1)
    optimizer = BayesianOptimizer(lambda x: {"foo": Dataset(x, x[:1])}, search_space)
    rule = FixedAcquisitionRule(tf.constant([[0.0]]))
    with pytest.raises(ValueError):
        optimizer.optimize(10, datasets, models, rule)


def test_bayesian_optimizer_optimize_raises_for_invalid_rule_keys_and_default_acquisition() -> None:
    optimizer = BayesianOptimizer(lambda x: x[:1], one_dimensional_range(-1, 1))
    with pytest.raises(ValueError):
        optimizer.optimize(3, {"foo": zero_dataset()}, {"foo": _PseudoTrainableQuadratic()})


@pytest.mark.parametrize(
    "starting_state, expected_states_received, final_acquisition_state",
    [(None, [None, 1, 2], 3), (3, [3, 4, 5], 6)],
)
def test_bayesian_optimizer_uses_specified_acquisition_state(
    starting_state: Optional[int],
    expected_states_received: List[Optional[int]],
    final_acquisition_state: Optional[int],
) -> None:
    class Rule(AcquisitionRule[int, Box]):
        def __init__(self):
            self.states_received = []

        def acquire(
            self,
            search_space: Box,
            datasets: Mapping[str, Dataset],
            models: Mapping[str, ProbabilisticModel],
            state: Optional[int],
        ) -> Tuple[TensorType, int]:
            self.states_received.append(state)

            if state is None:
                state = 0

            return tf.constant([[0.0]]), state + 1

    rule = Rule()

    final_state, history = (
        BayesianOptimizer(lambda x: {"": Dataset(x, x ** 2)}, one_dimensional_range(-1, 1))
        .optimize(3, {"": zero_dataset()}, {"": _PseudoTrainableQuadratic()}, rule, starting_state)
        .astuple()
    )

    assert rule.states_received == expected_states_received
    assert final_state.unwrap().acquisition_state == final_acquisition_state
    assert [record.acquisition_state for record in history] == expected_states_received


def test_bayesian_optimizer_optimize_for_uncopyable_model() -> None:
    class _UncopyableModel(_PseudoTrainableQuadratic):
        _optimize_count = 0

        def optimize(self, dataset: Dataset) -> None:
            self._optimize_count += 1

        def __deepcopy__(self, memo: Dict[int, object]) -> _UncopyableModel:
            if self._optimize_count >= 3:
                raise _Whoops

            return self

    rule = FixedAcquisitionRule(tf.constant([[0.0]]))
    result, history = (
        BayesianOptimizer(quadratic, one_dimensional_range(0, 1))
        .optimize(10, {"": zero_dataset()}, {"": _UncopyableModel()}, rule)
        .astuple()
    )

    with pytest.raises(_Whoops):
        result.unwrap()

    assert len(history) == 4


def _broken_observer(x: tf.Tensor) -> NoReturn:
    raise _Whoops


class _BrokenModel(_PseudoTrainableQuadratic):
    def optimize(self, dataset: Dataset) -> NoReturn:
        raise _Whoops


class _BrokenRule(AcquisitionRule[None, SearchSpace]):
    def acquire(
        self,
        search_space: SearchSpace,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ProbabilisticModel],
        state: None,
    ) -> NoReturn:
        raise _Whoops


@pytest.mark.parametrize(
    "observer, model, rule",
    [
        (_broken_observer, _PseudoTrainableQuadratic(), FixedAcquisitionRule(tf.constant([[0.0]]))),
        (quadratic, _BrokenModel(), FixedAcquisitionRule(tf.constant([[0.0]]))),
        (quadratic, _PseudoTrainableQuadratic(), _BrokenRule()),
    ],
)
def test_bayesian_optimizer_optimize_for_failed_step(
    observer: Observer, model: TrainableProbabilisticModel, rule: AcquisitionRule
) -> None:
    optimizer = BayesianOptimizer(observer, one_dimensional_range(0, 1))
    result, history = optimizer.optimize(3, {"": zero_dataset()}, {"": model}, rule).astuple()

    with pytest.raises(_Whoops):
        result.unwrap()

    assert len(history) == 1


@pytest.mark.parametrize("num_steps", [-3, -1])
def test_bayesian_optimizer_optimize_raises_for_negative_steps(num_steps: int) -> None:
    optimizer = BayesianOptimizer(quadratic, one_dimensional_range(-1, 1))

    with pytest.raises(ValueError, match="num_steps"):
        optimizer.optimize(num_steps, {"": zero_dataset()}, {"": _PseudoTrainableQuadratic()})


def test_bayesian_optimizer_optimize_is_noop_for_zero_steps() -> None:
    class _UnusableModel(TrainableProbabilisticModel):
        def predict(self, query_points: TensorType) -> NoReturn:
            assert False

        def sample(self, query_points: TensorType, num_samples: int) -> NoReturn:
            assert False

        def update(self, dataset: Dataset) -> NoReturn:
            assert False

        def optimize(self, dataset: Dataset) -> NoReturn:
            assert False

    class _UnusableRule(AcquisitionRule[None, Box]):
        def acquire(
            self,
            search_space: Box,
            datasets: Mapping[str, Dataset],
            models: Mapping[str, ProbabilisticModel],
            state: None,
        ) -> NoReturn:
            assert False

    def _unusable_observer(x: tf.Tensor) -> NoReturn:
        assert False

    data = {"": zero_dataset()}
    result, history = (
        BayesianOptimizer(_unusable_observer, one_dimensional_range(-1, 1))
        .optimize(0, data, {"": _UnusableModel()}, _UnusableRule())
        .astuple()
    )
    assert history == []
    final_data = result.unwrap().datasets
    assert len(final_data) == 1
    assert_datasets_allclose(final_data[""], data[""])


def test_bayesian_optimizer_can_use_two_gprs_for_objective_defined_by_two_dimensions() -> None:
    class ExponentialWithUnitVariance(GaussianMarginal, PseudoTrainableProbModel):
        def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
            return tf.exp(-query_points), tf.ones_like(query_points)

    class LinearWithUnitVariance(GaussianMarginal, PseudoTrainableProbModel):
        def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
            return 2 * query_points, tf.ones_like(query_points)

    LINEAR = "linear"
    EXPONENTIAL = "exponential"

    class AdditionRule(AcquisitionRule[int, Box]):
        def acquire(
            self,
            search_space: Box,
            datasets: Mapping[str, Dataset],
            models: Mapping[str, ProbabilisticModel],
            previous_state: Optional[int],
        ) -> Tuple[TensorType, int]:
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
            EXPONENTIAL: Dataset(query_points, tf.exp(-query_points)),
        }

    data: Mapping[str, Dataset] = {
        LINEAR: Dataset(tf.constant([[0.0]]), tf.constant([[0.0]])),
        EXPONENTIAL: Dataset(tf.constant([[0.0]]), tf.constant([[1.0]])),
    }

    models: Mapping[str, TrainableProbabilisticModel] = {
        LINEAR: LinearWithUnitVariance(),
        EXPONENTIAL: ExponentialWithUnitVariance(),
    }

    data = (
        BayesianOptimizer(linear_and_exponential, Box(tf.constant([-2.0]), tf.constant([2.0])))
        .optimize(20, data, models, AdditionRule())
        .try_get_final_datasets()
    )

    objective_values = data[LINEAR].observations + data[EXPONENTIAL].observations
    min_idx = tf.argmin(objective_values, axis=0)[0]
    npt.assert_allclose(data[LINEAR].query_points[min_idx], -tf.math.log(2.0), rtol=0.01)


def test_bayesian_optimizer_optimize_doesnt_track_state_if_told_not_to() -> None:
    class _UncopyableModel(_PseudoTrainableQuadratic):
        def __deepcopy__(self, memo: Dict[int, object]) -> NoReturn:
            assert False

    history = (
        BayesianOptimizer(quadratic, one_dimensional_range(-1, 1))
        .optimize(
            5, {OBJECTIVE: zero_dataset()}, {OBJECTIVE: _UncopyableModel()}, track_state=False
        )
        .history
    )
    assert len(history) == 0


def test_bayesian_optimizer_optimize_tracked_state() -> None:
    class _CountingRule(AcquisitionRule[int, Box]):
        def acquire(
            self,
            search_space: Box,
            datasets: Mapping[str, Dataset],
            models: Mapping[str, ProbabilisticModel],
            state: Optional[int],
        ) -> Tuple[TensorType, int]:
            new_state = 0 if state is None else state + 1
            return tf.constant([[10.0]]) + new_state, new_state

    class _DecreasingVarianceModel(QuadraticWithUnitVariance, TrainableProbabilisticModel):
        def __init__(self, data: Dataset):
            super().__init__()
            self._data = data

        def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
            mean, var = super().predict(query_points)
            return mean, var / len(self._data)

        def update(self, dataset: Dataset) -> None:
            self._data = dataset

        def optimize(self, dataset: Dataset) -> None:
            pass

    _, history = (
        BayesianOptimizer(quadratic, one_dimensional_range(0, 1))
        .optimize(
            3, {"": zero_dataset()}, {"": _DecreasingVarianceModel(zero_dataset())}, _CountingRule()
        )
        .astuple()
    )

    assert [record.acquisition_state for record in history] == [None, 0, 1]

    assert_datasets_allclose(history[0].datasets[""], zero_dataset())
    assert_datasets_allclose(
        history[1].datasets[""],
        Dataset(tf.constant([[0.0], [10.0]]), tf.constant([[0.0], [100.0]])),
    )
    assert_datasets_allclose(
        history[2].datasets[""],
        Dataset(tf.constant([[0.0], [10.0], [11.0]]), tf.constant([[0.0], [100.0], [121.0]])),
    )

    for step in range(3):
        _, variance_from_saved_model = history[step].models[""].predict(tf.constant([[0.0]]))
        npt.assert_allclose(variance_from_saved_model, 1.0 / (step + 1))

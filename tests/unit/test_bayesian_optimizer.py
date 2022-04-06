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

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn, Optional

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import (
    FixedAcquisitionRule,
    assert_datasets_allclose,
    empty_dataset,
    mk_dataset,
    quadratic,
)
from tests.util.models.gpflow.models import (
    GaussianProcess,
    PseudoTrainableProbModel,
    QuadraticMeanAndRBFKernel,
    rbf,
)
from trieste.acquisition.rule import AcquisitionRule
from trieste.bayesian_optimizer import BayesianOptimizer, FrozenRecord, OptimizationResult, Record
from trieste.data import Dataset
from trieste.models import ProbabilisticModel, TrainableProbabilisticModel
from trieste.observer import OBJECTIVE, Observer
from trieste.space import Box, SearchSpace
from trieste.types import State, TensorType
from trieste.utils import Err, Ok


def _quadratic_observer(x: tf.Tensor) -> Mapping[str, Dataset]:
    return {"": Dataset(x, quadratic(x))}


class _PseudoTrainableQuadratic(QuadraticMeanAndRBFKernel, PseudoTrainableProbModel):
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
    data = {"foo": empty_dataset([1], [1])}
    result: OptimizationResult[None] = OptimizationResult(
        Ok(Record(data, {"foo": _PseudoTrainableQuadratic()}, None)), []
    )
    assert result.try_get_final_datasets() is data
    assert result.try_get_final_dataset() is data["foo"]


def test_optimization_result_try_get_final_datasets_for_multiple_datasets() -> None:
    data = {"foo": empty_dataset([1], [1]), "bar": empty_dataset([2], [2])}
    models = {"foo": _PseudoTrainableQuadratic(), "bar": _PseudoTrainableQuadratic()}
    result: OptimizationResult[None] = OptimizationResult(Ok(Record(data, models, None)), [])
    assert result.try_get_final_datasets() is data
    with pytest.raises(ValueError):
        result.try_get_final_dataset()


def test_optimization_result_try_get_final_datasets_for_failed_optimization() -> None:
    result: OptimizationResult[object] = OptimizationResult(Err(_Whoops()), [])
    with pytest.raises(_Whoops):
        result.try_get_final_datasets()


def test_optimization_result_try_get_final_models_for_successful_optimization() -> None:
    models = {"foo": _PseudoTrainableQuadratic()}
    result: OptimizationResult[None] = OptimizationResult(
        Ok(Record({"foo": empty_dataset([1], [1])}, models, None)), []
    )
    assert result.try_get_final_models() is models
    assert result.try_get_final_model() is models["foo"]


def test_optimization_result_try_get_final_models_for_multiple_models() -> None:
    data = {"foo": empty_dataset([1], [1]), "bar": empty_dataset([2], [2])}
    models = {"foo": _PseudoTrainableQuadratic(), "bar": _PseudoTrainableQuadratic()}
    result: OptimizationResult[None] = OptimizationResult(Ok(Record(data, models, None)), [])
    assert result.try_get_final_models() is models
    with pytest.raises(ValueError):
        result.try_get_final_model()


def test_optimization_result_try_get_final_models_for_failed_optimization() -> None:
    result: OptimizationResult[object] = OptimizationResult(Err(_Whoops()), [])
    with pytest.raises(_Whoops):
        result.try_get_final_models()


def test_optimization_result_try_get_optimal_point_for_successful_optimization() -> None:
    data = {"foo": mk_dataset([[0.25, 0.25], [0.5, 0.4]], [[0.8], [0.7]])}
    result: OptimizationResult[None] = OptimizationResult(
        Ok(Record(data, {"foo": _PseudoTrainableQuadratic()}, None)), []
    )
    x, y, idx = result.try_get_optimal_point()
    npt.assert_allclose(x, [0.5, 0.4])
    npt.assert_allclose(y, [0.7])
    npt.assert_allclose(idx, 1)


def test_optimization_result_try_get_optimal_point_for_multiple_objectives() -> None:
    data = {"foo": mk_dataset([[0.25], [0.5]], [[0.8, 0.5], [0.7, 0.4]])}
    result: OptimizationResult[None] = OptimizationResult(
        Ok(Record(data, {"foo": _PseudoTrainableQuadratic()}, None)), []
    )
    with pytest.raises(ValueError):
        result.try_get_optimal_point()


def test_optimization_result_try_get_optimal_point_for_failed_optimization() -> None:
    result: OptimizationResult[object] = OptimizationResult(Err(_Whoops()), [])
    with pytest.raises(_Whoops):
        result.try_get_optimal_point()


def test_optimization_result_from_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        opt_result: OptimizationResult[None] = OptimizationResult(
            Err(_Whoops()), [Record({}, {}, None)] * 10
        )
        opt_result.save(tmpdirname)

        result, history = OptimizationResult[None].from_path(tmpdirname).astuple()
        assert result.is_err
        with pytest.raises(_Whoops):
            result.unwrap()
        assert len(history) == 10
        assert all(isinstance(record, FrozenRecord) for record in history)
        assert (
            r2.load() == r1
            for r1, r2 in zip(opt_result.history, history)
            if isinstance(r2, FrozenRecord)
        )


def test_optimization_result_from_path_partial_result() -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        opt_result: OptimizationResult[None] = OptimizationResult(
            Err(_Whoops()), [Record({}, {}, None)] * 10
        )
        opt_result.save(tmpdirname)
        (Path(tmpdirname) / OptimizationResult.RESULTS_FILENAME).unlink()
        (Path(tmpdirname) / OptimizationResult.step_filename(9, 10)).unlink()

        result, history = OptimizationResult[None].from_path(tmpdirname).astuple()
        assert result.is_err
        with pytest.raises(FileNotFoundError):
            result.unwrap()
        assert len(history) == 9
        assert all(isinstance(record, FrozenRecord) for record in history)
        assert (
            r2.load() == r1
            for r1, r2 in zip(opt_result.history, history)
            if isinstance(r2, FrozenRecord)
        )


@pytest.mark.parametrize("steps", [0, 1, 2, 5])
def test_bayesian_optimizer_calls_observer_once_per_iteration(steps: int) -> None:
    class _CountingObserver:
        call_count = 0

        def __call__(self, x: tf.Tensor) -> Dataset:
            self.call_count += 1
            return Dataset(x, tf.reduce_sum(x ** 2, axis=-1, keepdims=True))

    observer = _CountingObserver()
    optimizer = BayesianOptimizer(observer, Box([-1], [1]))
    data = mk_dataset([[0.5]], [[0.25]])

    optimizer.optimize(steps, data, _PseudoTrainableQuadratic()).final_result.unwrap()

    assert observer.call_count == steps


@pytest.mark.parametrize("fit_initial_model", [True, False])
def test_bayesian_optimizer_optimizes_initial_model(fit_initial_model: bool) -> None:
    class _CountingOptimizerModel(_PseudoTrainableQuadratic):
        _optimize_count = 0

        def optimize(self, dataset: Dataset) -> None:
            self._optimize_count += 1

    rule = FixedAcquisitionRule([[0.0]])
    model = _CountingOptimizerModel()

    final_opt_state, _ = (
        BayesianOptimizer(_quadratic_observer, Box([0], [1]))
        .optimize(
            1,
            {"": mk_dataset([[0.0]], [[0.0]])},
            {"": model},
            rule,
            fit_initial_model=fit_initial_model,
        )
        .astuple()
    )
    final_model = final_opt_state.unwrap().model

    if fit_initial_model:  # optimized at start and end of first BO step
        assert final_model._optimize_count == 2  # type: ignore
    else:  # optimized just at end of first BO step
        assert final_model._optimize_count == 1  # type: ignore


@pytest.mark.parametrize(
    "datasets, models",
    [
        ({}, {}),
        ({"foo": empty_dataset([1], [1])}, {}),
        ({"foo": empty_dataset([1], [1])}, {"bar": _PseudoTrainableQuadratic()}),
        (
            {"foo": empty_dataset([1], [1])},
            {"foo": _PseudoTrainableQuadratic(), "bar": _PseudoTrainableQuadratic()},
        ),
    ],
)
def test_bayesian_optimizer_optimize_raises_for_invalid_keys(
    datasets: dict[str, Dataset], models: dict[str, TrainableProbabilisticModel]
) -> None:
    search_space = Box([-1], [1])
    optimizer = BayesianOptimizer(lambda x: {"foo": Dataset(x, x)}, search_space)
    rule = FixedAcquisitionRule([[0.0]])
    with pytest.raises(ValueError):
        optimizer.optimize(10, datasets, models, rule)


def test_bayesian_optimizer_optimize_raises_for_invalid_rule_keys_and_default_acquisition() -> None:
    optimizer = BayesianOptimizer(lambda x: x[:1], Box([-1], [1]))
    data, models = {"foo": empty_dataset([1], [1])}, {"foo": _PseudoTrainableQuadratic()}
    with pytest.raises(ValueError):
        optimizer.optimize(3, data, models)


@pytest.mark.parametrize(
    "starting_state, expected_states_received, final_acquisition_state",
    [(None, [None, 1, 2], 3), (3, [3, 4, 5], 6)],
)
def test_bayesian_optimizer_uses_specified_acquisition_state(
    starting_state: int | None,
    expected_states_received: list[int | None],
    final_acquisition_state: int | None,
) -> None:
    class Rule(AcquisitionRule[State[Optional[int], TensorType], Box, ProbabilisticModel]):
        def __init__(self) -> None:
            self.states_received: list[int | None] = []

        def acquire(
            self,
            search_space: Box,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> State[int | None, TensorType]:
            def go(state: int | None) -> tuple[int | None, TensorType]:
                self.states_received.append(state)

                if state is None:
                    state = 0

                return state + 1, tf.constant([[0.0]], tf.float64)

            return go

    rule = Rule()

    data, models = {"": mk_dataset([[0.0]], [[0.0]])}, {"": _PseudoTrainableQuadratic()}
    final_state, history = (
        BayesianOptimizer(lambda x: {"": Dataset(x, x ** 2)}, Box([-1], [1]))
        .optimize(3, data, models, rule, starting_state)
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

        def __deepcopy__(self, memo: dict[int, object]) -> _UncopyableModel:
            if self._optimize_count >= 3:
                raise _Whoops

            return self

    rule = FixedAcquisitionRule([[0.0]])
    result, history = (
        BayesianOptimizer(_quadratic_observer, Box([0], [1]))
        .optimize(
            10,
            {"": mk_dataset([[0.0]], [[0.0]])},
            {"": _UncopyableModel()},
            rule,
            fit_initial_model=False,
        )
        .astuple()
    )

    with pytest.raises(NotImplementedError):
        result.unwrap()

    assert len(history) == 3


def _broken_observer(x: tf.Tensor) -> NoReturn:
    raise _Whoops


class _BrokenModel(_PseudoTrainableQuadratic):
    def optimize(self, dataset: Dataset) -> NoReturn:
        raise _Whoops


class _BrokenRule(AcquisitionRule[NoReturn, SearchSpace, ProbabilisticModel]):
    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[str, ProbabilisticModel],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> NoReturn:
        raise _Whoops


@pytest.mark.parametrize(
    "observer, model, rule",
    [
        (_broken_observer, _PseudoTrainableQuadratic(), FixedAcquisitionRule([[0.0]])),
        (_quadratic_observer, _BrokenModel(), FixedAcquisitionRule([[0.0]])),
        (_quadratic_observer, _PseudoTrainableQuadratic(), _BrokenRule()),
    ],
)
def test_bayesian_optimizer_optimize_for_failed_step(
    observer: Observer,
    model: TrainableProbabilisticModel,
    rule: AcquisitionRule[None, Box, ProbabilisticModel],
) -> None:
    optimizer = BayesianOptimizer(observer, Box([0], [1]))
    data, models = {"": mk_dataset([[0.0]], [[0.0]])}, {"": model}
    result, history = optimizer.optimize(3, data, models, rule).astuple()

    with pytest.raises(_Whoops):
        result.unwrap()

    assert len(history) == 1


@pytest.mark.parametrize("num_steps", [-3, -1])
def test_bayesian_optimizer_optimize_raises_for_negative_steps(num_steps: int) -> None:
    optimizer = BayesianOptimizer(_quadratic_observer, Box([-1], [1]))

    data, models = {"": empty_dataset([1], [1])}, {"": _PseudoTrainableQuadratic()}
    with pytest.raises(ValueError, match="num_steps"):
        optimizer.optimize(num_steps, data, models)


def test_bayesian_optimizer_optimize_is_noop_for_zero_steps() -> None:
    class _UnusableModel(TrainableProbabilisticModel):
        def predict(self, query_points: TensorType) -> NoReturn:
            assert False

        def predict_joint(self, query_points: TensorType) -> NoReturn:
            assert False

        def sample(self, query_points: TensorType, num_samples: int) -> NoReturn:
            assert False

        def update(self, dataset: Dataset) -> NoReturn:
            assert False

        def optimize(self, dataset: Dataset) -> NoReturn:
            assert False

    class _UnusableRule(AcquisitionRule[NoReturn, Box, ProbabilisticModel]):
        def acquire(
            self,
            search_space: Box,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> NoReturn:
            assert False

    def _unusable_observer(x: tf.Tensor) -> NoReturn:
        assert False

    data = {"": mk_dataset([[0.0]], [[0.0]])}
    result, history = (
        BayesianOptimizer(_unusable_observer, Box([-1], [1]))
        .optimize(0, data, {"": _UnusableModel()}, _UnusableRule())
        .astuple()
    )
    assert history == []
    final_data = result.unwrap().datasets
    assert len(final_data) == 1
    assert_datasets_allclose(final_data[""], data[""])


def test_bayesian_optimizer_can_use_two_gprs_for_objective_defined_by_two_dimensions() -> None:
    class ExponentialWithUnitVariance(GaussianProcess, PseudoTrainableProbModel):
        def __init__(self) -> None:
            super().__init__([lambda x: tf.exp(-x)], [rbf()])

    class LinearWithUnitVariance(GaussianProcess, PseudoTrainableProbModel):
        def __init__(self) -> None:
            super().__init__([lambda x: 2 * x], [rbf()])

    LINEAR = "linear"
    EXPONENTIAL = "exponential"

    class AdditionRule(AcquisitionRule[State[Optional[int], TensorType], Box, ProbabilisticModel]):
        def acquire(
            self,
            search_space: Box,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> State[int | None, TensorType]:
            def go(previous_state: int | None) -> tuple[int | None, TensorType]:
                if previous_state is None:
                    previous_state = 1

                candidate_query_points = search_space.sample(previous_state)
                linear_predictions, _ = models[LINEAR].predict(candidate_query_points)
                exponential_predictions, _ = models[EXPONENTIAL].predict(candidate_query_points)

                target = linear_predictions + exponential_predictions

                optimum_idx = tf.argmin(target, axis=0)[0]
                next_query_points = tf.expand_dims(candidate_query_points[optimum_idx, ...], axis=0)

                return previous_state * 2, next_query_points

            return go

    def linear_and_exponential(query_points: tf.Tensor) -> dict[str, Dataset]:
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
        def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
            assert False

    data, models = {OBJECTIVE: empty_dataset([1], [1])}, {OBJECTIVE: _UncopyableModel()}
    history = (
        BayesianOptimizer(_quadratic_observer, Box([-1], [1]))
        .optimize(5, data, models, track_state=False)
        .history
    )
    assert len(history) == 0


class _DecreasingVarianceModel(QuadraticMeanAndRBFKernel, TrainableProbabilisticModel):
    def __init__(self, data: Dataset):
        super().__init__()
        self._data = data

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        mean, var = super().predict(query_points)
        return mean, var / len(self._data)

    def update(self, dataset: Dataset) -> None:
        self._data = dataset

    def optimize(self, dataset: Dataset) -> None:
        pass


@pytest.mark.parametrize("save_to_disk", [False, True])
def test_bayesian_optimizer_optimize_tracked_state(save_to_disk: bool) -> None:
    class _CountingRule(AcquisitionRule[State[Optional[int], TensorType], Box, ProbabilisticModel]):
        def acquire(
            self,
            search_space: Box,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> State[int | None, TensorType]:
            def go(state: int | None) -> tuple[int | None, TensorType]:
                new_state = 0 if state is None else state + 1
                return new_state, tf.constant([[10.0]], tf.float64) + new_state

            return go

    with tempfile.TemporaryDirectory() as tmpdirname:
        initial_data = mk_dataset([[0.0]], [[0.0]])
        model = _DecreasingVarianceModel(initial_data)
        _, history = (
            BayesianOptimizer(_quadratic_observer, Box([0], [1]))
            .optimize(
                3,
                {"": initial_data},
                {"": model},
                _CountingRule(),
                track_path=Path(tmpdirname) if save_to_disk else None,
            )
            .astuple()
        )

        assert all(
            isinstance(record, FrozenRecord if save_to_disk else Record) for record in history
        )
        assert [record.acquisition_state for record in history] == [None, 0, 1]

        assert_datasets_allclose(history[0].datasets[""], initial_data)
        assert_datasets_allclose(
            history[1].datasets[""], mk_dataset([[0.0], [10.0]], [[0.0], [100.0]])
        )
        assert_datasets_allclose(
            history[2].datasets[""], mk_dataset([[0.0], [10.0], [11.0]], [[0.0], [100.0], [121.0]])
        )

        for step in range(3):
            record = history[step].load() if save_to_disk else history[step]  # type: ignore
            assert record.model == record.models[""]
            assert record.dataset == record.datasets[""]

            _, variance_from_saved_model = (
                history[step].models[""].predict(tf.constant([[0.0]], tf.float64))
            )
            npt.assert_allclose(variance_from_saved_model, 1.0 / (step + 1))

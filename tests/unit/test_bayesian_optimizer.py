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

from collections.abc import Mapping
from typing import NoReturn, cast

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
from tests.util.model import (
    GaussianProcess,
    PseudoTrainableProbModel,
    QuadraticMeanAndRBFKernel,
    rbf,
)
from trieste.acquisition.rule import AcquisitionRule
from trieste.bayesian_optimizer import BayesianOptimizer, OptimizationResult, Record
from trieste.data import Dataset
from trieste.models import ProbabilisticModel, TrainableProbabilisticModel
from trieste.observer import OBJECTIVE, Observer
from trieste.space import Box, SearchSpace
from trieste.type import TensorType
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


@pytest.mark.parametrize("fit_intial_model", [True, False])
def test_bayesian_optimizer_optimizes_initial_model(fit_intial_model: bool) -> None:
    class _CountingOptimizerModel(_PseudoTrainableQuadratic):
        _optimize_count = 0

        def optimize(self, dataset: Dataset) -> None:
            self._optimize_count += 1

    rule = FixedAcquisitionRule([[0.0]])
    model = _CountingOptimizerModel()

    final_opt_state, _ = cast(
        OptimizationResult[None],
        BayesianOptimizer(_quadratic_observer, Box([0], [1])).optimize(
            1,
            {"": mk_dataset([[0.0]], [[0.0]])},
            {"": model},
            rule,
            fit_intial_model=fit_intial_model,
        ),
    ).astuple()

    final_model = final_opt_state.unwrap().model

    if fit_intial_model:  # optimized at start and end of first BO step
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


def test_bayesian_optimizer_uses_specified_state() -> None:
    ...


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
    result, history = cast(
        OptimizationResult[None],
        BayesianOptimizer(_quadratic_observer, Box([0], [1])).optimize(
            10,
            {"": mk_dataset([[0.0]], [[0.0]])},
            {"": _UncopyableModel()},
            rule,
            fit_intial_model=False,
        ),
    ).astuple()

    with pytest.raises(_Whoops):
        result.unwrap()

    assert len(history) == 4


def _broken_observer(x: tf.Tensor) -> NoReturn:
    raise _Whoops


class _BrokenModel(_PseudoTrainableQuadratic):
    def optimize(self, dataset: Dataset) -> NoReturn:
        raise _Whoops


class _BrokenRule(AcquisitionRule[SearchSpace]):
    def acquire(
        self,
        search_space: SearchSpace,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ProbabilisticModel],
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
    observer: Observer, model: TrainableProbabilisticModel, rule: AcquisitionRule
) -> None:
    optimizer = BayesianOptimizer(observer, Box([0], [1]))
    data, models = {"": mk_dataset([[0.0]], [[0.0]])}, {"": model}
    result, history = cast(
        OptimizationResult[None], optimizer.optimize(3, data, models, rule)
    ).astuple()

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

    class _UnusableRule(AcquisitionRule[Box]):
        def acquire(
            self,
            search_space: Box,
            datasets: Mapping[str, Dataset],
            models: Mapping[str, ProbabilisticModel],
        ) -> NoReturn:
            assert False

    def _unusable_observer(x: tf.Tensor) -> NoReturn:
        assert False

    data = {"": mk_dataset([[0.0]], [[0.0]])}
    result, history = cast(
        OptimizationResult[None],
        BayesianOptimizer(_unusable_observer, Box([-1], [1])).optimize(
            0, data, {"": _UnusableModel()}, _UnusableRule()
        ),
    ).astuple()

    assert history == []
    final_data = result.unwrap().datasets
    assert len(final_data) == 1
    assert_datasets_allclose(final_data[""], data[""])


def test_bayesian_optimizer_can_use_two_gprs_for_objective_defined_by_two_dimensions() -> None:
    ...


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


def test_bayesian_optimizer_optimize_tracked_state() -> None:
    ...

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

from typing import Mapping, Optional

import pytest
import tensorflow as tf

from tests.util.misc import FixedAcquisitionRule, assert_datasets_allclose, mk_dataset
from tests.util.models.gpflow.models import GaussianProcess, PseudoTrainableProbModel, rbf
from trieste.acquisition.rule import AcquisitionRule
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.bayesian_optimizer import OptimizationResult, Record
from trieste.data import Dataset
from trieste.models.interfaces import ProbabilisticModel, TrainableProbabilisticModel
from trieste.observer import OBJECTIVE
from trieste.space import Box
from trieste.types import State, Tag, TensorType

# tags
TAG1: Tag = "1"
TAG2: Tag = "2"


class LinearWithUnitVariance(GaussianProcess, PseudoTrainableProbModel):
    def __init__(self) -> None:
        super().__init__([lambda x: 2 * x], [rbf()])
        self._optimize_count = 0

    def optimize(self, dataset: Dataset) -> None:
        self._optimize_count += 1

    @property
    def optimize_count(self) -> int:
        return self._optimize_count


@pytest.fixture
def search_space() -> Box:
    return Box([-1], [1])


@pytest.fixture
def init_dataset() -> Dataset:
    return mk_dataset([[0.0]], [[0.0]])


@pytest.fixture
def acquisition_rule() -> AcquisitionRule[TensorType, Box, ProbabilisticModel]:
    return FixedAcquisitionRule([[0.0]])


@pytest.fixture
def model() -> TrainableProbabilisticModel:
    return LinearWithUnitVariance()


def test_ask_tell_optimizer_suggests_new_point(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, acquisition_rule)

    new_point = ask_tell.ask()

    assert len(new_point) == 1


def test_ask_tell_optimizer_with_default_acquisition_suggests_new_point(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
) -> None:
    ask_tell = AskTellOptimizer(search_space, init_dataset, model)

    new_point = ask_tell.ask()

    assert len(new_point) == 1


@pytest.mark.parametrize("copy", [True, False])
def test_ask_tell_optimizer_returns_complete_state(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    copy: bool,
) -> None:
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, acquisition_rule)

    state_record: Record[None] = ask_tell.to_record(copy=copy)

    assert_datasets_allclose(state_record.dataset, init_dataset)
    assert isinstance(state_record.model, type(model))
    assert state_record.acquisition_state is None


@pytest.mark.parametrize("copy", [True, False])
def test_ask_tell_optimizer_loads_from_state(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    copy: bool,
) -> None:
    old_state: Record[None] = Record({OBJECTIVE: init_dataset}, {OBJECTIVE: model}, None)

    ask_tell = AskTellOptimizer.from_record(old_state, search_space, acquisition_rule)
    new_state: Record[None] = ask_tell.to_record(copy=copy)

    assert_datasets_allclose(old_state.dataset, new_state.dataset)
    assert isinstance(new_state.model, type(old_state.model))


@pytest.mark.parametrize("copy", [True, False])
def test_ask_tell_optimizer_returns_optimization_result(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    copy: bool,
) -> None:
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, acquisition_rule)

    result: OptimizationResult[None] = ask_tell.to_result(copy=copy)

    assert_datasets_allclose(result.try_get_final_dataset(), init_dataset)
    assert isinstance(result.try_get_final_model(), type(model))


def test_ask_tell_optimizer_updates_state_with_new_data(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    new_data = mk_dataset([[1.0]], [[1.0]])
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, acquisition_rule)

    ask_tell.tell(new_data)
    state_record: Record[None] = ask_tell.to_record()

    assert_datasets_allclose(state_record.dataset, init_dataset + new_data)


@pytest.mark.parametrize("copy", [True, False])
def test_ask_tell_optimizer_copies_state(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    copy: bool,
) -> None:
    new_data = mk_dataset([[1.0]], [[1.0]])
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, acquisition_rule)
    state_start: Record[None] = ask_tell.to_record(copy=copy)
    ask_tell.tell(new_data)
    state_end: Record[None] = ask_tell.to_record(copy=copy)

    assert_datasets_allclose(state_start.dataset, init_dataset if copy else init_dataset + new_data)
    assert_datasets_allclose(state_end.dataset, init_dataset + new_data)
    assert state_start.model is not model if copy else state_start.model is model


def test_ask_tell_optimizer_datasets_property(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, acquisition_rule)
    assert_datasets_allclose(ask_tell.datasets[OBJECTIVE], init_dataset)
    assert_datasets_allclose(ask_tell.dataset, init_dataset)


def test_ask_tell_optimizer_models_property(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, acquisition_rule)
    assert ask_tell.models[OBJECTIVE] is model
    assert ask_tell.model is model


def test_ask_tell_optimizer_models_setter(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, acquisition_rule)
    model2 = LinearWithUnitVariance()
    ask_tell.models = {OBJECTIVE: model2}
    assert ask_tell.models[OBJECTIVE] is model2 is not model


def test_ask_tell_optimizer_models_setter_errors(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, acquisition_rule)
    with pytest.raises(ValueError):
        ask_tell.models = {}
    with pytest.raises(ValueError):
        ask_tell.models = {OBJECTIVE: LinearWithUnitVariance(), "X": LinearWithUnitVariance()}
    with pytest.raises(ValueError):
        ask_tell.models = {"CONSTRAINT": LinearWithUnitVariance()}


def test_ask_tell_optimizer_model_setter(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, acquisition_rule)
    model2 = LinearWithUnitVariance()
    ask_tell.model = model2
    assert ask_tell.models[OBJECTIVE] is model2 is not model


def test_ask_tell_optimizer_model_setter_errors(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    one_model = AskTellOptimizer(search_space, {"X": init_dataset}, {"X": model}, acquisition_rule)
    with pytest.raises(ValueError):
        one_model.model = model
    two_models = AskTellOptimizer(
        search_space,
        {OBJECTIVE: init_dataset, "X": init_dataset},
        {OBJECTIVE: model, "X": model},
        acquisition_rule,
    )
    with pytest.raises(ValueError):
        two_models.model = model


def test_ask_tell_optimizer_trains_model(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    new_data = mk_dataset([[1.0]], [[1.0]])
    ask_tell = AskTellOptimizer(
        search_space, init_dataset, model, acquisition_rule, fit_model=False
    )

    ask_tell.tell(new_data)
    state_record: Record[None] = ask_tell.to_record()

    assert state_record.model.optimize_count == 1  # type: ignore


@pytest.mark.parametrize("fit_initial_model", [True, False])
def test_ask_tell_optimizer_optimizes_initial_model(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    fit_initial_model: bool,
) -> None:
    ask_tell = AskTellOptimizer(
        search_space, init_dataset, model, acquisition_rule, fit_model=fit_initial_model
    )
    state_record: Record[None] = ask_tell.to_record()

    if fit_initial_model:
        assert state_record.model.optimize_count == 1  # type: ignore
    else:
        assert state_record.model.optimize_count == 0  # type: ignore


def test_ask_tell_optimizer_from_state_does_not_train_model(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    old_state: Record[None] = Record({OBJECTIVE: init_dataset}, {OBJECTIVE: model}, None)

    ask_tell = AskTellOptimizer.from_record(old_state, search_space, acquisition_rule)
    state_record: Record[None] = ask_tell.to_record()

    assert state_record.model.optimize_count == 0  # type: ignore


@pytest.mark.parametrize(
    "starting_state, expected_state",
    [(None, 1), (0, 1), (3, 4)],
)
def test_ask_tell_optimizer_uses_specified_acquisition_state(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    starting_state: int | None,
    expected_state: int,
) -> None:
    class Rule(AcquisitionRule[State[Optional[int], TensorType], Box, ProbabilisticModel]):
        def __init__(self) -> None:
            self.states_received: list[int | None] = []

        def acquire(
            self,
            search_space: Box,
            models: Mapping[Tag, ProbabilisticModel],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
        ) -> State[int | None, TensorType]:
            def go(state: int | None) -> tuple[int | None, TensorType]:
                self.states_received.append(state)

                if state is None:
                    state = 0

                return state + 1, tf.constant([[0.0]], tf.float64)

            return go

    rule = Rule()

    ask_tell = AskTellOptimizer(
        search_space, init_dataset, model, rule, acquisition_state=starting_state
    )
    _ = ask_tell.ask()
    state_record: Record[State[int, TensorType]] = ask_tell.to_record()

    # mypy cannot see that this is in fact int
    assert state_record.acquisition_state == expected_state  # type: ignore
    assert ask_tell.acquisition_state == expected_state


def test_ask_tell_optimizer_does_not_accept_empty_datasets_or_models(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    with pytest.raises(ValueError):
        AskTellOptimizer(search_space, {}, model, acquisition_rule)  # type: ignore

    with pytest.raises(ValueError):
        AskTellOptimizer(search_space, init_dataset, {}, acquisition_rule)  # type: ignore


def test_ask_tell_optimizer_validates_keys(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    dataset_with_key_1 = {TAG1: init_dataset}
    model_with_key_2 = {TAG2: model}

    with pytest.raises(ValueError):
        AskTellOptimizer(search_space, dataset_with_key_1, model_with_key_2, acquisition_rule)


def test_ask_tell_optimizer_tell_validates_keys(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    dataset_with_key_1 = {TAG1: init_dataset}
    model_with_key_1 = {TAG1: model}
    new_data_with_key_2 = {TAG2: mk_dataset([[1.0]], [[1.0]])}

    ask_tell = AskTellOptimizer(
        search_space, dataset_with_key_1, model_with_key_1, acquisition_rule
    )
    with pytest.raises(ValueError):
        ask_tell.tell(new_data_with_key_2)


def test_ask_tell_optimizer_default_acquisition_requires_objective_tag(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
) -> None:
    wrong_tag: Tag = f"{OBJECTIVE}_WRONG"
    wrong_datasets = {wrong_tag: init_dataset}
    wrong_models = {wrong_tag: model}

    with pytest.raises(ValueError):
        AskTellOptimizer(search_space, wrong_datasets, wrong_models)


def test_ask_tell_optimizer_for_uncopyable_model(
    search_space: Box,
    init_dataset: Dataset,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    class _UncopyableModel(LinearWithUnitVariance):
        def __deepcopy__(self, memo: dict[int, object]) -> _UncopyableModel:
            raise MemoryError

    model = _UncopyableModel()
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, acquisition_rule)

    with pytest.raises(NotImplementedError):
        ask_tell.to_result()
    assert ask_tell.to_result(copy=False).final_result.is_ok

    ask_tell.tell(mk_dataset([[1.0]], [[1.0]]))

    with pytest.raises(NotImplementedError):
        ask_tell.to_result()
    assert ask_tell.to_result(copy=False).final_result.is_ok

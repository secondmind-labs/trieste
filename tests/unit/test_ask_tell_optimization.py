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
from trieste.bayesian_optimizer import Record
from trieste.data import Dataset
from trieste.models.interfaces import ProbabilisticModel, TrainableProbabilisticModel
from trieste.space import Box
from trieste.types import State, TensorType


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
def acquisition_rule() -> AcquisitionRule[TensorType, Box]:
    return FixedAcquisitionRule([[0.0]])


@pytest.fixture
def model() -> TrainableProbabilisticModel:
    return LinearWithUnitVariance()


def test_ask_tell_optimizer_suggests_new_point(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box],
) -> None:
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, True, acquisition_rule)

    new_point = ask_tell.ask()

    assert len(new_point) == 1


def test_ask_tell_optimizer_returns_complete_state(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box],
) -> None:
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, True, acquisition_rule)

    state_record = ask_tell.get_state()

    assert_datasets_allclose(state_record.dataset, init_dataset)
    assert isinstance(state_record.model, type(model))
    assert state_record.acquisition_state is None


def test_ask_tell_optimizer_loads_from_state(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box],
) -> None:
    old_state: Record[TensorType] = Record(
        datasets={"": init_dataset}, models={"": model}, acquisition_state=None
    )

    ask_tell = AskTellOptimizer.from_record(search_space, acquisition_rule, old_state)
    new_state = ask_tell.get_state()

    assert_datasets_allclose(old_state.dataset, new_state.dataset)
    assert type(old_state.model) == type(new_state.model)


def test_ask_tell_optimizer_updates_state_with_new_data(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box],
) -> None:
    new_data = mk_dataset([[1.0]], [[1.0]])
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, True, acquisition_rule)

    ask_tell.tell(new_data)
    state_record = ask_tell.get_state()

    assert_datasets_allclose(state_record.dataset, init_dataset + new_data)


def test_ask_tell_optimizer_trains_model(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box],
) -> None:
    new_data = mk_dataset([[1.0]], [[1.0]])
    ask_tell = AskTellOptimizer(search_space, init_dataset, model, False, acquisition_rule)

    ask_tell.tell(new_data)
    state_record = ask_tell.get_state()

    assert state_record.model.optimize_count == 1  # type: ignore


@pytest.mark.parametrize("fit_initial_model", [True, False])
def test_ask_tell_optimizer_optimizes_initial_model(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box],
    fit_initial_model: bool,
) -> None:
    ask_tell = AskTellOptimizer(
        search_space, init_dataset, model, fit_initial_model, acquisition_rule
    )
    state_record = ask_tell.get_state()

    if fit_initial_model:
        assert state_record.model.optimize_count == 1  # type: ignore
    else:
        assert state_record.model.optimize_count == 0  # type: ignore


def test_ask_tell_optimizer_from_state_does_not_train_model(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box],
) -> None:
    old_state: Record[TensorType] = Record(
        datasets={"": init_dataset}, models={"": model}, acquisition_state=None
    )

    ask_tell = AskTellOptimizer.from_record(search_space, acquisition_rule, old_state)
    state_record = ask_tell.get_state()

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
    class Rule(AcquisitionRule[State[Optional[int], TensorType], Box]):
        def __init__(self) -> None:
            self.states_received: list[int | None] = []

        def acquire(
            self,
            search_space: Box,
            datasets: Mapping[str, Dataset],
            models: Mapping[str, ProbabilisticModel],
        ) -> State[int | None, TensorType]:
            def go(state: int | None) -> tuple[int | None, TensorType]:
                self.states_received.append(state)

                if state is None:
                    state = 0

                return state + 1, tf.constant([[0.0]], tf.float64)

            return go

    rule = Rule()

    ask_tell = AskTellOptimizer(search_space, init_dataset, model, True, rule, starting_state)
    _ = ask_tell.ask()
    state_record = ask_tell.get_state()

    assert state_record.acquisition_state == expected_state


def test_ask_tell_optimizer_does_not_accept_empty_datasets_or_models(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box],
) -> None:
    with pytest.raises(ValueError):
        AskTellOptimizer(search_space, {}, model, True, acquisition_rule)

    with pytest.raises(ValueError):
        AskTellOptimizer(search_space, init_dataset, {}, True, acquisition_rule)


def test_ask_tell_optimizer_validates_keys(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box],
) -> None:
    dataset_with_key_1 = {"1": init_dataset}
    model_with_key_2 = {"2": model}

    with pytest.raises(ValueError):
        AskTellOptimizer(search_space, dataset_with_key_1, model_with_key_2, True, acquisition_rule)


def test_ask_tell_optimizer_tell_validates_keys(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box],
) -> None:
    dataset_with_key_1 = {"1": init_dataset}
    model_with_key_1 = {"1": model}
    new_data_with_key_2 = {"2": mk_dataset([[1.0]], [[1.0]])}

    ask_tell = AskTellOptimizer(
        search_space, dataset_with_key_1, model_with_key_1, True, acquisition_rule
    )
    with pytest.raises(ValueError):
        ask_tell.tell(new_data_with_key_2)

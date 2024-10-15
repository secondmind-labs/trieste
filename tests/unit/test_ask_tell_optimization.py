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

from itertools import zip_longest
from typing import Mapping, Optional, Sequence, Type, Union

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import (
    FixedAcquisitionRule,
    FixedLocalAcquisitionRule,
    assert_datasets_allclose,
    mk_dataset,
)
from tests.util.models.gpflow.models import (
    GaussianProcess,
    PseudoTrainableProbModel,
    QuadraticMeanAndRBFKernel,
    rbf,
)
from trieste.acquisition.rule import AcquisitionRule, LocalDatasetsAcquisitionRule
from trieste.acquisition.utils import copy_to_local_models
from trieste.ask_tell_optimization import (
    AskTellOptimizer,
    AskTellOptimizerNoTraining,
    AskTellOptimizerState,
)
from trieste.bayesian_optimizer import OptimizationResult, Record
from trieste.data import Dataset
from trieste.models.interfaces import ProbabilisticModel, TrainableProbabilisticModel
from trieste.objectives.utils import mk_batch_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box
from trieste.types import State, Tag, TensorType
from trieste.utils.misc import LocalizedTag

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
    return mk_dataset([[0.0], [0.5]], [[0.0], [0.5]])


@pytest.fixture
def acquisition_rule() -> AcquisitionRule[TensorType, Box, ProbabilisticModel]:
    return FixedAcquisitionRule([[0.0]])


@pytest.fixture
def local_acquisition_rule() -> LocalDatasetsAcquisitionRule[TensorType, Box, ProbabilisticModel]:
    return FixedLocalAcquisitionRule([[0.0]], 3)


@pytest.fixture
def model() -> TrainableProbabilisticModel:
    return LinearWithUnitVariance()


# most of the tests below should be run for both AskTellOptimizer and AskTellOptimizerNoTraining
OPTIMIZERS = [AskTellOptimizer, AskTellOptimizerNoTraining]
OptimizerType = Union[
    Type[AskTellOptimizer[Box, TrainableProbabilisticModel]],
    Type[AskTellOptimizerNoTraining[Box, TrainableProbabilisticModel]],
]


@pytest.mark.parametrize("track_data", [True, False])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_suggests_new_point(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
    track_data: bool,
) -> None:
    ask_tell = optimizer(search_space, init_dataset, model, acquisition_rule, track_data=track_data)

    new_point = ask_tell.ask()

    assert len(new_point) == 1


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_with_default_acquisition_suggests_new_point(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    optimizer: OptimizerType,
) -> None:
    ask_tell = optimizer(search_space, init_dataset, model)

    new_point = ask_tell.ask()

    assert len(new_point) == 1


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("copy", [True, False])
def test_ask_tell_optimizer_returns_complete_record(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
    copy: bool,
) -> None:
    ask_tell = optimizer(search_space, init_dataset, model, acquisition_rule)

    state_record: Record[None, TrainableProbabilisticModel] = ask_tell.to_record(copy=copy)

    assert_datasets_allclose(state_record.dataset, init_dataset)
    assert isinstance(state_record.model, type(model))
    assert state_record.acquisition_state is None


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("copy", [True, False])
def test_ask_tell_optimizer_loads_from_record(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
    copy: bool,
) -> None:
    old_state: Record[None, TrainableProbabilisticModel] = Record(
        {OBJECTIVE: init_dataset}, {OBJECTIVE: model}, None
    )

    ask_tell = optimizer.from_record(old_state, search_space, acquisition_rule)
    new_state: Record[None, TrainableProbabilisticModel] = ask_tell.to_record(copy=copy)

    assert_datasets_allclose(old_state.dataset, new_state.dataset)
    assert isinstance(new_state.model, type(old_state.model))


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_returns_complete_state(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    local_acquisition_rule: LocalDatasetsAcquisitionRule[
        TensorType, Box, TrainableProbabilisticModel
    ],
    optimizer: OptimizerType,
) -> None:
    ask_tell = optimizer(
        search_space, init_dataset, model, local_acquisition_rule, track_data=False
    )

    state: AskTellOptimizerState[None, TrainableProbabilisticModel] = ask_tell.to_state()

    assert_datasets_allclose(state.record.dataset, init_dataset)
    assert isinstance(state.record.model, type(model))
    assert state.record.acquisition_state is None
    assert state.local_data_ixs is not None
    assert state.local_data_len == 2
    npt.assert_array_equal(
        state.local_data_ixs,
        [
            tf.range(len(init_dataset.query_points))
            for _ in range(local_acquisition_rule.num_local_datasets)
        ],
    )


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_loads_from_state(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    local_acquisition_rule: LocalDatasetsAcquisitionRule[
        TensorType, Box, TrainableProbabilisticModel
    ],
    optimizer: OptimizerType,
) -> None:
    old_state: AskTellOptimizerState[None, TrainableProbabilisticModel] = AskTellOptimizerState(
        record=Record({OBJECTIVE: init_dataset}, {OBJECTIVE: model}, None),
        local_data_ixs=[
            tf.range(len(init_dataset.query_points))
            for _ in range(local_acquisition_rule.num_local_datasets)
        ],
        local_data_len=len(init_dataset.query_points),
    )

    ask_tell = optimizer.from_state(
        old_state,
        search_space,
        local_acquisition_rule,
        track_data=False,
    )
    new_state: AskTellOptimizerState[None, TrainableProbabilisticModel] = ask_tell.to_state()

    assert_datasets_allclose(new_state.record.dataset, old_state.record.dataset)
    assert old_state.record.model is new_state.record.model
    assert new_state.local_data_ixs is not None
    assert old_state.local_data_ixs is not None
    npt.assert_array_equal(new_state.local_data_ixs, old_state.local_data_ixs)
    assert old_state.local_data_len == new_state.local_data_len == len(init_dataset.query_points)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("copy", [True, False])
def test_ask_tell_optimizer_returns_optimization_result(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
    copy: bool,
) -> None:
    ask_tell = optimizer(search_space, init_dataset, model, acquisition_rule)

    result: OptimizationResult[None, TrainableProbabilisticModel] = ask_tell.to_result(copy=copy)

    assert_datasets_allclose(result.try_get_final_dataset(), init_dataset)
    assert isinstance(result.try_get_final_model(), type(model))


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_updates_state_with_new_data(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
) -> None:
    new_data = mk_dataset([[1.0]], [[1.0]])
    ask_tell = optimizer(search_space, init_dataset, model, acquisition_rule)

    ask_tell.tell(new_data)
    state_record: Record[None, TrainableProbabilisticModel] = ask_tell.to_record()

    assert_datasets_allclose(state_record.dataset, init_dataset + new_data)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_doesnt_update_state_with_new_data(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
) -> None:
    new_data = mk_dataset([[1.0]], [[1.0]])
    ask_tell = optimizer(search_space, init_dataset, model, acquisition_rule, track_data=False)

    ask_tell.tell(new_data)
    state_record: Record[None, TrainableProbabilisticModel] = ask_tell.to_record()

    assert_datasets_allclose(state_record.dataset, init_dataset)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("copy", [True, False])
def test_ask_tell_optimizer_copies_state(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
    copy: bool,
) -> None:
    new_data = mk_dataset([[1.0]], [[1.0]])
    ask_tell = optimizer(search_space, init_dataset, model, acquisition_rule)
    state_start: Record[None, TrainableProbabilisticModel] = ask_tell.to_record(copy=copy)
    ask_tell.tell(new_data)
    state_end: Record[None, TrainableProbabilisticModel] = ask_tell.to_record(copy=copy)

    assert_datasets_allclose(state_start.dataset, init_dataset if copy else init_dataset + new_data)
    assert_datasets_allclose(state_end.dataset, init_dataset + new_data)
    assert state_start.model is not model if copy else state_start.model is model


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_datasets_property(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
) -> None:
    ask_tell = optimizer(search_space, init_dataset, model, acquisition_rule)
    assert_datasets_allclose(ask_tell.datasets[OBJECTIVE], init_dataset)
    assert_datasets_allclose(ask_tell.dataset, init_dataset)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_models_property(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
) -> None:
    ask_tell = optimizer(search_space, init_dataset, model, acquisition_rule)
    assert ask_tell.models[OBJECTIVE] is model
    assert ask_tell.model is model


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_models_setter(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
) -> None:
    ask_tell = optimizer(search_space, init_dataset, model, acquisition_rule)
    model2 = LinearWithUnitVariance()
    ask_tell.models = {OBJECTIVE: model2}
    assert ask_tell.models[OBJECTIVE] is model2 is not model


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_models_setter_errors(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
) -> None:
    ask_tell = optimizer(search_space, init_dataset, model, acquisition_rule)
    with pytest.raises(ValueError):
        ask_tell.models = {}
    with pytest.raises(ValueError):
        ask_tell.models = {OBJECTIVE: LinearWithUnitVariance(), "X": LinearWithUnitVariance()}
    with pytest.raises(ValueError):
        ask_tell.models = {"CONSTRAINT": LinearWithUnitVariance()}


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_model_setter(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
) -> None:
    ask_tell = optimizer(search_space, init_dataset, model, acquisition_rule)
    model2 = LinearWithUnitVariance()
    ask_tell.model = model2
    assert ask_tell.models[OBJECTIVE] is model2 is not model


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_model_setter_errors(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
) -> None:
    one_model = optimizer(search_space, {"X": init_dataset}, {"X": model}, acquisition_rule)
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


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("track_data", [False, True])
def test_ask_tell_optimizer_local_data_ixs_property(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    local_acquisition_rule: LocalDatasetsAcquisitionRule[
        TensorType, Box, TrainableProbabilisticModel
    ],
    optimizer: OptimizerType,
    track_data: bool,
) -> None:
    local_data_ixs = [
        tf.range(min(i, len(init_dataset.query_points)))
        for i in range(local_acquisition_rule.num_local_datasets)
    ]
    ask_tell = optimizer(
        search_space,
        init_dataset,
        model,
        local_acquisition_rule,
        track_data=track_data,
        local_data_ixs=local_data_ixs,
    )
    if track_data:
        assert ask_tell.local_data_ixs is None
        assert ask_tell.local_data_len is None
    else:
        assert ask_tell.local_data_ixs is not None
        for expected, actual in zip_longest(local_data_ixs, ask_tell.local_data_ixs):
            npt.assert_array_equal(expected, actual)
        assert ask_tell.local_data_len == len(init_dataset.query_points)


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
    state_record: Record[None, TrainableProbabilisticModel] = ask_tell.to_record()

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
    state_record: Record[None, TrainableProbabilisticModel] = ask_tell.to_record()

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
    old_state: Record[None, TrainableProbabilisticModel] = Record(
        {OBJECTIVE: init_dataset}, {OBJECTIVE: model}, None
    )

    ask_tell = AskTellOptimizer.from_record(old_state, search_space, acquisition_rule)
    state_record: Record[None, TrainableProbabilisticModel] = ask_tell.to_record()

    assert state_record.model.optimize_count == 0  # type: ignore


def test_ask_tell_optimizer_no_training_does_not_train_model(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
) -> None:
    new_data = mk_dataset([[1.0]], [[1.0]])
    ask_tell = AskTellOptimizerNoTraining(
        search_space, init_dataset, model, acquisition_rule, fit_model=True
    )

    ask_tell.tell(new_data)
    state_record: Record[None, TrainableProbabilisticModel] = ask_tell.to_record()

    assert state_record.model.optimize_count == 0  # type: ignore


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize(
    "starting_state, expected_state",
    [(None, 1), (0, 1), (3, 4)],
)
def test_ask_tell_optimizer_uses_specified_acquisition_state(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    optimizer: OptimizerType,
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

    ask_tell = optimizer(search_space, init_dataset, model, rule, acquisition_state=starting_state)
    _ = ask_tell.ask()
    state_record: Record[State[int, TensorType], TrainableProbabilisticModel] = ask_tell.to_record()

    # mypy cannot see that this is in fact int
    assert state_record.acquisition_state == expected_state  # type: ignore
    assert ask_tell.acquisition_state == expected_state


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_does_not_accept_empty_datasets_or_models(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
) -> None:
    with pytest.raises(ValueError):
        optimizer(search_space, {}, model, acquisition_rule)  # type: ignore

    with pytest.raises(ValueError):
        optimizer(search_space, init_dataset, {}, acquisition_rule)  # type: ignore


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_validates_keys(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
) -> None:
    dataset_with_key_1 = {TAG1: init_dataset}
    model_with_key_2 = {TAG2: model}

    with pytest.raises(ValueError):
        optimizer(search_space, dataset_with_key_1, model_with_key_2, acquisition_rule)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_tell_validates_keys(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
) -> None:
    dataset_with_key_1 = {TAG1: init_dataset}
    model_with_key_1 = {TAG1: model}
    new_data_with_key_2 = {TAG2: mk_dataset([[1.0]], [[1.0]])}

    ask_tell = optimizer(search_space, dataset_with_key_1, model_with_key_1, acquisition_rule)
    with pytest.raises(ValueError):
        ask_tell.tell(new_data_with_key_2)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_default_acquisition_requires_objective_tag(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    optimizer: OptimizerType,
) -> None:
    wrong_tag: Tag = f"{OBJECTIVE}_WRONG"
    wrong_datasets = {wrong_tag: init_dataset}
    wrong_models = {wrong_tag: model}

    with pytest.raises(ValueError):
        optimizer(search_space, wrong_datasets, wrong_models)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_for_uncopyable_model(
    search_space: Box,
    init_dataset: Dataset,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    optimizer: OptimizerType,
) -> None:
    class _UncopyableModel(LinearWithUnitVariance):
        def __deepcopy__(self, memo: dict[int, object]) -> _UncopyableModel:
            raise MemoryError

    model = _UncopyableModel()
    ask_tell = optimizer(search_space, init_dataset, model, acquisition_rule)

    with pytest.raises(NotImplementedError):
        ask_tell.to_result()
    assert ask_tell.to_result(copy=False).final_result.is_ok

    ask_tell.tell(mk_dataset([[1.0]], [[1.0]]))

    with pytest.raises(NotImplementedError):
        ask_tell.to_result()
    assert ask_tell.to_result(copy=False).final_result.is_ok


class DatasetChecker(QuadraticMeanAndRBFKernel, PseudoTrainableProbModel):
    def __init__(
        self,
        use_global_model: bool,
        use_global_init_dataset: bool,
        init_data: Mapping[Tag, Dataset],
        query_points: TensorType,
    ) -> None:
        super().__init__()
        self.update_count = 0
        self._tag = OBJECTIVE
        self.use_global_model = use_global_model
        self.use_global_init_dataset = use_global_init_dataset
        self.init_data = init_data
        self.query_points = query_points

    def update(self, dataset: Dataset) -> None:
        if self.use_global_model:
            exp_init_qps = self.init_data[OBJECTIVE].query_points
        else:
            if self.use_global_init_dataset:
                exp_init_qps = self.init_data[OBJECTIVE].query_points
            else:
                exp_init_qps = self.init_data[self._tag].query_points

        if self.update_count == 0:
            # Initial model training.
            exp_qps = exp_init_qps
        else:
            # Subsequent model training.
            if self.use_global_model:
                exp_qps = tf.concat([exp_init_qps, tf.reshape(self.query_points, [-1, 1])], 0)
            else:
                index = LocalizedTag.from_tag(self._tag).local_index
                exp_qps = tf.concat([exp_init_qps, self.query_points[:, index]], 0)

        npt.assert_array_equal(exp_qps, dataset.query_points)
        self.update_count += 1


# Check that the correct dataset is routed to the model.
# Note: this test is almost identical to the one in test_bayesian_optimizer.py.
@pytest.mark.parametrize("use_global_model", [True, False])
@pytest.mark.parametrize("use_global_init_dataset", [True, False])
@pytest.mark.parametrize("num_query_points_per_batch", [1, 2])
def test_ask_tell_optimizer_creates_correct_datasets_for_rank3_points(
    use_global_model: bool, use_global_init_dataset: bool, num_query_points_per_batch: int
) -> None:
    batch_size = 4
    if use_global_init_dataset:
        init_data = {OBJECTIVE: mk_dataset([[0.5], [1.5]], [[0.25], [0.35]])}
    else:
        init_data = {
            LocalizedTag(OBJECTIVE, i): mk_dataset([[0.5 + i], [1.5 + i]], [[0.25], [0.35]])
            for i in range(batch_size)
        }
        init_data[OBJECTIVE] = mk_dataset([[0.5], [1.5]], [[0.25], [0.35]])

    query_points = tf.reshape(
        tf.constant(range(batch_size * num_query_points_per_batch), tf.float64),
        (num_query_points_per_batch, batch_size, 1),
    )

    search_space = Box([-1], [1])

    model = DatasetChecker(use_global_model, use_global_init_dataset, init_data, query_points)
    if use_global_model:
        models = {OBJECTIVE: model}
    else:
        models = copy_to_local_models(model, batch_size)  # type: ignore[assignment]
    for tag, model in models.items():
        model._tag = tag

    observer = mk_batch_observer(lambda x: Dataset(x, x))
    rule = FixedLocalAcquisitionRule(query_points, batch_size)
    ask_tell = AskTellOptimizer(search_space, init_data, models, rule)

    points = ask_tell.ask()
    new_data = observer(points)
    ask_tell.tell(new_data)


def test_ask_tell_optimizer_no_training_with_non_trainable_model(
    search_space: Box,
    init_dataset: Dataset,
    acquisition_rule: AcquisitionRule[TensorType, Box, ProbabilisticModel],
) -> None:
    model = GaussianProcess([lambda x: 2 * x], [rbf()])
    new_data = mk_dataset([[1.0]], [[1.0]])
    ask_tell = AskTellOptimizerNoTraining(search_space, init_dataset, model, acquisition_rule)

    new_point = ask_tell.ask()
    assert len(new_point) == 1

    ask_tell.tell(new_data)
    state_record: Record[None, ProbabilisticModel] = ask_tell.to_record()
    assert_datasets_allclose(state_record.dataset, init_dataset + new_data)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize(
    "new_data_ixs", [None, [tf.constant([2, 3, 4]), tf.constant([7]), tf.constant([3])]]
)
def test_ask_tell_optimizer_tracks_local_data_ixs(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    local_acquisition_rule: LocalDatasetsAcquisitionRule[
        TensorType, Box, TrainableProbabilisticModel
    ],
    optimizer: OptimizerType,
    new_data_ixs: Optional[Sequence[TensorType]],
) -> None:
    ask_tell = optimizer(
        search_space, init_dataset, model, local_acquisition_rule, track_data=False
    )
    new_data = mk_dataset(
        [[x / 100] for x in range(75, 75 + 6)], [[x / 100] for x in range(75, 75 + 6)]
    )
    ask_tell.tell(init_dataset + new_data, new_data_ixs=new_data_ixs)

    if new_data_ixs is None:
        # default is to assign new points round-robin
        expected_indices = [[0, 1, 2, 5], [0, 1, 3, 6], [0, 1, 4, 7]]
    else:
        expected_indices = [[0, 1, 2, 3, 4], [0, 1, 7], [0, 1, 3]]

    assert ask_tell.local_data_ixs is not None
    for ixs, expected_ixs in zip_longest(ask_tell.local_data_ixs, expected_indices):
        assert ixs.numpy().tolist() == expected_ixs
    assert ask_tell.local_data_len == len(init_dataset + new_data)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_raises_when_round_robin_fails(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    local_acquisition_rule: LocalDatasetsAcquisitionRule[
        TensorType, Box, TrainableProbabilisticModel
    ],
    optimizer: OptimizerType,
) -> None:
    ask_tell = optimizer(
        search_space, init_dataset, model, local_acquisition_rule, track_data=False
    )
    # five points can't be round-robined properly across three datasets
    new_data = mk_dataset(
        [[x / 100] for x in range(75, 75 + 5)], [[x / 100] for x in range(75, 75 + 5)]
    )
    with pytest.raises(ValueError, match="Cannot infer new data points"):
        ask_tell.tell(init_dataset + new_data)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_raises_with_badly_shaped_new_data_idxs(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    local_acquisition_rule: LocalDatasetsAcquisitionRule[
        TensorType, Box, TrainableProbabilisticModel
    ],
    optimizer: OptimizerType,
) -> None:
    ask_tell = optimizer(
        search_space, init_dataset, model, local_acquisition_rule, track_data=False
    )
    new_data = mk_dataset(
        [[x / 100] for x in range(75, 75 + 6)], [[x / 100] for x in range(75, 75 + 6)]
    )
    with pytest.raises(ValueError, match="new_data_ixs has 1"):
        ask_tell.tell(init_dataset + new_data, new_data_ixs=[tf.constant([[4]])])


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize(
    "local_data_len,expected_indices",
    [
        (None, [[0, 1], [0], [1]]),
        (2, [[0, 1, 2, 5], [0, 3, 6], [1, 4, 7]]),  # extended via round-robin
        (8, [[0, 1], [0], [1]]),
    ],
)
def test_ask_tell_optimizer_local_data_len(
    search_space: Box,
    model: TrainableProbabilisticModel,
    local_acquisition_rule: LocalDatasetsAcquisitionRule[
        TensorType, Box, TrainableProbabilisticModel
    ],
    optimizer: OptimizerType,
    local_data_len: Optional[int],
    expected_indices: Sequence[list[int]],
) -> None:
    dataset = mk_dataset(
        [[x / 100] for x in range(75, 75 + 8)], [[x / 100] for x in range(75, 75 + 8)]
    )
    local_data_ixs = [tf.constant([0, 1]), tf.constant([0]), tf.constant([1])]
    ask_tell = optimizer(
        search_space,
        dataset,
        model,
        local_acquisition_rule,
        track_data=False,
        local_data_ixs=local_data_ixs,
        local_data_len=local_data_len,
    )

    assert ask_tell.local_data_ixs is not None
    for ixs, expected_ixs in zip_longest(ask_tell.local_data_ixs, expected_indices):
        assert ixs.numpy().tolist() == expected_ixs
    assert ask_tell.local_data_len == len(dataset)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_raises_with_inconsistent_local_data_len(
    search_space: Box,
    model: TrainableProbabilisticModel,
    local_acquisition_rule: LocalDatasetsAcquisitionRule[
        TensorType, Box, TrainableProbabilisticModel
    ],
    optimizer: OptimizerType,
) -> None:
    dataset = mk_dataset(
        [[x / 100] for x in range(75, 75 + 8)], [[x / 100] for x in range(75, 75 + 8)]
    )
    local_data_ixs = [tf.constant([0, 1]), tf.constant([0]), tf.constant([1])]
    with pytest.raises(ValueError, match="Cannot infer new data points"):
        optimizer(
            search_space,
            dataset,
            model,
            local_acquisition_rule,
            track_data=False,
            local_data_ixs=local_data_ixs,
            local_data_len=6,
        )


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_uses_pre_filter_state_in_to_record(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    local_acquisition_rule: LocalDatasetsAcquisitionRule[
        TensorType, Box, TrainableProbabilisticModel
    ],
    optimizer: OptimizerType,
) -> None:
    ask_tell = optimizer(
        search_space, init_dataset, model, local_acquisition_rule, track_data=False
    )
    new_data = mk_dataset(
        [[x / 100] for x in range(75, 75 + 6)], [[x / 100] for x in range(75, 75 + 6)]
    )

    # the internal acquisition state is incremented every time we call either ask or tell
    # and once at initialisation; however, the state reported in to_record() is only updated
    # after calling ask
    assert ask_tell.to_record().acquisition_state is None
    ask_tell.ask()
    assert ask_tell.to_record().acquisition_state == 2
    ask_tell.tell(init_dataset + new_data)
    assert ask_tell.to_record().acquisition_state == 2
    ask_tell.ask()
    assert ask_tell.to_record().acquisition_state == 4
    ask_tell.tell(init_dataset + new_data + new_data)
    assert ask_tell.to_record().acquisition_state == 4

    # the pattern continues for a copy made using the reported state
    ask_tell_copy = optimizer.from_record(
        ask_tell.to_record(), search_space, local_acquisition_rule, track_data=False
    )
    assert ask_tell_copy.to_record().acquisition_state == 4
    ask_tell_copy.ask()
    assert ask_tell_copy.to_record().acquisition_state == 6


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_calls_initialize_subspaces(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    local_acquisition_rule: LocalDatasetsAcquisitionRule[
        TensorType, Box, TrainableProbabilisticModel
    ],
    optimizer: OptimizerType,
) -> None:
    assert isinstance(local_acquisition_rule, FixedLocalAcquisitionRule)
    assert local_acquisition_rule._initialize_subspaces_calls == 0
    optimizer(search_space, init_dataset, model, local_acquisition_rule, track_data=False)
    assert local_acquisition_rule._initialize_subspaces_calls == 1


@pytest.mark.parametrize("variable", [False, True])
def test_ask_tell_optimizer_dataset_len_variables(
    init_dataset: Dataset,
    variable: bool,
) -> None:
    if variable:
        dataset = Dataset(
            tf.Variable(
                init_dataset.query_points, shape=[None, *init_dataset.query_points.shape[1:]]
            ),
            tf.Variable(
                init_dataset.observations, shape=[None, *init_dataset.observations.shape[1:]]
            ),
        )
    else:
        dataset = init_dataset

    assert AskTellOptimizer.dataset_len({"tag": dataset}) == 2
    assert AskTellOptimizer.dataset_len({"tag1": dataset, "tag2": dataset}) == 2


def test_ask_tell_optimizer_dataset_len_raises_on_inconsistently_sized_datasets(
    init_dataset: Dataset,
) -> None:
    with pytest.raises(ValueError):
        AskTellOptimizer.dataset_len(
            {"tag": init_dataset, "empty": Dataset(tf.zeros([0, 2]), tf.zeros([0, 2]))}
        )
    with pytest.raises(ValueError):
        AskTellOptimizer.dataset_len({})


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_ask_tell_optimizer_doesnt_blow_up_with_no_local_datasets(
    search_space: Box,
    init_dataset: Dataset,
    model: TrainableProbabilisticModel,
    optimizer: OptimizerType,
) -> None:
    pseudo_local_acquisition_rule = FixedLocalAcquisitionRule([[0.0]], 0)
    ask_tell = optimizer(
        search_space, init_dataset, model, pseudo_local_acquisition_rule, track_data=False
    )
    ask_tell._datasets[OBJECTIVE] += mk_dataset(
        [[x / 100] for x in range(75, 75 + 5)], [[x / 100] for x in range(75, 75 + 5)]
    )
    ask_tell.tell(ask_tell.dataset)
    optimizer.from_state(
        ask_tell.to_state(),
        search_space,
        pseudo_local_acquisition_rule,
        track_data=False,
    )

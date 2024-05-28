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

import copy
import pickle
import tempfile
from dataclasses import replace
from typing import Callable, Mapping, Tuple, Union

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import random_seed
from trieste.acquisition import LocalPenalization, ParallelContinuousThompsonSampling
from trieste.acquisition.rule import (
    AcquisitionRule,
    AsynchronousGreedy,
    AsynchronousRuleState,
    BatchTrustRegionBox,
    BatchTrustRegionState,
    EfficientGlobalOptimization,
    SingleObjectiveTrustRegionBox,
    TREGOBox,
    UpdatableTrustRegionBox,
)
from trieste.acquisition.utils import copy_to_local_models
from trieste.ask_tell_optimization import AskTellOptimizer, AskTellOptimizerState
from trieste.bayesian_optimizer import OptimizationResult
from trieste.data import Dataset
from trieste.logging import set_step_number, tensorboard_writer
from trieste.models import TrainableProbabilisticModel
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.objectives import ScaledBranin, SimpleQuadratic
from trieste.objectives.utils import mk_batch_observer, mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box, SearchSpace
from trieste.types import State, Tag, TensorType

# Optimizer parameters for testing against the branin function.
# We use a copy of these for a quicker test against a simple quadratic function
# (copying is necessary as some of the acquisition rules are stateful).
OPTIMIZER_PARAMS = (
    "num_steps, reload_state, acquisition_rule_fn",
    [
        pytest.param(
            20, False, lambda: EfficientGlobalOptimization(), id="EfficientGlobalOptimization"
        ),
        pytest.param(
            20,
            True,
            lambda: EfficientGlobalOptimization(),
            id="EfficientGlobalOptimization/reload_state",
        ),
        pytest.param(
            15,
            False,
            lambda: BatchTrustRegionBox(TREGOBox(ScaledBranin.search_space)),
            id="TREGO",
        ),
        pytest.param(
            16,
            True,
            lambda: BatchTrustRegionBox(TREGOBox(ScaledBranin.search_space)),
            id="TREGO/reload_state",
        ),
        pytest.param(
            10,
            False,
            lambda: BatchTrustRegionBox(
                [SingleObjectiveTrustRegionBox(ScaledBranin.search_space) for _ in range(3)],
                EfficientGlobalOptimization(
                    ParallelContinuousThompsonSampling(),
                    num_query_points=3,
                ),
            ),
            id="BatchTrustRegionBox",
        ),
        pytest.param(
            10,
            False,
            (
                lambda: BatchTrustRegionBox(
                    [SingleObjectiveTrustRegionBox(ScaledBranin.search_space) for _ in range(3)],
                    EfficientGlobalOptimization(
                        ParallelContinuousThompsonSampling(),
                        num_query_points=2,
                    ),
                ),
                3,
            ),
            id="BatchTrustRegionBox/LocalModels",
        ),
        pytest.param(
            10,
            False,
            lambda: EfficientGlobalOptimization(
                LocalPenalization(
                    ScaledBranin.search_space,
                ).using(OBJECTIVE),
                num_query_points=3,
            ),
            id="LocalPenalization",
        ),
        pytest.param(
            30,
            False,
            lambda: AsynchronousGreedy(
                LocalPenalization(
                    ScaledBranin.search_space,
                ).using(OBJECTIVE),
            ),
            id="LocalPenalization/AsynchronousGreedy",
        ),
    ],
)


AcquisitionRuleFunction = Union[
    Callable[[], AcquisitionRule[TensorType, SearchSpace, TrainableProbabilisticModel]],
    Callable[
        [],
        AcquisitionRule[
            State[
                TensorType,
                Union[AsynchronousRuleState, BatchTrustRegionState[UpdatableTrustRegionBox]],
            ],
            Box,
            TrainableProbabilisticModel,
        ],
    ],
]


@random_seed
@pytest.mark.slow  # to run this, add --runslow yes to the pytest command
@pytest.mark.parametrize(*OPTIMIZER_PARAMS)
def test_ask_tell_optimizer_finds_minima_of_the_scaled_branin_function(
    num_steps: int,
    reload_state: bool,
    acquisition_rule_fn: AcquisitionRuleFunction | Tuple[AcquisitionRuleFunction, int],
) -> None:
    _test_ask_tell_optimization_finds_minima(True, num_steps, reload_state, acquisition_rule_fn)


@random_seed
@pytest.mark.parametrize("track_data", [True, False])
@pytest.mark.parametrize(*copy.deepcopy(OPTIMIZER_PARAMS))
def test_ask_tell_optimizer_finds_minima_of_simple_quadratic(
    track_data: bool,
    num_steps: int,
    reload_state: bool,
    acquisition_rule_fn: AcquisitionRuleFunction | Tuple[AcquisitionRuleFunction, int],
) -> None:
    # for speed reasons we sometimes test with a simple quadratic defined on the same search space
    # branin; currently assume that every rule should be able to solve this in 5 steps
    _test_ask_tell_optimization_finds_minima(
        False,
        min(num_steps, 5),
        reload_state,
        acquisition_rule_fn,
        track_data=track_data,
    )


def _test_ask_tell_optimization_finds_minima(
    optimize_branin: bool,
    num_steps: int,
    reload_state: bool,
    acquisition_rule_fn: AcquisitionRuleFunction | Tuple[AcquisitionRuleFunction, int],
    track_data: bool = True,
) -> None:
    # For the case when optimization state is saved and reload on each iteration
    # we need to use new acquisition function object to imitate real life usage
    # hence acquisition rule factory method is passed in, instead of a rule object itself
    # it is then called to create a new rule whenever needed in the test
    search_space = ScaledBranin.search_space
    initial_query_points = search_space.sample(5)
    observer = mk_observer(ScaledBranin.objective if optimize_branin else SimpleQuadratic.objective)
    batch_observer = mk_batch_observer(observer)
    initial_data = observer(initial_query_points)

    if isinstance(acquisition_rule_fn, tuple):
        acquisition_rule_fn, num_models = acquisition_rule_fn
    else:
        num_models = 1

    model = GaussianProcessRegression(
        build_gpr(initial_data, search_space, likelihood_variance=1e-7)
    )
    models = copy_to_local_models(model, num_models) if num_models > 1 else {OBJECTIVE: model}
    initial_dataset = {OBJECTIVE: initial_data}

    with tempfile.TemporaryDirectory() as tmpdirname:
        summary_writer = tf.summary.create_file_writer(tmpdirname)
        with tensorboard_writer(summary_writer):
            set_step_number(0)
            ask_tell = AskTellOptimizer(
                search_space, initial_dataset, models, acquisition_rule_fn(), track_data=track_data
            )

            for i in range(1, num_steps + 1):
                # two scenarios are tested here, depending on `reload_state` parameter
                # in first the same optimizer object is always used
                # in second new optimizer is created at each step from saved state
                set_step_number(i)
                new_point = ask_tell.ask()

                if reload_state:
                    state: AskTellOptimizerState[
                        None
                        | State[
                            TensorType,
                            AsynchronousRuleState | BatchTrustRegionState[UpdatableTrustRegionBox],
                        ],
                        GaussianProcessRegression,
                    ] = ask_tell.to_state()
                    written_state = pickle.dumps(state)

                # If query points are rank 3, then use a batched observer.
                if tf.rank(new_point) == 3:
                    new_data_point: Union[Mapping[Tag, Dataset], Dataset] = batch_observer(
                        new_point
                    )
                else:
                    new_data_point = observer(new_point)

                if reload_state:
                    state = pickle.loads(written_state)
                    state_record = state.record
                    if not track_data:
                        # reload using the up-to-date dataset
                        state_record = replace(state_record, datasets=initial_dataset)
                    ask_tell = AskTellOptimizer.from_record(
                        state_record,
                        search_space,
                        acquisition_rule_fn(),
                        track_data=track_data,
                        local_data_ixs=state.local_data_ixs,
                    )

                if track_data:
                    ask_tell.tell(new_data_point)
                else:
                    if isinstance(new_data_point, Dataset):
                        new_data_point = {OBJECTIVE: new_data_point}
                    for tag in initial_dataset.keys():
                        initial_dataset[tag] += new_data_point[tag]
                    ask_tell.tell(initial_dataset)

    result: OptimizationResult[
        None
        | State[TensorType, AsynchronousRuleState | BatchTrustRegionState[UpdatableTrustRegionBox]],
        GaussianProcessRegression,
    ] = ask_tell.to_result()
    dataset = result.try_get_final_dataset()

    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]

    if optimize_branin:
        relative_minimizer_err = tf.abs(
            (best_x - ScaledBranin.minimizers) / ScaledBranin.minimizers
        )
        # these accuracies are the current best for the given number of optimization steps,
        # which makes this is a regression test
        assert tf.reduce_any(tf.reduce_all(relative_minimizer_err < 0.05, axis=-1), axis=0)
        npt.assert_allclose(best_y, ScaledBranin.minimum, rtol=0.005)
    else:
        absolute_minimizer_err = tf.abs(best_x - SimpleQuadratic.minimizers)
        assert tf.reduce_any(tf.reduce_all(absolute_minimizer_err < 0.05, axis=-1), axis=0)
        npt.assert_allclose(best_y, SimpleQuadratic.minimum, rtol=0.05)

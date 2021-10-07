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

import pickle
from typing import Callable, List, Tuple, Union, cast

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import random_seed
from trieste.acquisition.function import LocalPenalizationAcquisitionFunction
from trieste.acquisition.rule import (
    AcquisitionRule,
    AsyncEfficientGlobalOptimization,
    EfficientGlobalOptimization,
    TrustRegion,
)
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.bayesian_optimizer import OptimizationResult, Record
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.objectives import (
    BRANIN_MINIMIZERS,
    BRANIN_SEARCH_SPACE,
    SCALED_BRANIN_MINIMUM,
    scaled_branin,
)
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box, SearchSpace
from trieste.types import State, TensorType


@random_seed
@pytest.mark.parametrize(
    "num_steps, reload_state, acquisition_rule_fn",
    cast(
        List[
            Tuple[
                int,
                bool,
                Union[
                    Callable[[], AcquisitionRule[TensorType, Box]],
                    Callable[
                        [],
                        AcquisitionRule[
                            State[
                                TensorType,
                                Union[AsyncEfficientGlobalOptimization.State, TrustRegion.State],
                            ],
                            Box,
                        ],
                    ],
                ],
            ]
        ],
        [
            (20, False, lambda: EfficientGlobalOptimization()),
            (20, True, lambda: EfficientGlobalOptimization()),
            (15, False, lambda: TrustRegion()),
            (15, True, lambda: TrustRegion()),
            (
                10,
                False,
                lambda: EfficientGlobalOptimization(
                    LocalPenalizationAcquisitionFunction(
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
            (
                30,
                False,
                lambda: AsyncEfficientGlobalOptimization(
                    LocalPenalizationAcquisitionFunction(
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=1,
                ),
            ),
        ],
    ),
)
def test_ask_tell_optimization_finds_minima_of_the_scaled_branin_function(
    num_steps: int,
    reload_state: bool,
    acquisition_rule_fn: Callable[[], AcquisitionRule[TensorType, SearchSpace]]
    | Callable[
        [],
        AcquisitionRule[
            State[TensorType, AsyncEfficientGlobalOptimization.State | TrustRegion.State], Box
        ],
    ],
) -> None:
    # For the case when optimization state is saved and reload on each iteration
    # we need to use new acquisition function object to imitate real life usage
    # hence acquisition rule factory method is passed in, instead of a rule object itself
    # it is then called to create a new rule whenever needed in the test

    search_space = BRANIN_SEARCH_SPACE

    def build_model(data: Dataset) -> GaussianProcessRegression:
        variance = tf.math.reduce_variance(data.observations)
        kernel = gpflow.kernels.Matern52(variance, tf.constant([0.2, 0.2], tf.float64))
        scale = tf.constant(1.0, dtype=tf.float64)
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.constant(-2.0, dtype=tf.float64), scale
        )
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(kernel.lengthscales), scale
        )
        gpr = gpflow.models.GPR((data.query_points, data.observations), kernel, noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        return GaussianProcessRegression(gpr)

    initial_query_points = search_space.sample(5)
    observer = mk_observer(scaled_branin)
    initial_data = observer(initial_query_points)
    model = build_model(initial_data)

    ask_tell = AskTellOptimizer(search_space, initial_data, model, acquisition_rule_fn())

    for _ in range(num_steps):
        # two scenarios are tested here, depending on `reload_state` parameter
        # in first the same optimizer object is always used
        # in second new optimizer is created at each step from saved state
        new_point = ask_tell.ask()

        if reload_state:
            state: Record[
                None | State[TensorType, AsyncEfficientGlobalOptimization.State | TrustRegion.State]
            ] = ask_tell.to_record()
            written_state = pickle.dumps(state)

        new_data_point = observer(new_point)

        if reload_state:
            state = pickle.loads(written_state)
            ask_tell = AskTellOptimizer.from_record(state, search_space, acquisition_rule_fn())

        ask_tell.tell(new_data_point)

    result: OptimizationResult[
        None | State[TensorType, AsyncEfficientGlobalOptimization.State | TrustRegion.State]
    ] = ask_tell.to_result()
    dataset = result.try_get_final_dataset()

    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]

    relative_minimizer_err = tf.abs((best_x - BRANIN_MINIMIZERS) / BRANIN_MINIMIZERS)
    # these accuracies are the current best for the given number of optimization steps, which makes
    # this is a regression test
    assert tf.reduce_any(tf.reduce_all(relative_minimizer_err < 0.05, axis=-1), axis=0)
    npt.assert_allclose(best_y, SCALED_BRANIN_MINIMUM, rtol=0.005)

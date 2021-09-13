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

from typing import List, Tuple, Union, cast

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import random_seed
from trieste.acquisition.function import LocalPenalizationAcquisitionFunction
from trieste.acquisition.rule import AcquisitionRule, EfficientGlobalOptimization, TrustRegion
from trieste.ask_tell_optimization import AskTellOptimizer
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
    "num_steps, reload_state, acquisition_rule",
    cast(
        List[
            Tuple[
                int,
                bool,
                Union[
                    AcquisitionRule[TensorType, Box],
                    AcquisitionRule[State[TensorType, TrustRegion.State], Box],
                ],
            ]
        ],
        [
            (20, False, EfficientGlobalOptimization()),
            (20, True, EfficientGlobalOptimization()),
            (
                10,
                False,
                EfficientGlobalOptimization(
                    LocalPenalizationAcquisitionFunction(
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
            (
                10,
                True,
                EfficientGlobalOptimization(
                    LocalPenalizationAcquisitionFunction(
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
            (15, False, TrustRegion()),
            (15, True, TrustRegion()),
        ],
    ),
)
def test_ask_tell_optimization_finds_minima_of_the_scaled_branin_function(
    num_steps: int,
    reload_state: bool,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace]
    | AcquisitionRule[State[TensorType, TrustRegion.State], Box],
) -> None:
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

    ask_tell = AskTellOptimizer(search_space, initial_data, model, True, acquisition_rule)

    for _ in range(num_steps):
        # two scenarios can be tested here, depending on `reload_state` parameter
        # in first the same optimizer object is always used
        # in second new optimizer is created at each step (e.g. from serialized state)
        new_point = ask_tell.ask()

        if reload_state:
            state = ask_tell.get_state()

        new_data = observer(new_point)

        if reload_state:
            ask_tell = AskTellOptimizer.from_record(search_space, acquisition_rule, state)

        ask_tell.tell(new_data)

    final_state = ask_tell.get_state()
    dataset = final_state.dataset

    print(dataset)
    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]

    relative_minimizer_err = tf.abs((best_x - BRANIN_MINIMIZERS) / BRANIN_MINIMIZERS)
    # these accuracies are the current best for the given number of optimization steps, which makes
    # this is a regression test
    assert tf.reduce_any(tf.reduce_all(relative_minimizer_err < 0.05, axis=-1), axis=0)
    npt.assert_allclose(best_y, SCALED_BRANIN_MINIMUM, rtol=0.005)

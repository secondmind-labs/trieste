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

from typing import List, Tuple, cast

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import random_seed
from trieste.acquisition import (
    AcquisitionFunctionClass,
    BatchMonteCarloExpectedImprovement,
    LocalPenalization,
)
from trieste.acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.models import TrainableProbabilisticModel
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.objectives import (
    BRANIN_MINIMIZERS,
    BRANIN_SEARCH_SPACE,
    SCALED_BRANIN_MINIMUM,
    scaled_branin,
)
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box, DiscreteSearchSpace, TaggedProductSearchSpace
from trieste.types import TensorType


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    cast(
        List[
            Tuple[
                int,
                AcquisitionRule[TensorType, TaggedProductSearchSpace, TrainableProbabilisticModel],
            ]
        ],
        [
            (25, EfficientGlobalOptimization()),
            (
                5,
                EfficientGlobalOptimization(
                    BatchMonteCarloExpectedImprovement(sample_size=500).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
            (
                8,
                EfficientGlobalOptimization(
                    LocalPenalization(
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
        ],
    ),
)
def test_optimizer_finds_minima_of_the_scaled_branin_function(
    num_steps: int,
    acquisition_rule: AcquisitionRule[
        TensorType, TaggedProductSearchSpace, TrainableProbabilisticModel
    ],
) -> None:
    search_space = TaggedProductSearchSpace(
        spaces=[Box([0], [1]), DiscreteSearchSpace(tf.linspace(0, 1, 15)[:, None])],
        tags=["continuous", "discrete"],
    )

    initial_query_points = search_space.sample(5)
    observer = mk_observer(scaled_branin)
    initial_data = observer(initial_query_points)
    model = GaussianProcessRegression(
        build_gpr(initial_data, search_space, likelihood_variance=1e-8)
    )

    dataset = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, model, acquisition_rule)
        .try_get_final_dataset()
    )

    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]

    relative_minimizer_err = tf.abs((best_x - BRANIN_MINIMIZERS) / BRANIN_MINIMIZERS)
    # these accuracies are the current best for the given number of optimization steps, which makes
    # this is a regression test
    assert tf.reduce_any(tf.reduce_all(relative_minimizer_err < 0.1, axis=-1), axis=0)
    npt.assert_allclose(best_y, SCALED_BRANIN_MINIMUM, rtol=0.005)

    # check that acquisition functions defined as classes aren't being retraced unnecessarily
    # They should be retraced once for the optimzier's starting grid and once for L-BFGS.
    if isinstance(acquisition_rule, EfficientGlobalOptimization):
        acquisition_function = acquisition_rule._acquisition_function
        if isinstance(acquisition_function, AcquisitionFunctionClass):
            assert acquisition_function.__call__._get_tracing_count() <= 3  # type: ignore

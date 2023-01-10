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

import math

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import random_seed
from trieste.acquisition import (
    ExpectedConstrainedImprovement,
    ExpectedImprovement,
    ProbabilityOfFeasibility,
)
from trieste.acquisition.function import FastConstraintsFeasibility
from trieste.acquisition.interface import AcquisitionFunctionBuilder
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.objectives import ConstrainedScaledBranin, ScaledBranin
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box
from trieste.types import Tag, TensorType


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_function_builder",
    [
        pytest.param(12, ExpectedConstrainedImprovement, id="ExpectedConstrainedImprovement"),
    ],
)
def test_optimizer_finds_minima_of_Gardners_Simulation_1(
    num_steps: int,
    acquisition_function_builder: type[ExpectedConstrainedImprovement[ProbabilisticModel]],
) -> None:
    """
    Test that tests the covergence of constrained BO algorithms on the
    synthetic "simulation 1" experiment of :cite:`gardner14`.
    """
    search_space = Box([0, 0], [6, 6])

    def objective(input_data: TensorType) -> TensorType:
        x, y = input_data[..., -2], input_data[..., -1]
        z = tf.cos(2.0 * x) * tf.cos(y) + tf.sin(x)
        return z[:, None]

    def constraint(input_data: TensorType) -> TensorType:
        x, y = input_data[:, -2], input_data[:, -1]
        z = tf.cos(x) * tf.cos(y) - tf.sin(x) * tf.sin(y)
        return z[:, None]

    MINIMUM = -2.0
    MINIMIZER = [math.pi * 1.5, 0.0]

    CONSTRAINT = "CONSTRAINT"

    # observe both objective and constraint data
    def observer(query_points: TensorType) -> dict[Tag, Dataset]:
        return {
            OBJECTIVE: Dataset(query_points, objective(query_points)),
            CONSTRAINT: Dataset(query_points, constraint(query_points)),
        }

    num_initial_points = 6
    initial_data = observer(search_space.sample(num_initial_points))

    models = {
        OBJECTIVE: GaussianProcessRegression(build_gpr(initial_data[OBJECTIVE], search_space)),
        CONSTRAINT: GaussianProcessRegression(build_gpr(initial_data[CONSTRAINT], search_space)),
    }

    pof = ProbabilityOfFeasibility(threshold=0.5)
    acq = acquisition_function_builder(OBJECTIVE, pof.using(CONSTRAINT))
    rule: EfficientGlobalOptimization[Box, ProbabilisticModel] = EfficientGlobalOptimization(acq)

    dataset = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, models, rule)
        .try_get_final_datasets()[OBJECTIVE]
    )

    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]

    relative_minimizer_err = tf.abs(best_x - MINIMIZER)
    # these accuracies are the current best for the given number of optimization steps, which makes
    # this is a regression test

    assert tf.reduce_all(relative_minimizer_err < 0.03, axis=-1)
    npt.assert_allclose(best_y, MINIMUM, rtol=0.005)


@random_seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_steps, acquisition_function_builder",
    [
        pytest.param(12, ExpectedImprovement, id="ExpectedImprovement"),
        pytest.param(12, ExpectedConstrainedImprovement, id="ExpectedConstrainedImprovement"),
    ],
)
def test_constrained_optimizer_finds_minima_of_custom_problem(
    num_steps: int,
    acquisition_function_builder: type[AcquisitionFunctionBuilder[ProbabilisticModel]],
) -> None:
    """
    Test the covergence of constrained algorithms on the constrained scaled Branin problem.
    """
    observer = mk_observer(ConstrainedScaledBranin.objective)
    search_space = ConstrainedScaledBranin.search_space

    num_initial_points = 5
    initial_query_points = search_space.sample(num_initial_points)
    initial_data = observer(initial_query_points)

    MINIMUM = ConstrainedScaledBranin.minimum
    MINIMIZER = ConstrainedScaledBranin.minimizers

    OBJECTIVE = "OBJECTIVE"

    model = GaussianProcessRegression(
        build_gpr(initial_data, search_space, likelihood_variance=1e-7)
    )

    if acquisition_function_builder is ExpectedConstrainedImprovement:
        feas = FastConstraintsFeasibility(search_space)  # Search space with constraints.
        eci = acquisition_function_builder(OBJECTIVE, feas.using(OBJECTIVE))  # type: ignore
        rule: EfficientGlobalOptimization[Box, ProbabilisticModel] = EfficientGlobalOptimization(
            eci
        )
        # Note: use the search space without constraints for the penalty method.
        bo_search_space = ScaledBranin.search_space
    else:
        ei = acquisition_function_builder(search_space)  # type: ignore
        rule = EfficientGlobalOptimization(ei)
        bo_search_space = search_space

    dataset = (
        BayesianOptimizer(observer, bo_search_space)
        .optimize(num_steps, initial_data, model, rule)
        .try_get_final_datasets()[OBJECTIVE]
    )

    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]

    relative_minimizer_err = tf.abs(best_x - MINIMIZER)
    # these accuracies are the current best for the given number of optimization steps, which makes
    # this is a regression test

    assert tf.reduce_all(relative_minimizer_err < 0.03, axis=-1)
    npt.assert_allclose(best_y, MINIMUM, rtol=0.1)

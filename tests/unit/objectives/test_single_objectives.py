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

import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.objectives import (
    Ackley5,
    Branin,
    GramacyLee,
    Hartmann3,
    Hartmann6,
    LogarithmicGoldsteinPrice,
    Michalewicz2,
    Michalewicz5,
    Michalewicz10,
    Rosenbrock4,
    ScaledBranin,
    Shekel4,
    SimpleQuadratic,
    SingleObjectiveTestProblem,
    Trid10,
)


@pytest.mark.parametrize(
    "problem",
    [
        Branin,
        ScaledBranin,
        SimpleQuadratic,
        GramacyLee,
        Michalewicz2,
        Michalewicz5,
        Michalewicz10,
        LogarithmicGoldsteinPrice,
        Hartmann3,
        Rosenbrock4,
        Shekel4,
        Ackley5,
        Hartmann6,
        Trid10,
    ],
)
def test_objective_maps_minimizers_to_minimum(
    problem: SingleObjectiveTestProblem,
) -> None:
    objective = problem.objective
    minimizers = problem.minimizers
    minimum = problem.minimum
    objective_values_at_minimizers = objective(minimizers)
    tf.debugging.assert_shapes([(objective_values_at_minimizers, [len(minimizers), 1])])
    npt.assert_allclose(objective_values_at_minimizers, tf.squeeze(minimum), atol=1e-4)


@pytest.mark.parametrize(
    "problem",
    [
        Branin,
        ScaledBranin,
        SimpleQuadratic,
        GramacyLee,
        Michalewicz2,
        Michalewicz5,
        Michalewicz10,
        LogarithmicGoldsteinPrice,
        Hartmann3,
        Rosenbrock4,
        Shekel4,
        Ackley5,
        Hartmann6,
        Trid10,
    ],
)
def test_no_function_values_are_less_than_global_minimum(
    problem: SingleObjectiveTestProblem,
) -> None:
    objective = problem.objective
    space = problem.search_space
    minimum = problem.minimum
    samples = space.sample(1000 * len(space.lower))
    npt.assert_array_less(tf.squeeze(minimum) - 1e-6, objective(samples))

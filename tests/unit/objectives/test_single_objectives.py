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

from typing import Any, Tuple

import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.objectives import (
    Ackley5,
    Branin,
    GramacyLee,
    Hartmann3,
    Hartmann6,
    Levy8,
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
from trieste.space import Box, SearchSpaceType


@pytest.fixture(
    name="problem",
    params=[
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
        Levy8,
    ],
)
def _problem_fixture(request: Any) -> Tuple[SingleObjectiveTestProblem[SearchSpaceType], int]:
    return request.param


def test_objective_maps_minimizers_to_minimum(
    problem: SingleObjectiveTestProblem[SearchSpaceType],
) -> None:
    objective = problem.objective
    minimizers = problem.minimizers
    minimum = problem.minimum
    objective_values_at_minimizers = objective(minimizers)
    tf.debugging.assert_shapes([(objective_values_at_minimizers, [len(minimizers), 1])])
    npt.assert_allclose(objective_values_at_minimizers, tf.squeeze(minimum), atol=1e-4)


def test_no_function_values_are_less_than_global_minimum(
    problem: SingleObjectiveTestProblem[Box],
) -> None:
    objective = problem.objective
    space = problem.search_space
    minimum = problem.minimum
    samples = space.sample_sobol(100_000 * len(space.lower), skip=0)
    npt.assert_array_less(tf.squeeze(minimum) - 1e-6, objective(samples))


@pytest.mark.parametrize("num_obs", [5, 1])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_objective_has_correct_shape_and_dtype(
    problem: SingleObjectiveTestProblem[SearchSpaceType],
    num_obs: int,
    dtype: tf.DType,
) -> None:
    x = problem.search_space.sample(num_obs)
    x = tf.cast(x, dtype)
    y = problem.objective(x)

    assert y.dtype == x.dtype
    tf.debugging.assert_shapes([(y, [num_obs, 1])])


@pytest.mark.parametrize(
    "problem, input_dim",
    [
        (Branin, 2),
        (ScaledBranin, 2),
        (SimpleQuadratic, 2),
        (GramacyLee, 1),
        (Michalewicz2, 2),
        (Michalewicz5, 5),
        (Michalewicz10, 10),
        (LogarithmicGoldsteinPrice, 2),
        (Hartmann3, 3),
        (Rosenbrock4, 4),
        (Shekel4, 4),
        (Ackley5, 5),
        (Hartmann6, 6),
        (Trid10, 10),
        (Levy8, 8),
    ],
)
@pytest.mark.parametrize("num_obs", [5, 1])
def test_search_space_has_correct_shape_and_default_dtype(
    problem: SingleObjectiveTestProblem[SearchSpaceType],
    input_dim: int,
    num_obs: int,
) -> None:
    x = problem.search_space.sample(num_obs)

    assert x.dtype == tf.float64
    tf.debugging.assert_shapes([(x, [num_obs, input_dim])])

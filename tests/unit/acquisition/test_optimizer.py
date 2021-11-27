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

from typing import Callable, TypeVar

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, quadratic, random_seed
from trieste.acquisition import AcquisitionFunction
from trieste.acquisition.optimizer import (
    AcquisitionOptimizer,
    FailedOptimizationError,
    automatic_optimizer_selector,
    batchify,
    generate_continuous_optimizer,
    generate_random_search_optimizer,
    get_bounds_of_box_relaxation_around_point,
    optimize_discrete,
)
from trieste.objectives import (
    ACKLEY_5_MINIMIZER,
    ACKLEY_5_MINIMUM,
    ACKLEY_5_SEARCH_SPACE,
    HARTMANN_3_MINIMIZER,
    HARTMANN_3_MINIMUM,
    HARTMANN_3_SEARCH_SPACE,
    HARTMANN_6_MINIMIZER,
    HARTMANN_6_MINIMUM,
    HARTMANN_6_SEARCH_SPACE,
    ackley_5,
    hartmann_3,
    hartmann_6,
)
from trieste.space import Box, DiscreteSearchSpace, SearchSpace, TaggedProductSearchSpace
from trieste.types import TensorType


def _quadratic_sum(shift: list[float]) -> AcquisitionFunction:
    return lambda x: tf.reduce_sum(0.5 - quadratic(x - shift), axis=-2)


def _delta_function(power: float) -> AcquisitionFunction:
    return lambda x: tf.reduce_sum((1 / (x ** power)), -1)


def test_generate_random_search_optimizer_raises_with_invalid_sample_size() -> None:
    with pytest.raises(ValueError):
        generate_random_search_optimizer(num_samples=-5)


SP = TypeVar("SP", bound=SearchSpace)


@random_seed
@pytest.mark.parametrize(
    "search_space, shift, expected_maximizer, optimizers",
    [
        (
            DiscreteSearchSpace(tf.constant([[-0.5], [0.2], [1.2], [1.7]])),
            [1.0],
            [[1.2]],
            [optimize_discrete, generate_random_search_optimizer()],
        ),  # 1D
        (
            DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [-0.2, 0.3], [0.2, -0.3], [1.2, 0.4]])),
            [0.3, -0.4],
            [[0.2, -0.3]],
            [optimize_discrete, generate_random_search_optimizer()],
        ),  # 2D
        (
            Box([-1], [2]),
            [1.0],
            [[1.0]],
            [generate_random_search_optimizer(10_000)],
        ),  # 1D
        (
            Box(tf.constant([-1], dtype=tf.float64), tf.constant([2], dtype=tf.float64)),
            [1.0],
            [[1.0]],
            [generate_random_search_optimizer(10_000)],
        ),  # 1D with tf bounds
        (
            Box([-1, -2], [1.5, 2.5]),
            [0.3, -0.4],
            [[0.3, -0.4]],
            [generate_random_search_optimizer(10_000)],
        ),  # 2D
        (
            Box([-1, -2], [1.5, 2.5]),
            [1.0, 4],
            [[1.0, 2.5]],
            [generate_random_search_optimizer(10_000)],
        ),  # 2D with maximum outside search space
    ],
)
def test_discrete_and_random_optimizer_on_quadratic(
    search_space: SP,
    shift: list[float],
    expected_maximizer: list[list[float]],
    optimizers: list[AcquisitionOptimizer[SP]],
) -> None:
    for optimizer in optimizers:
        maximizer = optimizer(search_space, _quadratic_sum(shift))
        if optimizer is optimize_discrete:
            npt.assert_allclose(maximizer, expected_maximizer, rtol=1e-4)
        else:
            npt.assert_allclose(maximizer, expected_maximizer, rtol=1e-1)


@random_seed
@pytest.mark.parametrize(
    "neg_function, expected_maximizer, expected_minimum, search_space",
    [
        (ackley_5, ACKLEY_5_MINIMIZER, ACKLEY_5_MINIMUM, ACKLEY_5_SEARCH_SPACE),
        (hartmann_3, HARTMANN_3_MINIMIZER, HARTMANN_3_MINIMUM, HARTMANN_3_SEARCH_SPACE),
        (hartmann_6, HARTMANN_6_MINIMIZER, HARTMANN_6_MINIMUM, HARTMANN_6_SEARCH_SPACE),
    ],
)
def test_random_search_optimizer_on_toy_problems(
    neg_function: Callable[[TensorType], TensorType],
    expected_maximizer: TensorType,
    expected_minimum: TensorType,
    search_space: SearchSpace,
) -> None:
    def target_function(x: TensorType) -> TensorType:
        return -1 * neg_function(tf.squeeze(x, 1))

    optimizer: AcquisitionOptimizer[SearchSpace] = generate_random_search_optimizer(1_000_000)
    maximizer = optimizer(search_space, target_function)
    npt.assert_allclose(maximizer, expected_maximizer, rtol=2e-1)


def test_generate_continuous_optimizer_raises_with_invalid_init_params() -> None:
    with pytest.raises(ValueError):
        generate_continuous_optimizer(num_initial_samples=-5)
    with pytest.raises(ValueError):
        generate_continuous_optimizer(num_optimization_runs=-5)
    with pytest.raises(ValueError):
        generate_continuous_optimizer(num_optimization_runs=5, num_initial_samples=4)
    with pytest.raises(ValueError):
        generate_continuous_optimizer(num_recovery_runs=-5)


@pytest.mark.parametrize("num_optimization_runs", [1, 10])
@pytest.mark.parametrize("num_recovery_runs", [1, 10])
def test_optimize_continuous_raises_for_impossible_optimization(
    num_optimization_runs: int, num_recovery_runs: int
) -> None:
    search_space = Box([-1, -1], [1, 2])
    optimizer = generate_continuous_optimizer(
        num_optimization_runs=num_optimization_runs, num_recovery_runs=num_recovery_runs
    )
    with pytest.raises(FailedOptimizationError) as e:
        optimizer(search_space, _delta_function(10))
    assert (
        str(e.value)
        == f"""
                    Acquisition function optimization failed,
                    even after {num_recovery_runs + num_optimization_runs} restarts.
                    """
    )


@pytest.mark.parametrize("num_optimization_runs", [1, 10])
@pytest.mark.parametrize("num_initial_samples", [1000, 5000])
def test_optimize_continuous_uses_correctly_uses_init_params(
    num_optimization_runs: int, num_initial_samples: int
) -> None:

    querying_initial_sample = True

    def _target_fn(x: TensorType) -> TensorType:
        nonlocal querying_initial_sample

        if querying_initial_sample:  # check size of initial sample
            assert tf.shape(x)[0] == num_initial_samples
        else:  # check evaluations are in parallel with correct batch size
            assert tf.shape(x)[0] == num_optimization_runs

        querying_initial_sample = False
        return _quadratic_sum([0.5, 0.5])(x)

    optimizer = generate_continuous_optimizer(num_initial_samples, num_optimization_runs)
    optimizer(Box([-1], [1]), _target_fn)


@pytest.mark.parametrize("failed_first_optimziation", [True, False])
@pytest.mark.parametrize("num_recovery_runs", [0, 2, 10])
def test_optimize_continuous_recovery_runs(
    failed_first_optimziation: bool, num_recovery_runs: int
) -> None:

    currently_failing = failed_first_optimziation
    num_batch_evals = 0

    def _target_fn(x: TensorType) -> TensorType:
        nonlocal currently_failing
        nonlocal num_batch_evals

        if (
            tf.shape(x)[0] > 1
        ):  # count when batch eval (i.e. random init or when doing recovery runs)
            num_batch_evals += 1

        if (
            num_batch_evals > 2
        ):  # after random init, the next batch eval will be start of recovery run
            assert (
                tf.shape(x)[0] == num_recovery_runs
            )  # check that we do correct number of recovery runs
            currently_failing = False

        if currently_failing:  # return function that is impossible to optimize
            return _delta_function(10)
        else:
            return _quadratic_sum([0.5, 0.5])  # return function that is easy to optimize

        optimizer = generate_continuous_optimizer(
            num_optimization_runs=1, num_recovery_runs=num_recovery_runs
        )
        if failed_first_optimziation and (num_recovery_runs == 0):
            with pytest.raises(FailedOptimizationError):
                optimizer(Box([-1], [1]), _target_fn)
        else:
            optimizer(Box([-1], [1]), _target_fn)


@random_seed
@pytest.mark.parametrize(
    "search_space, shift, expected_maximizer",
    [
        (
            Box([-1], [2]),
            [1.0],
            [[1.0]],
        ),  # 1D
        (
            Box([-1, -2], [1.5, 2.5]),
            [0.3, -0.4],
            [[0.3, -0.4]],
        ),  # 2D
        (
            Box([-1, -2], [1.5, 2.5]),
            [1.0, 4],
            [[1.0, 2.5]],
        ),  # 2D with maximum outside search space
        (
            Box([-1, -2, 1], [1.5, 2.5, 1.5]),
            [0.3, -0.4, 0.5],
            [[0.3, -0.4, 1.0]],
        ),  # 3D
        (
            TaggedProductSearchSpace([Box([-1, -2], [1.5, 2.5])]),
            [0.3, -0.4],
            [[0.3, -0.4]],
        ),  # Tagged space of just 2D Box
        (
            TaggedProductSearchSpace(
                [
                    DiscreteSearchSpace(
                        tf.constant([[0.4, -2.0], [0.3, -0.4], [0.0, 2.5]], dtype=tf.float64)
                    )
                ]
            ),
            [0.3, -0.4],
            [[0.3, -0.4]],
        ),  # Tagged space of just 2D discrete
        (
            TaggedProductSearchSpace(
                [
                    Box([-1], [1.5]),
                    DiscreteSearchSpace(tf.constant([[-2.0], [-0.4], [2.5]], dtype=tf.float64)),
                ]
            ),
            [0.3, -0.4],
            [[0.3, -0.4]],
        ),  # Tagged space of 1D Box, 1D discrete
        (
            TaggedProductSearchSpace(
                [
                    Box([-1, -2], [1.5, 2.5]),
                    DiscreteSearchSpace(tf.constant([[1.0], [1.25], [1.5]], dtype=tf.float64)),
                ]
            ),
            [0.3, -0.4, 0.5],
            [[0.3, -0.4, 1.0]],
        ),  # Tagged space of 2D Box, 1D discrete
        (
            TaggedProductSearchSpace(
                [
                    Box([-1], [1.5]),
                    DiscreteSearchSpace(
                        tf.constant([[-0.4, 1.0], [0.0, 1.25], [1.0, 1.5]], dtype=tf.float64)
                    ),
                ]
            ),
            [0.3, -0.4, 0.5],
            [[0.3, -0.4, 1.0]],
        ),  # Tagged space of 1D Box, 2D discrete
        (
            TaggedProductSearchSpace(
                [
                    Box([-1], [1.5]),
                    DiscreteSearchSpace(tf.constant([[-0.4], [0.0], [1.0]], dtype=tf.float64)),
                    DiscreteSearchSpace(tf.constant([[1.0], [1.25], [1.5]], dtype=tf.float64)),
                ]
            ),
            [0.3, -0.4, 0.5],
            [[0.3, -0.4, 1.0]],
        ),  # Tagged space of 1D Box, 1D discrete, 1D discrete
    ],
)
@pytest.mark.parametrize(
    "optimizer",
    [
        generate_continuous_optimizer(num_optimization_runs=3),
        generate_continuous_optimizer(num_optimization_runs=3, num_recovery_runs=0),
        generate_continuous_optimizer(num_optimization_runs=1, num_initial_samples=5),
    ],
)
def test_continuous_optimizer_on_quadratic(
    search_space: Box,
    shift: list[float],
    expected_maximizer: list[list[float]],
    optimizer: AcquisitionOptimizer[Box],
) -> None:
    maximizer = optimizer(search_space, _quadratic_sum(shift))
    npt.assert_allclose(maximizer, expected_maximizer, rtol=1e-3)


@random_seed
@pytest.mark.parametrize(
    "neg_function, expected_maximizer, expected_minimum, search_space",
    [
        (ackley_5, ACKLEY_5_MINIMIZER, ACKLEY_5_MINIMUM, ACKLEY_5_SEARCH_SPACE),
        (hartmann_3, HARTMANN_3_MINIMIZER, HARTMANN_3_MINIMUM, HARTMANN_3_SEARCH_SPACE),
        (hartmann_6, HARTMANN_6_MINIMIZER, HARTMANN_6_MINIMUM, HARTMANN_6_SEARCH_SPACE),
    ],
)
def test_continuous_optimizer_on_toy_problems(
    neg_function: Callable[[TensorType], TensorType],
    expected_maximizer: TensorType,
    expected_minimum: TensorType,
    search_space: Box,
) -> None:
    def target_function(x: TensorType) -> TensorType:
        return -1 * neg_function(tf.squeeze(x, 1))

    optimizer = generate_continuous_optimizer(num_initial_samples=1_000, num_optimization_runs=10)
    maximizer = optimizer(search_space, target_function)
    npt.assert_allclose(maximizer, expected_maximizer, rtol=1e-1)


@pytest.mark.parametrize(
    "search_space, point",
    [
        (Box([-1], [2]), tf.constant([[0.0]], dtype=tf.float64)),
        (Box([-1, -2], [1.5, 2.5]), tf.constant([[0.0, 0.0]])),
        (DiscreteSearchSpace(tf.constant([[-0.5], [0.2], [1.2], [1.7]])), tf.constant([[0.2]])),
    ],
)
def test_get_bounds_of_box_relaxation_around_point_raises_for_not_product_spaces(
    search_space: DiscreteSearchSpace | Box,
    point: TensorType,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        get_bounds_of_box_relaxation_around_point(search_space, point)  # type: ignore


@pytest.mark.parametrize(
    "search_space, point, lower, upper",
    [
        (
            TaggedProductSearchSpace(
                [
                    Box([-1.0], [1.5]),
                    DiscreteSearchSpace(tf.constant([[-2.0], [-0.4], [2.5]], dtype=tf.float64)),
                ]
            ),
            tf.constant([[0.0, -0.4]], dtype=tf.float64),
            [-1, -0.4],
            [1.5, -0.4],
        ),  # Tagged space of 1D Box and 1D discrete
        (
            TaggedProductSearchSpace(
                [
                    Box([-1.0, -2.0], [1.5, 2.5]),
                    DiscreteSearchSpace(tf.constant([[1.0], [1.25], [1.5]], dtype=tf.float64)),
                ]
            ),
            tf.constant([[0.0, 1.0, 1.25]], dtype=tf.float64),
            [-1.0, -2.0, 1.25],
            [1.5, 2.5, 1.25],
        ),  # Tagged space of 2D Box and 1D discrete
        (
            TaggedProductSearchSpace(
                [
                    Box([-1.0], [1.5]),
                    DiscreteSearchSpace(
                        tf.constant([[-0.4, 1.0], [0.0, 1.25], [1.0, 1.5]], dtype=tf.float64)
                    ),
                ]
            ),
            tf.constant([[-1.0, 1.0, 1.5]], dtype=tf.float64),
            [-1.0, 1.0, 1.5],
            [1.5, 1.0, 1.5],
        ),  # Tagged space of 1D Box and 2D discrete
        (
            TaggedProductSearchSpace(
                [
                    Box([-1.0], [1.5]),
                    DiscreteSearchSpace(tf.constant([[-0.4], [0.0], [1.0]], dtype=tf.float64)),
                    DiscreteSearchSpace(tf.constant([[1.0], [1.25], [1.5]], dtype=tf.float64)),
                ]
            ),
            tf.constant([[-1.0, 1.0, 1.5]], dtype=tf.float64),
            [-1.0, 1.0, 1.5],
            [1.5, 1.0, 1.5],
        ),  # Tagged space of 1D Box, 1D discrete and 1D discrete
    ],
)
def test_get_bounds_of_box_relaxation_around_point(
    search_space: TaggedProductSearchSpace,
    point: TensorType,
    lower: TensorType,
    upper: TensorType,
) -> None:
    bounds = get_bounds_of_box_relaxation_around_point(search_space, point)
    npt.assert_array_equal(bounds.lb, lower)
    npt.assert_array_equal(bounds.ub, upper)


def test_optimize_batch_raises_with_invalid_batch_size() -> None:
    batch_size_one_optimizer = generate_continuous_optimizer()
    with pytest.raises(ValueError):
        batchify(batch_size_one_optimizer, -5)


@random_seed
@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "search_space, acquisition, maximizer",
    [
        (Box([-1], [1]), _quadratic_sum([0.5]), ([[0.5]])),
        (Box([-1, -1, -1], [1, 1, 1]), _quadratic_sum([0.5, -0.5, 0.2]), ([[0.5, -0.5, 0.2]])),
    ],
)
def test_optimize_batch(
    search_space: Box, acquisition: AcquisitionFunction, maximizer: TensorType, batch_size: int
) -> None:
    batch_size_one_optimizer = generate_continuous_optimizer()
    batch_optimizer = batchify(batch_size_one_optimizer, batch_size)
    points = batch_optimizer(search_space, acquisition)
    assert points.shape == [batch_size] + search_space.lower.shape
    for point in points:
        npt.assert_allclose(tf.expand_dims(point, 0), maximizer, rtol=2e-4)


@random_seed
@pytest.mark.parametrize(
    "search_space, acquisition, maximizer",
    [
        (
            DiscreteSearchSpace(tf.constant([[-0.5], [0.2], [1.2], [1.7]])),
            _quadratic_sum([1.0]),
            [[1.2]],
        ),
        (Box([0], [1]), _quadratic_sum([0.5]), ([[0.5]])),
        (Box([-1, -1, -1], [1, 1, 1]), _quadratic_sum([0.5, -0.5, 0.2]), ([[0.5, -0.5, 0.2]])),
        (
            TaggedProductSearchSpace(
                [
                    Box([-1, -1], [1, 1]),
                    DiscreteSearchSpace(tf.constant([[-0.2], [0.0], [0.2]], dtype=tf.float64)),
                ]
            ),
            _quadratic_sum([0.5, -0.5, 0.2]),
            ([[0.5, -0.5, 0.2]]),
        ),
    ],
)
def test_automatic_optimizer_selector(
    search_space: Box,
    acquisition: AcquisitionFunction,
    maximizer: TensorType,
) -> None:
    optimizer = automatic_optimizer_selector
    point = optimizer(search_space, acquisition)
    npt.assert_allclose(point, maximizer, rtol=2e-4)

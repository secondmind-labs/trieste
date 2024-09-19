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

import unittest
import unittest.mock
from math import ceil
from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar, Union
from unittest.mock import MagicMock

import numpy.testing as npt
import pytest
import scipy.optimize as spo
import tensorflow as tf

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, quadratic, random_seed
from trieste.acquisition import AcquisitionFunction
from trieste.acquisition.optimizer import (
    AcquisitionOptimizer,
    FailedOptimizationError,
    automatic_optimizer_selector,
    batchify_joint,
    batchify_vectorize,
    generate_continuous_optimizer,
    generate_initial_points,
    generate_random_search_optimizer,
    get_bounds_of_box_relaxation_around_point,
    get_bounds_of_optimization,
    optimize_discrete,
    sample_from_space,
)
from trieste.acquisition.utils import split_acquisition_function_calls
from trieste.logging import tensorboard_writer
from trieste.objectives import Ackley5, Branin, Hartmann3, Hartmann6, ScaledBranin, SimpleQuadratic
from trieste.space import (
    Box,
    DiscreteSearchSpace,
    LinearConstraint,
    SearchSpace,
    TaggedMultiSearchSpace,
    TaggedProductSearchSpace,
)
from trieste.types import TensorType


def _quadratic_sum(shift: list[float]) -> AcquisitionFunction:
    return lambda x: tf.reduce_sum(0.5 - quadratic(x - shift), axis=-2)


def _delta_function(power: float) -> AcquisitionFunction:
    return lambda x: tf.reduce_sum((1 / (x**power)), -1)


def test_generate_random_search_optimizer_raises_with_invalid_sample_size() -> None:
    with pytest.raises(ValueError):
        generate_random_search_optimizer(num_samples=-5)


@pytest.mark.parametrize("batch_size", [0, -2])
def test_optimize_discrete_raises_with_invalid_vectorized_batch_size(batch_size: int) -> None:
    search_space = DiscreteSearchSpace(tf.constant([[-0.5], [0.2], [1.2], [1.7]]))
    acq_fn = _quadratic_sum([1.0])
    with pytest.raises(ValueError):
        optimize_discrete(search_space, (acq_fn, batch_size))


@pytest.mark.parametrize("batch_size", [0, -2])
def test_random_optimizer_raises_with_invalid_vectorized_batch_size(batch_size: int) -> None:
    search_space = Box([-1], [2])
    acq_fn = _quadratic_sum([1.0])
    with pytest.raises(ValueError):
        generate_random_search_optimizer()(search_space, (acq_fn, batch_size))


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
@pytest.mark.parametrize("split_acquisition_function", [False, True])
def test_discrete_and_random_optimizer_on_quadratic(
    search_space: SP,
    shift: list[float],
    expected_maximizer: list[list[float]],
    optimizers: list[AcquisitionOptimizer[SP]],
    split_acquisition_function: bool,
) -> None:
    for optimizer in optimizers:
        if split_acquisition_function:
            optimizer = split_acquisition_function_calls(optimizer, 97)
        maximizer = optimizer(search_space, _quadratic_sum(shift))
        if optimizer is optimize_discrete:
            npt.assert_allclose(maximizer, expected_maximizer, rtol=1e-4)
        else:
            npt.assert_allclose(maximizer, expected_maximizer, rtol=1e-1)


@random_seed
@pytest.mark.parametrize(
    "neg_function, expected_maximizer, search_space",
    [
        (Ackley5.objective, Ackley5.minimizers, Ackley5.search_space),
        (Hartmann3.objective, Hartmann3.minimizers, Hartmann3.search_space),
        (Hartmann6.objective, Hartmann6.minimizers, Hartmann6.search_space),
    ],
)
def test_random_search_optimizer_on_toy_problems(
    neg_function: Callable[[TensorType], TensorType],
    expected_maximizer: TensorType,
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


@pytest.mark.parametrize("batch_size", [0, -2])
def test_optimize_continuous_raises_with_invalid_vectorized_batch_size(batch_size: int) -> None:
    search_space = Box([-1], [2])
    acq_fn = _quadratic_sum([1.0])
    with pytest.raises(ValueError):
        generate_continuous_optimizer()(search_space, (acq_fn, batch_size))


def test_optimize_continuous_raises_with_mismatch_multi_search_space() -> None:
    space_A = Box([-1], [2])
    space_B = Box([3], [4])
    multi_space = TaggedMultiSearchSpace(spaces=[space_A, space_B])
    acq_fn = _quadratic_sum([1.0])
    with pytest.raises(
        TF_DEBUGGING_ERROR_TYPES, match="The vectorization of the target function 1 must be "
    ):
        generate_continuous_optimizer()(multi_space, acq_fn)


@pytest.mark.parametrize("points_per_box", [1, 3])
def test_optimize_continuous_finds_points_in_multi_search_space_boxes(points_per_box: int) -> None:
    # Test with non-overlapping grid of 2D boxes. Optimize them as a batch and check that each
    # point is only in the corresponding box (with potentially multiple points per box).
    boxes = [Box([x, y], [x + 0.7, y + 0.7]) for x in range(-2, 2) for y in range(-2, 2)]
    multi_space = TaggedMultiSearchSpace(spaces=boxes)
    batch_size = len(boxes) * points_per_box

    def target_function(x: TensorType) -> TensorType:  # [N, V, D] -> [N, V]
        individual_func = [_quadratic_sum([1.0])(x[:, i : i + 1, :]) for i in range(batch_size)]
        return tf.concat(individual_func, axis=-1)  # vectorize by repeating same function

    optimizer = generate_continuous_optimizer()
    optimizer = batchify_vectorize(optimizer, batch_size)
    max_points = optimizer(multi_space, target_function)

    # Ensure order of points is the same as the order of boxes and that each point is only in the
    # corresponding box.
    for i, point in enumerate(max_points):
        for j, box in enumerate(boxes):
            if i % len(boxes) == j:
                assert point in box
            else:
                assert point not in box


@pytest.mark.parametrize("num_optimization_runs", [1, 10])
@pytest.mark.parametrize("num_initial_samples", [1000, 5000])
def test_optimize_continuous_correctly_uses_init_params(
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


@unittest.mock.patch("trieste.logging.tf.summary.text")
@unittest.mock.patch("trieste.logging.tf.summary.scalar")
@pytest.mark.parametrize("failed_first_optimization", [True, False])
@pytest.mark.parametrize("num_recovery_runs", [0, 2, 10])
def test_optimize_continuous_recovery_runs(
    mocked_summary_scalar: unittest.mock.MagicMock,
    mocked_summary_text: unittest.mock.MagicMock,
    failed_first_optimization: bool,
    num_recovery_runs: int,
) -> None:
    currently_failing = failed_first_optimization
    num_batch_evals = 0
    num_evals = 0

    def _target_fn(x: TensorType) -> TensorType:
        nonlocal currently_failing
        nonlocal num_batch_evals
        nonlocal num_evals

        num_evals += 1

        if (
            tf.shape(x)[0] > 1
        ):  # count when batch eval (i.e. random init or when doing recovery runs)
            num_batch_evals += 1

        if (
            num_batch_evals > 1
        ):  # after random init, the next batch eval will be start of recovery run
            assert tf.shape(x)[0] in (
                num_recovery_runs,
                1,  # when generating improvement_on_initial_samples log
            )  # check that we do correct number of recovery runs
            currently_failing = False

        if currently_failing:  # use function that is impossible to optimize
            return _delta_function(10)(x)
        else:
            return _quadratic_sum([0.5, 0.5])(x)  # use function that is easy to optimize

    with tensorboard_writer(unittest.mock.MagicMock()):
        optimizer = generate_continuous_optimizer(
            num_optimization_runs=1, num_recovery_runs=num_recovery_runs
        )
        if failed_first_optimization and (num_recovery_runs == 0):
            with pytest.raises(FailedOptimizationError):
                optimizer(Box([-1], [1]), _target_fn)
        else:
            optimizer(Box([-1], [1]), _target_fn)

    # check we also generated the expected tensorboard logs
    scalar_logs = {call[0][0]: call[0][1:] for call in mocked_summary_scalar.call_args_list}
    if failed_first_optimization and (num_recovery_runs == 0):
        assert not scalar_logs
    else:
        assert set(scalar_logs) == {
            "spo_af_evaluations",
            "spo_improvement_on_initial_samples",
        }
        # also evaluated once for the initial points, and again when generating the log
        assert scalar_logs["spo_af_evaluations"][0] == num_evals - 2

    text_logs = {call[0][0]: call[0][1:] for call in mocked_summary_text.call_args_list}
    if failed_first_optimization and (num_recovery_runs > 0):
        assert set(text_logs) == {"spo_recovery_run"}
    else:
        assert not text_logs


def test_optimize_continuous_when_target_raises_exception() -> None:
    num_queries = 0

    def _target_fn(x: TensorType) -> TensorType:
        nonlocal num_queries

        if num_queries > 1:  # after initial sample return inf
            return -1 * Hartmann3.objective(tf.squeeze(x, 1)) / 0.0

        num_queries += 1
        return -1 * Hartmann3.objective(tf.squeeze(x, 1))

    optimizer = generate_continuous_optimizer(optimizer_args={"options": {"maxiter": 10}})
    with pytest.raises(FailedOptimizationError):
        optimizer(Hartmann3.search_space, _target_fn)


def test_continuous_optimizer_returns_raise_on_infeasible_points() -> None:
    def target_function(x: TensorType) -> TensorType:
        return -1 * ScaledBranin.objective(tf.squeeze(x, 1))

    search_space = Box([0.0, 0.0], [1.0, 1.0], [LinearConstraint(A=tf.eye(2), lb=0.5, ub=0.5)])
    optimizer = generate_continuous_optimizer(
        num_initial_samples=1_000, num_optimization_runs=10, optimizer_args=dict(method="l-bfgs-b")
    )
    with pytest.raises(FailedOptimizationError):
        optimizer(search_space, target_function)


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
    "neg_function, expected_maximizer, search_space",
    [
        (Ackley5.objective, Ackley5.minimizers, Ackley5.search_space),
        (Hartmann3.objective, Hartmann3.minimizers, Hartmann3.search_space),
        (Hartmann6.objective, Hartmann6.minimizers, Hartmann6.search_space),
    ],
)
def test_continuous_optimizer_on_toy_problems(
    neg_function: Callable[[TensorType], TensorType],
    expected_maximizer: TensorType,
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


def test_get_bounds_of_optimization_raises_on_incorrect_num_subspaces() -> None:
    search_space = TaggedMultiSearchSpace([Box([-1], [2]), Box([3], [4])])
    points = tf.ones([4, 3, 1], dtype=tf.float64)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES, match="The vectorization of the target function"):
        get_bounds_of_optimization(search_space, points)


@pytest.mark.parametrize(
    "search_space, exp_bounds",
    [
        (Box([-1], [2]), [(-1, 2)] * 12),
        (TaggedProductSearchSpace([Box([-1], [2]), Box([3], [4])]), [([-1, 3], [2, 4])] * 12),
        (TaggedMultiSearchSpace([Box([-1], [2]), Box([3], [4])]), [(-1, 2), (3, 4)] * 6),
        (
            TaggedMultiSearchSpace([TaggedProductSearchSpace([Box([-1], [2]), Box([3], [4])])]),
            [([-1, 3], [2, 4])] * 12,
        ),
        (
            TaggedMultiSearchSpace(
                [
                    TaggedProductSearchSpace([Box([-1], [2]), Box([3], [4])]),
                    TaggedProductSearchSpace([Box([-3], [0]), Box([7], [8])]),
                ]
            ),
            [([-1, 3], [2, 4]), ([-3, 7], [0, 8])] * 6,
        ),
        (
            TaggedProductSearchSpace([Box([-1], [2]), DiscreteSearchSpace([[-0.5], [0.2]])]),
            [  # Expect use of get_bounds_of_box_relaxation_around_point.
                ([-1, 11], [2, 11]),
                ([-1, 13], [2, 13]),
                ([-1, 15], [2, 15]),
                ([-1, 17], [2, 17]),
                ([-1, 21], [2, 21]),
                ([-1, 23], [2, 23]),
                ([-1, 25], [2, 25]),
                ([-1, 27], [2, 27]),
                ([-1, 31], [2, 31]),
                ([-1, 33], [2, 33]),
                ([-1, 35], [2, 35]),
                ([-1, 37], [2, 37]),
            ],
        ),
        (
            TaggedMultiSearchSpace(
                [DiscreteSearchSpace([[-0.5], [0.2]]), DiscreteSearchSpace([[1.0], [-0.2], [3.7]])]
            ),
            [(-0.5, 0.2), (-0.2, 3.7)] * 6,
        ),
        (
            TaggedMultiSearchSpace(
                [TaggedProductSearchSpace([Box([-1], [2]), DiscreteSearchSpace([[-0.5], [0.2]])])]
            ),
            [  # Expect use of get_bounds_of_box_relaxation_around_point.
                ([-1, 11], [2, 11]),
                ([-1, 13], [2, 13]),
                ([-1, 15], [2, 15]),
                ([-1, 17], [2, 17]),
                ([-1, 21], [2, 21]),
                ([-1, 23], [2, 23]),
                ([-1, 25], [2, 25]),
                ([-1, 27], [2, 27]),
                ([-1, 31], [2, 31]),
                ([-1, 33], [2, 33]),
                ([-1, 35], [2, 35]),
                ([-1, 37], [2, 37]),
            ],
        ),
        (
            TaggedMultiSearchSpace(
                [
                    TaggedProductSearchSpace(
                        [Box([-1], [2]), DiscreteSearchSpace([[-0.5], [0.2]])]
                    ),
                    TaggedProductSearchSpace(
                        [Box([-3], [0]), DiscreteSearchSpace([[1.0], [-0.2], [3.7]])]
                    ),
                ]
            ),
            [  # Expect use of get_bounds_of_box_relaxation_around_point.
                ([-1, 11], [2, 11]),
                ([-3, 13], [0, 13]),
                ([-1, 15], [2, 15]),
                ([-3, 17], [0, 17]),
                ([-1, 21], [2, 21]),
                ([-3, 23], [0, 23]),
                ([-1, 25], [2, 25]),
                ([-3, 27], [0, 27]),
                ([-1, 31], [2, 31]),
                ([-3, 33], [0, 33]),
                ([-1, 35], [2, 35]),
                ([-3, 37], [0, 37]),
            ],
        ),
    ],
)
def test_get_bounds_of_optimization(
    search_space: SearchSpace, exp_bounds: list[Tuple[TensorType, TensorType]]
) -> None:
    points = tf.constant(
        [
            [[10, 11], [12, 13], [14, 15], [16, 17]],
            [[20, 21], [22, 23], [24, 25], [26, 27]],
            [[30, 31], [32, 33], [34, 35], [36, 37]],
        ],
        dtype=tf.float64,
    )
    bounds = get_bounds_of_optimization(search_space, points)

    assert len(bounds) == 12
    for exp, b in zip(exp_bounds, bounds):
        npt.assert_array_equal(exp[0], b.lb)
        npt.assert_array_equal(exp[1], b.ub)


def test_batchify_joint_raises_with_invalid_batch_size() -> None:
    batch_size_one_optimizer = generate_continuous_optimizer()
    with pytest.raises(ValueError):
        batchify_joint(batch_size_one_optimizer, -5)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
def test_batchify_joint_raises_with_vectorized_acquisition_function(batch_size: int) -> None:
    batch_size_one_optimizer = generate_continuous_optimizer()
    optimizer = batchify_joint(batch_size_one_optimizer, 5)
    search_space = Box([-1], [1])
    acq_fn = _quadratic_sum([0.5])
    with pytest.raises(ValueError):
        optimizer(search_space, (acq_fn, batch_size))


@random_seed
@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "search_space, acquisition, maximizer",
    [
        (Box([-1], [1]), _quadratic_sum([0.5]), ([[0.5]])),
        (Box([-1, -1, -1], [1, 1, 1]), _quadratic_sum([0.5, -0.5, 0.2]), ([[0.5, -0.5, 0.2]])),
    ],
)
def test_batchify_joint(
    search_space: Box, acquisition: AcquisitionFunction, maximizer: TensorType, batch_size: int
) -> None:
    batch_size_one_optimizer = generate_continuous_optimizer(num_optimization_runs=5)
    batch_optimizer = batchify_joint(batch_size_one_optimizer, batch_size)
    points = batch_optimizer(search_space, acquisition)
    assert points.shape == [batch_size] + search_space.lower.shape
    for point in points:
        npt.assert_allclose(tf.expand_dims(point, 0), maximizer, rtol=2e-4)


def test_batchify_vectorized_raises_with_invalid_batch_size() -> None:
    batch_size_one_optimizer = generate_continuous_optimizer()
    with pytest.raises(ValueError):
        batchify_vectorize(batch_size_one_optimizer, -5)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
def test_batchify_vectorize_raises_with_vectorized_acquisition_function(batch_size: int) -> None:
    batch_size_one_optimizer = generate_continuous_optimizer()
    optimizer = batchify_vectorize(batch_size_one_optimizer, 5)
    search_space = Box([-1], [1])
    acq_fn = _quadratic_sum([0.5])
    with pytest.raises(ValueError):
        optimizer(search_space, (acq_fn, batch_size))


@random_seed
@pytest.mark.parametrize(
    "optimizer", [generate_random_search_optimizer(10_000), generate_continuous_optimizer()]
)
@pytest.mark.parametrize("split_acquisition_function", [False, True])
def test_batchify_vectorized_for_random_and_continuous_optimizers_on_vectorized_quadratic(
    optimizer: AcquisitionOptimizer[Box],
    split_acquisition_function: bool,
) -> None:
    search_space = Box([-1, -2], [1.5, 2.5])
    shifts = [[0.3, -0.4], [1.0, 4]]
    expected_maximizers = [[0.3, -0.4], [1.0, 2.5]]
    vectorized_batch_size = 2

    def vectorized_target(x: TensorType) -> TensorType:  # [N, V, D] -> [N,V]
        individual_func = [
            _quadratic_sum(shifts[i])(x[:, i : i + 1, :]) for i in range(vectorized_batch_size)
        ]
        return tf.concat(individual_func, axis=-1)

    batched_optimizer = batchify_vectorize(optimizer, batch_size=vectorized_batch_size)
    if split_acquisition_function:
        batched_optimizer = split_acquisition_function_calls(batched_optimizer, 1000)
    maximizers = batched_optimizer(search_space, vectorized_target)
    npt.assert_allclose(maximizers, expected_maximizers, rtol=1e-1)


def test_batchify_vectorized_for_discrete_optimizer_on_vectorized_quadratic() -> None:
    search_space = DiscreteSearchSpace(
        tf.constant([[0.3, -0.4], [1.0, 2.5], [0.2, 0.5], [0.5, 2.0], [2.0, 0.1]])
    )
    shifts = [[0.3, -0.4], [1.0, 4]]
    expected_maximizers = [[0.3, -0.4], [1.0, 2.5]]
    vectorized_batch_size = 2

    def vectorized_target(x: TensorType) -> TensorType:  # [N, V, D] -> [N,V]
        individual_func = [
            _quadratic_sum(shifts[i])(x[:, i : i + 1, :]) for i in range(vectorized_batch_size)
        ]
        return tf.concat(individual_func, axis=-1)

    batched_optimizer = batchify_vectorize(optimize_discrete, batch_size=vectorized_batch_size)
    maximizers = batched_optimizer(search_space, vectorized_target)
    npt.assert_allclose(maximizers, expected_maximizers, rtol=1e-1)


@random_seed
@pytest.mark.parametrize("vectorization", [1, 5])
@pytest.mark.parametrize(
    "neg_function, expected_maximizer, search_space",
    [
        (Ackley5.objective, Ackley5.minimizers, Ackley5.search_space),
        (Hartmann3.objective, Hartmann3.minimizers, Hartmann3.search_space),
        (Hartmann6.objective, Hartmann6.minimizers, Hartmann6.search_space),
    ],
)
def test_batchify_vectorized_for_continuous_optimizer_on_duplicated_toy_problems(
    vectorization: int,
    neg_function: Callable[[TensorType], TensorType],
    expected_maximizer: TensorType,
    search_space: Box,
) -> None:
    def target_function(x: TensorType) -> TensorType:  # [N,V,D] -> [N, V]
        individual_func = [-1 * neg_function(x[:, i, :]) for i in range(vectorization)]
        return tf.concat(individual_func, axis=-1)  # vectorize by repeating same function

    optimizer = batchify_vectorize(
        generate_continuous_optimizer(num_initial_samples=20_000, num_optimization_runs=10),
        batch_size=vectorization,
    )
    maximizer = optimizer(search_space, target_function)
    npt.assert_allclose(maximizer, tf.tile(expected_maximizer, [vectorization, 1]), rtol=1e-1)


@random_seed
def test_batchify_vectorized_for_continuous_optimizer_on_vectorized_toy_problems() -> None:
    search_space = Branin.search_space
    functions = [Branin.objective, ScaledBranin.objective, SimpleQuadratic.objective]
    expected_maximimums = [-Branin.minimum, -ScaledBranin.minimum, -SimpleQuadratic.minimum]
    vectorized_batch_size = 3

    def target_function(x: TensorType) -> TensorType:  # [N,V,D] -> [N, V]
        individual_func = [-1 * functions[i](x[:, i, :]) for i in range(vectorized_batch_size)]
        return tf.concat(individual_func, axis=-1)  # vectorize by concatenating three functions

    optimizer = batchify_vectorize(
        generate_continuous_optimizer(num_initial_samples=1_000, num_optimization_runs=10),
        batch_size=vectorized_batch_size,
    )
    maximizer = optimizer(search_space, target_function)
    npt.assert_allclose(
        target_function(maximizer[None, :, :]), tf.transpose(expected_maximimums), rtol=1e-5
    )


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


def test_split_acquisition_function_calls_raises_with_invalid_batch_size() -> None:
    optimizer = generate_continuous_optimizer()
    with pytest.raises(ValueError):
        split_acquisition_function_calls(optimizer, -5)


@pytest.mark.parametrize("batch_size", [1, 2, 9, 10, 11, 19, 20, 21, 100])
def test_split_acquisition_function(batch_size: int) -> None:
    acquisition_function = MagicMock()
    acquisition_function.side_effect = lambda x: x

    def dummy_optimizer(
        search_space: SearchSpace,
        f: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    ) -> TensorType:
        af, n = f if isinstance(f, tuple) else (f, 1)
        return af(tf.linspace([0, 0], [1, 1], n))

    batched_optimizer = split_acquisition_function_calls(dummy_optimizer, batch_size)
    value = batched_optimizer(Box([0, 0], [1, 1]), (acquisition_function, 10))
    npt.assert_array_equal(value, tf.linspace([0, 0], [1, 1], 10))
    # since each row has two elements, actual batch size will always be even
    expected_batch_size = 2 * ceil(batch_size / 2)
    assert all(
        tf.size(call[0][0]) <= expected_batch_size for call in acquisition_function.call_args_list
    )
    assert acquisition_function.call_count == ceil(20 / expected_batch_size)


@unittest.mock.patch("scipy.optimize.minimize")
@pytest.mark.parametrize(
    "search_space, optimizer_args, expected_method, expected_constraints",
    [
        (Branin.search_space, None, "l-bfgs-b", []),
        (Branin.search_space, dict(method="trust-constr"), "trust-constr", []),
        (Branin.search_space, dict(constraints="dummy"), "l-bfgs-b", "dummy"),
        (
            Branin.search_space,
            dict(method="trust-constr", constraints="dummy"),
            "trust-constr",
            "dummy",
        ),
        (
            Box([0, 0], [1, 1], [LinearConstraint(A=tf.eye(2), lb=0, ub=1)]),
            None,
            "trust-constr",
            [LinearConstraint(A=tf.eye(2), lb=0, ub=1)],
        ),
    ],
)
def test_optimizer_scipy_method_select(
    mocked_minimize: MagicMock,
    search_space: Box,
    optimizer_args: Optional[dict[str, Any]],
    expected_method: str,
    expected_constraints: Optional[str],
) -> None:
    def target_function(x: TensorType) -> TensorType:
        return -1 * Branin.objective(tf.squeeze(x, 1))

    def side_effect(*args: Any, **kwargs: Any) -> spo.OptimizeResult:
        return spo.OptimizeResult(fun=0.0, nfev=0, x=Branin.minimizers[0].numpy(), success=True)

    mocked_minimize.side_effect = side_effect
    optimizer = generate_continuous_optimizer(
        num_initial_samples=2, num_optimization_runs=2, optimizer_args=optimizer_args
    )
    optimizer(search_space, target_function)

    received_method = mocked_minimize.call_args[1]["method"]
    assert received_method == expected_method

    if "constraints" in mocked_minimize.call_args[1]:
        received_constraints = mocked_minimize.call_args[1]["constraints"]
    elif search_space.has_constraints:
        received_constraints = search_space.constraints
    else:
        received_constraints = None
    assert received_constraints == expected_constraints


@pytest.mark.parametrize("num_initial_points", [0, 1, 2, 3, 4])
def test_generate_initial_points(num_initial_points: int) -> None:
    def sampler(space: SearchSpace) -> Iterable[TensorType]:
        assert space == Box([-1], [2])
        yield tf.range(-1, 2, 0.1)[:, None]

    best_four_samples = tf.constant([1.0, 0.9, 1.1, 0.8])
    points = generate_initial_points(
        num_initial_points, sampler, Box([-1], [2]), _quadratic_sum([1.0])
    )
    assert points.shape == [num_initial_points, 1, 1]
    npt.assert_allclose(points, best_four_samples[:num_initial_points, None, None], atol=1e-6)


@pytest.mark.parametrize("num_initial_points", [0, 1, 2, 3, 6, 10])
def test_generate_initial_points_batched_sampler(num_initial_points: int) -> None:
    def sampler(space: SearchSpace) -> Iterable[TensorType]:
        assert space == Box([-1], [2])
        yield tf.constant([[0.8], [0.9]])
        yield tf.constant([[1.0], [1.1]])
        yield tf.constant([[1.2], [1.3]])

    best_samples = tf.constant([1.0, 0.9, 1.1, 0.8, 1.2, 1.3])
    points = generate_initial_points(
        num_initial_points, sampler, Box([-1], [2]), _quadratic_sum([1.0])
    )
    assert points.shape == [min(num_initial_points, 6), 1, 1]
    npt.assert_allclose(points, best_samples[:num_initial_points, None, None], atol=1e-6)


def test_generate_initial_points_raises_if_empty() -> None:
    def sampler(space: SearchSpace) -> Iterable[TensorType]:
        return []

    with pytest.raises(ValueError):
        generate_initial_points(4, sampler, Box([-1], [2]), _quadratic_sum([1.0]))


@pytest.mark.parametrize("num_initial_points", [0, 1, 2, 10])
@pytest.mark.parametrize("vectorization", [1, 3, 4])
def test_generate_initial_points_vectorized(num_initial_points: int, vectorization: int) -> None:
    search_space = Box([-1, -2], [1.5, 2.5])

    def sampler(space: SearchSpace) -> Iterable[TensorType]:
        assert space == search_space
        yield tf.constant([[0], [0.5], [1.0]])

    def vectorized_target(x: TensorType) -> TensorType:  # [N, V, D] -> [N,V]
        shifts = [[0.0], [0.2], [0.5], [1.0]]
        individual_func = [
            _quadratic_sum(shifts[i])(x[:, i : i + 1, :]) for i in range(vectorization)
        ]
        return tf.concat(individual_func, axis=-1)

    best_samples = tf.constant(
        [[[0.0], [0.0], [0.5], [1.0]], [[0.5], [0.5], [0.0], [0.5]], [[1.0], [1.0], [1.0], [0.0]]]
    )
    points = generate_initial_points(
        num_initial_points, sampler, search_space, vectorized_target, vectorization
    )
    assert points.shape == [min(num_initial_points, 3), vectorization, 1]
    npt.assert_allclose(points, best_samples[:num_initial_points, :vectorization], atol=1e-6)


@pytest.mark.parametrize("num_samples,batch_size", [(1, None), (5, None), (5, 2), (5, 5), (5, 10)])
def test_sample_from_space(num_samples: int, batch_size: Optional[int]) -> None:
    batches = list(sample_from_space(num_samples, batch_size)(Box([0], [1])))
    assert len(batches) == ceil(num_samples / (batch_size or num_samples))
    assert sum(len(batch) for batch in batches) == num_samples
    assert all(0 <= x <= 1 for batch in batches for x in batch)
    assert len(set(float(x) for batch in batches for x in batch)) == num_samples


@pytest.mark.parametrize(
    "space", [Box([0], [1]), TaggedMultiSearchSpace([Box([0], [1]), Box([0], [1])])]
)
def test_sample_from_space_vectorization(space: SearchSpace) -> None:
    batches = list(sample_from_space(10, vectorization=4)(space))
    assert len(batches) == 1
    assert batches[0].shape == [10, 4, 1]
    assert 0 <= tf.reduce_min(batches[0]) <= tf.reduce_max(batches[0]) <= 1
    # check that the vector batches aren't all the same
    assert tf.reduce_any((batches[0] - batches[0][:, 0:1, :]) != 0)


@pytest.mark.parametrize(
    "num_samples,batch_size,vectorization",
    [(0, None, 1), (-5, None, 1), (5, 0, 1), (5, -5, 1), (5, 5, 0), (5, 5, -1)],
)
def test_sample_from_space_raises(
    num_samples: int, batch_size: Optional[int], vectorization: int
) -> None:
    with pytest.raises(ValueError):
        sample_from_space(
            num_samples=num_samples, batch_size=batch_size, vectorization=vectorization
        )


def test_sample_from_space_vectorization_raises_with_invalid_space() -> None:
    # vectorisation of 3 not possible with 2 subspace multisearchspace
    space = TaggedMultiSearchSpace([Box([0], [1]), Box([0], [1])])
    sampler = sample_from_space(10, vectorization=3)
    with pytest.raises(tf.errors.InvalidArgumentError):
        list(sampler(space))


def test_optimize_continuous_raises_for_insufficient_starting_points() -> None:
    search_space = Box([-1], [2])

    def sampler(space: SearchSpace) -> Iterable[TensorType]:
        assert space == search_space
        yield tf.constant([[0.8], [0.9]])

    optimizer = generate_continuous_optimizer(sampler, 3)
    with pytest.raises(ValueError) as e:
        optimizer(search_space, _quadratic_sum([1.0]))
    assert str(e.value) == "Not enough initial points generated (2 for 3 optimization runs)"

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

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, quadratic, random_seed
from tests.util.model import QuadraticMeanAndRBFKernel
from trieste.acquisition import AcquisitionFunction, ExpectedImprovement, MinValueEntropySearch
from trieste.acquisition.optimizer import (
    AcquisitionOptimizer,
    LocalPenalization,
    Penalizer,
    batchify,
    default,
    generate_continuous_optimizer,
    generate_random_search_optimizer,
    hard_local_penalizer,
    optimize_discrete,
    soft_local_penalizer,
)
from trieste.data import Dataset
from trieste.space import Box, DiscreteSearchSpace
from trieste.type import TensorType


def _quadratic_sum(shift: list[float]) -> AcquisitionFunction:
    return lambda x: tf.reduce_sum(0.5 - quadratic(x - shift), axis=-2)


def test_generate_random_search_optimizer_raises_with_invalid_sample_size() -> None:
    with pytest.raises(ValueError):
        generate_random_search_optimizer(num_samples=-5)


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
        (  # 2D
            DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [-0.2, 0.3], [0.2, -0.3], [1.2, 0.4]])),
            [0.3, -0.4],
            [[0.2, -0.3]],
            [optimize_discrete, generate_random_search_optimizer()],
        ),
        (
            Box([-1], [2]),
            [1.0],
            [[1.0]],
            [generate_random_search_optimizer(10_000)],
        ),  # 1D
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
def test_discrete_and_random_optimizer(
    search_space: DiscreteSearchSpace,
    shift: list[float],
    expected_maximizer: list[list[float]],
    optimizers: list[AcquisitionOptimizer],
) -> None:
    for optimizer in optimizers:
        maximizer = optimizer(search_space, _quadratic_sum(shift))
        if optimizer is optimize_discrete:
            npt.assert_allclose(maximizer, expected_maximizer, rtol=1e-4)
        else:
            npt.assert_allclose(maximizer, expected_maximizer, rtol=1e-1)


def test_generate_continuous_optimizer_raises_with_invalid_init_params() -> None:
    with pytest.raises(ValueError):
        generate_continuous_optimizer(num_initial_samples=-5)
    with pytest.raises(ValueError):
        generate_continuous_optimizer(num_restarts=-5)
    with pytest.raises(ValueError):
        generate_continuous_optimizer(num_restarts=5, num_initial_samples=4)


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
    ],
)
@pytest.mark.parametrize(
    "optimizer",
    [
        generate_continuous_optimizer(),
        generate_continuous_optimizer(num_restarts=3),
        generate_continuous_optimizer(sigmoid=True),
        generate_continuous_optimizer(sigmoid=True, num_restarts=3),
        generate_continuous_optimizer(sigmoid=True, num_restarts=1, num_initial_samples=1),
    ],
)
def test_continuous_optimizer(
    search_space: DiscreteSearchSpace,
    shift: list[float],
    expected_maximizer: list[list[float]],
    optimizer: AcquisitionOptimizer,
) -> None:

    maximizer = optimizer(search_space, _quadratic_sum(shift))
    npt.assert_allclose(maximizer, expected_maximizer, rtol=1e-3)


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
    ],
)
def test_default_optimizer(
    search_space: Box,
    acquisition: AcquisitionFunction,
    maximizer: TensorType,
) -> None:
    optimizer = default(1)
    point = optimizer(search_space, acquisition)
    npt.assert_allclose(point, maximizer, rtol=2e-4)


def test_locally_penalized_expected_improvement_builder_raises_for_empty_data() -> None:
    lp = LocalPenalization(default(1), batch_size=1)
    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))
    with pytest.raises(ValueError):
        lp.acquire(data, QuadraticMeanAndRBFKernel())


def test_locally_penalized_expected_improvement_builder_raises_for_invalid_num_samples() -> None:
    with pytest.raises(ValueError):
        LocalPenalization(default(1), batch_size=1, sample_size=-5)


@random_seed
@pytest.mark.parametrize(
    "base_builder",
    [
        ExpectedImprovement(),
        MinValueEntropySearch(Box([0, 0], [1, 1]), grid_size=10000, num_samples=10),
    ],
)
def test_locally_penalized_acquisitions_match_base_acquisition(
    base_builder,
) -> None:
    pass  # todo replace


@random_seed
@pytest.mark.parametrize("penalizer", [soft_local_penalizer, hard_local_penalizer])
@pytest.mark.parametrize(
    "base_builder",
    [ExpectedImprovement(), MinValueEntropySearch(Box([0, 0], [1, 1]), grid_size=5000)],
)
def test_locally_penalized_acquisitions_combine_base_and_penalization_correctly(
    penalizer: Penalizer,
    base_builder,
):
    pass  # todo replace


@pytest.mark.parametrize("penalizer", [soft_local_penalizer, hard_local_penalizer])
@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_lipschitz_penalizers_raises_for_invalid_batch_size(
    at: TensorType,
    penalizer: Penalizer,
) -> None:
    pending_points = tf.zeros([1, 2], dtype=tf.float64)
    best = tf.constant([0], dtype=tf.float64)
    lipshitz_constant = tf.constant([1], dtype=tf.float64)
    lp = penalizer(QuadraticMeanAndRBFKernel(), pending_points, lipshitz_constant, best)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        lp(at)


@pytest.mark.parametrize("penalizer", [soft_local_penalizer, hard_local_penalizer])
@pytest.mark.parametrize("pending_points", [tf.constant([0.0]), tf.constant([[[0.0], [1.0]]])])
def test_lipschitz_penalizers_raises_for_invalid_pending_points_shape(
    pending_points: TensorType,
    penalizer: Penalizer,
) -> None:
    best = tf.constant([0], dtype=tf.float64)
    lipshitz_constant = tf.constant([1], dtype=tf.float64)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        soft_local_penalizer(QuadraticMeanAndRBFKernel(), pending_points, lipshitz_constant, best)

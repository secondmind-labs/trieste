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

from tests.util.misc import quadratic, random_seed
from trieste.acquisition import AcquisitionFunction
from trieste.acquisition.optimizer import optimize_continuous, optimize_discrete, simultaneous_batch
from trieste.space import Box, DiscreteSearchSpace
from trieste.type import TensorType
from trieste.utils.objectives import branin, BRANIN_MINIMIZERS


def _quadratic_sum(shift: list[float]) -> AcquisitionFunction:
    return lambda x: tf.reduce_sum(0.5 - quadratic(x - shift), axis=-2)


@random_seed
@pytest.mark.parametrize(
    "search_space, shift, expected_maximizer",
    [
        (DiscreteSearchSpace(tf.constant([[-0.5], [0.2], [1.2], [1.7]])), [1.0], [[1.2]]),  # 1D
        (  # 2D
            DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [-0.2, 0.3], [0.2, -0.3], [1.2, 0.4]])),
            [0.3, -0.4],
            [[0.2, -0.3]],
        ),
    ],
)
def test_optimize_discrete(
    search_space: DiscreteSearchSpace, shift: list[float], expected_maximizer: list[list[float]],
) -> None:
    maximizer = optimize_discrete(search_space, _quadratic_sum(shift))
    npt.assert_allclose(maximizer, expected_maximizer, rtol=1e-4)


@random_seed
@pytest.mark.parametrize(
    "search_space, shift, expected_maximizer",
    [
        (Box([-1], [2]), [1.0], [[1.0]]),  # 1D
        (Box([-1, -2], [1.5, 2.5]), [0.3, -0.4], [[0.3, -0.4]]),  # 2D
        (Box([-1, -2], [1.5, 2.5]), [1.0, 4], [[1.0, 2.5]]),  # 2D with maximum outside search space
        (Box([-1, -2, 1], [1.5, 2.5, 1.5]), [0.3, -0.4, 0.5], [[0.3, -0.4, 1.0]]),  # 3D
    ],
)
def test_optimize_continuous(
    search_space: Box, shift: list[float], expected_maximizer: list[list[float]],
) -> None:
    maximizer = optimize_continuous(search_space, _quadratic_sum(shift))
    npt.assert_allclose(maximizer, expected_maximizer, rtol=2e-4)


def _branin_sum(x: TensorType) -> TensorType:
    return -tf.reduce_sum(branin(x), axis=-2)


@random_seed
@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
@pytest.mark.parametrize("search_space, acquisition, maximizers", [
    (Box([0], [1]), _quadratic_sum([0.5]), ([[0.5, -0.5]])),
    (Box([0, 0], [1, 1]), _branin_sum, BRANIN_MINIMIZERS),
    (Box([0, 0, 0], [1, 1, 1]), _quadratic_sum([0.5, -0.5, 0.2]), ([[0.5, -0.5, 0.2]])),
])
def test_batchify(
    search_space: Box, acquisition: AcquisitionFunction, maximizers: TensorType, batch_size: int
) -> None:
    batch_optimizer = simultaneous_batch(optimize_continuous, batch_size)
    points = batch_optimizer(search_space, acquisition)

    assert points.shape == [batch_size] + search_space.lower.shape

    for point in points:
        tf.reduce_any(point == maximizers, axis=0)

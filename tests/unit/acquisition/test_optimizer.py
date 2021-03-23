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
from trieste.acquisition.optimizer import optimize_continuous, optimize_discrete
from trieste.space import Box, DiscreteSearchSpace


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
    search_space: DiscreteSearchSpace,
    shift: list[float],
    expected_maximizer: list[list[float]],
) -> None:
    maximizer = optimize_discrete(search_space, lambda x: 0.5 - quadratic(x - shift))
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
    search_space: Box,
    shift: list[float],
    expected_maximizer: list[list[float]],
) -> None:
    maximizer = optimize_continuous(search_space, lambda x: 0.5 - quadratic(x[0, :] - shift))
    npt.assert_allclose(maximizer, expected_maximizer, rtol=2e-4)


@random_seed
@pytest.mark.parametrize(
    "search_space, shifts, batch_size, expected_maximizer",
    [
        (Box([-1], [2]), [1.0, 1.5], 2, [[1.0], [1.5]]),  # 1D space with batch of size 2
        (
            Box([-1, -2], [1.5, 2.5]),
            [[0.3, -0.4]] * 3,
            3,
            [[0.3, -0.4]] * 3,
        ),  # 2D space with batch of size 3
        (
            Box([-1, -2, 1], [1.5, 2.5, 1.5]),
            [[0.3, -0.4, 0.5]] * 5,
            5,
            [[0.3, -0.4, 1.0]] * 5,
        ),  # 3D space with a batch of size 5
    ],
)
def test_optimize_continuous_batch(
    search_space: Box,
    shifts: list[float],
    batch_size: int,
    expected_maximizer: list[list[float]],
) -> None:
    def batch_objective(x):
        obj = 0.5 - quadratic(x[:, 0] - shifts[0])
        for j in range(1, batch_size):
            obj += -quadratic(x[:, j] - shifts[j])
        return obj

    maximizer = optimize_continuous(search_space, batch_objective, batch_size)
    npt.assert_allclose(maximizer, expected_maximizer, rtol=2e-4)

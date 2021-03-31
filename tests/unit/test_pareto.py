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

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, ListN
from trieste.utils.pareto import Pareto, non_dominated


@pytest.mark.parametrize(
    "scores, pareto_set, dominance",
    [
        (
            tf.constant(
                [
                    [0.9575, 0.4218],
                    [0.9649, 0.9157],
                    [0.1576, 0.7922],
                    [0.9706, 0.9595],
                    [0.9572, 0.6557],
                    [0.4854, 0.0357],
                    [0.8003, 0.8491],
                    [0.1419, 0.9340],
                ]
            ),
            tf.constant([[0.1576, 0.7922], [0.4854, 0.0357], [0.1419, 0.934]]),
            tf.constant([1, 5, 0, 7, 1, 0, 2, 0]),
        ),
        (
            tf.constant(
                [
                    [0.9575, 0.4218],
                    [0.9649, 0.9157],
                    [0.1576, 0.7922],
                    [0.9706, 0.9595],
                    [0.9572, 0.6557],
                    [0.4854, 0.0357],
                    [0.4954, 0.0357],
                    [0.8003, 0.8491],
                    [0.1419, 0.9340],
                    [0.1419, 0.9440],
                ]
            ),
            tf.constant([[0.1576, 0.7922], [0.4854, 0.0357], [0.1419, 0.934]]),
            tf.constant([2, 6, 0, 9, 2, 0, 1, 3, 0, 1]),
        ),
        (
            tf.constant(
                [
                    [0.9575, 0.4218],
                    [0.9649, 0.9157],
                    [0.1576, 0.7922],
                    [0.9706, 0.9595],
                    [0.9572, 0.6557],
                    [0.4854, 0.0357],
                    [0.4854, 0.0357],
                    [0.8003, 0.8491],
                    [0.1419, 0.9340],
                    [0.1419, 0.9340],
                ]
            ),
            tf.constant(
                [
                    [0.1576, 0.7922],
                    [0.4854, 0.0357],
                    [0.4854, 0.0357],
                    [0.1419, 0.934],
                    [0.1419, 0.934],
                ]
            ),
            tf.constant([2, 6, 0, 9, 2, 0, 0, 3, 0, 0]),
        ),
    ],
)
def test_dominated_sort(scores: tf.Tensor, pareto_set: tf.Tensor, dominance: tf.Tensor) -> None:
    ret_pareto_set, ret_dominance = non_dominated(scores)
    npt.assert_allclose(ret_pareto_set, pareto_set)
    npt.assert_allclose(ret_dominance, dominance)


def test_pareto_2d_bounds() -> None:
    objectives = tf.constant(
        [
            [0.9575, 0.4218],
            [0.9649, 0.9157],
            [0.1576, 0.7922],
            [0.9706, 0.9595],
            [0.9572, 0.6557],
            [0.4854, 0.0357],
            [0.8003, 0.8491],
            [0.1419, 0.9340],
        ]
    )

    pareto_2d = Pareto(objectives)

    npt.assert_allclose(pareto_2d.bounds.lower_idx, tf.constant([[0, 0], [1, 0], [2, 0], [3, 0]]))
    npt.assert_allclose(pareto_2d.bounds.upper_idx, tf.constant([[1, 4], [2, 1], [3, 2], [4, 3]]))
    npt.assert_allclose(
        pareto_2d.front, tf.constant([[0.1419, 0.9340], [0.1576, 0.7922], [0.4854, 0.0357]])
    )


@pytest.mark.parametrize("reference", [0.0, [0.0], [[0.0]]])
def test_pareto_hypervolume_indicator_raises_for_reference_with_invalid_shape(
    reference: ListN[float],
) -> None:
    pareto = Pareto(tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        pareto.hypervolume_indicator(tf.constant(reference))


@pytest.mark.parametrize("reference", [[0.1, -0.65], [-0.7, -0.1]])
def test_pareto_hypervolume_indicator_raises_for_reference_below_anti_ideal_point(
    reference: list[float],
) -> None:
    pareto = Pareto(tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]]))

    with pytest.raises(tf.errors.InvalidArgumentError):
        pareto.hypervolume_indicator(tf.constant(reference))


@pytest.mark.parametrize(
    "objectives, reference, expected",
    [
        ([[1.0, 0.5]], [2.3, 2.0], 1.95),
        ([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]], [0.1, -0.1], 0.92),
        (  # reference point is equal to one pareto point in one dimension
            [[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]],
            [0.1, -0.6],
            0.37,
        ),
    ],
)
def test_pareto_hypervolume_indicator(
    objectives: list[list[float]], reference: list[float], expected: float
) -> None:
    pareto = Pareto(tf.constant(objectives))
    npt.assert_allclose(pareto.hypervolume_indicator(tf.constant(reference)), expected)


@pytest.mark.parametrize("reference", [0.0, [0.0], [[0.0]]])
def test_pareto_get_hypercell_bounds_raises_for_reference_with_invalid_shape(
    reference: ListN[float],
) -> None:
    pareto = Pareto(tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        pareto.get_hypercell_bounds(tf.constant([0.0, 0.0]), tf.constant(reference))


@pytest.mark.parametrize("anti_reference", [0.0, [0.0], [[0.0]]])
def test_pareto_get_hypercell_bounds_raises_for_anti_reference_with_invalid_shape(
    anti_reference: ListN[float],
) -> None:
    pareto = Pareto(tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        pareto.get_hypercell_bounds(tf.constant(anti_reference), tf.constant([0.0, 0.0]))


@pytest.mark.parametrize("reference", [[0.1, -0.65], [-0.7, -0.1]])
def test_pareto_get_hypercell_bounds_raises_for_reference_below_anti_ideal_point(
    reference: list[float],
) -> None:
    pareto = Pareto(tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]]))

    with pytest.raises(tf.errors.InvalidArgumentError):
        pareto.get_hypercell_bounds(tf.constant([-10.0, -10.0]), tf.constant(reference))


@pytest.mark.parametrize("anti_reference", [[0.1, -0.65], [-0.7, -0.1]])
def test_pareto_get_hypercell_bounds_raises_for_front_below_anti_reference_point(
    anti_reference: list[float],
) -> None:
    pareto = Pareto(tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]]))

    with pytest.raises(tf.errors.InvalidArgumentError):
        pareto.get_hypercell_bounds(tf.constant(anti_reference), tf.constant([10.0, 10.0]))


@pytest.mark.parametrize(
    "objectives, anti_reference, reference, expected",
    [
        (
            [[1.0, 0.5]],
            [-10.0, -8.0],
            [2.3, 2.0],
            ([[-10.0, -8.0], [1.0, -8.0]], [[1.0, 2.0], [2.3, 0.5]]),
        ),
        (
            [[-1.0, -0.6], [-0.8, -0.7]],
            [-2.0, -1.0],
            [0.1, -0.1],
            ([[-2.0, -1.0], [-1.0, -1.0], [-0.8, -1.0]], [[-1.0, -0.1], [-0.8, -0.6], [0.1, -0.7]]),
        ),
        (  # reference point is equal to one pareto point in one dimension
            # anti idea point is equal to two pareto point in one dimension
            [[-1.0, -0.6], [-0.8, -0.7]],
            [-1.0, -0.7],
            [0.1, -0.6],
            ([[-1.0, -0.7], [-1.0, -0.7], [-0.8, -0.7]], [[-1.0, -0.6], [-0.8, -0.6], [0.1, -0.7]]),
        ),
    ],
)
def test_pareto_get_hypercell_bounds(objectives, anti_reference, reference, expected):
    pareto = Pareto(tf.constant(objectives))
    npt.assert_allclose(
        pareto.get_hypercell_bounds(tf.constant(anti_reference), tf.constant(reference))[0],
        tf.constant(expected[0]),
    )
    npt.assert_allclose(
        pareto.get_hypercell_bounds(tf.constant(anti_reference), tf.constant(reference))[1],
        tf.constant(expected[1]),
    )

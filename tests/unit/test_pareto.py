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

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, SequenceN
from trieste.utils.multi_objective.dominance import non_dominated
from trieste.utils.multi_objective.pareto import Pareto, get_reference_point


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
        (
            tf.constant(
                [
                    [0.90234935, 0.02297473, 0.05389869],
                    [0.98328614, 0.44182944, 0.6975261],
                    [0.39555323, 0.3040712, 0.3433497],
                    [0.72582424, 0.55389977, 0.00330079],
                    [0.9590585, 0.03233206, 0.2403127],
                    [0.04540098, 0.22407162, 0.11227596],
                ]
            ),
            tf.constant(
                [
                    [0.90234935, 0.02297473, 0.05389869],
                    [0.72582424, 0.55389977, 0.00330079],
                    [0.04540098, 0.22407162, 0.11227596],
                ]
            ),
            tf.constant([0, 4, 1, 0, 1, 0]),
        ),
    ],
)
def test_dominated_sort(scores: tf.Tensor, pareto_set: tf.Tensor, dominance: tf.Tensor) -> None:
    ret_pareto_set, ret_dominance = non_dominated(scores)
    npt.assert_allclose(ret_pareto_set, pareto_set)
    npt.assert_array_equal(ret_dominance, dominance)


@pytest.mark.parametrize("reference", [0.0, [0.0], [[0.0]]])
def test_pareto_hypervolume_indicator_raises_for_reference_with_invalid_shape(
    reference: SequenceN[float],
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


@pytest.mark.parametrize("front, reference, screen_concentration_point",
                         [(tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]]),
                          [[0.1, -0.65], [-0.7, -0.1]],
                          tf.constant([[-1.0, -2]])),
                          ((tf.constant([[2.0, 2.0, 0.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]]),
                            [4.0, 4.0, 4.0],
                            tf.constant([[0.0, 0.0, 0.0]])))])
def test_pareto_hypervolume_indicator_raises_for_empty_front(
    front: tf.Tensor, reference: list[float], screen_concentration_point: tf.Tensor
) -> None:
    pareto = Pareto(front, concentration_point=screen_concentration_point)

    with pytest.raises(ValueError):
        pareto.hypervolume_indicator(tf.constant(reference))


@pytest.mark.parametrize(
    "objectives, screen_reference_point, reference, expected",
    [
        ([[1.0, 0.5]], None, [2.3, 2.0], 1.95),
        ([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]], None, [0.1, -0.1], 0.92),
        (  # reference point is equal to one pareto point in one dimension
            [[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]],
            None,
            [0.1, -0.6],
            0.37,
        ),
        ([[2.0, 2.0, 0.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]], None, [4.0, 4.0, 4.0], 29.0),
    ],
)
def test_pareto_hypervolume_indicator(
    objectives: list[list[float]], screen_reference_point: [None, tf.Tensor],
        reference: list[float], expected: float
) -> None:
    pareto = Pareto(tf.constant(objectives), concentration_point=screen_reference_point)
    npt.assert_allclose(pareto.hypervolume_indicator(tf.constant(reference)), expected, 1e-6)


@pytest.mark.parametrize(
    "front",
    [
        (tf.zeros(shape=(0, 2))),
        (tf.zeros(shape=(0, 3))),
        ( tf.constant([])),
    ],
)
def test_get_reference_point_raise_when_feed_empty_front(front):
    with pytest.raises(ValueError):
        get_reference_point(front)


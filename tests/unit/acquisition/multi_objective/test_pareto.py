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
from trieste.acquisition.multi_objective.pareto import Pareto, get_reference_point


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


@pytest.mark.parametrize(
    "front, reference",
    [
        (tf.zeros(shape=(0, 2)), [[0.1, -0.65], [-0.7, -0.1]]),
        ((tf.zeros(shape=(0, 3)), [4.0, 4.0, 4.0])),
    ],
)
def test_pareto_hypervolume_indicator_raises_for_empty_front(
    front: tf.Tensor, reference: list[float]
) -> None:
    pareto = Pareto(front)

    with pytest.raises(ValueError):
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
        ([[2.0, 2.0, 0.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]], [4.0, 4.0, 4.0], 29.0),
    ],
)
def test_pareto_hypervolume_indicator(
    objectives: list[list[float]],
    reference: list[float],
    expected: float,
) -> None:
    pareto = Pareto(tf.constant(objectives))
    npt.assert_allclose(pareto.hypervolume_indicator(tf.constant(reference)), expected, 1e-6)


@pytest.mark.parametrize(
    "front",
    [
        (tf.zeros(shape=(0, 2))),
        (tf.zeros(shape=(0, 3))),
        (tf.constant([])),
    ],
)
def test_get_reference_point_raise_when_feed_empty_front(front: tf.Tensor) -> None:
    with pytest.raises(ValueError):
        get_reference_point(front)


@pytest.mark.parametrize(
    "front, expected",
    [
        (tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([5.0, 6.0])),
        (
            tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[7.0, 2.0], [5.0, 4.0]]]),
            tf.constant([[5.0, 6.0], [9.0, 6.0]]),
        ),
    ],
)
def test_get_reference_point_with_different_front_shape(
    front: tf.Tensor, expected: tf.Tensor
) -> None:
    tf.debugging.assert_equal(get_reference_point(front), expected)

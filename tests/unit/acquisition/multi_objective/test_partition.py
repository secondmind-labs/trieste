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

from typing import Optional

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, SequenceN
from trieste.acquisition.multi_objective.partition import (
    DividedAndConquerNonDominated,
    ExactPartition2dNonDominated,
    prepare_default_non_dominated_partition_bounds,
)


@pytest.mark.parametrize(
    "reference, observations, anti_ref, expected",
    [
        (
            tf.constant([1.0, 1.0]),
            None,
            tf.constant([-1.0, -1.0]),
            (tf.constant([[-1.0, -1.0]]), tf.constant([[1.0, 1.0]])),
        ),
        (
            tf.constant([1.0, 1.0]),
            None,
            tf.constant([1.0, -1.0]),
            (tf.constant([[1.0, -1.0]]), tf.constant([[1.0, 1.0]])),
        ),
        (
            tf.constant([1.0, 1.0]),
            tf.constant([]),
            tf.constant([1.0, -1.0]),
            (tf.constant([[1.0, -1.0]]), tf.constant([[1.0, 1.0]])),
        ),
    ],
)
def test_default_non_dominated_partition_when_no_valid_obs(
    reference: tf.Tensor,
    observations: Optional[tf.Tensor],
    anti_ref: Optional[tf.Tensor],
    expected: tuple[tf.Tensor, tf.Tensor],
) -> None:
    npt.assert_array_equal(
        prepare_default_non_dominated_partition_bounds(reference, observations, anti_ref), expected
    )


def test_default_non_dominated_partition_raise_when_obs_below_default_anti_reference() -> None:
    objectives = tf.constant(
        [
            [-1e11, 0.7922],
            [0.4854, 0.0357],
            [0.1419, 0.9340],
        ]
    )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        prepare_default_non_dominated_partition_bounds(tf.constant([1.0, 1.0]), objectives)


@pytest.mark.parametrize(
    "ref, obs, anti_ref",
    [
        (
            tf.constant([-1e12, 1.0]),
            tf.constant(
                [
                    [0.4854, 0.7922],
                    [0.4854, 0.0357],
                    [0.1419, 0.9340],
                ]
            ),
            None,
        ),
        (tf.constant([-1e12, 1.0]), None, None),
        (tf.constant([-1e12, 1.0]), tf.constant([]), None),
    ],
)
def test_default_non_dominated_partition_raise_when_ref_below_default_anti_reference(
    ref: tf.Tensor, obs: Optional[tf.Tensor], anti_ref: Optional[tf.Tensor]
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        prepare_default_non_dominated_partition_bounds(ref, obs, anti_ref)


def test_exact_partition_2d_bounds() -> None:
    objectives = tf.constant(
        [
            [0.1576, 0.7922],
            [0.4854, 0.0357],
            [0.1419, 0.9340],
        ]
    )

    partition_2d = ExactPartition2dNonDominated(objectives)

    npt.assert_array_equal(
        partition_2d._bounds.lower_idx, tf.constant([[0, 0], [1, 0], [2, 0], [3, 0]])
    )
    npt.assert_array_equal(
        partition_2d._bounds.upper_idx, tf.constant([[1, 4], [2, 1], [3, 2], [4, 3]])
    )
    npt.assert_allclose(
        partition_2d.front, tf.constant([[0.1419, 0.9340], [0.1576, 0.7922], [0.4854, 0.0357]])
    )


def test_exact_partition_2d_raise_when_input_is_not_pareto_front() -> None:
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
    with pytest.raises(tf.errors.InvalidArgumentError):
        ExactPartition2dNonDominated(objectives)


@pytest.mark.parametrize(
    "reference",
    [0.0, [0.0], [[0.0]]],
)
def test_exact_partition_2d_partition_bounds_raises_for_reference_with_invalid_shape(
    reference: SequenceN[float],
) -> None:
    partition = ExactPartition2dNonDominated(
        tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]])
    )

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        partition.partition_bounds(tf.constant([0.0, 0.0]), tf.constant(reference))


@pytest.mark.parametrize("anti_reference", [-10.0, [-10.0], [[-10.0]]])
def test_exact_partition_2d_partition_bounds_raises_for_anti_reference_with_invalid_shape(
    anti_reference: SequenceN[float],
) -> None:
    partition = ExactPartition2dNonDominated(
        tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]])
    )

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        partition.partition_bounds(tf.constant(anti_reference), tf.constant([10.0, 10.0]))


@pytest.mark.parametrize("reference", [[0.1, -0.65], [-0.7, -0.1]])
def test_exact_partition_2d_partition_bounds_raises_for_reference_below_anti_ideal_point(
    reference: list[float],
) -> None:
    partition = ExactPartition2dNonDominated(
        tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]])
    )

    with pytest.raises(tf.errors.InvalidArgumentError):
        partition.partition_bounds(tf.constant([-10.0, -10.0]), tf.constant(reference))


@pytest.mark.parametrize("anti_reference", [[0.1, -0.65], [-0.7, -0.1]])
def test_exact_partition_2d_partition_bounds_raises_for_front_below_anti_reference_point(
    anti_reference: list[float],
) -> None:
    partition = ExactPartition2dNonDominated(
        tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]])
    )

    with pytest.raises(tf.errors.InvalidArgumentError):
        partition.partition_bounds(tf.constant(anti_reference), tf.constant([10.0, 10.0]))


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
def test_exact_partition_2d_partition_bounds(
    objectives: SequenceN[float],
    anti_reference: list[float],
    reference: list[float],
    expected: SequenceN[float],
) -> None:
    partition = ExactPartition2dNonDominated(tf.constant(objectives))
    npt.assert_allclose(
        partition.partition_bounds(tf.constant(anti_reference), tf.constant(reference))[0],
        tf.constant(expected[0]),
    )
    npt.assert_allclose(
        partition.partition_bounds(tf.constant(anti_reference), tf.constant(reference))[1],
        tf.constant(expected[1]),
    )


def test_divide_conquer_non_dominated_raise_when_input_is_not_pareto_front() -> None:
    objectives = tf.constant(
        [
            [0.0, 2.0, 1.0],
            [7.0, 6.0, 0.0],
            [9.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    with pytest.raises(tf.errors.InvalidArgumentError):
        DividedAndConquerNonDominated(objectives)


@pytest.mark.parametrize(
    "reference",
    [0.0, [0.0], [[0.0]]],
)
def test_divide_conquer_non_dominated_partition_bounds_raises_for_reference_with_invalid_shape(
    reference: SequenceN[float],
) -> None:
    partition = DividedAndConquerNonDominated(
        tf.constant(
            [
                [0.0, 2.0, 1.0],
                [7.0, 6.0, 0.0],
                [9.0, 0.0, 1.0],
            ]
        )
    )

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        partition.partition_bounds(tf.constant([0.0, 0.0, 0.0]), tf.constant(reference))


@pytest.mark.parametrize("reference", [[0.5, 0.65, 4], [11.0, 4.0, 2.0], [11.0, 11.0, 0.0]])
def test_divide_conquer_non_dominated_partition_bounds_raises_for_reference_below_anti_ideal_point(
    reference: list[float],
) -> None:
    partition = DividedAndConquerNonDominated(
        tf.constant(
            [
                [0.0, 2.0, 1.0],
                [7.0, 6.0, 0.0],
                [9.0, 0.0, 1.0],
            ]
        )
    )

    with pytest.raises(tf.errors.InvalidArgumentError):
        partition.partition_bounds(tf.constant([-10.0, -10.0, -10.0]), tf.constant(reference))


@pytest.mark.parametrize(
    "anti_reference", [[1.0, -2.0, -2.0], [-1.0, 3.0, -2.0], [-1.0, -3.0, 1.0]]
)
def test_divide_conquer_non_dominated_partition_bounds_raises_for_front_below_anti_reference_point(
    anti_reference: list[float],
) -> None:
    partition = DividedAndConquerNonDominated(
        tf.constant(
            [
                [0.0, 2.0, 1.0],
                [7.0, 6.0, 0.0],
                [9.0, 0.0, 1.0],
            ]
        )
    )

    with pytest.raises(tf.errors.InvalidArgumentError):
        partition.partition_bounds(tf.constant(anti_reference), tf.constant([10.0, 10.0, 10.0]))


def test_divide_conquer_non_dominated_three_dimension_case() -> None:
    objectives = tf.constant(
        [
            [0.0, 2.0, 1.0],
            [7.0, 6.0, 0.0],
            [9.0, 0.0, 1.0],
        ]
    )

    partition_nd = DividedAndConquerNonDominated(objectives)

    npt.assert_array_equal(
        partition_nd._bounds.lower_idx,
        tf.constant(
            [
                [3, 2, 0],
                [3, 1, 0],
                [2, 2, 0],
                [2, 1, 0],
                [3, 0, 1],
                [2, 0, 1],
                [2, 0, 0],
                [0, 1, 1],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ),
    )
    npt.assert_array_equal(
        partition_nd._bounds.upper_idx,
        tf.constant(
            [
                [4, 4, 2],
                [4, 2, 1],
                [3, 4, 2],
                [3, 2, 1],
                [4, 3, 4],
                [3, 1, 4],
                [4, 1, 1],
                [1, 4, 4],
                [2, 4, 1],
                [2, 1, 4],
            ]
        ),
    )
    npt.assert_allclose(
        partition_nd.front,
        tf.constant(
            [
                [0.0, 2.0, 1.0],
                [7.0, 6.0, 0.0],
                [9.0, 0.0, 1.0],
            ]
        ),
    )

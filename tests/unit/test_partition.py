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
import tensorflow as tf
import pytest

from trieste.utils.multi_objective.partition import \
    (ExactPartition2dNonDominated,
     DividedAndConquerNonDominated,
     HypervolumeBoxDecompositionIncrementalDominated,
     FlipTrickPartitionNonDominated)
from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, SequenceN


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


def test_exact_partition_2d_raise_when_input_is_not_pareto_front():
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
        reference: SequenceN[float]) -> None:
    partition = ExactPartition2dNonDominated(tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        partition.partition_bounds(tf.constant([0.0, 0.0]), tf.constant(reference))


@pytest.mark.parametrize("anti_reference", [-10.0, [-10.0], [[-10.0]]])
def test_exact_partition_2d_partition_bounds_raises_for_anti_reference_with_invalid_shape(
        anti_reference: SequenceN[float],
) -> None:
    partition = ExactPartition2dNonDominated(tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        partition.partition_bounds(tf.constant(anti_reference), tf.constant([10.0, 10.0]))


@pytest.mark.parametrize("reference", [[0.1, -0.65], [-0.7, -0.1]])
def test_exact_partition_2d_partition_bounds_raises_for_reference_below_anti_ideal_point(
        reference: list[float],
) -> None:
    partition = ExactPartition2dNonDominated(tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]]))

    with pytest.raises(tf.errors.InvalidArgumentError):
        partition.partition_bounds(tf.constant([-10.0, -10.0]), tf.constant(reference))


@pytest.mark.parametrize("anti_reference", [[0.1, -0.65], [-0.7, -0.1]])
def test_exact_partition_2d_partition_bounds_raises_for_front_below_anti_reference_point(
        anti_reference: list[float],
) -> None:
    partition = ExactPartition2dNonDominated(tf.constant([[-1.0, -0.6], [-0.8, -0.7], [-0.6, -1.1]]))

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
):
    partition = ExactPartition2dNonDominated(tf.constant(objectives))
    npt.assert_allclose(
        partition.partition_bounds(tf.constant(anti_reference), tf.constant(reference))[0],
        tf.constant(expected[0]),
    )
    npt.assert_allclose(
        partition.partition_bounds(tf.constant(anti_reference), tf.constant(reference))[1],
        tf.constant(expected[1]),
    )


def test_divide_conquer_non_dominated_raise_when_input_is_not_pareto_front():
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
        reference: SequenceN[float]) -> None:
    partition = DividedAndConquerNonDominated(
        tf.constant(
            [
                [0.0, 2.0, 1.0],
                [7.0, 6.0, 0.0],
                [9.0, 0.0, 1.0],
            ]
        ))

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
        ))

    with pytest.raises(tf.errors.InvalidArgumentError):
        partition.partition_bounds(tf.constant([-10.0, -10.0, -10.0]), tf.constant(reference))


@pytest.mark.parametrize("anti_reference", [[1.0, -2.0, -2.0], [-1.0, 3.0, -2.0], [-1.0, -3.0, 1.0]])
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
        ))

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


@pytest.mark.parametrize(
    "reference",
    [0.0, [0.0], [[0.0]]],
)
def test_hypervolume_box_decomposition_incremental_dominated_raise_for_reference_with_invalid_shape(
        reference: SequenceN[float]) -> None:
    objectives = tf.constant(
        [
            [0.0, 2.0, 1.0],
            [7.0, 6.0, 0.0],
            [9.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        HypervolumeBoxDecompositionIncrementalDominated(objectives, tf.constant(reference))


@pytest.mark.parametrize("reference", [[0.5, 0.65, 4], [11.0, 4.0, 2.0], [11.0, 11.0, 0.0]])
def test_hypervolume_box_decomposition_incremental_dominated_raises_for_reference_below_anti_ideal_point(
        reference: list[float],
) -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        HypervolumeBoxDecompositionIncrementalDominated(
            tf.constant(
                [
                    [0.0, 2.0, 1.0],
                    [7.0, 6.0, 0.0],
                    [9.0, 0.0, 1.0],
                ]
            ), tf.constant(reference))


def test_hypervolume_box_decomposition_incremental_dominated_raises_for_front_below_anti_reference_point() -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        HypervolumeBoxDecompositionIncrementalDominated(
            tf.constant(
                [
                    [-1e11, 2.0, 1.0],
                    [7.0, 6.0, 0.0],
                    [9.0, 0.0, 1.0],
                ]
            ), tf.constant([10.0, 10.0, 10.0]))


@pytest.mark.parametrize(
    "observations, reference, expected_lb, expected_ub",
    [
        pytest.param(tf.constant([[2.0, 2.0]]), tf.constant([10.0, 10.0]),
                     tf.constant([[2.0, 2.0]]), tf.constant([[10.0, 10.0]]),
                     id='HypervolumeBoxDecompositionIncrementalDominated_2d_only1points'),
        pytest.param(tf.constant([[2.0, 2.0], [5.0, 5.0]]), tf.constant([10.0, 10.0]),
                     tf.constant([[2.0, 2.0]]), tf.constant([[10.0, 10.0]]),
                     id='HypervolumeBoxDecompositionIncrementalDominated_2d_only1PFpoints'),
        pytest.param(tf.constant([[2.0, 2.0], [5.0, 5.0], [1.0, 10.0], [10.0, 1.0]]), tf.constant([10.0, 10.0]),
                     tf.constant([[2.0, 2.0]]), tf.constant([[10.0, 10.0]]),
                     id='HypervolumeBoxDecompositionIncrementalDominated_2d_pf_point_has_1d_same_as_reference'),
        pytest.param(tf.constant([[2.0, 2.0], [1.0, 3.0], [5.0, 10.0]]), tf.constant([10.0, 10.0]),
                     tf.constant([[1., 3.], [2., 2.]]), tf.constant([[10., 10.], [10., 3.]]),
                     id='HypervolumeBoxDecompositionIncrementalDominated_2d_pf_common_case'),
        pytest.param(tf.constant([[2.0, 2.0]]), tf.constant([2.0, 2.0]),
                     tf.zeros(shape=(0, 2)), tf.zeros(shape=(0, 2)),
                     id='HypervolumeBoxDecompositionIncrementalDominated_2d_pf_point_same_as_reference'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0]]), tf.constant([10.0, 10.0, 10.0]),
                     tf.constant([[2.0, 2.0, 2.0]]), tf.constant([[10.0, 10.0, 10.0]]),
                     id='HypervolumeBoxDecompositionIncrementalDominated_3d_only1points'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]), tf.constant([10.0, 10.0, 10.0]),
                     tf.constant([[2.0, 2.0, 2.0]]), tf.constant([[10.0, 10.0, 10.0]]),
                     id='HypervolumeBoxDecompositionIncrementalDominated_3d_only1PFpoints'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0], [1.0, 5.0, 10.0], [10.0, 1.0, 5.0], [1.0, 10.0, 1.0]]),
                     tf.constant([10.0, 10.0, 10.0]),
                     tf.constant([[2.0, 2.0, 2.0]]), tf.constant([[10.0, 10.0, 10.0]]),
                     id='HypervolumeBoxDecompositionIncrementalDominated_3d_pf_point_has_1d_same_as_reference'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0], [1.0, 10.0, 10.0], [10.0, 1.0, 10.0], [10.0, 10.0, 1.0]]),
                     tf.constant([10.0, 10.0, 10.0]),
                     tf.constant([[2.0, 2.0, 2.0]]), tf.constant([[10.0, 10.0, 10.0]]),
                     id='HypervolumeBoxDecompositionIncrementalDominated_3d_pf_point_has_2d_same_as_reference'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0]]),
                     tf.constant([2.0, 2.0, 2.0]),
                     tf.zeros(shape=(0, 3)), tf.zeros(shape=(0, 3)),
                     id='HypervolumeBoxDecompositionIncrementalDominated_3d_pf_point_same_as_reference'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [1.0, 3.0, 5.0]]),
                     tf.constant([10.0, 10.0, 10.0]),
                     tf.constant([[1., 3., 5.], [2., 2., 5.], [2., 2., 2.]]),
                     tf.constant([[10., 10., 10.], [10., 3., 10.], [10., 10., 5.]]),
                     id='HypervolumeBoxDecompositionIncrementalDominated_3d_pf_common_case'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [2.0, 3.0, 1.0], [3.5, 2.0, 1.0]]),
                     tf.constant([10.0, 10.0, 10.0]),
                     tf.constant([[2., 2., 2.], [2., 3., 1.], [3.5, 2., 1.]]),
                     tf.constant([[10., 10., 10.], [10., 10., 2.], [10., 3., 2.]]),
                     id='HypervolumeBoxDecompositionIncrementalDominated_3d_pf_have_same_at_1d_case'),
    ])
def test_hypervolume_box_decomposition_incremental_dominated(observations: tf.Tensor, reference: tf.Tensor,
                                                             expected_lb: tf.Tensor, expected_ub: tf.Tensor):
    lb, ub = HypervolumeBoxDecompositionIncrementalDominated(observations, reference).partition_bounds()
    npt.assert_allclose(lb, expected_lb)
    npt.assert_allclose(ub, expected_ub)


@pytest.mark.parametrize(
    "reference",
    [0.0, [0.0], [[0.0]]],
)
def test_flip_trick_non_dominated_partition_raise_for_reference_with_invalid_shape(
        reference: SequenceN[float]) -> None:
    objectives = tf.constant(
        [
            [0.0, 2.0, 1.0],
            [7.0, 6.0, 0.0],
            [9.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        FlipTrickPartitionNonDominated(objectives, tf.constant(reference) - 10, tf.constant(reference))


@pytest.mark.parametrize("reference", [[0.5, 0.65, 4], [11.0, 4.0, 2.0], [11.0, 11.0, 0.0]])
def test_flip_trick_non_dominated_partition_raises_for_reference_below_anti_ideal_point(
        reference: list[float],
) -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        FlipTrickPartitionNonDominated(
            tf.constant(
                [
                    [0.0, 2.0, 1.0],
                    [7.0, 6.0, 0.0],
                    [9.0, 0.0, 1.0],
                ]
            ), tf.constant([-10.0, -10.0, -10.0]), tf.constant(reference)).partition_bounds()


@pytest.mark.parametrize("anti_reference", [[1.0, -2.0, -2.0], [-1.0, 3.0, -2.0], [-1.0, -3.0, 1.0]])
def test_flip_trick_non_dominated_partition_raises_for_front_below_anti_reference_point(
        anti_reference: list[float],
) -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        FlipTrickPartitionNonDominated(
            tf.constant(
                [
                    [0.0, 2.0, 1.0],
                    [7.0, 6.0, 0.0],
                    [9.0, 0.0, 1.0],
                ]
            ), tf.constant(anti_reference), tf.constant([10.0, 10.0, 10.0])).partition_bounds()


@pytest.mark.parametrize(
    "observations, anti_reference, reference, expected_lb, expected_ub",
    [
        pytest.param(tf.constant([[2.0, 2.0]]),
                     tf.constant([-10.0, -10.0]),
                     tf.constant([10.0, 10.0]),
                     tf.constant([[-10., -10.],
                                  [-10., 2.]]),
                     tf.constant([[10., 2.],
                                  [2., 10.]]),
                     id='FlipTrickPartitionNonDominated_2d_only1points'),
        pytest.param(tf.constant([[2.0, 2.0], [5.0, 5.0]]),
                     tf.constant([-10.0, -10.0]),
                     tf.constant([10.0, 10.0]),
                     tf.constant([[-10., -10.],
                                  [-10., 2.]]),
                     tf.constant([[10., 2.],
                                  [2., 10.]]),
                     id='FlipTrickPartitionNonDominated_2d_only1PFpoints'),
        pytest.param(tf.constant([[2.0, 2.0], [5.0, 5.0], [1.0, 10.0], [10.0, 1.0]]),
                     tf.constant([-10.0, -10.0]),
                     tf.constant([10.0, 10.0]),
                     tf.constant([[-10., -10.],
                                  [-10., 2.]]),
                     tf.constant([[10., 2.],
                                  [2., 10.]]),
                     id='FlipTrickPartitionNonDominated_2d_pf_point_has_1d_same_as_reference'),
        pytest.param(tf.constant([[2.0, 2.0], [1.0, 3.0], [5.0, 10.0]]), tf.constant([-10.0, -10.0]),
                     tf.constant([10.0, 10.0]),
                     tf.constant([[-10., -10.],
                                  [-10., 2.],
                                  [-10., 3.]]),
                     tf.constant([[10., 2.],
                                  [2., 3.],
                                  [1., 10.]]),
                     id='FlipTrickPartitionNonDominated_2d_pf_common_case'),
        pytest.param(tf.constant([[2.0, 2.0]]), tf.constant([2.0, 2.0]), tf.constant([4.0, 4.0]),
                     tf.zeros(shape=(0, 2)), tf.zeros(shape=(0, 2)),
                     id='FlipTrickPartitionNonDominated_2d_pf_point_same_as_antireference'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0]]), tf.constant([-10.0, -10.0, -10.0]),
                     tf.constant([10.0, 10.0, 10.0]),
                     tf.constant([[-10., -10., -10.],
                                  [-10., 2., -10.],
                                  [-10., 2., 2.]]), tf.constant([[10., 2., 10.],
                                                                 [10., 10., 2.],
                                                                 [2., 10., 10.]]),
                     id='FlipTrickPartitionNonDominated_3d_only1points'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
                     tf.constant([-10.0, -10.0, -10.0]), tf.constant([10.0, 10.0, 10.0]),
                     tf.constant([[-10., -10., -10.],
                                  [-10., 2., -10.],
                                  [-10., 2., 2.]]), tf.constant([[10., 2., 10.],
                                                                 [10., 10., 2.],
                                                                 [2., 10., 10.]]),
                     id='FlipTrickPartitionNonDominated_3d_only1PFpoints'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0], [1.0, 5.0, 10.0], [10.0, 1.0, 5.0], [1.0, 10.0, 1.0]]),
                     tf.constant([-10.0, -10.0, -10.0]),
                     tf.constant([10.0, 10.0, 10.0]),
                     tf.constant([[-10., -10., -10.],
                                  [-10., 2., -10.],
                                  [-10., 2., 2.]]), tf.constant([[10., 2., 10.],
                                                                 [10., 10., 2.],
                                                                 [2., 10., 10.]]),
                     id='FlipTrickPartitionNonDominated_3d_pf_point_has_1d_same_as_reference'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0], [1.0, 10.0, 10.0], [10.0, 1.0, 10.0], [10.0, 10.0, 1.0]]),
                     tf.constant([-10.0, -10.0, -10.0]),
                     tf.constant([10.0, 10.0, 10.0]),
                     tf.constant([[-10., -10., -10.],
                                  [-10., 2., -10.],
                                  [-10., 2., 2.]]), tf.constant([[10., 2., 10.],
                                                                 [10., 10., 2.],
                                                                 [2., 10., 10.]]),
                     id='FlipTrickPartitionNonDominated_3d_pf_point_has_2d_same_as_reference'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0]]),
                     tf.constant([2.0, 2.0, 2.0]),
                     tf.constant([10.0, 10.0, 10.0]),
                     tf.zeros(shape=(0, 3)), tf.zeros(shape=(0, 3)),
                     id='FlipTrickPartitionNonDominated_3d_pf_point_same_as_reference'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [1.0, 3.0, 5.0]]),
                     tf.constant([-10.0, -10.0, -10.0]),
                     tf.constant([10.0, 10.0, 10.0]),
                     tf.constant([[-10., -10., -10.],
                                  [-10., 2., -10.],
                                  [-10., 2., 2.],
                                  [-10., 3., 2.],
                                  [-10., 3., 5.]]),
                     tf.constant([[10., 2., 10.],
                                  [10., 10., 2.],
                                  [2., 3., 10.],
                                  [2., 10., 5.],
                                  [1., 10., 10.]]),
                     id='FlipTrickPartitionNonDominated_3d_pf_common_case'),
        pytest.param(tf.constant([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [2.0, 3.0, 1.0], [3.5, 2.0, 1.0]]),
                     tf.constant([-10.0, -10.0, -10.0]),
                     tf.constant([10.0, 10.0, 10.0]),
                     tf.constant([[-10., -10., -10.],
                                  [-10., 2., -10.],
                                  [-10., 2., 1.],
                                  [-10., 3., 1.],
                                  [-10., 2., 2.]]),
                     tf.constant([[10., 2., 10.],
                                  [10., 10., 1.],
                                  [3.5, 3., 2.],
                                  [2., 10., 2.],
                                  [2., 10., 10.]]),
                     id='HypervolumeBoxDecompositionIncrementalDominated_3d_pf_have_same_at_1d_case'),
    ])
def test_flip_trick_non_dominated_partition(observations: tf.Tensor, anti_reference: tf.Tensor,
                                            reference: tf.Tensor,
                                            expected_lb: tf.Tensor, expected_ub: tf.Tensor):
    lb, ub = FlipTrickPartitionNonDominated(observations, anti_reference, reference).partition_bounds()

    npt.assert_allclose(lb, expected_lb)
    npt.assert_allclose(ub, expected_ub)

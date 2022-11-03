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

import numpy as np
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
    "observations",
    [
        (tf.zeros(shape=(0, 2))),
        (tf.zeros(shape=(0, 3))),
        (tf.constant([])),
    ],
)
def test_get_reference_point_raise_when_feed_empty_front(observations: tf.Tensor) -> None:
    with pytest.raises(ValueError):
        get_reference_point(observations)


@pytest.mark.parametrize(
    "observations, expected",
    [
        (tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([1.0, 2.0])),
        (tf.constant([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]]), tf.constant([3.0, 3.0])),
        (tf.constant([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 5.0]]), tf.constant([3.0, 3.0])),
    ],
)
def test_get_reference_point_extract_based_on_pareto_front(
    observations: tf.Tensor, expected: tf.Tensor
) -> None:
    tf.debugging.assert_equal(get_reference_point(observations), expected)


@pytest.mark.qhsri
def test_pareto_sample_diverse_subset_raises_too_large_sample_size() -> None:
    observations = tf.constant([[1.0, -1.0], [-1.0, 1.0]])
    pareto_set = Pareto(observations)
    with pytest.raises(ValueError):
        pareto_set.sample_diverse_subset(3, allow_repeats=False)


@pytest.mark.qhsri
def test_pareto_sample_diverse_subset_raises_zero_range() -> None:
    observations = tf.constant([[1.0, 1.0], [1.0, 1.0]])
    pareto_set = Pareto(observations)
    with pytest.raises(ValueError):
        pareto_set.sample_diverse_subset(1, bounds_min_delta=0.0)


@pytest.mark.qhsri
def test_pareto_sample_diverse_subset_get_bounds() -> None:
    observations = tf.constant([[1.0, -1.0], [-1.0, 1.0]])
    pareto_set = Pareto(observations)
    lower_bounds, reference_point = pareto_set._get_bounds(delta_scaling_factor=0.2, min_delta=1e-9)
    expected_lower_bounds = tf.constant([-1.4, -1.4])
    expected_reference_point = tf.constant([1.4, 1.4])
    npt.assert_allclose(expected_lower_bounds, lower_bounds)
    npt.assert_allclose(expected_reference_point, reference_point)


@pytest.mark.qhsri
def test_pareto_sample_diverse_subset_calculate_p() -> None:
    observations = tf.constant([[1.0, -1.0], [-1.0, 1.0]])
    lower_bound = tf.constant([-2.0, -2.0])
    reference_point = tf.constant([2.0, 2.0])
    pareto_set = Pareto(observations)
    output = pareto_set._calculate_p_matrix(lower_bound, reference_point)
    expected_output = tf.constant([[3 / 16, 1 / 16], [1 / 16, 3 / 16]])
    npt.assert_array_equal(expected_output, output)


@pytest.mark.qhsri
def test_pareto_sample_diverse_subset_choose_batch_no_repeats() -> None:
    observations = tf.constant([[2.0, -2.0], [1.0, -1.0], [0.0, 0.0], [-1.0, 1.0], [-2.0, 2.0]])
    x_star = tf.constant([[0.15], [0.25], [0.2], [0.3], [0.1]])
    pareto_set = Pareto(observations)
    sample, sample_ids = pareto_set._choose_batch_no_repeats(x_star, sample_size=2)
    expected_sample = tf.constant([[-1.0, 1.0], [1.0, -1.0]])
    expected_sample_ids = tf.constant([3, 1])
    npt.assert_array_equal(expected_sample, sample)
    npt.assert_array_equal(expected_sample_ids, sample_ids)


@pytest.mark.qhsri
def test_pareto_sample_diverse_subset_choose_batch_no_repeats_return_same_front() -> None:
    observations = tf.constant([[1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]])
    x_star = tf.constant([[0.4], [0.35], [0.25]])
    pareto_set = Pareto(observations)
    sample, sample_ids = pareto_set._choose_batch_no_repeats(x_star, sample_size=3)
    expected_sample = pareto_set.front
    expected_sample_ids = tf.constant([0, 1, 2])
    npt.assert_array_equal(expected_sample, sample)
    npt.assert_array_equal(expected_sample_ids, sample_ids)


@pytest.mark.parametrize(
    "x_star,expected_ids",
    (
        ([[0.25], [0.1], [0.09], [0.51], [0.05]], [0, 3, 3]),
        ([[0.25], [0.24], [0.25], [0.01], [0.25]], [0, 1, 2, 4]),
        ([[0.1], [0.2], [0.3], [0.4], [0.0]], [1, 2, 3, 3]),
    ),
)
@pytest.mark.qhsri
def test_pareto_sample_diverse_subset_choose_batch_with_repeats(
    x_star: list[list[float]], expected_ids: list[int]
) -> None:
    observations = tf.constant([[2.0, -2.0], [1.0, -1.0], [0.0, 0.0], [-1.0, 1.0], [-2.0, 2.0]])
    pareto_set = Pareto(observations)
    _, sample_ids = pareto_set._choose_batch_with_repeats(np.array(x_star), sample_size=4)
    sample_ids_list = list(sample_ids)
    for expected_id in expected_ids:
        assert expected_id in sample_ids_list
        sample_ids_list.remove(expected_id)

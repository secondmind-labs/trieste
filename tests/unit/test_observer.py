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
import numpy as np
import pytest
import tensorflow as tf

from tests.util.misc import ShapeLike, assert_datasets_allclose
from trieste.data import Dataset
from trieste.observer import filter_finite, map_is_finite


def _sum_with_nan_at_origin(t: tf.Tensor) -> tf.Tensor:
    """
    Example:

    >>> _sum_with_nan_at_origin(tf.constant([[0.0, 0.0], [0.1, 0.5]])).numpy()
    array([[nan],
           [0.6]], dtype=float32)

    :param t: A tensor of N two-dimensional points.
    :return: A tensor of N one-dimensional points. For each of the N points, if the point in ``t``
        is at the origin [[0.0]], the result is `np.nan`. Otherwise, it's the sum across dimensions
        of the point in ``t``.
    """
    is_at_origin = tf.reduce_all(t == [[0.0, 0.0]], axis=-1, keepdims=True)
    sums = tf.reduce_sum(t, axis=-1, keepdims=True)
    return tf.where(is_at_origin, [[np.nan]], sums)


@pytest.mark.parametrize(
    "query_points, expected",
    [
        (  # one failure point
            tf.constant([[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 0.0], [1.0, 3.0]]),
            Dataset(
                tf.constant([[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 3.0]]),
                tf.constant([[-1.0], [1.0], [2.0], [4.0]]),
            ),
        ),
        (  # no failure points
            tf.constant([[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 3.0]]),
            Dataset(
                tf.constant([[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 3.0]]),
                tf.constant([[-1.0], [1.0], [2.0], [4.0]]),
            ),
        ),
        (  # only failure points
            tf.constant([[0.0, 0.0]]),
            Dataset(tf.zeros([0, 2]), tf.zeros([0, 1])),
        ),
        (tf.zeros([0, 2]), Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))),  # empty data
    ],
)
def test_filter_finite(query_points: tf.Tensor, expected: Dataset) -> None:
    observations = _sum_with_nan_at_origin(query_points)
    assert_datasets_allclose(filter_finite(query_points, observations), expected)


@pytest.mark.parametrize(
    "qp_shape, obs_shape",
    [
        ([3, 4], [3, 2]),  # observations not N x 1
        ([3, 4], [4, 1]),  # different leading dims
        ([3], [3, 1]),  # query_points missing a dimension
        ([3, 4, 2], [3, 1]),  # query_points have too many dimensions
    ],
)
def test_filter_finite_raises_for_invalid_shapes(qp_shape: ShapeLike, obs_shape: ShapeLike) -> None:
    with pytest.raises(ValueError):
        filter_finite(tf.ones(qp_shape), tf.ones(obs_shape))


def test_map_is_finite() -> None:
    query_points = tf.constant([[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 0.0], [1.0, 3.0]])
    observations = _sum_with_nan_at_origin(query_points)
    expected = Dataset(query_points, tf.constant([[1], [1], [1], [0], [1]], tf.uint8))
    assert_datasets_allclose(map_is_finite(query_points, observations), expected)


def test_map_is_finite_with_empty_data() -> None:
    query_points, observations = tf.zeros([0, 2]), tf.zeros([0, 1])
    expected = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1], tf.uint8))
    assert_datasets_allclose(map_is_finite(query_points, observations), expected)

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

from typing import List

import numpy.testing as npt
import numpy as np
import pytest
import tensorflow as tf

from trieste.observer import map_is_finite, filter_finite


def _nan_at_origin(t: tf.Tensor) -> tf.Tensor:
    is_at_origin = tf.reduce_all(t == [[0., 0.]], axis=-1, keepdims=True)
    sums = tf.reduce_sum(t, axis=-1, keepdims=True)
    return tf.where(is_at_origin, tf.constant([[np.nan]]), sums)


def test_filter_finite() -> None:
    ok_query_points = [[-1., 0.], [1., 0.], [0., 2.], [1., 3.]]
    query_points = tf.constant([[0., 0.]] + ok_query_points)
    finite_values = filter_finite(query_points, _nan_at_origin(query_points))

    npt.assert_array_almost_equal(finite_values.query_points, ok_query_points)
    npt.assert_array_almost_equal(finite_values.observations, [[-1.], [1.], [2.], [4.]])


@pytest.mark.parametrize('qp_shape, obs_shape', [
    ([3, 4], [3, 2]),  # observations not N x 1
    ([3, 4], [4, 1]),  # different leading dims
    ([3], [3, 1]),  # query_points missing a dimension
    ([3, 4, 2], [3, 1]),  # query_points have too many dimensions
])
def test_filter_finite_raises_for_invalid_shapes(qp_shape: List[int], obs_shape: List[int]) -> None:
    with pytest.raises(ValueError):
        filter_finite(tf.ones(qp_shape), tf.ones(obs_shape))


def test_map_is_finite() -> None:
    query_points = tf.constant([[0., 0.]] + [[-1., 0.], [1., 0.], [0., 2.], [1., 3.]])
    is_finite = map_is_finite(query_points, _nan_at_origin(query_points))

    npt.assert_array_almost_equal(is_finite.query_points, query_points)
    npt.assert_array_almost_equal(is_finite.observations, [[0.], [1.], [1.], [1.], [1.]])

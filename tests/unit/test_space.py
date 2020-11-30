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
from typing import Tuple, List

import pytest
import numpy.testing as npt
import tensorflow as tf

from tests.util.misc import ShapeLike, various_shapes
from trieste.space import SearchSpace, DiscreteSearchSpace, Box


def _points_in_2D_search_space() -> tf.Tensor:
    return tf.constant([[-1.0, 0.4], [-1.0, 0.6], [0.0, 0.4], [0.0, 0.6], [1.0, 0.4], [1.0, 0.6]])


@pytest.mark.parametrize("shape", various_shapes(excluding_ranks=[2]))
def test_discrete_search_space_raises_for_invalid_shapes(shape: ShapeLike) -> None:
    with pytest.raises(ValueError):
        DiscreteSearchSpace(tf.random.uniform(shape))


def test_discrete_search_space_points() -> None:
    space = DiscreteSearchSpace(_points_in_2D_search_space())
    npt.assert_array_equal(space.points, _points_in_2D_search_space())


@pytest.mark.parametrize("point", list(_points_in_2D_search_space()))
def test_discrete_search_space_contains_all_its_points(point: tf.Tensor) -> None:
    assert point in DiscreteSearchSpace(_points_in_2D_search_space())


@pytest.mark.parametrize(
    "point",
    [
        tf.constant([-1.0, -0.4]),
        tf.constant([-1.0, 0.5]),
        tf.constant([-2.0, 0.4]),
        tf.constant([-2.0, 0.7]),
    ],
)
def test_discrete_search_space_does_not_contain_other_points(point: tf.Tensor) -> None:
    assert point not in DiscreteSearchSpace(_points_in_2D_search_space())


@pytest.mark.parametrize(
    "points, test_point",
    [
        (tf.constant([[0.0]]), tf.constant([0.0, 0.0])),
        (tf.constant([[0.0, 0.0]]), tf.constant(0.0)),
        (tf.constant([[0.0, 0.0]]), tf.constant([0.0])),
        (tf.constant([[0.0, 0.0]]), tf.constant([0.0, 0.0, 0.0])),
    ],
)
def test_discrete_search_space_contains_raises_for_invalid_shapes(
    points: tf.Tensor, test_point: tf.Tensor
) -> None:
    space = DiscreteSearchSpace(points)
    with pytest.raises(ValueError):
        _ = test_point in space


def _assert_correct_number_of_unique_constrained_samples(
    num_samples: int, search_space: SearchSpace, samples: tf.Tensor
) -> None:
    assert all(sample in search_space for sample in samples)
    assert len(samples) == num_samples

    unique_samples = set(tuple(sample.numpy().tolist()) for sample in samples)

    assert len(unique_samples) == len(samples)


@pytest.mark.parametrize("num_samples", [0, 1, 3, 5, 6])
def test_discrete_search_space_sampling(num_samples: int) -> None:
    search_space = DiscreteSearchSpace(_points_in_2D_search_space())
    samples = search_space.sample(num_samples)
    _assert_correct_number_of_unique_constrained_samples(num_samples, search_space, samples)


@pytest.mark.parametrize("num_samples", [7, 8, 10])
def test_discrete_search_space_sampling_raises_when_too_many_samples_are_requested(
    num_samples: int,
) -> None:
    search_space = DiscreteSearchSpace(_points_in_2D_search_space())

    with pytest.raises(ValueError, match="samples"):
        search_space.sample(num_samples)


def _pairs_of_different_shapes() -> List[Tuple[ShapeLike, ShapeLike]]:
    return [
        ((), (1,)),
        ((1,), (1, 2)),
        ((1, 2), (1, 2, 3)),
    ]


@pytest.mark.parametrize("lower_shape, upper_shape", _pairs_of_different_shapes())
def test_box_raises_if_bounds_have_different_shape(
    lower_shape: ShapeLike, upper_shape: ShapeLike
) -> None:
    lower, upper = tf.zeros(lower_shape), tf.ones(upper_shape)

    with pytest.raises(ValueError, match="bound"):
        Box(lower, upper)


@pytest.mark.parametrize(
    "lower_dtype, upper_dtype",
    [
        (tf.int8, tf.uint16),
        (tf.uint32, tf.float32),
        (tf.float32, tf.float64),
        (tf.float64, tf.bfloat16),
    ],
)
def test_box_raises_if_bounds_have_different_dtypes(
    lower_dtype: Tuple[tf.DType, tf.DType], upper_dtype: Tuple[tf.DType, tf.DType]
) -> None:
    lower, upper = tf.zeros((1, 2), dtype=lower_dtype), tf.ones((1, 2), dtype=upper_dtype)

    with pytest.raises(TypeError, match="dtype"):
        Box(lower, upper)


@pytest.mark.parametrize(
    "lower, upper",
    [
        (tf.ones((3,)), tf.ones((3,))),  # all equal
        (tf.ones((3,)) + 1, tf.ones((3,))),  # lower all higher than upper
        (  # one lower higher than upper
            tf.constant([2.3, -0.1, 8.0]),
            tf.constant([3.0, -0.2, 8.0]),
        ),
        (tf.constant([2.3, -0.1, 8.0]), tf.constant([3.0, -0.1, 8.0])),  # one lower equal to upper
    ],
)
def test_box_raises_if_any_lower_bound_is_not_less_than_upper_bound(
    lower: tf.Tensor, upper: tf.Tensor
) -> None:
    with pytest.raises(ValueError):
        Box(lower, upper)


def test_box_bounds_attributes() -> None:
    lower, upper = tf.zeros([2]), tf.ones([2])
    box = Box(lower, upper)
    npt.assert_array_equal(box.lower, lower)
    npt.assert_array_equal(box.upper, upper)


@pytest.mark.parametrize(
    "point",
    [
        tf.constant([-1.0, 0.0, -2.0]),  # lower bound
        tf.constant([2.0, 1.0, -0.5]),  # upper bound
        tf.constant([0.5, 0.5, -1.5]),  # approx centre
        tf.constant([-1.0, 0.0, -1.9]),  # near the edge
    ],
)
def test_box_contains_point(point: tf.Tensor) -> None:
    assert point in Box(tf.constant([-1.0, 0.0, -2.0]), tf.constant([2.0, 1.0, -0.5]))


@pytest.mark.parametrize(
    "point",
    [
        tf.constant([-1.1, 0.0, -2.0]),  # just outside
        tf.constant([-0.5, -0.5, 1.5]),  # negative of a contained point
        tf.constant([10.0, -10.0, 10.0]),  # well outside
    ],
)
def test_box_does_not_contain_point(point: tf.Tensor) -> None:
    assert point not in Box(tf.constant([-1.0, 0.0, -2.0]), tf.constant([2.0, 1.0, -0.5]))


@pytest.mark.parametrize("bound_shape, point_shape", _pairs_of_different_shapes())
def test_box_contains_raises_on_point_of_different_shape(
    bound_shape: ShapeLike,
    point_shape: ShapeLike,
) -> None:
    box = Box(tf.zeros(bound_shape), tf.ones(bound_shape))
    point = tf.zeros(point_shape)

    with pytest.raises(ValueError):
        _ = point in box


@pytest.mark.parametrize("num_samples", [0, 1, 10])
def test_box_sampling(num_samples: int) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)))
    samples = box.sample(num_samples)
    _assert_correct_number_of_unique_constrained_samples(num_samples, box, samples)


@pytest.mark.parametrize("num_samples", [0, 1, 10])
def test_box_discretize_returns_search_space_with_only_points_contained_within_box(
    num_samples: int,
) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)))
    dss = box.discretize(num_samples)

    samples = dss.sample(num_samples)

    assert all(sample in box for sample in samples)


@pytest.mark.parametrize("num_samples", [0, 1, 10])
def test_box_discretize_returns_search_space_with_correct_number_of_points(
    num_samples: int,
) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)))
    dss = box.discretize(num_samples)

    samples = dss.sample(num_samples)

    assert len(samples) == num_samples

    with pytest.raises(ValueError):
        dss.sample(num_samples + 1)


def test_box_combined_with_itself_returns_new_box_with_bounds_twice_as_large() -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)))
    new_box = box * box

    assert len(new_box.lower) == 6
    assert len(new_box.upper) == 6


def test_box_product_bounds_are_the_concatenation_of_original_bounds() -> None:
    box1 = Box(tf.constant([0.0, 1.0]), tf.constant([2.0, 3.0]))
    box2 = Box(tf.constant([4.0, 5.0, 6.0]), tf.constant([7.0, 8.0, 9.0]))

    res = box1 * box2
    npt.assert_allclose(res.lower, [0, 1, 4, 5, 6])
    npt.assert_allclose(res.upper, [2, 3, 7, 8, 9])


def test_box_product_raises_if_bounds_have_different_types() -> None:
    box1 = Box(tf.constant([0.0, 1.0]), tf.constant([2.0, 3.0]))
    box2 = Box(tf.constant([4.0, 5.0], tf.float64), tf.constant([6.0, 7.0], tf.float64))

    with pytest.raises(TypeError):
        _ = box1 * box2


def test_discrete_search_space_product_points_is_the_concatenation_of_original_points() -> None:
    dss1 = DiscreteSearchSpace(tf.constant([[-1.0, -1.4], [-1.5, -3.6], [-0.5, -0.6]]))
    dss2 = DiscreteSearchSpace(tf.constant([[1.0, 1.4], [1.5, 3.6]]))
    [n1, d1] = dss1.points.shape
    [n2, d2] = dss2.points.shape
    res = dss1 * dss2

    assert res.points.shape[0] == n1 * n2
    assert res.points.shape[1] == d1 + d2
    assert all(point in dss1 for point in res.points[:, :2])
    assert all(point in dss2 for point in res.points[:, 2:])


def test_discrete_search_space_product_raises_if_points_have_different_types() -> None:
    dss1 = DiscreteSearchSpace(_points_in_2D_search_space())
    dss2 = DiscreteSearchSpace(tf.constant([[1.0, 1.4], [-1.5, 3.6]], tf.float64))

    with pytest.raises(TypeError):
        _ = dss1 * dss2

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

import copy
import itertools
import operator
from functools import reduce
from typing import Any, Container, List, Optional, Sequence, Type

import numpy.testing as npt
import pytest
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from typing_extensions import Final

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, ShapeLike, various_shapes
from trieste.space import (
    Box,
    CategoricalSearchSpace,
    CollectionSearchSpace,
    Constraint,
    DiscreteSearchSpace,
    GeneralDiscreteSearchSpace,
    LinearConstraint,
    NonlinearConstraint,
    SearchSpace,
    TaggedMultiSearchSpace,
    TaggedProductSearchSpace,
    cast_encoder,
    one_hot_encoder,
)
from trieste.types import TensorType


class Integers(SearchSpace):
    def __init__(self, exclusive_limit: int):
        assert exclusive_limit > 0
        self.limit: Final[int] = exclusive_limit

    @property
    def has_bounds(self) -> bool:
        return True

    @property
    def lower(self) -> None:
        pass

    @property
    def upper(self) -> None:
        pass

    def sample(self, num_samples: int, seed: Optional[int] = None) -> tf.Tensor:
        return tf.random.shuffle(tf.range(self.limit), seed=seed)[:num_samples]

    def _contains(self, point: tf.Tensor) -> bool | TensorType:
        tf.debugging.assert_integer(point)
        return 0 <= point < self.limit

    def product(self, other: Integers) -> Integers:
        return Integers(self.limit * other.limit)

    @property
    def dimension(self) -> TensorType:
        pass

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Integers):
            return NotImplemented
        return self.limit == other.limit


@pytest.mark.parametrize("exponent", [0, -2])
def test_search_space___pow___raises_for_non_positive_exponent(exponent: int) -> None:
    space = Integers(3)
    with pytest.raises(tf.errors.InvalidArgumentError):
        space**exponent


def test_search_space___pow___multiplies_correct_number_of_search_spaces() -> None:
    assert (Integers(5) ** 7).limit == 5**7


def _points_in_2D_search_space() -> tf.Tensor:
    return tf.constant([[-1.0, 0.4], [-1.0, 0.6], [0.0, 0.4], [0.0, 0.6], [1.0, 0.4], [1.0, 0.6]])


@pytest.mark.parametrize("shape", various_shapes(excluding_ranks=[2]))
def test_discrete_search_space_raises_for_invalid_shapes(shape: ShapeLike) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        DiscreteSearchSpace(tf.random.uniform(shape))


def test_discrete_search_space_points() -> None:
    space = DiscreteSearchSpace(_points_in_2D_search_space())
    npt.assert_array_equal(space.points, _points_in_2D_search_space())


@pytest.mark.parametrize("point", list(_points_in_2D_search_space()))
def test_discrete_search_space_contains_all_its_points(point: tf.Tensor) -> None:
    space = DiscreteSearchSpace(_points_in_2D_search_space())
    assert point in space
    assert space.contains(point)


def test_discrete_search_space_contains_all_its_points_at_once() -> None:
    points = _points_in_2D_search_space()
    space = DiscreteSearchSpace(points)
    contains = space.contains(points)
    assert len(contains) == len(points)
    assert tf.reduce_all(contains)


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
    space = DiscreteSearchSpace(_points_in_2D_search_space())
    assert point not in space
    assert not space.contains(point)


def test_discrete_search_space_contains_some_points_but_not_others() -> None:
    points = tf.constant([[-1.0, -0.4], [-1.0, 0.4], [-1.0, 0.5]])
    space = DiscreteSearchSpace(_points_in_2D_search_space())
    contains = space.contains(points)
    assert list(contains) == [False, True, False]


@pytest.mark.parametrize(
    "test_points, contains",
    [
        (tf.constant([[0.0, 0.0], [1.0, 1.0]]), tf.constant([True, False])),
        (tf.constant([[[0.0, 0.0]]]), tf.constant([[True]])),
    ],
)
def test_discrete_search_space_contains_handles_broadcast(
    test_points: tf.Tensor, contains: tf.Tensor
) -> None:
    space = DiscreteSearchSpace(tf.constant([[0.0, 0.0]]))
    tf.assert_equal(contains, space.contains(test_points))
    # point in space raises (because python insists on a bool)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        _ = test_points in space


@pytest.mark.parametrize(
    "space, dimension",
    [
        (DiscreteSearchSpace(tf.constant([[-0.5], [0.2], [1.2], [1.7]])), 1),  # 1d
        (DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [1.2, 0.4]])), 2),  # 2d
    ],
)
def test_discrete_search_space_returns_correct_dimension(
    space: DiscreteSearchSpace, dimension: int
) -> None:
    assert space.dimension == dimension


@pytest.mark.parametrize(
    "space, lower, upper",
    [
        (
            DiscreteSearchSpace(tf.constant([[-0.5], [0.2], [1.2], [1.7]])),
            tf.constant([-0.5]),
            tf.constant([1.7]),
        ),  # 1d
        (
            DiscreteSearchSpace(tf.constant([[-0.5, 0.3], [1.2, -0.4]])),
            tf.constant([-0.5, -0.4]),
            tf.constant([1.2, 0.3]),
        ),  # 2d
    ],
)
def test_discrete_search_space_returns_correct_bounds(
    space: DiscreteSearchSpace, lower: tf.Tensor, upper: tf.Tensor
) -> None:
    assert space.has_bounds
    npt.assert_array_equal(space.lower, lower)
    npt.assert_array_equal(space.upper, upper)


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
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        _ = test_point in space
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        _ = space.contains(test_point)


@pytest.mark.parametrize("num_samples", [0, 1, 3, 5, 6, 10, 20])
@pytest.mark.parametrize(
    "search_space",
    [
        pytest.param(DiscreteSearchSpace(_points_in_2D_search_space()), id="DiscreteSearchSpace"),
        pytest.param(CategoricalSearchSpace([3, 2]), id="CategoricalSearchSpace"),
    ],
)
def test_discrete_search_space_sampling(
    search_space: GeneralDiscreteSearchSpace, num_samples: int
) -> None:
    samples = search_space.sample(num_samples)
    assert all(sample in search_space for sample in samples)
    assert len(samples) == num_samples


@pytest.mark.parametrize("seed", [1, 42, 123])
@pytest.mark.parametrize(
    "search_space",
    [
        pytest.param(DiscreteSearchSpace(_points_in_2D_search_space()), id="DiscreteSearchSpace"),
        pytest.param(CategoricalSearchSpace([3, 2]), id="CategoricalSearchSpace"),
    ],
)
def test_discrete_search_space_sampling_returns_same_points_for_same_seed(
    search_space: GeneralDiscreteSearchSpace, seed: int
) -> None:
    random_samples_1 = search_space.sample(num_samples=100, seed=seed)
    random_samples_2 = search_space.sample(num_samples=100, seed=seed)
    npt.assert_allclose(random_samples_1, random_samples_2)


@pytest.mark.parametrize(
    "search_space",
    [
        pytest.param(DiscreteSearchSpace(_points_in_2D_search_space()), id="DiscreteSearchSpace"),
        pytest.param(CategoricalSearchSpace([3, 2]), id="CategoricalSearchSpace"),
    ],
)
def test_discrete_search_space_sampling_returns_different_points_for_different_call(
    search_space: GeneralDiscreteSearchSpace,
) -> None:
    search_space = DiscreteSearchSpace(_points_in_2D_search_space())
    random_samples_1 = search_space.sample(num_samples=100)
    random_samples_2 = search_space.sample(num_samples=100)
    npt.assert_raises(AssertionError, npt.assert_allclose, random_samples_1, random_samples_2)


def test_discrete_search_space___mul___points_is_the_concatenation_of_original_points() -> None:
    dss1 = DiscreteSearchSpace(tf.constant([[-1.0, -1.4], [-1.5, -3.6], [-0.5, -0.6]]))
    dss2 = DiscreteSearchSpace(tf.constant([[1.0, 1.4], [1.5, 3.6]]))

    product = dss1 * dss2

    all_expected_points = tf.constant(
        [
            [-1.0, -1.4, 1.0, 1.4],
            [-1.0, -1.4, 1.5, 3.6],
            [-1.5, -3.6, 1.0, 1.4],
            [-1.5, -3.6, 1.5, 3.6],
            [-0.5, -0.6, 1.0, 1.4],
            [-0.5, -0.6, 1.5, 3.6],
        ]
    )

    assert len(product.points) == len(all_expected_points)
    assert all(point in product for point in all_expected_points)


def test_discrete_search_space___mul___for_empty_search_space() -> None:
    dss = DiscreteSearchSpace(tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    empty = DiscreteSearchSpace(tf.zeros([0, 1]))

    npt.assert_array_equal((empty * dss).points, tf.zeros([0, 3]))
    npt.assert_array_equal((dss * empty).points, tf.zeros([0, 3]))


def test_discrete_search_space___mul___for_identity_space() -> None:
    dss = DiscreteSearchSpace(tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    identity = DiscreteSearchSpace(tf.zeros([1, 0]))

    npt.assert_array_equal((dss * identity).points, dss.points)
    npt.assert_array_equal((identity * dss).points, dss.points)


def test_discrete_search_space___mul___raises_if_points_have_different_types() -> None:
    dss1 = DiscreteSearchSpace(_points_in_2D_search_space())
    dss2 = DiscreteSearchSpace(tf.constant([[1.0, 1.4], [-1.5, 3.6]], tf.float64))

    with pytest.raises(TypeError):
        _ = dss1 * dss2


def test_discrete_search_space_deepcopy() -> None:
    dss = DiscreteSearchSpace(_points_in_2D_search_space())
    npt.assert_allclose(copy.deepcopy(dss).points, _points_in_2D_search_space())


@pytest.mark.parametrize(
    "lower, upper",
    [
        pytest.param([0.0, 1.0], [1.0, 2.0], id="lists"),
        pytest.param((0.0, 1.0), (1.0, 2.0), id="tuples"),
        pytest.param(range(2), range(1, 3), id="ranges"),
    ],
)
def test_box_converts_sequences_to_float64_tensors(
    lower: Sequence[float], upper: Sequence[float]
) -> None:
    box = Box(lower, upper)
    assert tf.as_dtype(box.lower.dtype) is tf.float64
    assert tf.as_dtype(box.upper.dtype) is tf.float64
    npt.assert_array_equal(box.lower, [0.0, 1.0])
    npt.assert_array_equal(box.upper, [1.0, 2.0])


def _pairs_of_shapes(
    *, excluding_ranks: Container[int] = ()
) -> frozenset[tuple[ShapeLike, ShapeLike]]:
    shapes = various_shapes(excluding_ranks=excluding_ranks)
    return frozenset(itertools.product(shapes, shapes))


@pytest.mark.parametrize(
    "lower_shape, upper_shape", _pairs_of_shapes(excluding_ranks={1}) | {((1,), (2,)), ((0,), (0,))}
)
def test_box_raises_if_bounds_have_invalid_shape(
    lower_shape: ShapeLike, upper_shape: ShapeLike
) -> None:
    lower, upper = tf.zeros(lower_shape), tf.ones(upper_shape)

    if lower_shape == upper_shape == (0,):
        Box(lower, upper)  # empty box is ok
    else:
        with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
            Box(lower, upper)


def test_box___mul___for_empty_search_space() -> None:
    empty = Box(tf.zeros(0, dtype=tf.float64), tf.zeros(0, dtype=tf.float64))
    cube = Box([0, 0, 0], [1, 1, 1])
    npt.assert_array_equal((cube * empty).lower, cube.lower)
    npt.assert_array_equal((cube * empty).upper, cube.upper)
    npt.assert_array_equal((empty * cube).lower, cube.lower)
    npt.assert_array_equal((empty * cube).upper, cube.upper)


@pytest.mark.parametrize(
    "lower_dtype, upper_dtype",
    [
        (tf.uint32, tf.uint32),  # same dtypes
        (tf.int8, tf.uint16),  # different dtypes ...
        (tf.uint32, tf.float32),
        (tf.float32, tf.float64),
        (tf.float64, tf.bfloat16),
    ],
)
def test_box_raises_if_bounds_have_invalid_dtypes(
    lower_dtype: tf.DType, upper_dtype: tf.DType
) -> None:
    lower, upper = tf.zeros([3], dtype=lower_dtype), tf.ones([3], dtype=upper_dtype)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        Box(lower, upper)


@pytest.mark.parametrize(
    "lower, upper",
    [
        (tf.ones((3,)) + 1, tf.ones((3,))),  # lower all higher than upper
        (  # one lower higher than upper
            tf.constant([2.3, -0.1, 8.0]),
            tf.constant([3.0, -0.2, 8.0]),
        ),
    ],
)
def test_box_raises_if_any_lower_bound_is_not_less_than_upper_bound(
    lower: tf.Tensor, upper: tf.Tensor
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        Box(lower, upper)


@pytest.mark.parametrize(
    "space, dimension",
    [
        (Box([-1], [2]), 1),  # 1d
        (Box([-1], [-1]), 1),  # still 1d (though width 0)
        (Box([-1, -2], [1.5, 2.5]), 2),  # 2d
        (Box([-1, -2, -3], [1.5, 2.5, 3.5]), 3),  # 3d
    ],
)
def test_box_returns_correct_dimension(space: Box, dimension: int) -> None:
    assert space.dimension == dimension


def test_box_bounds_attributes() -> None:
    lower, upper = tf.zeros([2]), tf.ones([2])
    box = Box(lower, upper)
    assert box.has_bounds
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
    box = Box(tf.constant([-1.0, 0.0, -2.0]), tf.constant([2.0, 1.0, -0.5]))
    assert point in box
    assert box.contains(point)


@pytest.mark.parametrize(
    "lower,upper",
    [
        (tf.constant([]), tf.constant([])),
        (tf.constant([0.0]), tf.constant([0.0])),
        (tf.constant([-1.0, 0.0, -2.0]), tf.constant([2.0, 1.0, -0.5])),
        (tf.constant([-1.0, 0.0, -2.0]), tf.constant([2.0, 1.0, -2.0])),
    ],
)
def test_box_with_zero_width(lower: tf.Tensor, upper: tf.Tensor) -> None:
    box = Box(lower, upper)
    assert lower in box
    assert upper in box
    assert (lower + upper) / 2 in box
    if box.dimension > 0:
        assert lower - 1 not in box
        assert upper + 1 not in box
    assert tf.reduce_all(box.contains(box.sample(10)))


@pytest.mark.parametrize(
    "point",
    [
        tf.constant([-1.1, 0.0, -2.0]),  # just outside
        tf.constant([-0.5, -0.5, 1.5]),  # negative of a contained point
        tf.constant([10.0, -10.0, 10.0]),  # well outside
    ],
)
def test_box_does_not_contain_point(point: tf.Tensor) -> None:
    box = Box(tf.constant([-1.0, 0.0, -2.0]), tf.constant([2.0, 1.0, -0.5]))
    assert point not in box
    assert not box.contains(point)


@pytest.mark.parametrize(
    "points, contains",
    [
        (tf.constant([[-1.0, 0.0, -2.0], [-1.1, 0.0, -2.0]]), tf.constant([True, False])),
        (tf.constant([[[0.5, 0.5, -1.5]]]), tf.constant([[True]])),
    ],
)
def test_box_contains_broadcasts(points: tf.Tensor, contains: tf.Tensor) -> None:
    box = Box(tf.constant([-1.0, 0.0, -2.0]), tf.constant([2.0, 1.0, -0.5]))
    npt.assert_array_equal(contains, box.contains(points))
    # point in space raises (because python insists on a bool)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        _ = points in box


@pytest.mark.parametrize(
    "bound_shape, point_shape",
    (
        (bs, ps)
        for bs, ps in _pairs_of_shapes()
        if bs[-1:] != ps[-1:] and len(bs) == 1 and bs != (0,)
    ),
)
def test_box_contains_raises_on_point_of_different_shape(
    bound_shape: ShapeLike,
    point_shape: ShapeLike,
) -> None:
    box = Box(tf.zeros(bound_shape), tf.ones(bound_shape))
    point = tf.zeros(point_shape)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        _ = point in box
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        _ = box.contains(point)


def _assert_correct_number_of_unique_constrained_samples(
    num_samples: int, search_space: SearchSpace, samples: tf.Tensor
) -> None:
    assert all(sample in search_space for sample in samples)
    assert len(samples) == num_samples

    unique_samples = set(tuple(sample.numpy().tolist()) for sample in samples)

    assert len(unique_samples) == len(samples)


def _box_sampling_constraints() -> Sequence[LinearConstraint]:
    return [LinearConstraint(A=tf.eye(3), lb=tf.zeros((3)) + 0.3, ub=tf.ones((3)) - 0.3)]


@pytest.mark.parametrize("num_samples", [0, 1, 10])
@pytest.mark.parametrize("constraints", [None, _box_sampling_constraints()])
def test_box_sampling_returns_correct_shape(
    num_samples: int,
    constraints: Sequence[LinearConstraint],
) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), constraints)
    samples = box.sample_feasible(num_samples)
    _assert_correct_number_of_unique_constrained_samples(num_samples, box, samples)


@pytest.mark.parametrize("num_samples", [0, 1, 10])
@pytest.mark.parametrize("constraints", [None, _box_sampling_constraints()])
def test_box_sobol_sampling_returns_correct_shape(
    num_samples: int,
    constraints: Sequence[LinearConstraint],
) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), constraints)
    sobol_samples = box.sample_sobol_feasible(num_samples)
    _assert_correct_number_of_unique_constrained_samples(num_samples, box, sobol_samples)


@pytest.mark.parametrize("num_samples", [0, 1, 10])
@pytest.mark.parametrize("constraints", [None, _box_sampling_constraints()])
def test_box_halton_sampling_returns_correct_shape(
    num_samples: int,
    constraints: Sequence[LinearConstraint],
) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), constraints)
    halton_samples = box.sample_halton_feasible(num_samples)
    _assert_correct_number_of_unique_constrained_samples(num_samples, box, halton_samples)


@pytest.mark.parametrize("num_samples", [-1, -10])
@pytest.mark.parametrize("constraints", [None, _box_sampling_constraints()])
def test_box_sampling_raises_for_invalid_sample_size(
    num_samples: int,
    constraints: Sequence[LinearConstraint],
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        box = Box(tf.zeros((3,)), tf.ones((3,)), constraints)
        box.sample_feasible(num_samples)


@pytest.mark.parametrize("num_samples", [-1, -10])
@pytest.mark.parametrize("constraints", [None, _box_sampling_constraints()])
def test_box_sobol_sampling_raises_for_invalid_sample_size(
    num_samples: int,
    constraints: Sequence[LinearConstraint],
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        box = Box(tf.zeros((3,)), tf.ones((3,)), constraints)
        box.sample_sobol_feasible(num_samples)


@pytest.mark.parametrize("num_samples", [-1, -10])
@pytest.mark.parametrize("constraints", [None, _box_sampling_constraints()])
def test_box_halton_sampling_raises_for_invalid_sample_size(
    num_samples: int,
    constraints: Sequence[LinearConstraint],
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        box = Box(tf.zeros((3,)), tf.ones((3,)), constraints)
        box.sample_halton_feasible(num_samples)


@pytest.mark.parametrize("seed", [1, 42, 123])
@pytest.mark.parametrize("constraints", [None, _box_sampling_constraints()])
def test_box_sampling_returns_same_points_for_same_seed(
    seed: int,
    constraints: Sequence[LinearConstraint],
) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), constraints)
    random_samples_1 = box.sample_feasible(num_samples=100, seed=seed)
    random_samples_2 = box.sample_feasible(num_samples=100, seed=seed)
    npt.assert_allclose(random_samples_1, random_samples_2)


@pytest.mark.parametrize("skip", [1, 10, 100])
@pytest.mark.parametrize("constraints", [None, _box_sampling_constraints()])
def test_box_sobol_sampling_returns_same_points_for_same_skip(
    skip: int,
    constraints: Sequence[LinearConstraint],
) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), constraints)
    sobol_samples_1 = box.sample_sobol_feasible(num_samples=100, skip=skip)
    sobol_samples_2 = box.sample_sobol_feasible(num_samples=100, skip=skip)
    npt.assert_allclose(sobol_samples_1, sobol_samples_2)


@pytest.mark.parametrize("seed", [1, 42, 123])
@pytest.mark.parametrize("constraints", [None, _box_sampling_constraints()])
def test_box_halton_sampling_returns_same_points_for_same_seed(
    seed: int,
    constraints: Sequence[LinearConstraint],
) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), constraints)
    halton_samples_1 = box.sample_halton_feasible(num_samples=100, seed=seed)
    halton_samples_2 = box.sample_halton_feasible(num_samples=100, seed=seed)
    npt.assert_allclose(halton_samples_1, halton_samples_2)


@pytest.mark.parametrize("constraints", [None, _box_sampling_constraints()])
def test_box_sampling_returns_different_points_for_different_call(
    constraints: Sequence[LinearConstraint],
) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), constraints)
    random_samples_1 = box.sample_feasible(num_samples=100)
    random_samples_2 = box.sample_feasible(num_samples=100)
    npt.assert_raises(AssertionError, npt.assert_allclose, random_samples_1, random_samples_2)


@pytest.mark.parametrize("constraints", [None, _box_sampling_constraints()])
def test_box_sobol_sampling_returns_different_points_for_different_call(
    constraints: Sequence[LinearConstraint],
) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), constraints)
    sobol_samples_1 = box.sample_sobol_feasible(num_samples=100)
    sobol_samples_2 = box.sample_sobol_feasible(num_samples=100)
    npt.assert_raises(AssertionError, npt.assert_allclose, sobol_samples_1, sobol_samples_2)


@pytest.mark.parametrize("constraints", [None, _box_sampling_constraints()])
def test_box_halton_sampling_returns_different_points_for_different_call(
    constraints: Sequence[LinearConstraint],
) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), constraints)
    halton_samples_1 = box.sample_halton_feasible(num_samples=100)
    halton_samples_2 = box.sample_halton_feasible(num_samples=100)
    npt.assert_raises(AssertionError, npt.assert_allclose, halton_samples_1, halton_samples_2)


def test_box_sampling_with_constraints_returns_feasible_points() -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), _box_sampling_constraints())
    samples = box.sample_feasible(num_samples=100)
    assert all(box.is_feasible(samples))


def test_box_sobol_sampling_with_constraints_returns_feasible_points() -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), _box_sampling_constraints())
    samples = box.sample_sobol_feasible(num_samples=100)
    assert all(box.is_feasible(samples))


def test_box_halton_sampling_with_constraints_returns_feasible_points() -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)), _box_sampling_constraints())
    samples = box.sample_halton_feasible(num_samples=100)
    assert all(box.is_feasible(samples))


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


def test_box___mul___bounds_are_the_concatenation_of_original_bounds() -> None:
    box1 = Box(tf.constant([0.0, 1.0]), tf.constant([2.0, 3.0]))
    box2 = Box(tf.constant([4.1, 5.1, 6.1]), tf.constant([7.2, 8.2, 9.2]))

    product = box1 * box2

    npt.assert_allclose(product.lower, [0, 1, 4.1, 5.1, 6.1])
    npt.assert_allclose(product.upper, [2, 3, 7.2, 8.2, 9.2])


def test_box___mul___raises_if_bounds_have_different_types() -> None:
    box1 = Box(tf.constant([0.0, 1.0]), tf.constant([2.0, 3.0]))
    box2 = Box(tf.constant([4.0, 5.0], tf.float64), tf.constant([6.0, 7.0], tf.float64))

    with pytest.raises(TypeError):
        _ = box1 * box2


def test_box_deepcopy() -> None:
    box = Box(tf.constant([1.2, 3.4]), tf.constant([5.6, 7.8]))
    box_copy = copy.deepcopy(box)
    npt.assert_allclose(box.lower, box_copy.lower)
    npt.assert_allclose(box.upper, box_copy.upper)


@pytest.fixture(name="search_space_type", params=[TaggedMultiSearchSpace, TaggedProductSearchSpace])
def _default_search_space_type_fixture(request: Any) -> Type[CollectionSearchSpace]:
    return request.param


def test_collection_space_raises_for_non_unqique_subspace_names(
    search_space_type: Type[CollectionSearchSpace],
) -> None:
    space_A = Box([-1, -2], [2, 3])
    space_B = DiscreteSearchSpace(tf.constant([[-0.5, 0.5]]))
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES, match="Subspace names must be unique"):
        search_space_type(spaces=[space_A, space_B], tags=["A", "A"])


def test_collection_space_raises_for_length_mismatch_between_spaces_and_tags(
    search_space_type: Type[CollectionSearchSpace],
) -> None:
    space_A = Box([-1, -2], [2, 3])
    space_B = DiscreteSearchSpace(tf.constant([[-0.5, 0.5]]))
    with pytest.raises(
        TF_DEBUGGING_ERROR_TYPES, match="Number of tags must match number of subspaces"
    ):
        search_space_type(spaces=[space_A, space_B], tags=["A", "B", "C"])


def test_collection_space_subspace_tags_attribute(
    search_space_type: Type[CollectionSearchSpace],
) -> None:
    decision_space = Box([-1, -2], [2, 3])
    context_space = DiscreteSearchSpace(tf.constant([[-0.5, 0.5]]))
    multi_space = search_space_type(
        spaces=[context_space, decision_space], tags=["context", "decision"]
    )

    npt.assert_array_equal(multi_space.subspace_tags, ["context", "decision"])


def test_collection_space_subspace_tags_default_behaviour(
    search_space_type: Type[CollectionSearchSpace],
) -> None:
    decision_space = Box([-1, -2], [2, 3])
    context_space = DiscreteSearchSpace(tf.constant([[-0.5, 0.5]]))
    multi_space = search_space_type(spaces=[context_space, decision_space])

    npt.assert_array_equal(multi_space.subspace_tags, ["0", "1"])


def test_collection_space_get_subspace_raises_for_invalid_tag(
    search_space_type: Type[CollectionSearchSpace],
) -> None:
    space_A = Box([-1, -2], [2, 3])
    space_B = DiscreteSearchSpace(tf.constant([[-0.5, 0.5]]))
    multi_space = search_space_type(spaces=[space_A, space_B], tags=["A", "B"])

    with pytest.raises(
        TF_DEBUGGING_ERROR_TYPES, match="Attempted to access a subspace that does not exist"
    ):
        multi_space.get_subspace("dummy")


def test_collection_space_get_subspace(search_space_type: Type[CollectionSearchSpace]) -> None:
    space_A = Box([-1, -2], [2, 3])
    space_B = DiscreteSearchSpace(tf.constant([[-0.5, 0.5]]))
    space_C = Box([-1, -3], [2, 2])
    multi_space = search_space_type(spaces=[space_A, space_B, space_C], tags=["A", "B", "C"])

    subspace_A = multi_space.get_subspace("A")
    assert isinstance(subspace_A, Box)
    npt.assert_array_equal(subspace_A.lower, [-1, -2])
    npt.assert_array_equal(subspace_A.upper, [2, 3])

    subspace_B = multi_space.get_subspace("B")
    assert isinstance(subspace_B, DiscreteSearchSpace)
    npt.assert_array_equal(subspace_B.points, tf.constant([[-0.5, 0.5]]))

    subspace_C = multi_space.get_subspace("C")
    assert isinstance(subspace_C, Box)
    npt.assert_array_equal(subspace_C.lower, [-1, -3])
    npt.assert_array_equal(subspace_C.upper, [2, 2])


@pytest.mark.parametrize(
    "search_space_type, point",
    [
        (TaggedMultiSearchSpace, tf.constant([-1.0, 0.0], dtype=tf.float64)),
        (TaggedMultiSearchSpace, tf.constant([2.0, -2.0], dtype=tf.float64)),
        (TaggedMultiSearchSpace, tf.constant([-0.5, 0.5], dtype=tf.float64)),
        (TaggedProductSearchSpace, tf.constant([-1.0, 0.0, -0.5, 0.5], dtype=tf.float64)),
        (TaggedProductSearchSpace, tf.constant([2.0, -2.0, -0.5, 0.5], dtype=tf.float64)),
    ],
)
def test_collection_space_contains_point(
    search_space_type: Type[CollectionSearchSpace],
    point: tf.Tensor,
) -> None:
    space_A = Box([-1.0, -2.0], [2.0, 0.0])
    space_B = DiscreteSearchSpace(tf.constant([[-0.5, 0.5]], dtype=tf.float64))
    collection_space = search_space_type(spaces=[space_A, space_B])
    assert point in collection_space
    assert collection_space.contains(point)


@pytest.mark.parametrize(
    "search_space_type, point",
    [
        pytest.param(
            TaggedMultiSearchSpace,
            tf.constant([-1.1, 0.0], dtype=tf.float64),
            id="just outside box",
        ),
        pytest.param(
            TaggedMultiSearchSpace,
            tf.constant([-10, 10.0], dtype=tf.float64),
            id="well outside box",
        ),
        pytest.param(
            TaggedMultiSearchSpace,
            tf.constant([-0.5, 0.51], dtype=tf.float64),
            id="just outside discrete",
        ),
        pytest.param(
            TaggedProductSearchSpace,
            tf.constant([-1.1, 0.0, -0.5, 0.5], dtype=tf.float64),
            id="just outside context space",
        ),
        pytest.param(
            TaggedProductSearchSpace,
            tf.constant([-10, 10.0, -0.5, 0.5], dtype=tf.float64),
            id="well outside context space",
        ),
        pytest.param(
            TaggedProductSearchSpace,
            tf.constant([2.0, 0.0, 2.0, 7.0], dtype=tf.float64),
            id="outside decision space",
        ),
        pytest.param(
            TaggedProductSearchSpace,
            tf.constant([-10.0, -10.0, -10.0, -10.0], dtype=tf.float64),
            id="outside both",
        ),
        pytest.param(
            TaggedProductSearchSpace,
            tf.constant([-0.5, 0.5, 1.0, 0.0], dtype=tf.float64),
            id="swap order of components",
        ),
    ],
)
def test_collection_space_does_not_contain_point(
    search_space_type: Type[CollectionSearchSpace],
    point: tf.Tensor,
) -> None:
    space_A = Box([-1.0, -2.0], [2.0, 0.0])
    space_B = DiscreteSearchSpace(tf.constant([[-0.5, 0.5]], dtype=tf.float64))
    collection_space = search_space_type(spaces=[space_A, space_B])
    assert point not in collection_space
    assert not collection_space.contains(point)


@pytest.mark.parametrize(
    "search_space_type, points",
    [
        (TaggedMultiSearchSpace, tf.constant([[-1.1, 0.0], [-0.5, 0.5]], dtype=tf.float64)),
        (
            TaggedProductSearchSpace,
            tf.constant([[-1.1, 0.0, -0.5, 0.5], [-1.0, 0.0, -0.5, 0.5]], dtype=tf.float64),
        ),
    ],
)
def test_collection_space_contains_broadcasts(
    search_space_type: Type[CollectionSearchSpace],
    points: tf.Tensor,
) -> None:
    space_A = Box([-1.0, -2.0], [2.0, 3.0])
    space_B = DiscreteSearchSpace(tf.constant([[-0.5, 0.5]], dtype=tf.float64))
    collection_space = search_space_type(spaces=[space_A, space_B])
    tf.assert_equal(collection_space.contains(points), [False, True])
    # point in space raises (because python insists on a bool)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        _ = points in collection_space


@pytest.mark.parametrize(
    "spaces",
    [
        [DiscreteSearchSpace(tf.constant([[-0.5]]))],
        [
            DiscreteSearchSpace(tf.constant([[-0.5, 1.3]])),
            DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [1.2, 0.4]])),
        ],
        [
            Box([-2], [3]),
            DiscreteSearchSpace(tf.constant([[-0.5]])),
            Box([-1], [2]),
        ],
    ],
)
def test_collection_space_contains_raises_on_point_of_different_shape(
    search_space_type: Type[CollectionSearchSpace],
    spaces: Sequence[SearchSpace],
) -> None:
    space = search_space_type(spaces=spaces)
    dimension = space.dimension
    for wrong_input_shape in [dimension - 1, dimension + 1]:
        point = tf.zeros([wrong_input_shape])
        with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
            _ = point in space
        with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
            _ = space.contains(point)


@pytest.mark.parametrize(
    "search_space_type, space_A, exp_shape_tail",
    [
        (TaggedMultiSearchSpace, Box([-1, -2], [2, 3]), [2, 2]),
        (TaggedProductSearchSpace, Box([-1], [2]), [3]),
        (TaggedProductSearchSpace, CategoricalSearchSpace(["A", "B", "C"]), [3]),
    ],
)
@pytest.mark.parametrize("num_samples", [0, 1, 10])
def test_collection_space_sampling_returns_correct_shape(
    search_space_type: Type[CollectionSearchSpace],
    space_A: SearchSpace,
    exp_shape_tail: List[int],
    num_samples: int,
) -> None:
    space_B = DiscreteSearchSpace(tf.ones([100, 2], dtype=tf.float64))
    if search_space_type is TaggedMultiSearchSpace:
        product: SearchSpace = TaggedMultiSearchSpace([space_A]) * TaggedMultiSearchSpace([space_B])
    else:
        product = space_A * space_B
    for collection_space in (search_space_type(spaces=[space_A, space_B]), product):
        samples = collection_space.sample(num_samples)
        npt.assert_array_equal(tf.shape(samples), [num_samples] + exp_shape_tail)


@pytest.mark.parametrize(
    "search_space_type, space_A",
    [
        (TaggedMultiSearchSpace, Box([-1, -2], [2, 3])),
        (TaggedProductSearchSpace, Box([-1], [2])),
    ],
)
@pytest.mark.parametrize("num_samples", [-1, -10])
def test_collection_space_sampling_raises_for_invalid_sample_size(
    search_space_type: Type[CollectionSearchSpace],
    space_A: SearchSpace,
    num_samples: int,
) -> None:
    space_B = DiscreteSearchSpace(tf.ones([100, 2], dtype=tf.float64))
    collection_space = search_space_type(spaces=[space_A, space_B])
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        collection_space.sample(num_samples)


@pytest.mark.parametrize("search_space_type", [TaggedMultiSearchSpace, TaggedProductSearchSpace])
def test_collection_space_sampling_uses_different_seeds_for_subspaces(
    search_space_type: Type[CollectionSearchSpace],
) -> None:
    box = Box([0], [1])
    collection_space = search_space_type(spaces=[box, box])
    samples = collection_space.sample(5, seed=42)
    # check that all the points are unique despite the seed
    flattened = samples.numpy().flatten()
    assert len(flattened) == len(set(flattened))


@pytest.mark.parametrize(
    "search_space_type, space_A",
    [
        (TaggedMultiSearchSpace, Box([-1, -2], [2, 3])),
        (TaggedProductSearchSpace, Box([-1], [2])),
    ],
)
@pytest.mark.parametrize("num_samples", [0, 1, 10])
def test_collection_space_discretize_returns_search_space_with_only_points_contained_within_box(
    search_space_type: Type[CollectionSearchSpace],
    space_A: SearchSpace,
    num_samples: int,
) -> None:
    space_B = DiscreteSearchSpace(tf.ones([100, 2], dtype=tf.float64))
    collection_space = search_space_type(spaces=[space_A, space_B])

    dss = collection_space.discretize(num_samples)
    samples = dss.sample(num_samples)

    assert all(sample in collection_space for sample in samples)


@pytest.mark.parametrize(
    "search_space_type, space_A",
    [
        (TaggedMultiSearchSpace, Box([-1, -2], [2, 3])),
        (TaggedProductSearchSpace, Box([-1], [2])),
    ],
)
@pytest.mark.parametrize("num_samples", [0, 1, 10])
def test_collection_space_discretize_returns_search_space_with_correct_number_of_points(
    search_space_type: Type[CollectionSearchSpace],
    space_A: SearchSpace,
    num_samples: int,
) -> None:
    space_B = DiscreteSearchSpace(tf.ones([100, 2], dtype=tf.float64))
    collection_space = search_space_type(spaces=[space_A, space_B])

    dss = collection_space.discretize(num_samples)
    samples = dss.sample(num_samples)

    assert len(samples) == num_samples


@pytest.mark.parametrize(
    "search_space_type, space_A",
    [
        (TaggedMultiSearchSpace, Box([-1, -2], [2, 3])),
        (TaggedProductSearchSpace, Box([-1], [2])),
    ],
)
@pytest.mark.parametrize("seed", [1, 42, 123])
def test_collection_space_sampling_returns_same_points_for_same_seed(
    search_space_type: Type[CollectionSearchSpace], space_A: SearchSpace, seed: int
) -> None:
    space_B = DiscreteSearchSpace(tf.random.uniform([100, 2], dtype=tf.float64, seed=42))
    collection_space = search_space_type(spaces=[space_A, space_B])
    random_samples_1 = collection_space.sample(num_samples=100, seed=seed)
    random_samples_2 = collection_space.sample(num_samples=100, seed=seed)
    npt.assert_allclose(random_samples_1, random_samples_2)


@pytest.mark.parametrize(
    "search_space_type, space_A",
    [
        (TaggedMultiSearchSpace, Box([-1, -2], [2, 3])),
        (TaggedProductSearchSpace, Box([-1], [2])),
    ],
)
def test_collection_space_sampling_returns_different_points_for_different_call(
    search_space_type: Type[CollectionSearchSpace],
    space_A: SearchSpace,
) -> None:
    space_B = DiscreteSearchSpace(tf.random.uniform([100, 2], dtype=tf.float64, seed=42))
    collection_space = search_space_type(spaces=[space_A, space_B])
    random_samples_1 = collection_space.sample(num_samples=100)
    random_samples_2 = collection_space.sample(num_samples=100)
    npt.assert_raises(AssertionError, npt.assert_allclose, random_samples_1, random_samples_2)


@pytest.mark.parametrize(
    "search_space_type, space_A",
    [
        (TaggedMultiSearchSpace, Box([-1, -2], [2, 3])),
        (TaggedProductSearchSpace, Box([-1], [2])),
    ],
)
def test_collection_space_deepcopy(
    search_space_type: Type[CollectionSearchSpace],
    space_A: SearchSpace,
) -> None:
    space_B = DiscreteSearchSpace(tf.ones([100, 2], dtype=tf.float64))
    collection_space = search_space_type(spaces=[space_A, space_B], tags=["A", "B"])

    copied_space = copy.deepcopy(collection_space)
    npt.assert_allclose(copied_space.get_subspace("A").lower, space_A.lower)
    npt.assert_allclose(copied_space.get_subspace("A").upper, space_A.upper)
    npt.assert_allclose(copied_space.get_subspace("B").points, space_B.points)  # type: ignore


@pytest.mark.parametrize(
    "spaces, dimension",
    [
        ([DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [1.2, 0.4]]))], 2),
        ([DiscreteSearchSpace(tf.constant([[-0.5]])), Box([-1], [2])], 2),
        ([Box([-1, -2], [2, 3]), DiscreteSearchSpace(tf.constant([[-0.5]]))], 3),
        ([Box([-1, -2], [2, 3]), Box([-1, -2], [2, 3]), Box([-1], [2])], 5),
    ],
)
def test_product_space_returns_correct_dimension(
    spaces: Sequence[SearchSpace], dimension: int
) -> None:
    for space in (TaggedProductSearchSpace(spaces=spaces), reduce(operator.mul, spaces)):
        assert space.dimension == dimension


@pytest.mark.parametrize(
    "spaces, lower, upper",
    [
        (
            [DiscreteSearchSpace(tf.constant([[-0.5, 0.4], [1.2, -0.3]]))],
            tf.constant([-0.5, -0.3]),
            tf.constant([1.2, 0.4]),
        ),
        (
            [DiscreteSearchSpace(tf.constant([[-0.5]], dtype=tf.float64)), Box([-1.0], [2.0])],
            tf.constant([-0.5, -1.0]),
            tf.constant([-0.5, 2.0]),
        ),
        (
            [Box([-1, -2], [2, 3]), DiscreteSearchSpace(tf.constant([[-0.5]], dtype=tf.float64))],
            tf.constant([-1.0, -2.0, -0.5]),
            tf.constant([2.0, 3.0, -0.5]),
        ),
        (
            [Box([-1, -2], [2, 3]), Box([-1, -2], [2, 3]), Box([-1], [2])],
            tf.constant([-1.0, -2.0, -1.0, -2.0, -1.0]),
            tf.constant([2.0, 3.0, 2.0, 3.0, 2.0]),
        ),
    ],
)
def test_product_space_returns_correct_bounds(
    spaces: Sequence[SearchSpace], lower: tf.Tensor, upper: tf.Tensor
) -> None:
    for space in (TaggedProductSearchSpace(spaces=spaces), reduce(operator.mul, spaces)):
        assert space.has_bounds
        npt.assert_array_equal(space.lower, lower)
        npt.assert_array_equal(space.upper, upper)


@pytest.mark.parametrize(
    "points",
    [
        tf.ones((1, 5), dtype=tf.float64),
        tf.ones((2, 3), dtype=tf.float64),
    ],
)
def test_product_space_fix_subspace_fixes_desired_subspace(points: tf.Tensor) -> None:
    spaces = [
        Box([-1, -2], [2, 3]),
        DiscreteSearchSpace(tf.constant([[-0.5]], dtype=tf.float64)),
        Box([-1], [2]),
    ]
    tags = ["A", "B", "C"]
    product_space = TaggedProductSearchSpace(spaces=spaces, tags=tags)

    for tag in tags:
        product_space_with_fixed_subspace = product_space.fix_subspace(tag, points)
        new_subspace = product_space_with_fixed_subspace.get_subspace(tag)
        assert isinstance(new_subspace, DiscreteSearchSpace)
        npt.assert_array_equal(new_subspace.points, points)


@pytest.mark.parametrize(
    "points",
    [
        tf.ones((1, 5), dtype=tf.float64),
        tf.ones((2, 3), dtype=tf.float64),
    ],
)
def test_product_space_fix_subspace_doesnt_fix_undesired_subspace(points: tf.Tensor) -> None:
    spaces = [
        Box([-1, -2], [2, 3]),
        DiscreteSearchSpace(tf.constant([[-0.5]], dtype=tf.float64)),
        Box([-1], [2]),
    ]
    tags = ["A", "B", "C"]
    product_space = TaggedProductSearchSpace(spaces=spaces, tags=tags)

    for tag in tags:
        product_space_with_fixed_subspace = product_space.fix_subspace(tag, points)
        for other_tag in tags:
            if other_tag != tag:
                assert isinstance(
                    product_space_with_fixed_subspace.get_subspace(other_tag),
                    type(product_space.get_subspace(other_tag)),
                )


@pytest.mark.parametrize(
    "spaces, tags, subspace_dim_range",
    [
        ([DiscreteSearchSpace(tf.constant([[-0.5]]))], ["A"], {"A": [0, 1]}),
        (
            [
                DiscreteSearchSpace(tf.constant([[-0.5]])),
                DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [1.2, 0.4]])),
            ],
            ["A", "B"],
            {"A": [0, 1], "B": [1, 3]},
        ),
        (
            [
                Box([-1, -2], [2, 3]),
                DiscreteSearchSpace(tf.constant([[-0.5]])),
                CategoricalSearchSpace([3, 2]),
            ],
            ["A", "B", "C"],
            {"A": [0, 2], "B": [2, 3], "C": [3, 5]},
        ),
    ],
)
def test_product_space_can_get_subspace_components(
    spaces: list[SearchSpace],
    tags: list[str],
    subspace_dim_range: dict[str, list[int]],
) -> None:
    space = TaggedProductSearchSpace(spaces, tags)
    points = tf.random.uniform([10, space.dimension])

    for tag in space.subspace_tags:
        subspace_points = points[:, subspace_dim_range[tag][0] : subspace_dim_range[tag][1]]
        npt.assert_array_equal(space.get_subspace_component(tag, points), subspace_points)


def test_product_space___mul___() -> None:
    space_A = Box([-1], [2])
    space_B = DiscreteSearchSpace(tf.ones([100, 2], dtype=tf.float64))
    product_space_1 = TaggedProductSearchSpace(spaces=[space_A, space_B], tags=["A", "B"])

    space_C = Box([-2, -2], [2, 3])
    space_D = DiscreteSearchSpace(tf.ones([5, 3], dtype=tf.float64))
    product_space_2 = TaggedProductSearchSpace(spaces=[space_C, space_D], tags=["C", "D"])

    product_of_product_spaces = product_space_1 * product_space_2

    subspace_0 = product_of_product_spaces.get_subspace("0")
    subspace_0_A = subspace_0.get_subspace("A")  # type: ignore
    assert isinstance(subspace_0_A, Box)
    npt.assert_array_equal(subspace_0_A.lower, [-1])
    npt.assert_array_equal(subspace_0_A.upper, [2])
    subspace_0_B = subspace_0.get_subspace("B")  # type: ignore
    assert isinstance(subspace_0_B, DiscreteSearchSpace)
    npt.assert_array_equal(subspace_0_B.points, tf.ones([100, 2], dtype=tf.float64))

    subspace_1 = product_of_product_spaces.get_subspace("1")
    subspace_1_C = subspace_1.get_subspace("C")  # type: ignore
    assert isinstance(subspace_1_C, Box)
    npt.assert_array_equal(subspace_1_C.lower, [-2, -2])
    npt.assert_array_equal(subspace_1_C.upper, [2, 3])
    subspace_1_D = subspace_1.get_subspace("D")  # type: ignore
    assert isinstance(subspace_1_D, DiscreteSearchSpace)
    npt.assert_array_equal(subspace_1_D.points, tf.ones([5, 3], dtype=tf.float64))


def test_product_space_handles_empty_spaces() -> None:
    space_A = Box([-1, -2], [2, 3])
    tag_A = TaggedProductSearchSpace(spaces=[space_A], tags=["A"])
    tag_B = TaggedProductSearchSpace(spaces=[], tags=[])
    tag_C = TaggedProductSearchSpace(spaces=[tag_A, tag_B], tags=["AA", "BB"])

    assert tag_C.dimension == 2
    npt.assert_array_equal(tag_C.lower, [-1, -2])
    npt.assert_array_equal(tag_C.upper, [2, 3])
    npt.assert_array_equal(tag_C.subspace_tags, ["AA", "BB"])


def test_multi_space_raises_for_empty_collection() -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES, match="At least one subspace is required"):
        TaggedMultiSearchSpace([])


@pytest.mark.parametrize(
    "spaces",
    [
        ([Box([-1], [2]), Box([-1, -2], [2, 3])]),
        ([Box([-1, -2], [2, 3]), DiscreteSearchSpace(tf.constant([[-0.5]]))]),
    ],
)
def test_multi_space_raises_for_mixed_dimensions(spaces: Sequence[SearchSpace]) -> None:
    with pytest.raises(
        TF_DEBUGGING_ERROR_TYPES, match="All subspaces must have the same dimension"
    ):
        TaggedMultiSearchSpace(spaces)


@pytest.mark.parametrize(
    "spaces, dimension",
    [
        ([DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [1.2, 0.4]]))], 2),
        ([DiscreteSearchSpace(tf.constant([[-0.5]])), Box([-1], [2])], 1),
        ([Box([-1, -2], [2, 3]), DiscreteSearchSpace(tf.constant([[-0.5, -1.5]]))], 2),
        ([Box([-1, -2], [2, 3]), Box([-1, -2], [2, 3]), Box([-1, 1], [2, 2])], 2),
    ],
)
def test_multi_space_returns_correct_dimension(
    spaces: Sequence[SearchSpace], dimension: int
) -> None:
    for space in (
        TaggedMultiSearchSpace(spaces=spaces),
        reduce(operator.mul, [TaggedMultiSearchSpace([s]) for s in spaces]),
    ):
        assert space.dimension == dimension


@pytest.mark.parametrize(
    "spaces, lower, upper",
    [
        (
            [DiscreteSearchSpace(tf.constant([[-0.5, 0.4], [1.2, -0.3]]))],
            tf.constant([[-0.5, -0.3]]),
            tf.constant([[1.2, 0.4]]),
        ),
        (
            [DiscreteSearchSpace(tf.constant([[-0.5]], dtype=tf.float64)), Box([-1.0], [2.0])],
            tf.constant([[-0.5], [-1.0]]),
            tf.constant([[-0.5], [2.0]]),
        ),
        (
            [
                Box([-1, -2], [2, 3]),
                DiscreteSearchSpace(tf.constant([[-0.5, 1.5]], dtype=tf.float64)),
            ],
            tf.constant([[-1.0, -2.0], [-0.5, 1.5]]),
            tf.constant([[2.0, 3.0], [-0.5, 1.5]]),
        ),
        (
            [Box([-1, -2], [2, 3]), Box([-1, -2], [2, 3]), Box([-1, 1], [2, 2])],
            tf.constant([[-1.0, -2.0], [-1.0, -2.0], [-1.0, 1.0]]),
            tf.constant([[2.0, 3.0], [2.0, 3.0], [2.0, 2.0]]),
        ),
    ],
)
def test_multi_space_returns_correct_bounds(
    spaces: Sequence[SearchSpace], lower: tf.Tensor, upper: tf.Tensor
) -> None:
    for space in (
        TaggedMultiSearchSpace(spaces=spaces),
        reduce(operator.mul, [TaggedMultiSearchSpace([s]) for s in spaces]),
    ):
        npt.assert_array_equal(space.lower, lower)
        npt.assert_array_equal(space.upper, upper)


def test_multi_space___mul___() -> None:
    space_A = Box([-1, -2], [2, 3])
    space_B = DiscreteSearchSpace(tf.ones([100, 2], dtype=tf.float64))
    product_space_1 = TaggedMultiSearchSpace(spaces=[space_A, space_B], tags=["A", "B"])

    space_C = Box([-2, -2], [2, 3])
    space_D = DiscreteSearchSpace(tf.ones([5, 2], dtype=tf.float64))
    product_space_2 = TaggedMultiSearchSpace(spaces=[space_C, space_D], tags=["C", "D"])

    product_of_product_spaces = product_space_1 * product_space_2

    subspace_0_A = product_of_product_spaces.get_subspace("0")
    assert isinstance(subspace_0_A, Box)
    npt.assert_array_equal(subspace_0_A.lower, [-1, -2])
    npt.assert_array_equal(subspace_0_A.upper, [2, 3])
    subspace_0_B = product_of_product_spaces.get_subspace("1")
    assert isinstance(subspace_0_B, DiscreteSearchSpace)
    npt.assert_array_equal(subspace_0_B.points, tf.ones([100, 2], dtype=tf.float64))

    subspace_1_C = product_of_product_spaces.get_subspace("2")
    assert isinstance(subspace_1_C, Box)
    npt.assert_array_equal(subspace_1_C.lower, [-2, -2])
    npt.assert_array_equal(subspace_1_C.upper, [2, 3])
    subspace_1_D = product_of_product_spaces.get_subspace("3")
    assert isinstance(subspace_1_D, DiscreteSearchSpace)
    npt.assert_array_equal(subspace_1_D.points, tf.ones([5, 2], dtype=tf.float64))


def test_multi_space_handles_empty_spaces() -> None:
    tag_A = TaggedProductSearchSpace(spaces=[], tags=[])
    tag_B = TaggedProductSearchSpace(spaces=[], tags=[])
    tag_C = TaggedMultiSearchSpace(spaces=[tag_A, tag_B], tags=["AA", "BB"])

    assert tag_C.dimension == 0
    npt.assert_array_equal(tag_C.lower, [[], []])
    npt.assert_array_equal(tag_C.upper, [[], []])
    npt.assert_array_equal(tag_C.subspace_tags, ["AA", "BB"])


def _nlc_func(x: TensorType) -> TensorType:
    c0 = x[..., 0] - tf.sin(x[..., 1])
    c0 = tf.expand_dims(c0, axis=-1)
    return c0


@pytest.mark.parametrize(
    "a, b, equal",
    [
        (Box([-1], [2]), Box([-1], [2]), True),
        (Box([-1], [2]), Box([0], [2]), False),
        (Box([-1], [2]), DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [1.2, 0.4]])), False),
        (
            DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [1.2, 0.4]])),
            DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [1.2, 0.4]])),
            True,
        ),
        (
            DiscreteSearchSpace(tf.constant([[-0.5, -0.3]])),
            DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [1.2, 0.4]])),
            False,
        ),
        (
            DiscreteSearchSpace(tf.constant([[-0.5, -0.3], [1.2, 0.4]])),
            DiscreteSearchSpace(tf.constant([[1.2, 0.4], [-0.5, -0.3]])),
            True,
        ),
        (
            TaggedProductSearchSpace([Box([-1], [1]), Box([1], [2])]),
            TaggedProductSearchSpace([Box([-1], [1]), Box([1], [2])]),
            True,
        ),
        (
            TaggedProductSearchSpace([Box([-1], [1]), Box([1], [2])]),
            TaggedProductSearchSpace([Box([-1], [1]), Box([3], [4])]),
            False,
        ),
        (
            TaggedProductSearchSpace([Box([-1], [1]), Box([1], [2])], tags=["A", "B"]),
            TaggedProductSearchSpace([Box([-1], [1]), Box([1], [2])], tags=["B", "A"]),
            False,
        ),
        (
            TaggedProductSearchSpace([Box([-1], [1]), Box([1], [2])], tags=["A", "B"]),
            TaggedProductSearchSpace([Box([1], [2]), Box([-1], [1])], tags=["B", "A"]),
            False,
        ),
        (
            TaggedMultiSearchSpace([Box([-1], [1]), Box([1], [2])]),
            TaggedMultiSearchSpace([Box([-1], [1]), Box([1], [2])]),
            True,
        ),
        (
            TaggedMultiSearchSpace([Box([-1], [1]), Box([1], [2])]),
            TaggedMultiSearchSpace([Box([-1], [1]), Box([3], [4])]),
            False,
        ),
        (
            TaggedMultiSearchSpace([Box([-1], [1]), Box([1], [2])], tags=["A", "B"]),
            TaggedMultiSearchSpace([Box([-1], [1]), Box([1], [2])], tags=["B", "A"]),
            False,
        ),
        (
            TaggedMultiSearchSpace([Box([-1], [1]), Box([1], [2])], tags=["A", "B"]),
            TaggedMultiSearchSpace([Box([1], [2]), Box([-1], [1])], tags=["B", "A"]),
            False,
        ),
        (
            TaggedProductSearchSpace([Box([-1], [1]), Box([1], [2])]),
            TaggedMultiSearchSpace([Box([-1], [1]), Box([1], [2])]),
            False,
        ),
        (
            Box(
                [-1],
                [2],
                [
                    NonlinearConstraint(_nlc_func, -1.0, 0.0),
                    LinearConstraint(A=tf.eye(2), lb=tf.zeros((2)), ub=tf.ones((2))),
                ],
            ),
            Box(
                [-1],
                [2],
                [
                    NonlinearConstraint(_nlc_func, -1.0, 0.0),
                    LinearConstraint(A=tf.eye(2), lb=tf.zeros((2)), ub=tf.ones((2))),
                ],
            ),
            True,
        ),
        (
            Box(
                [-1],
                [2],
                [
                    NonlinearConstraint(_nlc_func, -1.0, 0.0),
                    LinearConstraint(A=tf.eye(2), lb=tf.zeros((2)), ub=tf.ones((2))),
                ],
            ),
            Box(
                [-1],
                [2],
                [
                    NonlinearConstraint(_nlc_func, -1.0, 0.1),
                    LinearConstraint(A=tf.eye(2), lb=tf.zeros((2)), ub=tf.ones((2))),
                ],
            ),
            False,
        ),
        (CategoricalSearchSpace([3, 2]), CategoricalSearchSpace([3, 2]), True),
        (CategoricalSearchSpace([3]), CategoricalSearchSpace([3, 2]), False),
        (CategoricalSearchSpace(3), CategoricalSearchSpace(["0", "1", "2"]), True),
        (CategoricalSearchSpace(3), CategoricalSearchSpace(["R", "G", "B"]), False),
        (CategoricalSearchSpace(3), DiscreteSearchSpace(tf.constant([[0], [1]])), False),
    ],
)
def test___eq___search_spaces(a: SearchSpace, b: SearchSpace, equal: bool) -> None:
    assert (a == b) is equal
    assert (a != b) is (not equal)
    assert (a == a) and (b == b)


def test_linear_constraints_residual() -> None:
    points = tf.constant([[-1.0, 0.4], [-1.0, 0.6], [0.0, 0.4]])
    lc = LinearConstraint(
        A=tf.constant([[-1.0, 1.0], [1.0, 0.0]]),
        lb=tf.constant([-0.4, 0.5]),
        ub=tf.constant([-0.2, 0.9]),
    )
    got = lc.residual(points)
    expected = tf.constant([[1.8, -1.5, -1.6, 1.9], [2.0, -1.5, -1.8, 1.9], [0.8, -0.5, -0.6, 0.9]])
    npt.assert_allclose(expected, got)


def test_nonlinear_constraints_residual() -> None:
    points = tf.constant([[-1.0, 0.4], [-1.0, 0.6], [0.0, 0.4]])
    nlc = NonlinearConstraint(
        lambda x: tf.expand_dims(x[..., 0] - tf.math.sin(x[..., 1]), -1), -1.4, 1.9
    )
    got = nlc.residual(points)
    expected = tf.constant(
        [[0.01058163, 3.28941832], [-0.1646425, 3.46464245], [1.01058163, 2.28941832]]
    )
    npt.assert_allclose(expected, got, atol=1e-7)


@pytest.mark.parametrize(
    "constraints, points",
    [
        (
            [
                LinearConstraint(
                    A=tf.constant([[-1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
                    lb=tf.constant([-0.4, 0.15, 0.2]),
                    ub=tf.constant([0.6, 0.9, 0.9]),
                ),
                NonlinearConstraint(_nlc_func, tf.constant(-1.0), tf.constant(0.0)),
                LinearConstraint(A=tf.eye(2), lb=tf.zeros((2)), ub=tf.ones((2))),
            ],
            tf.constant([[0.820, 0.057], [0.3, 0.4], [0.582, 0.447], [0.15, 0.75]]),
        ),
    ],
)
def test_box_constraints_residuals_and_feasibility(
    constraints: Sequence[Constraint], points: tf.Tensor
) -> None:
    space = Box(tf.constant([0.0, 0.0]), tf.constant([1.0, 1.0]), constraints)
    got = space.constraints_residuals(points)
    expected = tf.constant(
        [
            [
                -0.363,
                0.66999996,
                -0.143,
                1.363,
                0.07999998,
                0.843,
                1.7630308,
                -0.7630308,
                0.82,
                0.057,
                0.18,
                0.943,
            ],
            [
                0.5,
                0.15,
                0.2,
                0.5,
                0.59999996,
                0.49999997,
                0.9105817,
                0.08941832,
                0.3,
                0.4,
                0.7,
                0.6,
            ],
            [
                0.265,
                0.432,
                0.247,
                0.735,
                0.31799996,
                0.45299998,
                1.1497378,
                -0.14973778,
                0.582,
                0.447,
                0.41799998,
                0.553,
            ],
            [
                1.0,
                0.0,
                0.55,
                0.0,
                0.75,
                0.14999998,
                0.46836126,
                0.53163874,
                0.15,
                0.75,
                0.85,
                0.25,
            ],
        ]
    )
    print(got)

    npt.assert_array_equal(expected, got)
    npt.assert_array_equal(tf.constant([False, True, False, True]), space.is_feasible(points))


def test_discrete_search_space_raises_if_has_constraints() -> None:
    space = Box(
        tf.zeros((2)),
        tf.ones((2)),
        [LinearConstraint(A=tf.eye(2), lb=tf.zeros((2)), ub=tf.ones((2)))],
    )
    with pytest.raises(NotImplementedError):
        _ = space.discretize(2)


def test_nonlinear_constraints_multioutput_raises() -> None:
    points = tf.constant([[-1.0, 0.4], [-1.0, 0.6], [0.0, 0.4]])
    nlc = NonlinearConstraint(
        lambda x: tf.broadcast_to(tf.expand_dims(x[..., 0] - x[..., 1], -1), (x.shape[0], 2)),
        -1.4,
        1.9,
    )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        nlc.residual(points)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_box_empty_sobol_sampling_returns_correct_dtype(dtype: tf.DType) -> None:
    box = Box(tf.zeros((3,), dtype=dtype), tf.ones((3,), dtype=dtype))
    sobol_samples = box.sample_sobol(0)
    assert sobol_samples.dtype == dtype


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_box_empty_halton_sampling_returns_correct_dtype(dtype: tf.DType) -> None:
    box = Box(tf.zeros((3,), dtype=dtype), tf.ones((3,), dtype=dtype))
    sobol_samples = box.sample_halton(0)
    assert sobol_samples.dtype == dtype


@pytest.mark.parametrize(
    "categories, points",
    [
        pytest.param([], tf.zeros([0, 0])),
        pytest.param(3, tf.constant([[0.0], [1.0], [2.0]])),
        pytest.param([3], tf.constant([[0.0], [1.0], [2.0]])),
        pytest.param(
            [3, 2],
            tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]]),
        ),
        pytest.param(["R", "G", "B"], tf.constant([[0.0], [1.0], [2.0]])),
        pytest.param([["R", "G", "B"]], tf.constant([[0.0], [1.0], [2.0]])),
        pytest.param(
            [["R", "G", "B"], ["Y", "N"]],
            tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]]),
        ),
    ],
)
def test_categorical_search_space__points(
    categories: int | Sequence[int] | Sequence[str] | Sequence[Sequence[str]], points: TensorType
) -> None:
    space = CategoricalSearchSpace(categories, dtype=tf.float32)
    npt.assert_array_equal(space.points, points)
    contains = space.contains(points)
    assert len(contains) == len(points)
    assert tf.reduce_all(contains)


@pytest.mark.parametrize(
    "categories, exception",
    [
        pytest.param([3, 0, 2], ValueError, id="Empty category size"),
        pytest.param([-2, 1], ValueError, id="Negative category size"),
        pytest.param([["R", "G", "B"], [], ["Y", "N"]], ValueError, id="Empty category list"),
        pytest.param([3, ["A", "B"]], TypeError, id="Mixed description types"),
    ],
)
def test_categorical_search_space__raises(categories: Any, exception: type) -> None:
    with pytest.raises(exception):
        CategoricalSearchSpace(categories)


@pytest.mark.parametrize(
    "space1, space2, expected_product",
    [
        (CategoricalSearchSpace(3), CategoricalSearchSpace(2), CategoricalSearchSpace([3, 2])),
        (CategoricalSearchSpace([]), CategoricalSearchSpace(2), CategoricalSearchSpace([2])),
        (
            CategoricalSearchSpace(["R", "G", "B"]),
            CategoricalSearchSpace(["Y", "N"]),
            CategoricalSearchSpace([["R", "G", "B"], ["Y", "N"]]),
        ),
    ],
)
def test_categorical_search_space__product(
    space1: CategoricalSearchSpace,
    space2: CategoricalSearchSpace,
    expected_product: CategoricalSearchSpace,
) -> None:
    product = space1 * space2
    assert product == expected_product
    npt.assert_array_equal(product.points, expected_product.points)
    assert product.tags == expected_product.tags


@pytest.mark.parametrize(
    "categories, tags",
    [
        ([], []),
        (3, [("0", "1", "2")]),
        ([2, 3], [("0", "1"), ("0", "1", "2")]),
        (["R", "G", "B"], [("R", "G", "B")]),
        ([["R", "G", "B"], ["Y", "N"]], [("R", "G", "B"), ("Y", "N")]),
    ],
)
def test_categorical_search_space__tags(
    categories: int | Sequence[int] | Sequence[str] | Sequence[Sequence[str]],
    tags: Sequence[Sequence[str]],
) -> None:
    search_space = CategoricalSearchSpace(categories)
    assert search_space.tags == tags


@pytest.mark.parametrize(
    "categories, indices, expected_tags",
    [
        (3, tf.constant([[0], [2], [2]]), tf.constant([["0"], ["2"], ["2"]])),
        (["A", "B", "C"], tf.constant([[0.0], [2.0], [2.0]]), tf.constant([["A"], ["C"], ["C"]])),
        (
            (3, 2),
            tf.constant([[0, 1.0], [2.0, 0.0], [2.0, 1.0]]),
            tf.constant([["0", "1"], ["2", "0"], ["2", "1"]]),
        ),
        (
            [("A", "B", "C"), ("Y", "N")],
            tf.constant([[0.0, 1.0], [2.0, 0.0], [2.0, 1.0]]),
            tf.constant([["A", "N"], ["C", "Y"], ["C", "N"]]),
        ),
    ],
)
def test_categorical_search_space__to_tags(
    categories: int | Sequence[int] | Sequence[str] | Sequence[Sequence[str]],
    indices: TensorType,
    expected_tags: TensorType,
) -> None:
    search_space = CategoricalSearchSpace(categories)
    tags = search_space.to_tags(indices)
    npt.assert_array_equal(tags, expected_tags)


def test_categorical_search_space__to_tags_raises_for_non_integers() -> None:
    search_space = CategoricalSearchSpace(["A", "B", "C"], dtype=tf.float32)
    with pytest.raises(ValueError):
        search_space.to_tags(tf.constant([[1.0], [1.2]]))


@pytest.mark.parametrize(
    "search_space, query_points, encoded_points",
    [
        (
            CategoricalSearchSpace(["V"]),
            tf.constant([[0], [0]], dtype=tf.float64),
            tf.constant([[1], [1]], dtype=tf.float64),
        ),
        (
            CategoricalSearchSpace(["Y", "N"]),
            tf.constant([[0], [1], [0]], dtype=tf.float64),
            tf.constant([[0], [1], [0]], dtype=tf.float64),
        ),
        (
            CategoricalSearchSpace(["R", "G", "B"], dtype=tf.float32),
            tf.constant([[0], [2], [1]], dtype=tf.float32),
            tf.constant([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=tf.float32),
        ),
        (
            CategoricalSearchSpace(["R", "G", "B"]),
            tf.constant([[[[[0]]]]], dtype=tf.float64),
            tf.constant([[[[[1, 0, 0]]]]], dtype=tf.float64),
        ),
        (
            CategoricalSearchSpace(["R", "G", "B", "A"], dtype=tf.float32),
            tf.constant([[0], [2], [2]], dtype=tf.float32),
            tf.constant([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=tf.float32),
        ),
        (
            CategoricalSearchSpace([["R", "G", "B"], ["Y", "N"]]),
            tf.constant([[0, 0], [2, 0], [1, 1]], dtype=tf.float64),
            tf.constant([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]], dtype=tf.float64),
        ),
        (
            CategoricalSearchSpace([["R", "G", "B"], ["Y", "N"]]),
            tf.constant([[[0, 0], [0, 0]], [[2, 0], [1, 1]]], dtype=tf.float64),
            tf.constant(
                [[[1, 0, 0, 0], [1, 0, 0, 0]], [[0, 0, 1, 0], [0, 1, 0, 1]]],
                dtype=tf.float64,
            ),
        ),
        (
            TaggedProductSearchSpace([Box([0.0], [1.0]), CategoricalSearchSpace(["R", "G", "B"])]),
            tf.constant([[0.5, 0], [0.3, 2]], dtype=tf.float64),
            tf.constant([[0.5, 1, 0, 0], [0.3, 0, 0, 1]], dtype=tf.float64),
        ),
        (
            TaggedProductSearchSpace([Box([0.0], [1.0]), CategoricalSearchSpace(["R", "G", "B"])]),
            tf.constant([[[0.5, 0]], [[0.3, 2]]], dtype=tf.float64),
            tf.constant([[[0.5, 1, 0, 0]], [[0.3, 0, 0, 1]]], dtype=tf.float64),
        ),
        (
            Box([0.0], [1.0]),
            tf.constant([[0.5], [0.3]], dtype=tf.float64),
            tf.constant([[0.5], [0.3]], dtype=tf.float64),
        ),
    ],
)
def test_categorical_search_space_one_hot_encoding(
    search_space: SearchSpace, query_points: TensorType, encoded_points: TensorType
) -> None:
    encoder = one_hot_encoder(search_space)
    points = encoder(query_points)
    npt.assert_array_equal(encoded_points, points)


@pytest.mark.parametrize(
    "search_space, query_points, exception",
    [
        pytest.param(
            CategoricalSearchSpace(["Y", "N"]),
            tf.constant([0, 2, 1]),
            InvalidArgumentError,
            id="Wrong input rank",
        ),
        pytest.param(
            CategoricalSearchSpace(["Y", "N"]),
            tf.constant([[0], [2], [1]]),
            InvalidArgumentError,
            id="Out of range binary input value",
        ),
        pytest.param(
            CategoricalSearchSpace(["Y", "N", "maybe"]),
            tf.constant([[0], [3], [1]]),
            InvalidArgumentError,
            id="Out of range input value",
        ),
        pytest.param(
            CategoricalSearchSpace([["R", "G", "B"], ["Y", "N"]]),
            tf.constant([[0], [1], [1]]),
            InvalidArgumentError,
            id="Wrong input shape",
        ),
    ],
)
def test_categorical_search_space_one_hot_encoding__raises(
    search_space: CategoricalSearchSpace, query_points: TensorType, exception: type
) -> None:
    encoder = one_hot_encoder(search_space)
    with pytest.raises(exception):
        encoder(query_points)


@pytest.mark.parametrize(
    "space",
    [
        CategoricalSearchSpace([3, 2]),
        CategoricalSearchSpace(["R", "G", "B"]),
        TaggedProductSearchSpace([Box([-1, -2], [2, 3]), CategoricalSearchSpace(2)]),
    ],
)
def test_unbound_search_spaces(
    space: SearchSpace,
) -> None:
    assert not space.has_bounds
    with pytest.raises(AttributeError):
        space.lower
    with pytest.raises(AttributeError):
        space.upper


@pytest.mark.parametrize("input_dtype", [None, tf.float64, tf.float32])
@pytest.mark.parametrize("output_dtype", [None, tf.float64, tf.float32])
def test_cast_encoder(input_dtype: Optional[tf.DType], output_dtype: Optional[tf.DType]) -> None:

    query_points = tf.constant([1, 2, 3], dtype=tf.int32)

    def add_encoder(x: TensorType) -> TensorType:
        assert x.dtype is (input_dtype or tf.int32)
        return x + 1

    encoder = cast_encoder(add_encoder, input_dtype=input_dtype, output_dtype=output_dtype)
    points = encoder(query_points)
    assert points.dtype is (output_dtype or input_dtype or tf.int32)
    npt.assert_array_equal(tf.cast(query_points + 1, points.dtype), points)

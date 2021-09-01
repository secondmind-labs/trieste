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
from collections import Sequence
from collections.abc import Container

import numpy.testing as npt
import pytest
import tensorflow as tf
from typing_extensions import Final

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, ShapeLike, various_shapes
from trieste.space import Box, DiscreteSearchSpace, OrdinalSearchSpace, SearchSpace


class Integers(SearchSpace):
    def __init__(self, exclusive_limit: int):
        assert exclusive_limit > 0
        self.limit: Final[int] = exclusive_limit

    def sample(self, num_samples: int) -> tf.Tensor:
        return tf.random.shuffle(tf.range(self.limit))[:num_samples]

    def __contains__(self, point: tf.Tensor) -> tf.Tensor:
        tf.debugging.assert_integer(point)
        return 0 <= point < self.limit

    def __mul__(self, other: Integers) -> Integers:
        return Integers(self.limit * other.limit)


@pytest.mark.parametrize("exponent", [0, -2])
def test_search_space___pow___raises_for_non_positive_exponent(exponent: int) -> None:
    space = Integers(3)
    with pytest.raises(tf.errors.InvalidArgumentError):
        space ** exponent


def test_search_space___pow___multiplies_correct_number_of_search_spaces() -> None:
    assert (Integers(5) ** 7).limit == 5 ** 7


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
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
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

    with pytest.raises(tf.errors.InvalidArgumentError):
        search_space.sample(num_samples)


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

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        Box(lower, upper)


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
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
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


@pytest.mark.parametrize(
    "bound_shape, point_shape",
    ((bs, ps) for bs, ps in _pairs_of_shapes() if bs != ps and len(bs) == 1 and bs != (0,)),
)
def test_box_contains_raises_on_point_of_different_shape(
    bound_shape: ShapeLike,
    point_shape: ShapeLike,
) -> None:
    box = Box(tf.zeros(bound_shape), tf.ones(bound_shape))
    point = tf.zeros(point_shape)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        _ = point in box


@pytest.mark.parametrize("num_samples", [0, 1, 10])
def test_box_sampling_returns_correct_shape(num_samples: int) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)))
    samples = box.sample(num_samples)
    _assert_correct_number_of_unique_constrained_samples(num_samples, box, samples)


@pytest.mark.parametrize("num_samples", [0, 1, 10])
def test_box_sobol_sampling_returns_correct_shape(num_samples: int) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)))
    sobol_samples = box.sample_sobol(num_samples)
    _assert_correct_number_of_unique_constrained_samples(num_samples, box, sobol_samples)


@pytest.mark.parametrize("num_samples", [0, 1, 10])
def test_box_halton_sampling_returns_correct_shape(num_samples: int) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)))
    halton_samples = box.sample_halton(num_samples)
    _assert_correct_number_of_unique_constrained_samples(num_samples, box, halton_samples)


@pytest.mark.parametrize("num_samples", [-1, -10])
def test_box_sampling_raises_for_invalid_sample_size(num_samples: int) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        box = Box(tf.zeros((3,)), tf.ones((3,)))
        box.sample(num_samples)


@pytest.mark.parametrize("num_samples", [-1, -10])
def test_box_sobol_sampling_raises_for_invalid_sample_size(num_samples: int) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        box = Box(tf.zeros((3,)), tf.ones((3,)))
        box.sample_sobol(num_samples)


@pytest.mark.parametrize("num_samples", [-1, -10])
def test_box_halton_sampling_raises_for_invalid_sample_size(num_samples: int) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        box = Box(tf.zeros((3,)), tf.ones((3,)))
        box.sample_halton(num_samples)


@pytest.mark.parametrize("skip", [1, 10, 100])
def test_box_sobol_sampling_returns_same_points_for_same_skip(skip: int) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)))
    sobol_samples_1 = box.sample_sobol(num_samples=100, skip=skip)
    sobol_samples_2 = box.sample_sobol(num_samples=100, skip=skip)
    npt.assert_allclose(sobol_samples_1, sobol_samples_2)


@pytest.mark.parametrize("seed", [1, 42, 123])
def test_box_halton_sampling_returns_same_points_for_same_seed(seed: int) -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)))
    halton_samples_1 = box.sample_halton(num_samples=100, seed=seed)
    halton_samples_2 = box.sample_halton(num_samples=100, seed=seed)
    npt.assert_allclose(halton_samples_1, halton_samples_2)


def test_box_sobol_sampling_returns_different_points_for_different_call() -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)))
    sobol_samples_1 = box.sample_sobol(num_samples=100)
    sobol_samples_2 = box.sample_sobol(num_samples=100)
    npt.assert_raises(AssertionError, npt.assert_allclose, sobol_samples_1, sobol_samples_2)


def test_box_haltom_sampling_returns_different_points_for_different_call() -> None:
    box = Box(tf.zeros((3,)), tf.ones((3,)))
    halton_samples_1 = box.sample_halton(num_samples=100)
    halton_samples_2 = box.sample_halton(num_samples=100)
    npt.assert_raises(AssertionError, npt.assert_allclose, halton_samples_1, halton_samples_2)


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

    with pytest.raises(tf.errors.InvalidArgumentError):
        dss.sample(num_samples + 1)


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


@pytest.mark.parametrize(
    "lower, upper, stepsizes",
    [
        pytest.param([0.0, 1.0], [1.0, 2.0], [0.1, 0.2], id="lists"),
        pytest.param((0.0, 1.0), (1.0, 2.0), (0.1, 0.2), id="tuples"),
        pytest.param(range(2), range(1, 3), [0.1, 0.2], id="ranges"),
    ],
)
def test_ordinalsearchspace_converts_sequences_to_float64_tensors(
    lower: Sequence[float], upper: Sequence[float], stepsizes: Sequence[float]
) -> None:
    ordinalsp = OrdinalSearchSpace(lower, upper, stepsizes)
    assert tf.as_dtype(ordinalsp.lower.dtype) is tf.float64
    assert tf.as_dtype(ordinalsp.upper.dtype) is tf.float64
    assert tf.as_dtype(ordinalsp.stepsizes.dtype) is tf.float64
    npt.assert_array_equal(ordinalsp.lower, [0.0, 1.0])
    npt.assert_array_equal(ordinalsp.upper, [1.0, 2.0])
    npt.assert_array_equal(ordinalsp.stepsizes, [0.1, 0.2])


@pytest.mark.parametrize(
    "lower, upper, stepsizes, lower_ord, upper_ord",
    [
        pytest.param([0.0, 1.0], [1.0, 2.0], [0.1, 0.2], [0.0, 1.0], [1.0, 2.0]),
        pytest.param(
            [0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [0.0, 1.0, 2.1], [1.0, 2.0, 3.0]
        ),
        pytest.param([1.0, 10.0], [10.0, 100.0], [2.0, 10.0], [2.0, 10.0], [10.0, 100.0]),
        pytest.param([0.1, 0.9], [1.0, 2.0], [0.2, 0.2], [0.2, 1.0], [1.0, 2.0]),
        pytest.param([0.0, 1.0], [1.0, 2.0], [0.3, 0.3], [0.0, 1.2], [0.9, 1.8]),
        pytest.param([0.01, 1.01], [1.0, 2.0], [0.03, 0.03], [0.03, 1.02], [0.99, 1.98]),
        pytest.param([-1.0, -2.0], [1.0, 2.0], [0.3, 0.3], [-0.9, -1.8], [0.9, 1.8]),
        pytest.param([-1.0, -2.0], [1.0, 2.0], [0.6, 0.6], [-0.6, -1.8], [0.6, 1.8]),
        pytest.param([5 / 3], [2.0], [1 / 3], [5 / 3], [2.0]),
    ],
)
def test_ordinalsearchspace_bounds_changes_to_nearest_multiples(
    lower: Sequence[float],
    upper: Sequence[float],
    stepsizes: Sequence[float],
    lower_ord: Sequence[float],
    upper_ord: Sequence[float],
) -> None:

    ordinalsp = OrdinalSearchSpace(lower, upper, stepsizes)
    npt.assert_allclose(ordinalsp.lower, lower_ord)
    npt.assert_allclose(ordinalsp.upper, upper_ord)


@pytest.mark.parametrize(
    "lower, upper, stepsizes",
    [
        pytest.param([0.0, 1.0], [1.0, 2.0], [0.1, 0.2]),
        pytest.param([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]),
        pytest.param([1.0, 10.0], [10.0, 100.0], [2, 10]),
    ],
)
def test_ordinalsearchspace_sampling_return_correct_rounded_points(
    lower: Sequence[float], upper: Sequence[float], stepsizes: Sequence[float]
) -> None:
    ordinalsp = OrdinalSearchSpace(lower, upper, stepsizes)
    boxsp = Box(lower, upper)
    tf.random.set_seed(0)
    ordinalsample = ordinalsp.sample(5)
    tf.random.set_seed(0)
    boxsample = boxsp.sample(5)
    npt.assert_allclose(ordinalsample, tf.round(boxsample / stepsizes) * stepsizes)


@pytest.mark.parametrize(
    "bound_shape, stepsizes_shape",
    _pairs_of_shapes(excluding_ranks={1}) | {((1,), (2,)), ((0,), (0,))},
)
def test_ordinal_raises_if_stepsizes_have_invalid_shape(
    bound_shape: ShapeLike, stepsizes_shape: ShapeLike
) -> None:
    lower, upper = tf.zeros(bound_shape), tf.ones(bound_shape)
    stepsizes = tf.fill(stepsizes_shape, 0.1)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        OrdinalSearchSpace(lower, upper, stepsizes)


def test_ordinal_bounds_and_stepsizes_attributes() -> None:
    lower, upper, stepsizes = tf.zeros([2]), tf.ones([2]), tf.fill([2], 0.1)
    ordinalsp = OrdinalSearchSpace(lower, upper, stepsizes)
    npt.assert_array_equal(ordinalsp.lower, lower)
    npt.assert_array_equal(ordinalsp.upper, upper)
    npt.assert_array_equal(ordinalsp.stepsizes, stepsizes)


@pytest.mark.parametrize(
    "point",
    [
        tf.constant([-1.0, 0.0, -2.0]),  # lower bound
        tf.constant([2.0, 1.0, -0.5]),  # upper bound
        tf.constant([0.6, 0.5, -1.5]),  # approx centre
        tf.constant([-1.0, 0.0, -1.9]),  # near the edge
        tf.constant([-0.2, 0.3, -1.3]),  # checking step sizes
    ],
)
def test_ordinal_contains_point(point: tf.Tensor) -> None:
    assert point in OrdinalSearchSpace(
        tf.constant([-1.0, 0.0, -2.0]), tf.constant([2.0, 1.0, -0.5]), tf.constant([0.2, 0.1, 0.1])
    )


@pytest.mark.parametrize(
    "point",
    [
        tf.constant([-1.1, 0.0, -2.0]),  # just outside
        tf.constant([-0.5, -0.5, 1.5]),  # negative of a contained point
        tf.constant([10.0, -10.0, 10.0]),  # well outside
        tf.constant([0.5, 0.5, -1.5]),  # approx centre with different step size
        tf.constant([0.5, 0.55, -1.11]),  # inside with totally different step sizes
    ],
)
def test_ordinal_does_not_contain_point(point: tf.Tensor) -> None:
    assert point not in OrdinalSearchSpace(
        tf.constant([-1.0, 0.0, -2.0]), tf.constant([2.0, 1.0, -0.5]), tf.constant([0.2, 0.1, 0.1])
    )


def test_ordinal___mul___bounds_are_the_concatenation_of_original_bounds() -> None:
    ordinalsp1 = OrdinalSearchSpace(
        tf.constant([0.0, 1.0]), tf.constant([2.0, 3.0]), tf.constant([0.1, 0.2])
    )
    ordinalsp2 = OrdinalSearchSpace(
        tf.constant([4.2, 5.6, 6.5]), tf.constant([7.5, 8.0, 9.5]), tf.constant([0.3, 0.4, 0.5])
    )

    product = ordinalsp1 * ordinalsp2

    npt.assert_allclose(product.lower, [0, 1, 4.2, 5.6, 6.5])
    npt.assert_allclose(product.upper, [2, 3, 7.5, 8.0, 9.5])
    npt.assert_allclose(product.stepsizes, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_ordinal___mul___raises_if_bounds_have_different_types() -> None:
    ordinalsp1 = OrdinalSearchSpace(
        tf.constant([0.0, 1.0]), tf.constant([2.0, 3.0]), tf.constant([0.1, 0.2])
    )
    ordinalsp2 = OrdinalSearchSpace(
        tf.constant([4.0, 5.0], tf.float64),
        tf.constant([6.0, 7.0], tf.float64),
        tf.constant([0.15, 0.25], tf.float64),
    )

    with pytest.raises(TypeError):
        _ = ordinalsp1 * ordinalsp2

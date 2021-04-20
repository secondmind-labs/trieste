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

import pytest
import tensorflow as tf

from tests.util.misc import ShapeLike, assert_datasets_allclose
from trieste.data import Dataset
from trieste.utils import shapes_equal


@pytest.mark.parametrize(
    "query_points, observations",
    [
        (tf.constant([[]]), tf.constant([[]])),
        (tf.constant([[0.0], [1.0], [2.0]]), tf.constant([[], [], []])),
        (tf.constant([[], [], []]), tf.constant([[0.0], [1.0], [2.0]])),
    ],
)
def test_dataset_raises_for_zero_dimensional_data(
    query_points: tf.Tensor, observations: tf.Tensor
) -> None:
    with pytest.raises(ValueError):
        Dataset(query_points, observations)


@pytest.mark.parametrize(
    "query_points_leading_shape, observations_leading_shape",
    [
        ((1,), (2,)),
        ((2,), (1,)),
        ((5, 6), (5, 4)),
        ((5, 6), (4, 6)),
        ((5, 6), (4, 4)),
    ],
)
@pytest.mark.parametrize("last_dim_size", [1, 5])
def test_dataset_raises_for_different_leading_shapes(
    query_points_leading_shape: tuple[int, ...],
    observations_leading_shape: tuple[int, ...],
    last_dim_size: int,
) -> None:
    query_points = tf.zeros(query_points_leading_shape + (last_dim_size,))
    observations = tf.ones(observations_leading_shape + (last_dim_size,))

    with pytest.raises(ValueError, match="(L|l)eading"):
        Dataset(query_points, observations)


@pytest.mark.parametrize(
    "query_points_shape, observations_shape",
    [
        ((1, 2), (1,)),
        ((1, 2), (1, 2, 3)),
    ],
)
def test_dataset_raises_for_different_ranks(
    query_points_shape: ShapeLike, observations_shape: ShapeLike
) -> None:
    query_points = tf.zeros(query_points_shape)
    observations = tf.ones(observations_shape)

    with pytest.raises(ValueError):
        Dataset(query_points, observations)


@pytest.mark.parametrize(
    "query_points_shape, observations_shape",
    [
        ((), ()),
        ((), (10,)),
        ((10,), (10,)),
        ((1, 2), (1,)),
        ((1, 2), (1, 2, 3)),
    ],
)
def test_dataset_raises_for_invalid_ranks(
    query_points_shape: ShapeLike, observations_shape: ShapeLike
) -> None:
    query_points = tf.zeros(query_points_shape)
    observations = tf.ones(observations_shape)

    with pytest.raises(ValueError):
        Dataset(query_points, observations)


def test_dataset_getters() -> None:
    query_points, observations = tf.constant([[0.0]]), tf.constant([[1.0]])
    dataset = Dataset(query_points, observations)
    assert dataset.query_points.dtype == query_points.dtype
    assert dataset.observations.dtype == observations.dtype

    assert shapes_equal(dataset.query_points, query_points)
    assert shapes_equal(dataset.observations, observations)

    assert tf.reduce_all(dataset.query_points == query_points)
    assert tf.reduce_all(dataset.observations == observations)


@pytest.mark.parametrize(
    "lhs, rhs, expected",
    [
        (  # lhs and rhs populated
            Dataset(tf.constant([[1.2, 3.4], [5.6, 7.8]]), tf.constant([[1.1], [2.2]])),
            Dataset(tf.constant([[5.0, 6.0], [7.0, 8.0]]), tf.constant([[-1.0], [-2.0]])),
            Dataset(
                # fmt: off
                tf.constant([[1.2, 3.4], [5.6, 7.8], [5.0, 6.0], [7.0, 8.0]]),
                tf.constant([[1.1], [2.2], [-1.0], [-2.0]]),
                # fmt: on
            ),
        ),
        (  # lhs populated
            Dataset(tf.constant([[1.2, 3.4], [5.6, 7.8]]), tf.constant([[1.1], [2.2]])),
            Dataset(tf.zeros([0, 2]), tf.zeros([0, 1])),
            Dataset(tf.constant([[1.2, 3.4], [5.6, 7.8]]), tf.constant([[1.1], [2.2]])),
        ),
        (  # rhs populated
            Dataset(tf.zeros([0, 2]), tf.zeros([0, 1])),
            Dataset(tf.constant([[1.2, 3.4], [5.6, 7.8]]), tf.constant([[1.1], [2.2]])),
            Dataset(tf.constant([[1.2, 3.4], [5.6, 7.8]]), tf.constant([[1.1], [2.2]])),
        ),
        (  # both empty
            Dataset(tf.zeros([0, 2]), tf.zeros([0, 1])),
            Dataset(tf.zeros([0, 2]), tf.zeros([0, 1])),
            Dataset(tf.zeros([0, 2]), tf.zeros([0, 1])),
        ),
    ],
)
def test_dataset_concatenation(lhs: Dataset, rhs: Dataset, expected: Dataset) -> None:
    assert_datasets_allclose(lhs + rhs, expected)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        (  # incompatible query points shape
            Dataset(tf.constant([[0.0]]), tf.constant([[0.0]])),
            Dataset(tf.constant([[1.0, 1.0]]), tf.constant([[1.0]])),
        ),
        (  # incompatible observations shape
            Dataset(tf.constant([[0.0]]), tf.constant([[0.0]])),
            Dataset(tf.constant([[1.0]]), tf.constant([[1.0, 1.0]])),
        ),
        (  # incompatible query points dtype
            Dataset(tf.constant([[0.0]]), tf.constant([[0.0]])),
            Dataset(tf.constant([[1.0]], dtype=tf.float64), tf.constant([[1.0]])),
        ),
        (  # incompatible observations dtype
            Dataset(tf.constant([[0.0]]), tf.constant([[0.0]])),
            Dataset(tf.constant([[1.0]]), tf.constant([[1.0]], dtype=tf.float64)),
        ),
    ],
)
def test_dataset_concatentation_raises_for_incompatible_data(lhs: Dataset, rhs: Dataset) -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        lhs + rhs

    with pytest.raises(tf.errors.InvalidArgumentError):
        rhs + lhs


@pytest.mark.parametrize(
    "data, length",
    [
        (Dataset(tf.ones((7, 8, 10)), tf.ones((7, 8, 13))), 7),
        (Dataset(tf.ones([0, 2]), tf.ones([0, 1])), 0),
        (Dataset(tf.ones([1, 0, 2]), tf.ones([1, 0, 1])), 1),
    ],
)
def test_dataset_length(data: Dataset, length: int) -> None:
    assert len(data) == length


def test_dataset_deepcopy() -> None:
    data = Dataset(tf.constant([[0.0, 1.0]]), tf.constant([[2.0]]))
    assert_datasets_allclose(data, copy.deepcopy(data))


def test_dataset_astuple() -> None:
    qp, obs = tf.constant([[0.0]]), tf.constant([[1.0]])
    qp_from_astuple, obs_from_astuple = Dataset(qp, obs).astuple()
    assert qp_from_astuple is qp
    assert obs_from_astuple is obs

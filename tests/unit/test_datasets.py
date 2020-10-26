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

from typing import Tuple

import pytest
import tensorflow as tf
from trieste.data import Dataset


@pytest.mark.parametrize('query_points_leading_shape, observations_leading_shape', [
    ((1,), (2,)),
    ((2,), (1,)),
    ((5, 6), (5, 4)),
    ((5, 6), (4, 6)),
    ((5, 6), (4, 4)),
])
@pytest.mark.parametrize('last_dim_size', [0, 1, 5])
def test_dataset_raises_on_initialisation_for_different_leading_shapes(
        query_points_leading_shape: Tuple[int, ...],
        observations_leading_shape: Tuple[int, ...],
        last_dim_size: int
) -> None:
    query_points = tf.zeros(query_points_leading_shape + (last_dim_size,))
    observations = tf.ones(observations_leading_shape + (last_dim_size,))

    with pytest.raises(ValueError, match='(L|l)eading'):
        Dataset(query_points, observations)


@pytest.mark.parametrize('query_points_shape, observations_shape', [
    ((1, 2), (1,)),
    ((1, 2), (1, 2, 3)),
])
def test_dataset_raises_on_initialisation_for_different_ranks(
        query_points_shape: Tuple[int, ...],
        observations_shape: Tuple[int, ...]
) -> None:
    query_points = tf.zeros(query_points_shape)
    observations = tf.ones(observations_shape)

    with pytest.raises(ValueError):
        Dataset(query_points, observations)


@pytest.mark.parametrize('query_points_shape, observations_shape', [
    ((), ()),
    ((), (10,)),
    ((10,), (10,)),
    ((1, 2), (1,)),
    ((1, 2), (1, 2, 3)),
])
def test_dataset_raises_on_initialisation_for_invalid_ranks(
        query_points_shape: Tuple[int, ...],
        observations_shape: Tuple[int, ...]
) -> None:
    query_points = tf.zeros(query_points_shape)
    observations = tf.ones(observations_shape)

    with pytest.raises(ValueError):
        Dataset(query_points, observations)


def test_dataset_getters() -> None:
    query_points, observations = tf.zeros((3, 3)), tf.zeros((3, 3))
    dataset = Dataset(query_points, observations)
    assert tf.reduce_all(dataset.query_points == query_points)
    assert tf.reduce_all(dataset.observations == observations)


def test_concatenate_datasets() -> None:
    qp_this = [[1.2, 3.4], [5.6, 7.8]]
    qp_that = [[5., 6.], [7., 8.]]

    obs_this = [[1.1, 2.2], [3.3, 4.4]]
    obs_that = [[-1., -2.], [-3., -4.]]

    this = Dataset(tf.constant(qp_this), tf.constant(obs_this))
    that = Dataset(tf.constant(qp_that), tf.constant(obs_that))
    merged = this + that
    assert tf.reduce_all(merged.query_points == tf.constant(qp_this + qp_that))
    assert tf.reduce_all(merged.observations == tf.constant(obs_this + obs_that))


def test_dataset_length() -> None:
    assert len(Dataset(tf.ones((7, 8, 10)), tf.ones((7, 8, 13)))) == 7

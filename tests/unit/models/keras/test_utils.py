# Copyright 2021 The Bellman Contributors
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

from tests.util.misc import ShapeLike, empty_dataset, random_seed
from trieste.data import Dataset
from trieste.models.keras.utils import (
    get_tensor_spec_from_data,
    sample_model_index,
    sample_with_replacement,
)


def test_get_tensor_spec_from_data_raises_for_incorrect_dataset() -> None:
    dataset = empty_dataset([1], [1])

    with pytest.raises(ValueError):
        get_tensor_spec_from_data(dataset.query_points)


@pytest.mark.parametrize(
    "query_point_shape, observation_shape",
    [([1], [1]), ([2], [1]), ([5], [1]), ([5], [2]), ([3, 2], [3, 1])],
)
def test_get_tensor_spec_from_data(
    query_point_shape: ShapeLike, observation_shape: ShapeLike
) -> None:
    dataset = empty_dataset(query_point_shape, observation_shape)
    input_spec, output_spec = get_tensor_spec_from_data(dataset)

    assert input_spec.shape == query_point_shape
    assert input_spec.dtype == dataset.query_points.dtype
    assert input_spec.name == "query_points"

    assert output_spec.shape == observation_shape
    assert output_spec.dtype == dataset.observations.dtype
    assert output_spec.name == "observations"


def test_sample_with_replacement_raises_for_invalid_dataset() -> None:
    dataset = empty_dataset([1], [1])

    with pytest.raises(ValueError):
        sample_with_replacement(dataset.query_points)


def test_sample_with_replacement_raises_for_empty_dataset() -> None:
    dataset = empty_dataset([1], [1])

    with pytest.raises(tf.errors.InvalidArgumentError):
        sample_with_replacement(dataset)


@random_seed
@pytest.mark.parametrize("rank", [2, 3])
def test_sample_with_replacement_seems_correct(rank: int) -> None:
    n_rows = 100
    if rank == 2:
        x = tf.constant(np.arange(0, n_rows, 1), shape=[n_rows, 1])
        y = tf.constant(np.arange(0, n_rows, 1), shape=[n_rows, 1])
    elif rank == 3:
        x = tf.constant(np.arange(0, n_rows, 1).repeat(2), shape=[n_rows, 2, 1])
        y = tf.constant(np.arange(0, n_rows, 1).repeat(2), shape=[n_rows, 2, 1])
    dataset = Dataset(x, y)

    dataset_resampled = sample_with_replacement(dataset)

    # basic check that original dataset has not been changed
    assert tf.reduce_all(dataset.query_points == x)
    assert tf.reduce_all(dataset.observations == y)

    # x and y should be resampled the same, and should differ from the original
    assert tf.reduce_all(dataset_resampled.query_points == dataset_resampled.observations)
    assert tf.reduce_any(dataset_resampled.query_points != x)
    assert tf.reduce_any(dataset_resampled.observations != y)

    # values are likely to repeat due to replacement
    _, _, count = tf.unique_with_counts(tf.squeeze(dataset_resampled.query_points[:, 0]))
    assert tf.reduce_any(count > 1)

    # mean of bootstrap samples should be close to true mean
    mean = [
        tf.reduce_mean(
            tf.cast(sample_with_replacement(dataset).query_points[:, 0], dtype=tf.float32)
        )
        for _ in range(100)
    ]
    x = tf.cast(x[:, 0], dtype=tf.float32)
    assert (tf.reduce_mean(mean) - tf.reduce_mean(x)) < 1
    assert tf.math.abs(tf.math.reduce_std(mean) - tf.math.reduce_std(x) / 10.0) < 0.1


@pytest.mark.parametrize("size", [2, 10])
@pytest.mark.parametrize("num_samples", [0, 1, 10])
def test_sample_model_index_call_shape(size: int, num_samples: int) -> None:
    indices = sample_model_index(size, num_samples)

    assert indices.shape == (num_samples,)


@random_seed
@pytest.mark.parametrize("size", [2, 5, 10, 20])
def test_sample_model_index_size(size: int) -> None:
    indices = sample_model_index(size, 1000)

    assert tf.math.reduce_variance(tf.cast(indices, tf.float32)) > 0
    assert tf.reduce_min(indices) >= 0
    assert tf.reduce_max(indices) < size


@pytest.mark.parametrize("size", [10, 20, 50, 100])
def test_sample_model_index_no_replacement(size: int) -> None:
    indices = sample_model_index(size, size)

    assert tf.reduce_sum(indices) == tf.reduce_sum(tf.range(size))
    assert tf.reduce_all(tf.unique_with_counts(indices)[2] == 1)

# Copyright 2022 The Trieste Contributors
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

from typing import Any, Mapping, Optional, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from trieste.acquisition import AcquisitionFunction
from trieste.acquisition.utils import (
    copy_to_local_models,
    get_local_dataset,
    get_unique_points_mask,
    select_nth_output,
    split_acquisition_function,
    with_local_datasets,
)
from trieste.data import Dataset
from trieste.space import Box, SearchSpaceType
from trieste.types import Tag, TensorType
from trieste.utils.misc import LocalizedTag


@pytest.mark.parametrize(
    "f",
    [
        lambda x: x**2,
        lambda x: tf.cast(x, tf.float64),
    ],
)
@pytest.mark.parametrize(
    "x, split_size, expected_batches",
    [
        (np.zeros((0,)), 2, 1),
        (np.array([1]), 2, 1),
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), 2, 6),
        (np.array([1, 2, 3, 4]), 3, 2),
        (np.array([1, 2, 3, 4]), 4, 1),
        (np.array([1, 2, 3, 4]), 1, 4),
        (np.array([1, 2, 3, 4]), 10, 1),
    ],
)
def test_split_acquisition_function(
    f: AcquisitionFunction, x: "np.ndarray[Any, Any]", split_size: int, expected_batches: int
) -> None:
    mock_f = MagicMock()
    mock_f.side_effect = f
    batch_f = split_acquisition_function(mock_f, split_size=split_size)
    np.testing.assert_allclose(f(x), batch_f(x))
    assert expected_batches == mock_f.call_count


@pytest.mark.parametrize("split_size", [0, -1])
def test_split_acquisition_function__invalid_split_size(split_size: int) -> None:
    with pytest.raises(ValueError):
        split_acquisition_function(MagicMock(), split_size=split_size)


def test_select_nth_output() -> None:
    a = tf.random.normal([5, 6])

    assert np.all(select_nth_output(a) == a[..., 0])
    assert np.all(select_nth_output(a, 3) == a[..., 3])


@pytest.mark.parametrize(
    "space, dataset",
    [
        (Box([0], [1]), Dataset(tf.constant([[0, 1], [0, 1]]), tf.constant([[1], [1]]))),
        (Box([0, 0], [1, 1]), Dataset(tf.constant([[1], [1]]), tf.constant([[1], [1]]))),
    ],
)
def test_get_local_dataset_raises_for_invalid_input(
    space: SearchSpaceType, dataset: Dataset
) -> None:
    with pytest.raises(ValueError):
        get_local_dataset(space, dataset)


def test_get_local_dataset_works() -> None:
    search_space_1 = Box([0, 0, 0], [1, 1, 1])
    search_space_2 = Box([5, 5, 5], [10, 10, 10])
    points_1 = search_space_1.sample(10)
    points_2 = search_space_2.sample(20)
    dataset_1 = Dataset(points_1, points_1[:, 0:1])
    dataset_2 = Dataset(points_2, points_2[:, 0:1])
    combined = dataset_1 + dataset_2

    assert tf.shape(get_local_dataset(search_space_1, combined).query_points)[0] == 10
    assert tf.shape(get_local_dataset(search_space_2, combined).query_points)[0] == 20


@pytest.mark.parametrize("num_local_models", [1, 3])
@pytest.mark.parametrize("key", [None, "a"])
def test_copy_to_local_models(num_local_models: int, key: Optional[Tag]) -> None:
    global_model = MagicMock()
    local_models = copy_to_local_models(global_model, num_local_models=num_local_models, key=key)
    assert len(local_models) == num_local_models
    for i, (k, m) in enumerate(local_models.items()):
        assert k == LocalizedTag(key, i)
        assert isinstance(m, MagicMock)
        assert m is not global_model


@pytest.mark.parametrize(
    "datasets, num_other_datasets",
    [
        ({"a": Dataset(tf.constant([[1.0, 2.0]]), tf.constant([[3.0]]))}, 0),
        (
            {
                "a": Dataset(tf.constant([[1.0, 2.0]]), tf.constant([[3.0]])),
                "b": Dataset(tf.constant([[3.0, 7.0], [3.0, 4.0]]), tf.constant([[5.0], [6.0]])),
            },
            0,
        ),
        (
            {
                "a": Dataset(tf.constant([[1.0, 2.0]]), tf.constant([[3.0]])),
                "b": Dataset(tf.constant([[3.0, 7.0], [3.0, 4.0]]), tf.constant([[5.0], [6.0]])),
                LocalizedTag("a", 0): Dataset(tf.constant([[0.0]]), tf.constant([[0.0]])),
            },
            0,
        ),
        (
            {
                "a": Dataset(tf.constant([[1.0, 2.0]]), tf.constant([[3.0]])),
                "b": Dataset(tf.constant([[3.0, 7.0], [3.0, 4.0]]), tf.constant([[5.0], [6.0]])),
                LocalizedTag("c", 2): Dataset(tf.constant([[0.0]]), tf.constant([[0.0]])),
            },
            1,
        ),
    ],
)
@pytest.mark.parametrize("num_local_datasets", [1, 3])
def test_with_local_datasets(
    datasets: Mapping[Tag, Dataset], num_other_datasets: int, num_local_datasets: int
) -> None:
    original_datasets = dict(datasets).copy()
    global_tags = {t for t in original_datasets if not LocalizedTag.from_tag(t).is_local}
    num_global_datasets = len(global_tags)

    datasets = with_local_datasets(datasets, num_local_datasets)
    assert len(datasets) == num_global_datasets * (1 + num_local_datasets) + num_other_datasets

    for global_tag in global_tags:
        assert datasets[global_tag] is original_datasets[global_tag]
        for i in range(num_local_datasets):
            ltag = LocalizedTag(global_tag, i)
            if ltag in original_datasets:
                assert datasets[ltag] is original_datasets[ltag]
            else:
                assert datasets[ltag] is original_datasets[global_tag]


@pytest.mark.parametrize(
    "datasets, indices",
    [
        (
            {
                "a": Dataset(tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([[5.0], [6.0]])),
                "b": Dataset(tf.constant([[7.0, 8.0], [9.0, 1.0]]), tf.constant([[2.0], [3.0]])),
            },
            [tf.constant([0]), tf.constant([0, 1])],
        ),
        (
            {
                "a": Dataset(tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([[5.0], [6.0]])),
                "b": Dataset(tf.constant([[7.0, 8.0], [9.0, 1.0]]), tf.constant([[2.0], [3.0]])),
            },
            [tf.constant([], dtype=tf.int32), tf.constant([0])],
        ),
    ],
)
def test_with_local_datasets_indices(
    datasets: Mapping[Tag, Dataset], indices: Sequence[TensorType]
) -> None:
    original_datasets = dict(datasets).copy()
    global_tags = {t for t in original_datasets if not LocalizedTag.from_tag(t).is_local}
    num_global_datasets = len(global_tags)

    num_local_datasets = len(indices)
    datasets = with_local_datasets(datasets, num_local_datasets, indices)
    assert len(datasets) == num_global_datasets * (1 + num_local_datasets)

    for global_tag in global_tags:
        assert datasets[global_tag] is original_datasets[global_tag]
        for i in range(num_local_datasets):
            ltag = LocalizedTag(global_tag, i)
            if ltag in original_datasets:
                assert datasets[ltag] is original_datasets[ltag]
            else:
                assert len(datasets[ltag].query_points) == len(indices[i])
                assert len(datasets[ltag].observations) == len(indices[i])


@pytest.mark.parametrize(
    "points, tolerance, expected_mask",
    [
        (
            tf.constant([[1.0, 1.0], [1.2, 1.1], [2.0, 2.0], [2.2, 2.2], [3.0, 3.0]]),
            0.5,
            tf.constant([True, False, True, False, True]),
        ),
        (
            tf.constant([[1.0, 2.0], [2.0, 3.0], [1.0, 2.1]]),
            0.2,
            tf.constant([True, True, False]),
        ),
        (
            tf.constant([[1.0], [2.0], [1.0], [3.0], [1.71], [1.699999], [3.29], [3.300001]]),
            0.3,
            tf.constant([True, True, False, True, False, True, False, True]),
        ),
        (
            tf.constant([[1.0], [2.0], [1.0], [3.0], [1.699999], [1.71], [3.300001], [3.29]]),
            0.3,
            tf.constant([True, True, False, True, True, False, True, False]),
        ),
        (tf.constant([[1], [2], [3], [4]]), 1, tf.constant([True, False, True, False])),
        (tf.constant([[1]]), 0, tf.constant([True])),
        (tf.zeros([0, 2]), 0, tf.constant([])),
    ],
)
def test_get_unique_points_mask(
    points: tf.Tensor, tolerance: float, expected_mask: tf.Tensor
) -> None:
    mask = get_unique_points_mask(points, tolerance)
    np.testing.assert_array_equal(mask, expected_mask)

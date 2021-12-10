# Copyright 2021 The Trieste Contributors
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

from typing import Iterable, Union

import tensorflow as tf

from ...data import Dataset
from ...types import TensorType


def get_tensor_spec_from_data(dataset: Dataset) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    """
    Extract tensor specifications for inputs and outputs of neural network models, based on the
    dataset. This utility faciliates constructing neural networks.

    :param dataset: A dataset with ``query_points`` and ``observations`` tensors.
    :return: Tensor specification objects for the ``query_points`` and ``observations`` tensors.
    :raise ValueError: If the dataset is not an instance of :class:`~trieste.data.Dataset`.
    """
    if not isinstance(dataset, Dataset):
        raise ValueError(
            f"This function works only on trieste.data.Dataset objects, however got"
            f"{type(dataset)} which is incompatible."
        )
    input_tensor_spec = tf.TensorSpec(
        shape=(dataset.query_points.shape[-1],),
        dtype=dataset.query_points.dtype,
        name="query_points",
    )
    output_tensor_spec = tf.TensorSpec(
        shape=(dataset.observations.shape[-1],),
        dtype=dataset.observations.dtype,
        name="observations",
    )
    return input_tensor_spec, output_tensor_spec


def sample_with_replacement(dataset: Dataset) -> Dataset:
    """
    Create a new ``dataset`` with data sampled with replacement. This
    function is useful for creating bootstrap samples of data for training ensembles.

    :param dataset: The data that should be sampled.
    :return: A (new) ``dataset`` with sampled data.
    :raise ValueError (or InvalidArgumentError): If the dataset is not an instance of
        :class:`~trieste.data.Dataset` or it is empty.
    """
    if not isinstance(dataset, Dataset):
        raise ValueError(
            f"This function works only on trieste.data.Dataset objects, however got"
            f"{type(dataset)} which is incompatible."
        )
    tf.debugging.assert_positive(len(dataset), message="Dataset must not be empty.")

    n_rows = dataset.observations.shape[0]

    index_tensor = tf.random.uniform(
        (n_rows,), maxval=n_rows, dtype=tf.dtypes.int32
    )  # pylint: disable=all

    observations = tf.gather(dataset.observations, index_tensor)  # pylint: disable=all
    query_points = tf.gather(dataset.query_points, index_tensor)  # pylint: disable=all

    return Dataset(query_points=query_points, observations=observations)

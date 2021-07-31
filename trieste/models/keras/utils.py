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

import numpy as np
import tensorflow as tf

from ...data import Dataset


def size(tensor_spec: tf.TensorSpec) -> int:
    """
    Equivalent to `np.size` for `TensorSpec` objects.
    """
    return int(np.prod(tensor_spec.shape))


def get_tensor_spec_from_data(data: Dataset) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    """
    Extract tensor specifications for neural network inputs and outputs based on the data.
    """
    input_tensor_spec = tf.TensorSpec(
        shape=(data.query_points.shape[-1],),
        dtype=data.query_points.dtype,
        name="query_points",
    )
    output_tensor_spec = tf.TensorSpec(
        shape=(data.observations.shape[-1],),
        dtype=data.observations.dtype,
        name="observations",
    )
    return input_tensor_spec, output_tensor_spec


def sample_with_replacement(dataset: Dataset) -> Dataset:
    """
    Create a new ``dataset`` with data sampled with replacement. This
    function is useful for creating bootstrap samples of data for training ensembles.

    :param dataset: The data whose observations should be sampled.
    :return: A (new) ``dataset`` with sampled data.
    """
    tf.debugging.assert_equal(dataset.observations.shape[0], dataset.query_points.shape[0])

    n_rows = dataset.observations.shape[0]

    index_tensor = tf.random.uniform(
        (n_rows,), maxval=n_rows, dtype=tf.dtypes.int64
    )  # pylint: disable=all

    observations = tf.gather(dataset.observations, index_tensor)  # pylint: disable=all
    query_points = tf.gather(dataset.query_points, index_tensor)  # pylint: disable=all

    return Dataset(query_points=query_points, observations=observations)

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

from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...types import TensorType


def get_tensor_spec_from_data(dataset: Dataset) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    r"""
    Extract tensor specifications for inputs and outputs of neural network models, based on the
    dataset. This utility faciliates constructing neural networks, providing the required
    dimensions for the input and the output of the network. For example

    >>> data = Dataset(
    ...     tf.constant([[0.1, 0.2], [0.3, 0.4]]),
    ...     tf.constant([[0.5], [0.7]])
    ... )
    >>> input_spec, output_spec = get_tensor_spec_from_data(data)
    >>> input_spec
    TensorSpec(shape=(2,), dtype=tf.float32, name='query_points')
    >>> output_spec
    TensorSpec(shape=(1,), dtype=tf.float32, name='observations')

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
        shape=(dataset.query_points.shape[1:]),
        dtype=dataset.query_points.dtype,
        name="query_points",
    )
    output_tensor_spec = tf.TensorSpec(
        shape=(dataset.observations.shape[1:]),
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

    index_tensor = tf.random.uniform((n_rows,), maxval=n_rows, dtype=tf.dtypes.int32)

    observations = tf.gather(dataset.observations, index_tensor, axis=0)
    query_points = tf.gather(dataset.query_points, index_tensor, axis=0)

    return Dataset(query_points=query_points, observations=observations)


def sample_model_index(
    size: TensorType,
    num_samples: TensorType,
    seed: Optional[int] = None,
) -> TensorType:
    """
    Returns samples of indices of individual models in the ensemble.

    If ``num_samples`` is smaller or equal to ``size`` (i.e. the ensemble size) indices are sampled
    without replacement. When ``num_samples`` is larger than ``size`` then until ``size`` is reached
    we sample without replacement, while after that we sample with replacement. The rationale of
    this mixed scheme is that typically one wants to exhaust all networks and then resample them
    only if required.

    :param size: The maximum index, effectively the number of models in the ensemble.
    :param num_samples: The number of samples to take.
    :param seed: Optional RNG seed.
    :return: A tensor with indices.
    """
    shuffle_indices = tf.random.shuffle(tf.range(size), seed=seed)
    if num_samples > size:
        random_indices = tf.random.uniform(
            shape=(tf.cast(num_samples - size, tf.int32),),
            maxval=size,
            dtype=tf.int32,
            seed=seed,
        )
        indices = tf.concat([shuffle_indices, random_indices], 0)
    else:
        indices = shuffle_indices[:num_samples]

    return indices


def negative_log_likelihood(
    y_true: TensorType, y_pred: tfp.distributions.Distribution
) -> TensorType:
    """
    Maximum likelihood objective function for training neural networks.

    :param y_true: The output variable values.
    :param y_pred: The output layer of the model. It has to be a probabilistic neural network
        with a distribution as a final layer.
    :return: Negative log likelihood values.
    """
    return -y_pred.log_prob(y_true)

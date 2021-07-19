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
""" This module contains utilities for :class:`~trieste.observer.Observer` data. """
from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from trieste.type import TensorType


@dataclass(frozen=True)
class Dataset:
    """
    Container for the query points and corresponding observations from an
    :class:`~trieste.observer.Observer`.
    """

    query_points: TensorType
    """ The points at which the :class:`~trieste.observer.Observer` was queried. """

    observations: TensorType
    """ The observed output of the :class:`~trieste.observer.Observer` for each query point. """

    def __post_init__(self) -> None:
        """
        :raise tf.errors.InvalidArgumentError: If ``query_points`` or ``observations`` have
            rank less than two, or they have unequal shape in any but their first dimension.
        """
        tf.debugging.assert_rank_at_least(self.query_points, 2)
        tf.debugging.assert_rank_at_least(self.observations, 2)

        for x, name in ((self.query_points, "query_points"), (self.observations, "observations")):
            tf.debugging.assert_positive(
                x.shape[1:],
                message=f"{name} cannot have empty trailing dimensions, got shape {x.shape}",
            )

        tf.debugging.assert_equal(len(self.query_points), len(self.observations))

    def __add__(self, rhs: Dataset) -> Dataset:
        r"""
        Return the :class:`Dataset` whose query points are the result of concatenating the
        `query_points` in each :class:`Dataset` along the zeroth axis, and the same for the
        `observations`. For example:

        >>> d1 = Dataset(
        ...     tf.constant([[0.1, 0.2], [0.3, 0.4]]),
        ...     tf.constant([[0.5, 0.6], [0.7, 0.8]])
        ... )
        >>> d2 = Dataset(tf.constant([[0.9, 1.0]]), tf.constant([[1.1, 1.2]]))
        >>> (d1 + d2).query_points
        <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
        array([[0.1, 0.2],
               [0.3, 0.4],
               [0.9, 1. ]], dtype=float32)>
        >>> (d1 + d2).observations
        <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
        array([[0.5, 0.6],
               [0.7, 0.8],
               [1.1, 1.2]], dtype=float32)>

        :param rhs: A :class:`Dataset` with the same shapes as this one, except in the zeroth
            dimension, which can have any size.
        :return: The result of concatenating the :class:`Dataset`\ s.
        :raise InvalidArgumentError: If the shapes of the `query_points` in each :class:`Dataset`
            differ in any but the zeroth dimension. The same applies for `observations`.
        """
        return Dataset(
            tf.concat([self.query_points, rhs.query_points], axis=0),
            tf.concat([self.observations, rhs.observations], axis=0),
        )

    def __len__(self) -> TensorType:
        """
        :return: The number of query points, or equivalently the number of observations.
        """
        return tf.shape(self.observations)[0]

    def __getitem__(self, idx: slice) -> Dataset:
        """
        Slice the :class:`Dataset` along the first axis. For example:

        >>> dataset = Dataset(
        ...     tf.constant([[0.0], [1.0], [2.0], [3.0]]),
        ...     tf.constant([[0.0, 0.0], [0.1, 0.01], [0.2, 0.02], [0.3, 0.03]])
        ... )
        >>> dataset[1:3].query_points.numpy()
        array([[1.],
               [2.]], dtype=float32)
        >>> dataset[1:3].observations.numpy()
        array([[0.1, 0.01],
               [0.2, 0.02]], dtype=float32)

        **Note:** Indexing with integers, like `data[0]`, is not supported.

        :param idx: The `slice` of indices to return.
        :return: The points specified by ``idx`` from this :class:`Dataset`.
        :raise tf.errors.InvalidArgumentError: If the slice bounds extend beyond the data.
        """
        return Dataset(self.query_points[idx], self.observations[idx])

    def __deepcopy__(self, memo: dict[int, object]) -> Dataset:
        return self

    def astuple(self) -> tuple[TensorType, TensorType]:
        """
        **Note:** Unlike the standard library function `dataclasses.astuple`, this method does
        **not** deepcopy the attributes.

        :return: A 2-tuple of the :attr:`query_points` and :attr:`observations`.
        """
        return self.query_points, self.observations

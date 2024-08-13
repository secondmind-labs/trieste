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
from typing import Optional, Sequence

import tensorflow as tf

from trieste.types import TensorType


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
        :raise ValueError (or InvalidArgumentError): If ``query_points`` or ``observations`` have \
            rank less than two, or they have unequal shape in any but their last dimension.
        """
        tf.debugging.assert_rank_at_least(self.query_points, 2)
        tf.debugging.assert_rank_at_least(self.observations, 2)

        if 0 in (self.query_points.shape[-1], self.observations.shape[-1]):
            raise ValueError(
                f"query_points and observations cannot have dimension 0, got shapes"
                f" {self.query_points.shape} and {self.observations.shape}."
            )

        if (
            self.query_points.shape[:-1] != self.observations.shape[:-1]
            # can't check dynamic shapes, so trust that they're ok (if not, they'll fail later)
            and None not in self.query_points.shape[:-1]
            and None not in self.observations.shape[:-1]
        ):
            raise ValueError(
                f"Leading shapes of query_points and observations must match. Got shapes"
                f" {self.query_points.shape}, {self.observations.shape}."
            )

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

    def __len__(self) -> tf.Tensor:
        """
        :return: The number of query points, or equivalently the number of observations.
        """
        return tf.shape(self.observations)[0]

    def __deepcopy__(self, memo: dict[int, object]) -> Dataset:
        return self

    def astuple(self) -> tuple[TensorType, TensorType]:
        """
        **Note:** Unlike the standard library function `dataclasses.astuple`, this method does
        **not** deepcopy the attributes.

        :return: A 2-tuple of the :attr:`query_points` and :attr:`observations`.
        """
        return self.query_points, self.observations


def check_and_extract_fidelity_query_points(
    query_points: TensorType, max_fidelity: Optional[int] = None
) -> tuple[TensorType, TensorType]:
    """Check whether the final column of a tensor is close enough to ints
    to be reasonably considered to represent fidelities.

    The final input column of multi-fidelity data should be a reference to
    the fidelity of the query point. We cannot have mixed type tensors, but
    we can check that thhe final column values are suitably close to integers.

    :param query_points: Data to check final column of.
    :raise: ValueError: If there are not enough columns to be multifidelity data
    :raise InvalidArgumentError: If any value in the final column is far from an integer
    :return: Query points without fidelity column
        and the fidelities of each of the query points
    """
    # Check we have sufficient columns
    if query_points.shape[-1] < 2:
        raise ValueError(
            "Query points do not have enough columns to be multifidelity,"
            f" need at least 2, got {query_points.shape[1]}"
        )
    input_points = query_points[..., :-1]
    fidelity_col = query_points[..., -1:]
    # Check fidelity column values are close to ints
    tf.debugging.assert_equal(
        tf.round(fidelity_col),
        fidelity_col,
        message="Fidelity column should be float(int), but got a float that"
        " was not close to an int",
    )
    # Check fidelity column values are non-negative

    tf.debugging.assert_non_negative(fidelity_col, message="Fidelity must be non-negative")
    if max_fidelity is not None:
        max_input_fid = tf.reduce_max(fidelity_col)
        max_fidelity_float = tf.cast(max_fidelity, dtype=query_points.dtype)
        tf.debugging.assert_less_equal(
            max_input_fid,
            max_fidelity_float,
            message=(
                f"Model only supports fidelities up to {max_fidelity},"
                f" but {max_input_fid} was passed"
            ),
        )

    return input_points, fidelity_col


def split_dataset_by_fidelity(dataset: Dataset, num_fidelities: int) -> Sequence[Dataset]:
    """Split dataset into individual datasets without fidelity information

    :param dataset: Dataset for which to split fidelities
    :param num_fidlities: Number of fidelities in the problem (not just dataset)
    :return: Ordered list of datasets with lowest fidelity at index 0 and highest at -1
    """
    if num_fidelities < 1:
        raise ValueError(f"Data must have 1 or more fidelities, got {num_fidelities}")
    datasets = [get_dataset_for_fidelity(dataset, fidelity) for fidelity in range(num_fidelities)]
    return datasets


def get_dataset_for_fidelity(dataset: Dataset, fidelity: int) -> Dataset:
    """Get a dataset with only the specified fidelity of data in

    :param dataset: The dataset from which to extract the single fidelity data
    :param fidelity: The fidelity to extract the data for
    :return: Dataset with a single fidelity and no fidelity column
    """

    input_points, fidelity_col = check_and_extract_fidelity_query_points(
        dataset.query_points
    )  # [..., D], [..., 1]
    mask = fidelity_col == fidelity  # [..., ]
    inds = tf.where(mask)[..., 0]  # [..., ]
    inputs_for_fidelity = tf.gather(input_points, inds, axis=0)  # [..., D]
    observations_for_fidelity = tf.gather(dataset.observations, inds, axis=0)  # [..., 1]
    return Dataset(query_points=inputs_for_fidelity, observations=observations_for_fidelity)


def add_fidelity_column(query_points: TensorType, fidelity: int) -> TensorType:
    """Add fidelity column to query_points without fidelity data

    :param query_points: query points without fidelity to add fidelity column to
    :param fidelity: fidelity to populate fidelity column with
    :return: TensorType of query points with fidelity column added
    """
    fidelity_col = tf.ones((tf.shape(query_points)[-2], 1), dtype=query_points.dtype) * fidelity
    query_points_for_fidelity = tf.concat([query_points, fidelity_col], axis=-1)
    return query_points_for_fidelity

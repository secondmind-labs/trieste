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
""" Definitions and utilities for observers of objective functions. """
from __future__ import annotations

from typing import Callable, Mapping, Union

import tensorflow as tf
from typing_extensions import Final

from .data import Dataset
from .types import Tag, TensorType

SingleObserver = Callable[[TensorType], Dataset]
"""
Type alias for an observer of the objective function (that takes query points and returns an
unlabelled dataset).
"""

MultiObserver = Callable[[TensorType], Mapping[Tag, Dataset]]
"""
Type alias for an observer of the objective function (that takes query points and returns labelled
datasets).
"""

Observer = Union[SingleObserver, MultiObserver]
"""
Type alias for an observer, returning either labelled datasets or a single unlabelled dataset.
"""

OBJECTIVE: Final[Tag] = "OBJECTIVE"
"""
A tag typically used by acquisition rules to denote the data sets and models corresponding to the
optimization objective.
"""


def _is_finite(t: TensorType) -> TensorType:
    return tf.logical_and(tf.math.is_finite(t), tf.logical_not(tf.math.is_nan(t)))


def filter_finite(query_points: TensorType, observations: TensorType) -> Dataset:
    """
    :param query_points: A tensor of shape (N x M).
    :param observations: A tensor of shape (N x 1).
    :return: A :class:`~trieste.data.Dataset` containing all the rows in ``query_points`` and
        ``observations`` where the ``observations`` are finite numbers.
    :raise ValueError or InvalidArgumentError: If ``query_points`` or ``observations`` have invalid
        shapes.
    """
    tf.debugging.assert_shapes([(observations, ("N", 1))])

    mask = tf.reshape(_is_finite(observations), [-1])
    return Dataset(tf.boolean_mask(query_points, mask), tf.boolean_mask(observations, mask))


def map_is_finite(query_points: TensorType, observations: TensorType) -> Dataset:
    """
    :param query_points: A tensor.
    :param observations: A tensor.
    :return: A :class:`~trieste.data.Dataset` containing all the rows in ``query_points``,
        along with the tensor result of mapping the elements of ``observations`` to: `1` if they are
        a finite number, else `0`, with dtype `tf.uint8`.
    :raise ValueError or InvalidArgumentError: If ``query_points`` and ``observations`` do not
        satisfy the shape constraints of :class:`~trieste.data.Dataset`.
    """
    return Dataset(query_points, tf.cast(_is_finite(observations), tf.uint8))

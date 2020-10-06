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
from typing import Callable, Dict

import tensorflow as tf

from .datasets import Dataset


Observer = Callable[[tf.Tensor], Dict[str, Dataset]]
"""
Type alias for an observer of the objective function (that takes query points and returns labelled
datasets).
"""


def _is_finite(t: tf.Tensor) -> tf.Tensor:
    return tf.logical_and(tf.math.is_finite(t), tf.logical_not(tf.math.is_nan(t)))


def filter_finite(query_points: tf.Tensor, observations: tf.Tensor) -> Dataset:
    """
    :param query_points: A tensor of shape (N x M).
    :param observations: A tensor of shape (N x 1).
    :return: A :class:`~trieste.datasets.Dataset` containing all the rows in ``query_points`` and
        ``observations`` where the ``observations`` are finite numbers.
    :raise ValueError or InvalidArgumentError: If ``query_points`` or ``observations`` have invalid
        shapes.
    """
    tf.debugging.assert_shapes([(observations, ("N", 1))])

    mask = tf.reshape(_is_finite(observations), [-1])
    return Dataset(tf.boolean_mask(query_points, mask), tf.boolean_mask(observations, mask))


def map_is_finite(query_points: tf.Tensor, observations: tf.Tensor) -> Dataset:
    """
    :param query_points: A tensor.
    :param observations: A tensor.
    :return: A :class:`~trieste.datasets.Dataset` containing all the rows in ``query_points``,
        along with the tensor result of mapping the elements of ``observations`` to: `1` if they are
        a finite number, else `0`.
    :raise ValueError or InvalidArgumentError: If ``query_points`` and ``observations`` do not
        satisfy the shape constraints of :class:`~trieste.datasets.Dataset`.
    """
    return Dataset(query_points, tf.cast(_is_finite(observations), tf.uint8))

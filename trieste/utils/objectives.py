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
""" This module contains toy objective functions, useful for experimentation. """
import math
from typing import Callable

import tensorflow as tf

from ..datasets import Dataset
from ..observer import Observer
from ..type import TensorType


def branin(x: TensorType) -> TensorType:
    """
    The Branin-Hoo function, rescaled to have zero mean and unit variance over :math:`[0, 1]^2`, see
    `here <https://www.sfu.ca/~ssurjano/branin.html>`_ for details.

    :param x: Array of two-dimensional points at which to evaluate the function. Shape (..., N, 2).
    :return: The values of the rescaled Branin-Hoo at points in ``x``. Shape (..., N, 1).
    :raise ValueError (or InvalidArgumentError): If the points in ``x`` are not two-dimensional.
    """
    tf.debugging.assert_shapes([(x, (..., "N", 2))])

    x0 = x[..., :1] * 15.0 - 5.0
    x1 = x[..., 1:] * 15.0

    a = 1
    b = 5.1 / (4 * math.pi ** 2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8 * math.pi)

    return a * (x1 - b * x0 ** 2 + c * x0 - r) ** 2 + s * (1 - t) * tf.cos(x0) + s


_ORIGINAL_BRANIN_ARGMIN = tf.constant([[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]])

BRANIN_GLOBAL_ARGMIN = (_ORIGINAL_BRANIN_ARGMIN + [5.0, 0.0]) / 15.0
""" The three global minimizers of the :func:`branin` function. """

BRANIN_GLOBAL_MINIMUM = tf.constant(0.397887)
""" The global miminum of the :func:`branin` function. """


def mk_observer(objective: Callable[[tf.Tensor], tf.Tensor], key: str) -> Observer:
    """
    :param objective: An objective function designed to be used with a single data set and model.
    :param key: The key to use to access the data from the observer result.
    :return: An observer returning the data from ``objective`` with the specified ``key``.
    """
    return lambda qp: {key: Dataset(qp, objective(qp))}

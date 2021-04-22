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
"""
This module contains toy objective functions, useful for experimentation. A number of them have been
taken from `this Virtual Library of Simulation Experiments
<https://www.sfu.ca/~ssurjano/optimization.html>`_.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import tensorflow as tf

from ..data import Dataset
from ..observer import Observer
from ..type import TensorType


def branin(x: TensorType) -> TensorType:
    """
    The Branin-Hoo function, rescaled to have zero mean and unit variance over :math:`[0, 1]^2`. See
    :cite:`Picheny2013` for details.

    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 2))])

    x0 = x[..., :1] * 15.0 - 5.0
    x1 = x[..., 1:] * 15.0

    a = 1
    b = 5.1 / (4 * math.pi ** 2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8 * math.pi)

    return a * (x1 - b * x0 ** 2 + c * x0 - r) ** 2 + s * (1 - t) * tf.cos(x0) + s


_ORIGINAL_BRANIN_MINIMIZERS = tf.constant(
    [[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]], tf.float64
)

BRANIN_MINIMIZERS = (_ORIGINAL_BRANIN_MINIMIZERS + [5.0, 0.0]) / 15.0
"""
The three global minimizers of the :func:`branin` function over :math:`[0, 1]^2`, with shape [3, 2]
and dtype float64.
"""

BRANIN_MINIMUM = tf.constant([0.397887], tf.float64)
""" The global minimum of the :func:`branin` function, with shape [1] and dtype float64. """


def gramacy_lee(x: TensorType) -> TensorType:
    """
    The Gramacy & Lee function, typically evaluated over :math:`[0.5, 2.5]`. See
    :cite:`gramacy2012cases` for details.

    :param x: Where to evaluate the function, with shape [..., 1].
    :return: The function values, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 1))])
    return tf.sin(10 * math.pi * x) / (2 * x) + (x - 1) ** 4


GRAMACY_LEE_MINIMIZER = tf.constant([[0.548562]], tf.float64)
"""
The global minimizer of the :func:`gramacy_lee` function over :math:`[0.5, 2.5]`, with shape [1, 1]
and dtype float64.
"""

GRAMACY_LEE_MINIMUM = tf.constant([-0.869011], tf.float64)
"""
The global minimum of the :func:`gramacy_lee` function over :math:`[0.5, 2.5]`, with shape [1] and
dtype float64.
"""


def logarithmic_goldstein_price(x: TensorType) -> TensorType:
    """
    A logarithmic form of the Goldstein-Price function, with zero mean and unit variance over
    :math:`[0, 1]^2`. See :cite:`Picheny2013` for details.

    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 2))])

    x0, x1 = tf.split(4 * x - 2, 2, axis=-1)

    a = (x0 + x1 + 1) ** 2
    b = 19 - 14 * x0 + 3 * x0 ** 2 - 14 * x1 + 6 * x0 * x1 + 3 * x1 ** 2
    c = (2 * x0 - 3 * x1) ** 2
    d = 18 - 32 * x0 + 12 * x0 ** 2 + 48 * x1 - 36 * x0 * x1 + 27 * x1 ** 2

    return (1 / 2.427) * (tf.math.log((1 + a * b) * (30 + c * d)) - 8.693)


LOGARITHMIC_GOLDSTEIN_PRICE_MINIMIZER = tf.constant([[0.5, 0.25]], tf.float64)
"""
The global minimizer for the :func:`logarithmic_goldstein_price` function, with shape [1, 2] and
dtype float64.
"""

LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM = tf.constant([-3.12913], tf.float64)
"""
The global minimum for the :func:`logarithmic_goldstein_price` function, with shape [1] and dtype
float64.
"""


def hartmann_3(x: TensorType) -> TensorType:
    """
    The Hartmann 3 test function over :math:`[0, 1]^3`. This function has 3 local
    and one global minima. See https://www.sfu.ca/~ssurjano/hart3.html for details

    :param x: The points at which to evaluate the function, with shape [..., 3].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 3))])

    a = [1.0, 1.2, 3.0, 3.2]
    A = [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]]
    P = [
        [0.3689, 0.1170, 0.2673],
        [0.4699, 0.4387, 0.7470],
        [0.1091, 0.8732, 0.5547],
        [0.0381, 0.5743, 0.8828],
    ]

    inner_sum = -tf.reduce_sum(A * (tf.expand_dims(x, 1) - P) ** 2, -1)
    return -tf.reduce_sum(a * tf.math.exp(inner_sum), -1, keepdims=True)


HARTMANN_3_MINIMIZER = tf.constant([[0.114614, 0.555649, 0.852547]], tf.float64)
"""
The global minimizer for the :func:`hartmann_3` function, with shape [1, 3] and
dtype float64.
"""


HARTMANN_3_MINIMUM = tf.constant([-3.86278], tf.float64)
"""
The global minimum for the :func:`hartmann_3` function, with shape [1] and dtype
float64.
"""


def hartmann_6(x: TensorType) -> TensorType:
    """
    The Hartmann 6 test function over :math:`[0, 1]^6`. This function has
    6 local and one global minima. See https://www.sfu.ca/~ssurjano/hart6.html
    for details.

    :param x: The points at which to evaluate the function, with shape [..., 6].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 6))])

    a = [1.0, 1.2, 3.0, 3.2]
    A = [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
    P = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]

    inner_sum = -tf.reduce_sum(A * (tf.expand_dims(x, 1) - P) ** 2, -1)
    return -tf.reduce_sum(a * tf.math.exp(inner_sum), -1, keepdims=True)


HARTMANN_6_MINIMIZER = tf.constant(
    [[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]], tf.float64
)
"""
The global minimizer for the :func:`hartmann_6` function, with shape [1, 6] and
dtype float64.
"""


HARTMANN_6_MINIMUM = tf.constant([-3.32237], tf.float64)
"""
The global minimum for the :func:`hartmann_6` function, with shape [1] and dtype
float64.
"""


def mk_observer(objective: Callable[[TensorType], TensorType], key: str) -> Observer:
    """
    :param objective: An objective function designed to be used with a single data set and model.
    :param key: The key to use to access the data from the observer result.
    :return: An observer returning the data from ``objective`` with the specified ``key``.
    """
    return lambda qp: {key: Dataset(qp, objective(qp))}

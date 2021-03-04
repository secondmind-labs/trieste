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
    :cite:`gramacy2010cases` and :cite:`Ranjan2013` for details.

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


def mk_observer(objective: Callable[[TensorType], TensorType], key: str) -> Observer:
    """
    :param objective: An objective function designed to be used with a single data set and model.
    :param key: The key to use to access the data from the observer result.
    :return: An observer returning the data from ``objective`` with the specified ``key``.
    """
    return lambda qp: {key: Dataset(qp, objective(qp))}

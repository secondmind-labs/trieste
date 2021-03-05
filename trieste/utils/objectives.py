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
from abc import ABC, abstractmethod
from functools import partial

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


class MultiObjectiveTestProblem(ABC):
    r"""Base class for test multi-objective test functions.
    between a provided point and the closest point on the true pareto front.
    """
    dim = None
    bounds = None

    @abstractmethod
    def prepare_benchmark(self):
        """
        Evaluate Original problem
        """

    @abstractmethod
    def gen_pareto_optimal_points(self, n: int):
        """
        Generate `n` pareto optimal points
        """


class VLMOP2(MultiObjectiveTestProblem):
    """
    The VLMOP2n, typically evaluated over :math:`[-2, 2]^2`. See
    :cite:`van1999multiobjective`  for details.
    """
    bounds = [[-2.0, 2.0]] * 2
    dim = 2

    def prepare_benchmark(self):
        return vlmop2

    def gen_pareto_optimal_points(self, n: int):
        """
        :cite: fonseca1995multiobjective
        True pareto lies on -1/sqrt(2)-1/sqrt(2) and x1=x2
        """
        tf.debugging.assert_greater(n, 0)
        _x = tf.linspace([-1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0)], n)
        return vlmop2(tf.concat([_x, _x], axis=1))


def vlmop2(x: TensorType) -> TensorType:
    tf.debugging.assert_shapes([(x, (..., 2))])
    transl = 1 / tf.sqrt(2.0)
    y1 = 1 - tf.exp(-1 * tf.reduce_sum((x - transl) ** 2, axis=1))
    y2 = 1 - tf.exp(-1 * tf.reduce_sum((x + transl) ** 2, axis=1))
    return tf.stack([y1, y2], axis=1)


class DTLZ(MultiObjectiveTestProblem):
    r"""DTLZ series multi-objective test functions.
    :cite: deb2002scalable
    """

    def __init__(self, input_dim: int, num_objective: int):
        tf.debugging.assert_greater(input_dim, 0)
        tf.debugging.assert_greater(num_objective, 0)
        tf.debugging.assert_greater(
            input_dim, num_objective,
            f'input dimension {input_dim}'
            f'  must be greater than function objective numbers {num_objective}')
        self.dim = input_dim
        self.M = num_objective
        self.k = self.dim - self.M + 1
        self.bounds = [[0, 1]] * input_dim
        super().__init__()


class DTLZ1(DTLZ):
    def prepare_benchmark(self):
        return partial(dtlz1, m=self.M, k=self.k, d=self.dim)

    def gen_pareto_optimal_points(self, n: int, seed=None):
        tf.debugging.assert_greater_equal(self.M, 2)
        rnd = tf.random.uniform([n, self.M - 1], minval=0, maxval=1)
        strnd = tf.sort(rnd, axis=-1)
        strnd = tf.concat([tf.zeros([n, 1]), strnd, tf.ones([n, 1])], axis=-1)
        return strnd[..., 1:] - strnd[..., :-1]


def dtlz1(x: TensorType, m: int, k: int, d: int) -> TensorType:
    tf.debugging.assert_equal(x.shape[-1], d,
                              f'input x dim: {x.shape[-1]} is not align with pre-specified dim: {d}')

    def g(xM):
        return 100 * \
               (k
                + tf.reduce_sum((xM - 0.5) ** 2
                - tf.cos(20 * math.pi * (xM - 0.5)), axis=-1, keepdims=True))

    f = None
    for i in range(m):
        xM = x[:, m - 1:]
        y = (1 + g(xM))
        y *= 1 / 2 * tf.reduce_prod(x[:, :m - 1 - i])
        if i > 0:
            y *= 1 - x[:, m - i - 1, tf.newaxis]
        f = y if f is None else tf.concat([f, y], 1)
    return f


class DTLZ2(DTLZ):
    def prepare_benchmark(self):
        return partial(dtlz2, m=self.M, d=self.dim)

    def gen_pareto_optimal_points(self, n: int, seed=None):
        tf.debugging.assert_greater_equal(self.M, 2)
        rnd = tf.random.normal([n, self.M], seed=seed)
        samples = tf.abs(rnd / tf.norm(rnd, axis=-1, keepdims=True))
        return samples


def dtlz2(x: TensorType, m: int, d: int) -> TensorType:
    tf.debugging.assert_equal(x.shape[-1], d,
                              f'input x dim: {x.shape[-1]} is not align with pre-specified dim: {d}')

    def g(xM):
        z = (xM - 0.5) ** 2
        return tf.reduce_sum(z, axis=1, keepdims=True)

    f = None
    for i in range(m):
        y = (1 + g(x[:, m - 1:]))
        for j in range(m - 1 - i):
            y *= tf.cos(math.pi / 2 * x[:, j, tf.newaxis])
        if i > 0:
            y *= tf.sin(math.pi / 2 * x[:, m - 1 - i, tf.newaxis])
        f = y if f is None else tf.concat([f, y], 1)
    return f


def mk_observer(objective: Callable[[TensorType], TensorType], key: str) -> Observer:
    """
    :param objective: An objective function designed to be used with a single data set and model.
    :param key: The key to use to access the data from the observer result.
    :return: An observer returning the data from ``objective`` with the specified ``key``.
    """
    return lambda qp: {key: Dataset(qp, objective(qp))}

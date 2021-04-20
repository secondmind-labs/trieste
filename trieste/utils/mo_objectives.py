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
This module contains toy multi-objective functions, useful for experimentation.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from functools import partial

import tensorflow as tf

from ..type import TensorType


class MultiObjectiveTestProblem(ABC):
    """base class for synthetic multi-objective test functions"""

    @abstractmethod
    def prepare_benchmark(self):
        """
        Evaluate original problem.
        """

    @abstractmethod
    def gen_pareto_optimal_points(self, n: int):
        """
        Generate `n` pareto optimal points.
        """


class VLMOP2(MultiObjectiveTestProblem):
    """
    The VLMOP2 function, typically evaluated over :math:`[-2, 2]^2`. See
    See :cite:`van1999multiobjective` for details.
    """

    bounds = [[-2.0] * 2, [2.0] * 2]
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
    tf.debugging.assert_shapes([(x, ("N", 2))], message="vlmop2 only allow 2d input")
    transl = 1 / tf.sqrt(2.0)
    y1 = 1 - tf.exp(-1 * tf.reduce_sum((x - transl) ** 2, axis=1))
    y2 = 1 - tf.exp(-1 * tf.reduce_sum((x + transl) ** 2, axis=1))
    return tf.stack([y1, y2], axis=1)


class DTLZ(MultiObjectiveTestProblem):
    """
    DTLZ series multi-objective test functions.
    See :cite:deb2002scalable for details.
    """

    def __init__(self, input_dim: int, num_objective: int):
        tf.debugging.assert_greater(input_dim, 0)
        tf.debugging.assert_greater(num_objective, 0)
        tf.debugging.assert_greater(
            input_dim,
            num_objective,
            f"input dimension {input_dim}"
            f"  must be greater than function objective numbers {num_objective}",
        )
        self.dim = input_dim
        self.M = num_objective
        self.k = self.dim - self.M + 1
        self.bounds = [[0] * input_dim, [1] * input_dim]


class DTLZ1(DTLZ):
    def prepare_benchmark(self):
        return partial(dtlz1, m=self.M, k=self.k, d=self.dim)

    def gen_pareto_optimal_points(self, n: int, seed=None):
        tf.debugging.assert_greater_equal(self.M, 2)
        rnd = tf.random.uniform([n, self.M - 1], minval=0, maxval=1)
        strnd = tf.sort(rnd, axis=-1)
        strnd = tf.concat([tf.zeros([n, 1]), strnd, tf.ones([n, 1])], axis=-1)
        return 0.5 * (strnd[..., 1:] - strnd[..., :-1])


def dtlz1(x: TensorType, m: int, k: int, d: int) -> TensorType:
    tf.debugging.assert_shapes(
        [(x, ("N", d))],
        message=f"input x dim: {x.shape[-1]} is not align with pre-specified dim: {d}",
    )

    def g(xM):
        return 100 * (
            k
            + tf.reduce_sum(
                (xM - 0.5) ** 2 - tf.cos(20 * math.pi * (xM - 0.5)), axis=-1, keepdims=True
            )
        )

    f = None
    for i in range(m):
        xM = x[:, m - 1 :]
        y = 1 + g(xM)
        y *= 1 / 2 * tf.reduce_prod(x[:, : m - 1 - i], axis=1, keepdims=True)
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
    tf.debugging.assert_shapes(
        [(x, ("N", d))],
        message=f"input x dim: {x.shape[-1]} is not align with pre-specified dim: {d}",
    )

    def g(xM):
        z = (xM - 0.5) ** 2
        return tf.reduce_sum(z, axis=1, keepdims=True)

    f = None
    for i in range(m):
        y = 1 + g(x[:, m - 1 :])
        for j in range(m - 1 - i):
            y *= tf.cos(math.pi / 2 * x[:, j, tf.newaxis])
        if i > 0:
            y *= tf.sin(math.pi / 2 * x[:, m - 1 - i, tf.newaxis])
        f = y if f is None else tf.concat([f, y], 1)
    return f

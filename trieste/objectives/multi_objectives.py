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
This module contains synthetic multi-objective functions, useful for experimentation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import tensorflow as tf
from typing_extensions import Protocol

from ..space import Box
from ..types import TensorType
from .single_objectives import ObjectiveTestProblem, branin


class NoAnalyticalParetoPointsError(Exception):
    pass


class GenParetoOptimalPoints(Protocol):
    """A Protocol representing a function that generates Pareto optimal points."""

    def __call__(self, n: int, seed: int | None = None) -> TensorType:
        """
        Generate `n` Pareto optimal points.

        :param n: The number of pareto optimal points to be generated.
        :param seed: An integer used to create a random seed for distributions that
         used to generate pareto optimal points.
        :return: The Pareto optimal points.
        """


class ConstraintTestProblem(Protocol):
    """A Protocol representing function returning constraint values given specified inputs."""

    def __call__(self, x: TensorType, threshold: Optional[float] = 0.0) -> TensorType:
        """
        return the constraint value given specified inputs `x` and `threshold`

        :param x: The points at which to evaluate the function, with shape [..., d].
        :param threshold: a feasibility threshold used to determine the constraint, by default 0 is
            used as in the original problem.
        :return: The constraint values.
        """


@dataclass(frozen=True)
class MultiObjectiveTestProblem(ObjectiveTestProblem):
    """
    Convenience container class for synthetic multi-objective test functions, containing
    a generator for the pareto optimal points, which can be used as a reference of performance
    measure of certain multi-objective optimization algorithms.
    """

    gen_pareto_optimal_points: GenParetoOptimalPoints
    """Function to generate Pareto optimal points, given the number of points and an optional
    random number seed."""


@dataclass(frozen=True)
class ConstrainedMultiObjectiveTestProblem(MultiObjectiveTestProblem):
    """
    Convenience container class for synthetic constrained multi-objective test functions, containing
    additionally a constraint function.
    """

    constraint: ConstraintTestProblem
    """The synthetic test function's constraints"""


def vlmop2(x: TensorType, d: int) -> TensorType:
    """
    The VLMOP2 synthetic function.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :param d: The dimensionality of the synthetic function.
    :return: The function values at ``x``, with shape [..., 2].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes(
        [(x, (..., d))],
        message=f"input x dim: {x.shape[-1]} does not align with pre-specified dim: {d}",
    )
    transl = 1 / tf.sqrt(tf.cast(d, x.dtype))
    y1 = 1 - tf.exp(-1 * tf.reduce_sum((x - transl) ** 2, axis=-1))
    y2 = 1 - tf.exp(-1 * tf.reduce_sum((x + transl) ** 2, axis=-1))
    return tf.stack([y1, y2], axis=-1)


def VLMOP2(input_dim: int) -> MultiObjectiveTestProblem:
    """
    The VLMOP2 problem, typically evaluated over :math:`[-2, 2]^d`.
    The idea pareto fronts lies on -1/sqrt(d) - 1/sqrt(d) and x1=...=xdim.

    See :cite:`van1999multiobjective` and :cite:`fonseca1995multiobjective`
    (the latter for discussion of pareto front property) for details.

    :param input_dim: The input dimensionality of the synthetic function.
    :return: The problem specification.
    """

    def gen_pareto_optimal_points(n: int, seed: int | None = None) -> TensorType:
        tf.debugging.assert_greater(n, 0)
        transl = 1 / tf.sqrt(tf.cast(input_dim, tf.float64))
        _x = tf.tile(tf.linspace([-transl], [transl], n), [1, input_dim])
        return vlmop2(_x, input_dim)

    return MultiObjectiveTestProblem(
        name=f"VLMOP2({input_dim})",
        objective=partial(vlmop2, d=input_dim),
        search_space=Box([-2.0], [2.0]) ** input_dim,
        gen_pareto_optimal_points=gen_pareto_optimal_points,
    )


def dtlz_mkd(input_dim: int, num_objective: int) -> tuple[int, int, int]:
    """Return m/k/d values for dtlz synthetic functions."""
    tf.debugging.assert_greater(input_dim, 0)
    tf.debugging.assert_greater(num_objective, 0)
    tf.debugging.assert_greater(
        input_dim,
        num_objective,
        f"input dimension {input_dim}"
        f"  must be greater than function objective numbers {num_objective}",
    )
    M = num_objective
    k = input_dim - M + 1
    d = input_dim
    return (M, k, d)


def dtlz1(x: TensorType, m: int, k: int, d: int) -> TensorType:
    """
    The DTLZ1 synthetic function.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :param m: The objective numbers.
    :param k: The input dimensionality for g.
    :param d: The dimensionality of the synthetic function.
    :return: The function values at ``x``, with shape [..., m].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes(
        [(x, (..., d))],
        message=f"input x dim: {x.shape[-1]} does not align with pre-specified dim: {d}",
    )
    tf.debugging.assert_greater(m, 0, message=f"positive objective numbers expected but found {m}")

    def g(xM: TensorType) -> TensorType:
        return 100 * (
            k
            + tf.reduce_sum(
                (xM - 0.5) ** 2 - tf.cos(20 * math.pi * (xM - 0.5)), axis=-1, keepdims=True
            )
        )

    ta = tf.TensorArray(x.dtype, size=m)
    for i in range(m):
        xM = x[..., m - 1 :]
        y = 1 + g(xM)
        y *= 1 / 2 * tf.reduce_prod(x[..., : m - 1 - i], axis=-1, keepdims=True)
        if i > 0:
            y *= 1 - x[..., m - i - 1, tf.newaxis]
        ta = ta.write(i, y)

    return tf.squeeze(tf.concat(tf.split(ta.stack(), m, axis=0), axis=-1), axis=0)


def DTLZ1(input_dim: int, num_objective: int) -> MultiObjectiveTestProblem:
    """
    The DTLZ1 problem, the idea pareto fronts lie on a linear hyper-plane.
    See :cite:`deb2002scalable` for details.

    :param input_dim: The input dimensionality of the synthetic function.
    :param num_objective: The number of objectives.
    :return: The problem specification.
    """
    M, k, d = dtlz_mkd(input_dim, num_objective)

    def gen_pareto_optimal_points(n: int, seed: int | None = None) -> TensorType:
        tf.debugging.assert_greater_equal(M, 2)
        rnd = tf.random.uniform([n, M - 1], minval=0, maxval=1, seed=seed, dtype=tf.float64)
        strnd = tf.sort(rnd, axis=-1)
        strnd = tf.concat(
            [tf.zeros([n, 1], dtype=tf.float64), strnd, tf.ones([n, 1], dtype=tf.float64)], axis=-1
        )
        return 0.5 * (strnd[..., 1:] - strnd[..., :-1])

    return MultiObjectiveTestProblem(
        name=f"DTLZ1({input_dim}, {num_objective})",
        objective=partial(dtlz1, m=M, k=k, d=d),
        search_space=Box([0.0], [1.0]) ** d,
        gen_pareto_optimal_points=gen_pareto_optimal_points,
    )


def dtlz2(x: TensorType, m: int, d: int) -> TensorType:
    """
    The DTLZ2 synthetic function.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :param m: The objective numbers.
    :param d: The dimensionality of the synthetic function.
    :return: The function values at ``x``, with shape [..., m].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes(
        [(x, (..., d))],
        message=f"input x dim: {x.shape[-1]} does not align with pre-specified dim: {d}",
    )
    tf.debugging.assert_greater(m, 0, message=f"positive objective numbers expected but found {m}")

    def g(xM: TensorType) -> TensorType:
        z = (xM - 0.5) ** 2
        return tf.reduce_sum(z, axis=-1, keepdims=True)

    ta = tf.TensorArray(x.dtype, size=m)
    for i in tf.range(m):
        y = 1 + g(x[..., m - 1 :])
        for j in tf.range(m - 1 - i):
            y *= tf.cos(math.pi / 2 * x[..., j, tf.newaxis])
        if i > 0:
            y *= tf.sin(math.pi / 2 * x[..., m - 1 - i, tf.newaxis])
        ta = ta.write(i, y)

    return tf.squeeze(tf.concat(tf.split(ta.stack(), m, axis=0), axis=-1), axis=0)


def DTLZ2(input_dim: int, num_objective: int) -> MultiObjectiveTestProblem:
    """
    The DTLZ2 problem, the idea pareto fronts lie on (part of) a unit hyper sphere.
    See :cite:`deb2002scalable` for details.

    :param input_dim: The input dimensionality of the synthetic function.
    :param num_objective: The number of objectives.
    :return: The problem specification.
    """
    M, k, d = dtlz_mkd(input_dim, num_objective)

    def gen_pareto_optimal_points(n: int, seed: int | None = None) -> TensorType:
        tf.debugging.assert_greater_equal(M, 2)
        rnd = tf.random.normal([n, M], seed=seed, dtype=tf.float64)
        samples = tf.abs(rnd / tf.norm(rnd, axis=-1, keepdims=True))
        return samples

    return MultiObjectiveTestProblem(
        name=f"DTLZ2({input_dim}, {num_objective})",
        objective=partial(dtlz2, m=M, d=d),
        search_space=Box([0.0], [1.0]) ** d,
        gen_pareto_optimal_points=gen_pareto_optimal_points,
    )


def ConstrainedBraninCurrin() -> ConstrainedMultiObjectiveTestProblem:
    """
    The ConstrainedBraninCurrin problem, typically evaluated over :math:`[0, 1]^2`.
    See :cite:`belakaria2019max` and :cite:`daulton2020differentiable`
    (the latter for adding the constraint) for details.

    :return: The problem specification.
    """

    def gen_pareto_optimal_points(n: int, seed: int | None = None) -> TensorType:
        """
        raise an `NoAnalyticalParetoPointsError` since there is no explicit way of defining
        this problem's Pareto frontier.
        """
        raise NoAnalyticalParetoPointsError(
            "No analytical approach to generate Pareto optimal points for this problem, "
            "an optimization-based approach may be utilized to approximate the Pareto "
            "optimal points"
        )

    def evaluate_constraint(x: TensorType, threshold: Optional[float] = 0.0) -> TensorType:
        """
        The constraint of branincurrin problem, < ``threshold`` is feasible.

        :param x: The points at which to evaluate the function, with shape [..., d].
        :param threshold: a feasibility threshold used to determine the constraint, by default 0 is
            used as in the original problem.
        :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
        """
        x = x * (
            tf.constant([10.0, 15.0], dtype=x.dtype) - tf.constant([-5.0, 0.0], dtype=x.dtype)
        ) + tf.constant([-5.0, 0.0], dtype=x.dtype)
        return (x[..., :1] - 2.5) ** 2 + (x[..., 1:] - 7.5) ** 2 - 50 - threshold

    return ConstrainedMultiObjectiveTestProblem(
        name="ConstrainedBraninCurrin",
        objective=branin_currin,
        constraint=evaluate_constraint,
        search_space=Box([0.0], [1.0]) ** 2,
        gen_pareto_optimal_points=gen_pareto_optimal_points,
    )


def branin_currin(x: TensorType) -> TensorType:
    """
    The branincurrin synthetic function.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 2))])
    return tf.concat([branin(x), currin(x)], axis=-1)


def currin(x: TensorType) -> TensorType:
    """
    The currin synthetic function

    :param x: The points at which to evaluate the function, with shape [..., d].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes([(x, (..., 2))])
    return (
        (1 - tf.math.exp(-0.5 * (1 / (x[..., 1] + 1e-100))))  # 1e-100 used for avoid zero division
        * (
            (2300 * x[..., 0] ** 3 + 1900 * x[..., 0] ** 2 + 2092 * x[..., 0] + 60)
            / (100 * x[..., 0] ** 3 + 500 * x[..., 0] ** 2 + 4 * x[..., 0] + 20)
        )
    )[..., tf.newaxis]

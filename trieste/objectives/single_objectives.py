# Copyright 2021 The Trieste Contributors
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
<https://web.archive.org/web/20211015101644/https://www.sfu.ca/~ssurjano/> (:cite:`ssurjano2021`)`_.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import pi
from typing import Callable, Generic, Sequence

import tensorflow as tf
from check_shapes import check_shapes

from ..space import Box, Constraint, LinearConstraint, NonlinearConstraint, SearchSpaceType
from ..types import TensorType

ObjectiveTestFunction = Callable[[TensorType], TensorType]
"""A synthetic test function"""


@dataclass(frozen=True)
class ObjectiveTestProblem(Generic[SearchSpaceType]):
    """
    Convenience container class for synthetic objective test functions.
    """

    name: str
    """The test function name"""

    objective: ObjectiveTestFunction
    """The synthetic test function"""

    search_space: SearchSpaceType
    """The (continuous) search space of the test function"""

    @property
    def dim(self) -> int:
        """The input dimensionality of the test function"""
        return self.search_space.dimension

    @property
    def bounds(self) -> list[list[float]]:
        """The input space bounds of the test function"""
        return [self.search_space.lower, self.search_space.upper]


@dataclass(frozen=True)
class SingleObjectiveTestProblem(ObjectiveTestProblem[SearchSpaceType]):
    """
    Convenience container class for synthetic single-objective test functions,
    including the global minimizers and minimum.
    """

    minimizers: TensorType
    """The global minimizers of the test function."""

    minimum: TensorType
    """The global minimum of the test function."""


def check_objective_shapes(d: int) -> Callable[[ObjectiveTestFunction], ObjectiveTestFunction]:
    """Returns a decorator for checking the shape of single objective test functions."""
    return check_shapes(f"x: [batch..., {d}]", "return: [batch..., 1]")


def _branin_internals(x: TensorType, scale: TensorType, translate: TensorType) -> TensorType:
    x0 = x[..., :1] * 15.0 - 5.0
    x1 = x[..., 1:] * 15.0

    b = 5.1 / (4 * math.pi**2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8 * math.pi)

    return scale * ((x1 - b * x0**2 + c * x0 - r) ** 2 + s * (1 - t) * tf.cos(x0) + translate)


@check_objective_shapes(d=2)
def branin(x: TensorType) -> TensorType:
    """
    The Branin-Hoo function over :math:`[0, 1]^2`. See
    :cite:`Picheny2013` for details.

    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    return _branin_internals(x, 1, 10)


@check_objective_shapes(d=2)
def scaled_branin(x: TensorType) -> TensorType:
    """
    The Branin-Hoo function, rescaled to have zero mean and unit variance over :math:`[0, 1]^2`. See
    :cite:`Picheny2013` for details.

    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    return _branin_internals(x, 1 / 51.95, -44.81)


_ORIGINAL_BRANIN_MINIMIZERS = tf.constant(
    [[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]], tf.float64
)

Branin = SingleObjectiveTestProblem(
    name="Branin",
    objective=branin,
    search_space=Box([0.0], [1.0]) ** 2,
    minimizers=(_ORIGINAL_BRANIN_MINIMIZERS + [5.0, 0.0]) / 15.0,
    minimum=tf.constant([0.397887], tf.float64),
)
"""The Branin-Hoo function over :math:`[0, 1]^2`. See :cite:`Picheny2013` for details."""

ScaledBranin = SingleObjectiveTestProblem(
    name="Scaled Branin",
    objective=scaled_branin,
    search_space=Branin.search_space,
    minimizers=Branin.minimizers,
    minimum=tf.constant([-1.047393], tf.float64),
)
"""The Branin-Hoo function, rescaled to have zero mean and unit variance over :math:`[0, 1]^2`. See
:cite:`Picheny2013` for details."""


def _scaled_branin_constraints() -> Sequence[Constraint]:
    def _nlc_func0(x: TensorType) -> TensorType:
        c0 = x[..., 0] - 0.2 - tf.sin(x[..., 1])
        c0 = tf.expand_dims(c0, axis=-1)
        return c0

    def _nlc_func1(x: TensorType) -> TensorType:
        c1 = x[..., 0] - tf.cos(x[..., 1])
        c1 = tf.expand_dims(c1, axis=-1)
        return c1

    constraints: Sequence[Constraint] = [
        LinearConstraint(
            A=tf.constant([[-1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
            lb=tf.constant([-0.4, 0.15, 0.2]),
            ub=tf.constant([0.5, 0.9, 0.9]),
        ),
        NonlinearConstraint(_nlc_func0, tf.constant(-1.0), tf.constant(0.0)),
        NonlinearConstraint(_nlc_func1, tf.constant(-0.8), tf.constant(0.0)),
    ]

    return constraints


ConstrainedScaledBranin = SingleObjectiveTestProblem(
    name="Constrained Scaled Branin",
    objective=scaled_branin,
    search_space=Box(
        Branin.search_space.lower,
        Branin.search_space.upper,
        constraints=_scaled_branin_constraints(),
    ),
    minimizers=tf.constant([[0.16518, 0.66518]], tf.float64),
    minimum=tf.constant([-0.99888], tf.float64),
)
"""The rescaled Branin-Hoo function with a combination of linear and nonlinear constraints on the
search space."""


@check_objective_shapes(d=2)
def simple_quadratic(x: TensorType) -> TensorType:
    """
    A trivial quadratic function over :math:`[0, 1]^2`. Useful for quick testing.

    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    return -tf.math.reduce_sum(x, axis=-1, keepdims=True) ** 2


SimpleQuadratic = SingleObjectiveTestProblem(
    name="Simple Quadratic",
    objective=simple_quadratic,
    search_space=Branin.search_space,
    minimizers=tf.constant([[1.0, 1.0]], tf.float64),
    minimum=tf.constant([-4.0], tf.float64),
)
"""A trivial quadratic function over :math:`[0, 1]^2`. Useful for quick testing."""


@check_objective_shapes(d=1)
def gramacy_lee(x: TensorType) -> TensorType:
    """
    The Gramacy & Lee function, typically evaluated over :math:`[0.5, 2.5]`. See
    :cite:`gramacy2012cases` for details.

    :param x: Where to evaluate the function, with shape [..., 1].
    :return: The function values, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    return tf.sin(10 * math.pi * x) / (2 * x) + (x - 1) ** 4


GramacyLee = SingleObjectiveTestProblem(
    name="Gramacy & Lee",
    objective=gramacy_lee,
    search_space=Box([0.5], [2.5]),
    minimizers=tf.constant([[0.548562]], tf.float64),
    minimum=tf.constant([-0.869011], tf.float64),
)
"""The Gramacy & Lee function, typically evaluated over :math:`[0.5, 2.5]`. See
:cite:`gramacy2012cases` for details."""


@check_objective_shapes(d=2)
def logarithmic_goldstein_price(x: TensorType) -> TensorType:
    """
    A logarithmic form of the Goldstein-Price function, with zero mean and unit variance over
    :math:`[0, 1]^2`. See :cite:`Picheny2013` for details.

    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    x0, x1 = tf.split(4 * x - 2, 2, axis=-1)

    a = (x0 + x1 + 1) ** 2
    b = 19 - 14 * x0 + 3 * x0**2 - 14 * x1 + 6 * x0 * x1 + 3 * x1**2
    c = (2 * x0 - 3 * x1) ** 2
    d = 18 - 32 * x0 + 12 * x0**2 + 48 * x1 - 36 * x0 * x1 + 27 * x1**2

    return (1 / 2.427) * (tf.math.log((1 + a * b) * (30 + c * d)) - 8.693)


LogarithmicGoldsteinPrice = SingleObjectiveTestProblem(
    name="Logarithmic Goldstein-Price",
    objective=logarithmic_goldstein_price,
    search_space=Box([0.0], [1.0]) ** 2,
    minimizers=tf.constant([[0.5, 0.25]], tf.float64),
    minimum=tf.constant([-3.12913], tf.float64),
)
"""A logarithmic form of the Goldstein-Price function, with zero mean and unit variance over
:math:`[0, 1]^2`. See :cite:`Picheny2013` for details."""


@check_objective_shapes(d=3)
def hartmann_3(x: TensorType) -> TensorType:
    """
    The Hartmann 3 test function over :math:`[0, 1]^3`. This function has 3 local
    and one global minima. See https://www.sfu.ca/~ssurjano/hart3.html for details.

    :param x: The points at which to evaluate the function, with shape [..., 3].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
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


Hartmann3 = SingleObjectiveTestProblem(
    name="Hartmann 3",
    objective=hartmann_3,
    search_space=Box([0.0], [1.0]) ** 3,
    minimizers=tf.constant([[0.114614, 0.555649, 0.852547]], tf.float64),
    minimum=tf.constant([-3.86278], tf.float64),
)
"""The Hartmann 3 test function over :math:`[0, 1]^3`. This function has 3 local
and one global minima. See https://www.sfu.ca/~ssurjano/hart3.html for details."""


@check_objective_shapes(d=4)
def shekel_4(x: TensorType) -> TensorType:
    """
    The Shekel test function over :math:`[0, 1]^4`. This function has ten local
    minima and a single global minimum. See https://www.sfu.ca/~ssurjano/shekel.html for details.
    Note that we rescale the original problem, which is typically defined
    over `[0, 10]^4`.

    :param x: The points at which to evaluate the function, with shape [..., 4].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    y: TensorType = x * 10.0

    beta = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    C = [
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
    ]

    inner_sum = tf.reduce_sum((tf.expand_dims(y, -1) - C) ** 2, 1)
    inner_sum += tf.cast(tf.transpose(beta), dtype=inner_sum.dtype)
    return -tf.reduce_sum(inner_sum ** (-1), -1, keepdims=True)


Shekel4 = SingleObjectiveTestProblem(
    name="Shekel 4",
    objective=shekel_4,
    search_space=Box([0.0], [1.0]) ** 4,
    minimizers=tf.constant([[0.4, 0.4, 0.4, 0.4]], tf.float64),
    minimum=tf.constant([-10.5363], tf.float64),
)
"""The Shekel test function over :math:`[0, 1]^4`. This function has ten local
minima and a single global minimum. See https://www.sfu.ca/~ssurjano/shekel.html for details.
Note that we rescale the original problem, which is typically defined
over `[0, 10]^4`."""


def levy(x: TensorType, d: int) -> TensorType:
    """
    The Levy test function over :math:`[0, 1]^d`. This function has many local
    minima and a single global minimum. See https://www.sfu.ca/~ssurjano/levy.html for details.
    Note that we rescale the original problem, which is typically defined
    over `[-10, 10]^d`, to be defined over a unit hypercube :math:`[0, 1]^d`.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :param d: The dimension of the function.
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_greater_equal(d, 1)
    tf.debugging.assert_shapes([(x, (..., d))])

    w: TensorType = 1 + ((x * 20.0 - 10) - 1) / 4
    term1 = tf.pow(tf.sin(pi * w[..., 0:1]), 2)
    term3 = (w[..., -1:] - 1) ** 2 * (1 + tf.pow(tf.sin(2 * pi * w[..., -1:]), 2))
    wi = w[..., 0:-1]
    wi_sum = tf.reduce_sum(
        (wi - 1) ** 2 * (1 + 10 * tf.pow(tf.sin(pi * wi + 1), 2)), axis=-1, keepdims=True
    )
    return term1 + wi_sum + term3


@check_objective_shapes(d=8)
def levy_8(x: TensorType) -> TensorType:
    """
    Convenience function for the 8-dimensional :func:`levy` function, with output
    normalised to unit interval

    :param x: The points at which to evaluate the function, with shape [..., 8].
    :return: The function values at ``x``, with shape [..., 1].
    """
    return levy(x, d=8) / 450.0


Levy8 = SingleObjectiveTestProblem(
    name="Levy 8",
    objective=levy_8,
    search_space=Box([0.0], [1.0]) ** 8,
    minimizers=tf.constant([[11 / 20] * 8], tf.float64),
    minimum=tf.constant([0], tf.float64),
)
"""Convenience function for the 8-dimensional :func:`levy` function.
Taken from https://www.sfu.ca/~ssurjano/levy.html"""


def rosenbrock(x: TensorType, d: int) -> TensorType:
    """
    The Rosenbrock function, also known as the Banana function, is a unimodal function,
    however the minima lies in a narrow valley. Even though this valley is
    easy to find, convergence to the minimum is difficult. See
    https://www.sfu.ca/~ssurjano/rosen.html for details. Inputs are rescaled to
    be defined over a unit hypercube :math:`[0, 1]^d`.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :param d: The dimension of the function.
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_greater_equal(d, 1)
    tf.debugging.assert_shapes([(x, (..., d))])

    y: TensorType = x * 15.0 - 5
    unscaled_function = tf.reduce_sum(
        (100.0 * (y[..., 1:] - y[..., :-1]) ** 2 + (1 - y[..., :-1]) ** 2), axis=-1, keepdims=True
    )
    return unscaled_function


@check_objective_shapes(d=4)
def rosenbrock_4(x: TensorType) -> TensorType:
    """
    Convenience function for the 4-dimensional :func:`rosenbrock` function with steepness 10.
    It is rescaled to have zero mean and unit variance over :math:`[0, 1]^4. See
    :cite:`Picheny2013` for details.

    :param x: The points at which to evaluate the function, with shape [..., 4].
    :return: The function values at ``x``, with shape [..., 1].
    """
    return (rosenbrock(x, d=4) - 3.827 * 1e5) / (3.755 * 1e5)


Rosenbrock4 = SingleObjectiveTestProblem(
    name="Rosenbrock 4",
    objective=rosenbrock_4,
    search_space=Box([0.0], [1.0]) ** 4,
    minimizers=tf.constant([[0.4] * 4], tf.float64),
    minimum=tf.constant([-1.01917], tf.float64),
)
"""The Rosenbrock function, rescaled to have zero mean and unit variance over :math:`[0, 1]^4. See
:cite:`Picheny2013` for details.
This function (also known as the Banana function) is unimodal, however the minima
lies in a narrow valley."""


@check_objective_shapes(d=5)
def ackley_5(x: TensorType) -> TensorType:
    """
    The Ackley test function over :math:`[0, 1]^5`. This function has
    many local minima and a global minima. See https://www.sfu.ca/~ssurjano/ackley.html
    for details.
    Note that we rescale the original problem, which is typically defined
    over `[-32.768, 32.768]`.

    :param x: The points at which to evaluate the function, with shape [..., 5].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    x = (x - 0.5) * (32.768 * 2.0)

    exponent_1 = -0.2 * tf.math.sqrt((1 / 5.0) * tf.reduce_sum(x**2, -1))
    exponent_2 = (1 / 5.0) * tf.reduce_sum(tf.math.cos(2.0 * math.pi * x), -1)

    function = (
        -20.0 * tf.math.exp(exponent_1)
        - tf.math.exp(exponent_2)
        + 20.0
        + tf.cast(tf.math.exp(1.0), dtype=x.dtype)
    )

    return tf.expand_dims(function, -1)


Ackley5 = SingleObjectiveTestProblem(
    name="Ackley 5",
    objective=ackley_5,
    search_space=Box([0.0], [1.0]) ** 5,
    minimizers=tf.constant([[0.5, 0.5, 0.5, 0.5, 0.5]], tf.float64),
    minimum=tf.constant([0.0], tf.float64),
)
"""The Ackley test function over :math:`[0, 1]^5`. This function has
many local minima and a global minima. See https://www.sfu.ca/~ssurjano/ackley.html
for details.
Note that we rescale the original problem, which is typically defined
over `[-32.768, 32.768]`."""


@check_objective_shapes(d=6)
def hartmann_6(x: TensorType) -> TensorType:
    """
    The Hartmann 6 test function over :math:`[0, 1]^6`. This function has
    6 local and one global minima. See https://www.sfu.ca/~ssurjano/hart6.html
    for details.

    :param x: The points at which to evaluate the function, with shape [..., 6].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
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


Hartmann6 = SingleObjectiveTestProblem(
    name="Hartmann 6",
    objective=hartmann_6,
    search_space=Box([0.0], [1.0]) ** 6,
    minimizers=tf.constant([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]], tf.float64),
    minimum=tf.constant([-3.32237], tf.float64),
)
"""The Hartmann 6 test function over :math:`[0, 1]^6`. This function has
6 local and one global minima. See https://www.sfu.ca/~ssurjano/hart6.html
for details."""


def michalewicz(x: TensorType, d: int = 2, m: int = 10) -> TensorType:
    """
    The Michalewicz function over :math:`[0, \\pi]` for all i=1,...,d. Dimensionality is determined
    by the parameter ``d`` and it features steep ridges and drops. It has :math:`d!` local minima,
    and it is multimodal. The parameter ``m`` defines the steepness of they valleys and ridges; a
    larger ``m`` leads to a more difficult search. The recommended value of ``m`` is 10. See
    https://www.sfu.ca/~ssurjano/egg.html for details.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :param d: The dimension of the function.
    :param m: The steepness of the valleys/ridges.
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_greater_equal(d, 1)
    tf.debugging.assert_shapes([(x, (..., d))])

    xi = tf.range(1, (d + 1), delta=1, dtype=x.dtype) * tf.pow(x, 2)
    result = tf.reduce_sum(tf.sin(x) * tf.pow(tf.sin(xi / math.pi), 2 * m), axis=1, keepdims=True)

    return -result


@check_objective_shapes(d=2)
def michalewicz_2(x: TensorType) -> TensorType:
    """
    Convenience function for the 2-dimensional :func:`michalewicz` function with steepness 10.
    :param x: The points at which to evaluate the function, with shape [..., 2].
    :return: The function values at ``x``, with shape [..., 1].
    """
    return michalewicz(x, d=2)


@check_objective_shapes(d=5)
def michalewicz_5(x: TensorType) -> TensorType:
    """
    Convenience function for the 5-dimensional :func:`michalewicz` function with steepness 10.
    :param x: The points at which to evaluate the function, with shape [..., 5].
    :return: The function values at ``x``, with shape [..., 1].
    """
    return michalewicz(x, d=5)


@check_objective_shapes(d=10)
def michalewicz_10(x: TensorType) -> TensorType:
    """
    Convenience function for the 10-dimensional :func:`michalewicz` function with steepness 10.
    :param x: The points at which to evaluate the function, with shape [..., 10].
    :return: The function values at ``x``, with shape [..., 1].
    """
    return michalewicz(x, d=10)


Michalewicz2 = SingleObjectiveTestProblem(
    name="Michalewicz 2",
    objective=michalewicz_2,
    search_space=Box([0.0], [pi]) ** 2,
    minimizers=tf.constant([[2.202906, 1.570796]], tf.float64),
    minimum=tf.constant([-1.8013034], tf.float64),
)
"""Convenience function for the 2-dimensional :func:`michalewicz` function with steepness 10.
Taken from https://arxiv.org/abs/2003.09867"""

Michalewicz5 = SingleObjectiveTestProblem(
    name="Michalewicz 5",
    objective=michalewicz_5,
    search_space=Box([0.0], [pi]) ** 5,
    minimizers=tf.constant([[2.202906, 1.570796, 1.284992, 1.923058, 1.720470]], tf.float64),
    minimum=tf.constant([-4.6876582], tf.float64),
)
"""Convenience function for the 5-dimensional :func:`michalewicz` function with steepness 10.
Taken from https://arxiv.org/abs/2003.09867"""

Michalewicz10 = SingleObjectiveTestProblem(
    name="Michalewicz 10",
    objective=michalewicz_10,
    search_space=Box([0.0], [pi]) ** 10,
    minimizers=tf.constant(
        [
            [
                2.202906,
                1.570796,
                1.284992,
                1.923058,
                1.720470,
                1.570796,
                1.454414,
                1.756087,
                1.655717,
                1.570796,
            ]
        ],
        tf.float64,
    ),
    minimum=tf.constant([-9.6601517], tf.float64),
)
"""Convenience function for the 10-dimensional :func:`michalewicz` function with steepness 10.
Taken from https://arxiv.org/abs/2003.09867"""


def trid(x: TensorType, d: int = 10) -> TensorType:
    """
    The Trid function over :math:`[-d^2, d^2]` for all i=1,...,d. Dimensionality is determined
    by the parameter ``d`` and it has a global minimum. This function has large variation in
    output which makes it challenging for Bayesian optimisation with vanilla Gaussian processes
    with non-stationary kernels. Models that can deal with non-stationarities, such as deep
    Gaussian processes, can be useful for modelling these functions. See :cite:`hebbal2019bayesian`
    and https://www.sfu.ca/~ssurjano/trid.html for details.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :param d: Dimensionality.
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_greater_equal(d, 2)
    tf.debugging.assert_shapes([(x, (..., d))])

    result = tf.reduce_sum(tf.pow(x - 1, 2), 1, True) - tf.reduce_sum(x[:, 1:] * x[:, :-1], 1, True)
    return result


@check_objective_shapes(d=10)
def trid_10(x: TensorType) -> TensorType:
    """The Trid function with dimension 10.

    :param x: The points at which to evaluate the function, with shape [..., 10].
    :return: The function values at ``x``, with shape [..., 1].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    return trid(x, d=10)


Trid10 = SingleObjectiveTestProblem(
    name="Trid 10",
    objective=trid_10,
    search_space=Box([-(10**2)], [10**2]) ** 10,
    minimizers=tf.constant([[i * (10 + 1 - i) for i in range(1, 10 + 1)]], tf.float64),
    minimum=tf.constant([-10 * (10 + 4) * (10 - 1) / 6], tf.float64),
)
"""The Trid function with dimension 10."""

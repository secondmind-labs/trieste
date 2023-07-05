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
""" This module contains implementations of various types of search space. """
from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable, Optional, Sequence, Tuple, TypeVar, Union, overload

import numpy as np
import scipy.optimize as spo
import tensorflow as tf
import tensorflow_probability as tfp

from .types import TensorType

SearchSpaceType = TypeVar("SearchSpaceType", bound="SearchSpace")
""" A type variable bound to :class:`SearchSpace`. """

DEFAULT_DTYPE: tf.DType = tf.float64
""" Default dtype to use when none is provided. """


class SampleTimeoutError(Exception):
    """Raised when sampling from a search space has timed out."""


class NonlinearConstraint(spo.NonlinearConstraint):  # type: ignore[misc]
    """
    A wrapper class for nonlinear constraints on variables. The constraints expression is of the
    form::

        lb <= fun(x) <= ub

    :param fun: The function defining the nonlinear constraints; with input shape [..., D] and
        output shape [..., 1], returning a scalar value for each input point.
    :param lb: The lower bound of the constraint. Should be a scalar or of shape [1].
    :param ub: The upper bound of the constraint. Should be a scalar or of shape [1].
    :param keep_feasible: Keep the constraints feasible throughout optimization iterations if this
        is `True`.
    """

    def __init__(
        self,
        fun: Callable[[TensorType], TensorType],
        lb: Sequence[float] | TensorType,
        ub: Sequence[float] | TensorType,
        keep_feasible: bool = False,
    ):
        # Implement caching to avoid calling the constraint function multiple times to get value
        # and gradient.
        def _constraint_value_and_gradient(x: TensorType) -> Tuple[TensorType, TensorType]:
            val, grad = tfp.math.value_and_gradient(fun, x)

            tf.debugging.assert_shapes(
                [(val, [..., 1])],
                message="Nonlinear constraint only supports single output function.",
            )

            return tf.cast(val, dtype=x.dtype), tf.cast(grad, dtype=x.dtype)

        cache_x: TensorType = tf.constant([])
        cache_f: TensorType = tf.constant([])
        cache_df_dx: TensorType = tf.constant([])

        def val_fun(x: TensorType) -> TensorType:
            nonlocal cache_x, cache_f, cache_df_dx
            if not np.array_equal(x, cache_x):
                cache_f, cache_df_dx = _constraint_value_and_gradient(x)
                cache_x = x
            return cache_f

        def jac_fun(x: TensorType) -> TensorType:
            nonlocal cache_x, cache_f, cache_df_dx
            if not np.array_equal(x, cache_x):
                cache_f, cache_df_dx = _constraint_value_and_gradient(x)
                cache_x = x
            return cache_df_dx

        self._orig_fun = fun  # Used for constraints equality check.
        super().__init__(val_fun, lb, ub, jac=jac_fun, keep_feasible=keep_feasible)

    def residual(self, points: TensorType) -> TensorType:
        """
        Calculate the residuals between the constraint function and its lower/upper limits.

        :param points: The points to calculate the residuals for, with shape [..., D].
        :return: A tensor containing the lower and upper residual values with shape [..., 2].
        """
        tf.debugging.assert_rank_at_least(points, 2)
        non_d_axes = np.ones_like(points.shape)[:-1]  # Avoid adding axes shape to static graph.
        lb = tf.cast(tf.reshape(self.lb, (*non_d_axes, -1)), dtype=points.dtype)
        ub = tf.cast(tf.reshape(self.ub, (*non_d_axes, -1)), dtype=points.dtype)
        fval = self.fun(points)
        fval = tf.reshape(fval, (*points.shape[:-1], -1))  # Atleast 2D.
        fval = tf.cast(fval, dtype=points.dtype)
        values = [fval - lb, ub - fval]
        values = tf.concat(values, axis=-1)
        return values

    def __repr__(self) -> str:
        """"""
        return f"""
            NonlinearConstraint({self.fun!r}, {self.lb!r}, {self.ub!r}, {self.keep_feasible!r})"
        """

    def __eq__(self, other: object) -> bool:
        """
        :param other: A constraint.
        :return: Whether the constraint is identical to this one.
        """
        if not isinstance(other, NonlinearConstraint):
            return False
        return bool(
            self._orig_fun == other._orig_fun
            and tf.reduce_all(self.lb == other.lb)
            and tf.reduce_all(self.ub == other.ub)
            and self.keep_feasible == other.keep_feasible
        )


class LinearConstraint(spo.LinearConstraint):  # type: ignore[misc]
    """
    A wrapper class for linear constraints on variables. The constraints expression is of the form::

        lb <= A @ x <= ub

    :param A: The matrix defining the linear constraints with shape [M, D], where M is the
        number of constraints.
    :param lb: The lower bound of the constraint. Should be a scalar or of shape [M].
    :param ub: The upper bound of the constraint. Should be a scalar or of shape [M].
    :param keep_feasible: Keep the constraints feasible throughout optimization iterations if this
        is `True`.
    """

    def __init__(
        self,
        A: TensorType,
        lb: Sequence[float] | TensorType,
        ub: Sequence[float] | TensorType,
        keep_feasible: bool = False,
    ):
        super().__init__(A, lb, ub, keep_feasible=keep_feasible)

    def residual(self, points: TensorType) -> TensorType:
        """
        Calculate the residuals between the constraint function and its lower/upper limits.

        :param points: The points to calculate the residuals for, with shape [..., D].
        :return: A tensor containing the lower and upper residual values with shape [..., M*2].
        """
        tf.debugging.assert_rank_at_least(points, 2)
        non_d_axes = np.ones_like(points.shape)[:-1]  # Avoid adding axes shape to static graph.
        lb = tf.cast(tf.reshape(self.lb, (*non_d_axes, -1)), dtype=points.dtype)
        ub = tf.cast(tf.reshape(self.ub, (*non_d_axes, -1)), dtype=points.dtype)
        A = tf.cast(self.A, dtype=points.dtype)
        fval = tf.linalg.matmul(points, A, transpose_b=True)
        fval = tf.reshape(fval, (*points.shape[:-1], -1))  # Atleast 2D.
        values = [fval - lb, ub - fval]
        values = tf.concat(values, axis=-1)
        return values

    def __repr__(self) -> str:
        """"""
        return f"""
            LinearConstraint({self.A!r}, {self.lb!r}, {self.ub!r}, {self.keep_feasible!r})"
        """

    def __eq__(self, other: object) -> bool:
        """
        :param other: A constraint.
        :return: Whether the constraint is identical to this one.
        """
        if not isinstance(other, LinearConstraint):
            return False
        return bool(
            tf.reduce_all(self.A == other.A)
            and tf.reduce_all(self.lb == other.lb)
            and tf.reduce_all(self.ub == other.ub)
            and tf.reduce_all(self.keep_feasible == other.keep_feasible)
        )


Constraint = Union[LinearConstraint, NonlinearConstraint]
""" Type alias for constraints. """


class SearchSpace(ABC):
    """
    A :class:`SearchSpace` represents the domain over which an objective function is optimized.
    """

    @abstractmethod
    def sample(self, num_samples: int, seed: Optional[int] = None) -> TensorType:
        """
        :param num_samples: The number of points to sample from this search space.
        :param seed: Random seed for reproducibility.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly from this search space.
        """

    def contains(self, value: TensorType) -> TensorType:
        """Method for checking membership.

        :param value: A point or points to check for membership of this :class:`SearchSpace`.
        :return: A boolean array showing membership for each point in value.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``value`` has a different
            dimensionality points from this :class:`SearchSpace`.
        """
        tf.debugging.assert_equal(
            tf.rank(value) > 0 and tf.shape(value)[-1] == self.dimension,
            True,
            message=f"""
                Dimensionality mismatch: space is {self.dimension}, value is {tf.shape(value)[-1]}
                """,
        )
        return self._contains(value)

    @abstractmethod
    def _contains(self, value: TensorType) -> TensorType:
        """Space-specific implementation of membership. Can assume valid input shape.

        :param value: A point or points to check for membership of this :class:`SearchSpace`.
        :return: A boolean array showing membership for each point in value.
        """

    def __contains__(self, value: TensorType) -> bool:
        """Method called by `in` operator. Doesn't support broadcasting as Python insists
        on converting the result to a boolean.

        :param value: A single point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``value`` has a different
            dimensionality from this :class:`SearchSpace`.
        """
        tf.debugging.assert_equal(
            tf.rank(value) == 1,
            True,
            message=f"""
                Rank mismatch: expected 1, got {tf.rank(value)}. To get a tensor of boolean
                membership values from a tensor of points, use `space.contains(value)`
                rather than `value in space`.
                """,
        )
        return self.contains(value)

    @property
    @abstractmethod
    def dimension(self) -> TensorType:
        """The number of inputs in this search space."""

    @property
    @abstractmethod
    def lower(self) -> TensorType:
        """The lowest value taken by each search space dimension."""

    @property
    @abstractmethod
    def upper(self) -> TensorType:
        """The highest value taken by each search space dimension."""

    @abstractmethod
    def product(self: SearchSpaceType, other: SearchSpaceType) -> SearchSpaceType:
        """
        :param other: A search space of the same type as this search space.
        :return: The Cartesian product of this search space with the ``other``.
        """

    @overload
    def __mul__(self: SearchSpaceType, other: SearchSpaceType) -> SearchSpaceType:
        ...

    @overload
    def __mul__(self: SearchSpaceType, other: SearchSpace) -> SearchSpace:  # type: ignore[misc]
        # mypy complains that this is superfluous, but it seems to use it fine to infer
        # that Box * Box = Box, while Box * Discrete = SearchSpace.
        ...

    def __mul__(self, other: SearchSpace) -> SearchSpace:
        """
        :param other: A search space.
        :return: The Cartesian product of this search space with the ``other``.
            If both spaces are of the same type then this calls the :meth:`product` method.
            Otherwise, it generates a :class:`TaggedProductSearchSpace`.
        """
        # If the search space has any constraints, always return a tagged product search space.
        if not self.has_constraints and not other.has_constraints and isinstance(other, type(self)):
            return self.product(other)
        return TaggedProductSearchSpace((self, other))

    def __pow__(self: SearchSpaceType, other: int) -> SearchSpaceType:
        """
        Return the Cartesian product of ``other`` instances of this search space. For example, for
        an exponent of `3`, and search space `s`, this is `s ** 3`, which is equivalent to
        `s * s * s`.

        :param other: The exponent, or number of instances of this search space to multiply
            together. Must be strictly positive.
        :return: The Cartesian product of ``other`` instances of this search space.
        :raise tf.errors.InvalidArgumentError: If the exponent ``other`` is less than 1.
        """
        tf.debugging.assert_positive(other, message="Exponent must be strictly positive")
        return reduce(operator.mul, [self] * other)

    def discretize(self, num_samples: int) -> DiscreteSearchSpace:
        """
        :param num_samples: The number of points in the :class:`DiscreteSearchSpace`.
        :return: A discrete search space consisting of ``num_samples`` points sampled uniformly from
            this search space.
        :raise NotImplementedError: If this :class:`SearchSpace` has constraints.
        """
        if self.has_constraints:  # Constraints are not supported.
            raise NotImplementedError(
                "Discretization is currently not supported in the presence of constraints."
            )
        return DiscreteSearchSpace(points=self.sample(num_samples))

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        :param other: A search space.
        :return: Whether the search space is identical to this one.
        """

    @property
    def constraints(self) -> Sequence[Constraint]:
        """The sequence of explicit constraints specified in this search space."""
        return []

    def constraints_residuals(self, points: TensorType) -> TensorType:
        """
        Return residuals for all the constraints in this :class:`SearchSpace`.

        :param points: The points to get the residuals for, with shape [..., D].
        :return: A tensor of all the residuals with shape [..., C], where C is the total number of
            constraints.
        :raise NotImplementedError: If this :class:`SearchSpace` does not support constraints.
        """
        raise NotImplementedError("Constraints are currently not supported for this search space.")

    def is_feasible(self, points: TensorType) -> TensorType:
        """
        Checks if points satisfy the explicit constraints of this :class:`SearchSpace`.
        Note membership of the search space is not checked.

        :param points: The points to check constraints feasibility for, with shape [..., D].
        :return: A tensor of booleans. Returns `True` for each point if it is feasible in this
            search space, else `False`.
        :raise NotImplementedError: If this :class:`SearchSpace` has constraints.
        """
        # Everything is feasible in the absence of constraints. Must be overriden if there are
        # constraints.
        if self.has_constraints:
            raise NotImplementedError("Feasibility check is not implemented for this search space.")
        return tf.cast(tf.ones(points.shape[:-1]), dtype=bool)

    @property
    def has_constraints(self) -> bool:
        """Returns `True` if this search space has any explicit constraints specified."""
        # By default assume there are no constraints; can be overridden by a subclass.
        return False


class DiscreteSearchSpace(SearchSpace):
    r"""
    A discrete :class:`SearchSpace` representing a finite set of :math:`D`-dimensional points in
    :math:`\mathbb{R}^D`.

    For example:

        >>> points = tf.constant([[-1.0, 0.4], [-1.0, 0.6], [0.0, 0.4]])
        >>> search_space = DiscreteSearchSpace(points)
        >>> assert tf.constant([0.0, 0.4]) in search_space
        >>> assert tf.constant([1.0, 0.5]) not in search_space

    """

    def __init__(self, points: TensorType):
        """
        :param points: The points that define the discrete space, with shape ('N', 'D').
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``points`` has an invalid shape.
        """

        tf.debugging.assert_shapes([(points, ("N", "D"))])
        self._points = points
        self._dimension = tf.shape(self._points)[-1]

    def __repr__(self) -> str:
        """"""
        return f"DiscreteSearchSpace({self._points!r})"

    @property
    def lower(self) -> TensorType:
        """The lowest value taken across all points by each search space dimension."""
        return tf.reduce_min(self.points, -2)

    @property
    def upper(self) -> TensorType:
        """The highest value taken across all points by each search space dimension."""
        return tf.reduce_max(self.points, -2)

    @property
    def points(self) -> TensorType:
        """All the points in this space."""
        return self._points

    @property
    def dimension(self) -> TensorType:
        """The number of inputs in this search space."""
        return self._dimension

    def _contains(self, value: TensorType) -> TensorType:
        comparison = tf.math.equal(self._points, tf.expand_dims(value, -2))  # [..., N, D]
        return tf.reduce_any(tf.reduce_all(comparison, axis=-1), axis=-1)  # [...]

    def sample(self, num_samples: int, seed: Optional[int] = None) -> TensorType:
        """
        :param num_samples: The number of points to sample from this search space.
        :param seed: Random seed for reproducibility.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly,
            from this search space.
        """
        if seed is not None:  # ensure reproducibility
            tf.random.set_seed(seed)

        if num_samples == 0:
            return self.points[:0, :]
        else:
            sampled_indices = tf.random.categorical(
                tf.ones((1, tf.shape(self.points)[0])), num_samples, seed=seed
            )
            return tf.gather(self.points, sampled_indices)[0, :, :]  # [num_samples, D]

    def product(self, other: DiscreteSearchSpace) -> DiscreteSearchSpace:
        r"""
        Return the Cartesian product of the two :class:`DiscreteSearchSpace`\ s. For example:

            >>> sa = DiscreteSearchSpace(tf.constant([[0, 1], [2, 3]]))
            >>> sb = DiscreteSearchSpace(tf.constant([[4, 5, 6], [7, 8, 9]]))
            >>> (sa * sb).points.numpy()
            array([[0, 1, 4, 5, 6],
                   [0, 1, 7, 8, 9],
                   [2, 3, 4, 5, 6],
                   [2, 3, 7, 8, 9]], dtype=int32)

        :param other: A :class:`DiscreteSearchSpace` with :attr:`points` of the same dtype as this
            search space.
        :return: The Cartesian product of the two :class:`DiscreteSearchSpace`\ s.
        :raise TypeError: If one :class:`DiscreteSearchSpace` has :attr:`points` of a different
            dtype to the other.
        """
        if self.points.dtype is not other.points.dtype:
            return NotImplemented

        tile_self = tf.tile(self.points[:, None], [1, len(other.points), 1])
        tile_other = tf.tile(other.points[None], [len(self.points), 1, 1])
        cartesian_product = tf.concat([tile_self, tile_other], axis=2)
        product_space_dimension = self.points.shape[-1] + other.points.shape[-1]
        return DiscreteSearchSpace(tf.reshape(cartesian_product, [-1, product_space_dimension]))

    def __eq__(self, other: object) -> bool:
        """
        :param other: A search space.
        :return: Whether the search space is identical to this one.
        """
        if not isinstance(other, DiscreteSearchSpace):
            return NotImplemented
        return bool(tf.reduce_all(tf.sort(self.points, 0) == tf.sort(other.points, 0)))

    def __deepcopy__(self, memo: dict[int, object]) -> DiscreteSearchSpace:
        return self


class Box(SearchSpace):
    r"""
    Continuous :class:`SearchSpace` representing a :math:`D`-dimensional box in
    :math:`\mathbb{R}^D`. Mathematically it is equivalent to the Cartesian product of :math:`D`
    closed bounded intervals in :math:`\mathbb{R}`.
    """

    @overload
    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        constraints: Optional[Sequence[Constraint]] = None,
        ctol: float | TensorType = 1e-7,
    ):
        ...

    @overload
    def __init__(
        self,
        lower: TensorType,
        upper: TensorType,
        constraints: Optional[Sequence[Constraint]] = None,
        ctol: float | TensorType = 1e-7,
    ):
        ...

    def __init__(
        self,
        lower: Sequence[float] | TensorType,
        upper: Sequence[float] | TensorType,
        constraints: Optional[Sequence[Constraint]] = None,
        ctol: float | TensorType = 1e-7,
    ):
        r"""
        If ``lower`` and ``upper`` are `Sequence`\ s of floats (such as lists or tuples),
        they will be converted to tensors of dtype `DEFAULT_DTYPE`.

        :param lower: The lower (inclusive) bounds of the box. Must have shape [D] for positive D,
            and if a tensor, must have float type.
        :param upper: The upper (inclusive) bounds of the box. Must have shape [D] for positive D,
            and if a tensor, must have float type.
        :param constraints: Sequence of explicit input constraints for this search space.
        :param ctol: Tolerance to use to check constraints satisfaction.
        :raise ValueError (or tf.errors.InvalidArgumentError): If any of the following are true:

            - ``lower`` and ``upper`` have invalid shapes.
            - ``lower`` and ``upper`` do not have the same floating point type.
            - ``upper`` is not greater than ``lower`` across all dimensions.
        """

        tf.debugging.assert_shapes([(lower, ["D"]), (upper, ["D"])])
        tf.assert_rank(lower, 1)
        tf.assert_rank(upper, 1)
        tf.debugging.assert_non_negative(ctol, message="Tolerance must be non-negative")

        if isinstance(lower, Sequence):
            self._lower = tf.constant(lower, dtype=DEFAULT_DTYPE)
            self._upper = tf.constant(upper, dtype=DEFAULT_DTYPE)
        else:
            self._lower = tf.convert_to_tensor(lower)
            self._upper = tf.convert_to_tensor(upper)

            tf.debugging.assert_same_float_dtype([self._lower, self._upper])

        tf.debugging.assert_less(self._lower, self._upper)

        self._dimension = tf.shape(self._upper)[-1]

        if constraints is None:
            self._constraints: Sequence[Constraint] = []
        else:
            self._constraints = constraints
        self._ctol = ctol

    def __repr__(self) -> str:
        """"""
        return f"Box({self._lower!r}, {self._upper!r}, {self._constraints!r}, {self._ctol!r})"

    @property
    def lower(self) -> tf.Tensor:
        """The lower bounds of the box."""
        return self._lower

    @property
    def upper(self) -> tf.Tensor:
        """The upper bounds of the box."""
        return self._upper

    @property
    def dimension(self) -> TensorType:
        """The number of inputs in this search space."""
        return self._dimension

    @property
    def constraints(self) -> Sequence[Constraint]:
        """The sequence of explicit constraints specified in this search space."""
        return self._constraints

    def _contains(self, value: TensorType) -> TensorType:
        """
        For each point in ``value``, return `True` if the point is a member of this search space,
        else `False`. A point is a member if all of its coordinates lie in the closed intervals
        bounded by the lower and upper bounds.

        :param value: A point or points to check for membership of this :class:`SearchSpace`.
        :return: A boolean array showing membership for each point in value.
        """
        return tf.reduce_all(value >= self._lower, axis=-1) & tf.reduce_all(
            value <= self._upper, axis=-1
        )

    def _sample(self, num_samples: int, seed: Optional[int] = None) -> TensorType:
        # Internal common method to sample randomly from the space.
        dim = tf.shape(self._lower)[-1]
        return tf.random.uniform(
            (num_samples, dim),
            minval=self._lower,
            maxval=self._upper,
            dtype=self._lower.dtype,
            seed=seed,
        )

    def sample(self, num_samples: int, seed: Optional[int] = None) -> TensorType:
        """
        Sample randomly from the space.

        :param num_samples: The number of points to sample from this search space.
        :param seed: Random seed for reproducibility.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly,
            from this search space with shape '[num_samples, D]' , where D is the search space
            dimension.
        """
        tf.debugging.assert_non_negative(num_samples)
        if seed is not None:  # ensure reproducibility
            tf.random.set_seed(seed)
        return self._sample(num_samples, seed)

    def _sample_halton(
        self,
        start: int,
        num_samples: int,
        seed: Optional[int] = None,
    ) -> TensorType:
        # Internal common method to sample from the space using a Halton sequence.
        tf.debugging.assert_non_negative(num_samples)
        if num_samples == 0:
            return tf.constant([])
        if seed is not None:  # ensure reproducibility
            tf.random.set_seed(seed)
        dim = tf.shape(self._lower)[-1]
        sequence_indices = tf.range(start=start, limit=start + num_samples, dtype=tf.int32)
        return (self._upper - self._lower) * tfp.mcmc.sample_halton_sequence(
            dim=dim, sequence_indices=sequence_indices, dtype=self._lower.dtype, seed=seed
        ) + self._lower

    def sample_halton(self, num_samples: int, seed: Optional[int] = None) -> TensorType:
        """
        Sample from the space using a Halton sequence. The resulting samples are guaranteed to be
        diverse and are reproducible by using the same choice of ``seed``.

        :param num_samples: The number of points to sample from this search space.
        :param seed: Random seed for the halton sequence
        :return: ``num_samples`` of points, using halton sequence with shape '[num_samples, D]' ,
            where D is the search space dimension.
        """
        return self._sample_halton(0, num_samples, seed)

    def sample_sobol(self, num_samples: int, skip: Optional[int] = None) -> TensorType:
        """
        Sample a diverse set from the space using a Sobol sequence.
        If ``skip`` is specified, then the resulting samples are reproducible.

        :param num_samples: The number of points to sample from this search space.
        :param skip: The number of initial points of the Sobol sequence to skip
        :return: ``num_samples`` of points, using sobol sequence with shape '[num_samples, D]' ,
            where D is the search space dimension.
        """
        tf.debugging.assert_non_negative(num_samples)
        if num_samples == 0:
            return tf.constant([])
        if skip is None:  # generate random skip
            skip = tf.random.uniform([1], maxval=2**16, dtype=tf.int32)[0]
        dim = tf.shape(self._lower)[-1]
        return (self._upper - self._lower) * tf.math.sobol_sample(
            dim=dim, num_results=num_samples, dtype=self._lower.dtype, skip=skip
        ) + self._lower

    def _sample_feasible_loop(
        self,
        num_samples: int,
        sampler: Callable[[], TensorType],
        max_tries: int = 100,
    ) -> TensorType:
        """
        Rejection sampling using provided callable. Try ``max_tries`` number of times to find
        ``num_samples`` feasible points.

        :param num_samples: The number of feasible points to sample from this search space.
        :param sampler: Callable to return samples. Called potentially multiple times.
        :param max_tries: Maximum attempts to sample the requested number of points.
        :return: ``num_samples`` feasible points sampled using ``sampler``.
        :raise SampleTimeoutError: If ``max_tries`` are exhausted before ``num_samples`` are
            sampled.
        """
        xs = []
        count = 0
        tries = 0
        while count < num_samples and tries < max_tries:
            tries += 1
            xi = sampler()
            mask = self.is_feasible(xi)
            xo = tf.boolean_mask(xi, mask)
            xs.append(xo)
            count += xo.shape[0]

        if count < num_samples:
            raise SampleTimeoutError(
                f"""Failed to sample {num_samples} feasible point(s), even after {tries} attempts.
                    Sampled only {count} feasible point(s)."""
            )

        xs = tf.concat(xs, axis=0)[:num_samples]
        return xs

    def sample_feasible(
        self, num_samples: int, seed: Optional[int] = None, max_tries: int = 100
    ) -> TensorType:
        """
        Sample feasible points randomly from the space.

        :param num_samples: The number of feasible points to sample from this search space.
        :param seed: Random seed for reproducibility.
        :param max_tries: Maximum attempts to sample the requested number of points.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly,
            from this search space with shape '[num_samples, D]' , where D is the search space
            dimension.
        :raise SampleTimeoutError: If ``max_tries`` are exhausted before ``num_samples`` are
            sampled.
        """
        tf.debugging.assert_non_negative(num_samples)

        # Without constraints or zero-num-samples use the normal sample method directly.
        if not self.has_constraints or num_samples == 0:
            return self.sample(num_samples, seed)

        if seed is not None:  # ensure reproducibility
            tf.random.set_seed(seed)

        def _sampler() -> TensorType:
            return self._sample(num_samples, seed)

        return self._sample_feasible_loop(num_samples, _sampler, max_tries)

    def sample_halton_feasible(
        self, num_samples: int, seed: Optional[int] = None, max_tries: int = 100
    ) -> TensorType:
        """
        Sample feasible points from the space using a Halton sequence. The resulting samples are
        guaranteed to be diverse and are reproducible by using the same choice of ``seed``.

        :param num_samples: The number of feasible points to sample from this search space.
        :param seed: Random seed for the halton sequence
        :param max_tries: Maximum attempts to sample the requested number of points.
        :return: ``num_samples`` of points, using halton sequence with shape '[num_samples, D]' ,
            where D is the search space dimension.
        :raise SampleTimeoutError: If ``max_tries`` are exhausted before ``num_samples`` are
            sampled.
        """
        tf.debugging.assert_non_negative(num_samples)

        # Without constraints or zero-num-samples use the normal sample method directly.
        if not self.has_constraints or num_samples == 0:
            return self.sample_halton(num_samples, seed)

        start = 0

        def _sampler() -> TensorType:
            nonlocal start
            # Global seed is set on every call in _sample_halton() so that we always sample from
            # the same (randomised) sequence, and skip the relevant number of beginning samples.
            samples = self._sample_halton(start, num_samples, seed)
            start += num_samples
            return samples

        return self._sample_feasible_loop(num_samples, _sampler, max_tries)

    def sample_sobol_feasible(
        self, num_samples: int, skip: Optional[int] = None, max_tries: int = 100
    ) -> TensorType:
        """
        Sample a diverse set of feasible points from the space using a Sobol sequence.
        If ``skip`` is specified, then the resulting samples are reproducible.

        :param num_samples: The number of feasible points to sample from this search space.
        :param skip: The number of initial points of the Sobol sequence to skip
        :param max_tries: Maximum attempts to sample the requested number of points.
        :return: ``num_samples`` of points, using sobol sequence with shape '[num_samples, D]' ,
            where D is the search space dimension.
        :raise SampleTimeoutError: If ``max_tries`` are exhausted before ``num_samples`` are
            sampled.
        """
        tf.debugging.assert_non_negative(num_samples)

        # Without constraints or zero-num-samples use the normal sample method directly.
        if not self.has_constraints or num_samples == 0:
            return self.sample_sobol(num_samples, skip)

        if skip is None:  # generate random skip
            skip = tf.random.uniform([1], maxval=2**16, dtype=tf.int32)[0]
        _skip: TensorType = skip  # To keep mypy happy.

        def _sampler() -> TensorType:
            nonlocal _skip
            samples = self.sample_sobol(num_samples, skip=_skip)
            # Skip the relevant number of beginning samples from previous iterations.
            _skip += num_samples
            return samples

        return self._sample_feasible_loop(num_samples, _sampler, max_tries)

    def product(self, other: Box) -> Box:
        r"""
        Return the Cartesian product of the two :class:`Box`\ es (concatenating their respective
        lower and upper bounds). For example:

            >>> unit_interval = Box([0.0], [1.0])
            >>> square_at_origin = Box([-2.0, -2.0], [2.0, 2.0])
            >>> new_box = unit_interval * square_at_origin
            >>> new_box.lower.numpy()
            array([ 0., -2., -2.])
            >>> new_box.upper.numpy()
            array([1., 2., 2.])

        :param other: A :class:`Box` with bounds of the same type as this :class:`Box`.
        :return: The Cartesian product of the two :class:`Box`\ es.
        :raise TypeError: If the bounds of one :class:`Box` have different dtypes to those of
            the other :class:`Box`.
        """
        if self.lower.dtype is not other.lower.dtype:
            return NotImplemented

        product_lower_bound = tf.concat([self._lower, other.lower], axis=-1)
        product_upper_bound = tf.concat([self._upper, other.upper], axis=-1)

        return Box(product_lower_bound, product_upper_bound)

    def __eq__(self, other: object) -> bool:
        """
        :param other: A search space.
        :return: Whether the search space is identical to this one.
        """
        if not isinstance(other, Box):
            return NotImplemented
        return bool(
            tf.reduce_all(self.lower == other.lower)
            and tf.reduce_all(self.upper == other.upper)
            # Constraints match only if they are exactly the same (in the same order).
            and self._constraints == other._constraints
        )

    def __deepcopy__(self, memo: dict[int, object]) -> Box:
        return self

    def constraints_residuals(self, points: TensorType) -> TensorType:
        """
        Return residuals for all the constraints in this :class:`SearchSpace`.

        :param points: The points to get the residuals for, with shape [..., D].
        :return: A tensor of all the residuals with shape [..., C], where C is the total number of
            constraints.
        """
        residuals = [constraint.residual(points) for constraint in self._constraints]
        residuals = tf.concat(residuals, axis=-1)
        return residuals

    def is_feasible(self, points: TensorType) -> TensorType:
        """
        Checks if points satisfy the explicit constraints of this :class:`SearchSpace`.
        Note membership of the search space is not checked.

        :param points: The points to check constraints feasibility for, with shape [..., D].
        :return: A tensor of booleans. Returns `True` for each point if it is feasible in this
            search space, else `False`.
        """
        return tf.math.reduce_all(self.constraints_residuals(points) >= -self._ctol, axis=-1)

    @property
    def has_constraints(self) -> bool:
        """Returns `True` if this search space has any explicit constraints specified."""
        return len(self._constraints) > 0


class TaggedProductSearchSpace(SearchSpace):
    r"""
    Product :class:`SearchSpace` consisting of a product of
    multiple :class:`SearchSpace`. This class provides functionality for
    accessing either the resulting combined search space or each individual space.

    Note that this class assumes that individual points in product spaces are
    represented with their inputs in the same order as specified when initializing
    the space.
    """

    def __init__(self, spaces: Sequence[SearchSpace], tags: Optional[Sequence[str]] = None):
        r"""
        Build a :class:`TaggedProductSearchSpace` from a list ``spaces`` of other spaces. If
        ``tags`` are provided then they form the identifiers of the subspaces, otherwise the
        subspaces are labelled numerically.

        :param spaces: A sequence of :class:`SearchSpace` objects representing the space's subspaces
        :param tags: An optional list of tags giving the unique identifiers of
            the space's subspaces.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``spaces`` has a different
            length to ``tags`` when ``tags`` is provided or if ``tags`` contains duplicates.
        """

        number_of_subspaces = len(spaces)
        if tags is None:
            tags = [str(index) for index in range(number_of_subspaces)]
        else:
            number_of_tags = len(tags)
            tf.debugging.assert_equal(
                number_of_tags,
                number_of_subspaces,
                message=f"""
                    Number of tags must match number of subspaces but
                    received {number_of_tags} tags and {number_of_subspaces} subspaces.
                """,
            )
            number_of_unique_tags = len(set(tags))
            tf.debugging.assert_equal(
                number_of_tags,
                number_of_unique_tags,
                message=f"Subspace names must be unique but received {tags}.",
            )

        self._spaces = dict(zip(tags, spaces))

        subspace_sizes = [space.dimension for space in spaces]

        self._subspace_sizes_by_tag = {
            tag: subspace_size for tag, subspace_size in zip(tags, subspace_sizes)
        }

        self._subspace_starting_indices = dict(zip(tags, tf.cumsum(subspace_sizes, exclusive=True)))

        self._dimension = tf.cast(tf.reduce_sum(subspace_sizes), dtype=tf.int32)
        self._tags = tuple(tags)  # avoid accidental modification by users

    def __repr__(self) -> str:
        """"""
        return f"""TaggedProductSearchSpace(spaces =
                {[self.get_subspace(tag) for tag in self.subspace_tags]},
                tags = {self.subspace_tags})
                """

    @property
    def lower(self) -> TensorType:
        """The lowest values taken by each space dimension, concatenated across subspaces."""
        lower_for_each_subspace = [self.get_subspace(tag).lower for tag in self.subspace_tags]
        return (
            tf.concat(lower_for_each_subspace, axis=-1)
            if lower_for_each_subspace
            else tf.constant([], dtype=DEFAULT_DTYPE)
        )

    @property
    def upper(self) -> TensorType:
        """The highest values taken by each space dimension, concatenated across subspaces."""
        upper_for_each_subspace = [self.get_subspace(tag).upper for tag in self.subspace_tags]
        return (
            tf.concat(upper_for_each_subspace, axis=-1)
            if upper_for_each_subspace
            else tf.constant([], dtype=DEFAULT_DTYPE)
        )

    @property
    def subspace_tags(self) -> tuple[str, ...]:
        """Return the names of the subspaces contained in this product space."""
        return self._tags

    @property
    def dimension(self) -> TensorType:
        """The number of inputs in this product search space."""
        return self._dimension

    def get_subspace(self, tag: str) -> SearchSpace:
        """
        Return the domain of a particular subspace.

        :param tag: The tag specifying the target subspace.
        :return: Target subspace.
        """
        tf.debugging.assert_equal(
            tag in self.subspace_tags,
            True,
            message=f"""
                Attempted to access a subspace that does not exist. This space only contains
                subspaces with the tags {self.subspace_tags} but received {tag}.
            """,
        )
        return self._spaces[tag]

    def fix_subspace(self, tag: str, values: TensorType) -> TaggedProductSearchSpace:
        """
        Return a new :class:`TaggedProductSearchSpace` with the specified subspace replaced with
        a :class:`DiscreteSearchSpace` containing ``values`` as its points. This is useful if you
        wish to restrict subspaces to sets of representative points.

        :param tag: The tag specifying the target subspace.
        :param values: The  values used to populate the new discrete subspace.z
        :return: New :class:`TaggedProductSearchSpace` with the specified subspace replaced with
            a :class:`DiscreteSearchSpace` containing ``values`` as its points.
        """

        new_spaces = [
            self.get_subspace(t) if t != tag else DiscreteSearchSpace(points=values)
            for t in self.subspace_tags
        ]

        return TaggedProductSearchSpace(spaces=new_spaces, tags=self.subspace_tags)

    def get_subspace_component(self, tag: str, values: TensorType) -> TensorType:
        """
        Returns the components of ``values`` lying in a particular subspace.

        :param tag: Subspace tag.
        :param values: Points from the :class:`TaggedProductSearchSpace` of shape [N,Dprod].
        :return: The sub-components of ``values`` lying in the specified subspace, of shape
            [N, Dsub], where Dsub is the dimensionality of the specified subspace.
        """

        starting_index_of_subspace = self._subspace_starting_indices[tag]
        ending_index_of_subspace = starting_index_of_subspace + self._subspace_sizes_by_tag[tag]
        return values[..., starting_index_of_subspace:ending_index_of_subspace]

    def _contains(self, value: TensorType) -> TensorType:
        """
        Return `True` if ``value`` is a member of this search space, else `False`. A point is a
        member if each of its subspace components lie in each subspace.

        Recall that individual points in product spaces are represented with their inputs in the
        same order as specified when initializing the space.

        :param value: A point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`. May return a
            scalar boolean `TensorType` instead of the `bool` itself.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``value`` has a different
            dimensionality from the search space.
        """
        in_each_subspace = [
            self._spaces[tag].contains(self.get_subspace_component(tag, value))
            for tag in self._tags
        ]
        return tf.reduce_all(in_each_subspace, axis=0)

    def sample(self, num_samples: int, seed: Optional[int] = None) -> TensorType:
        """
        Sample randomly from the space by sampling from each subspace
        and concatenating the resulting samples.

        :param num_samples: The number of points to sample from this search space.
        :param seed: Optional tf.random seed.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly,
            from this search space with shape '[num_samples, D]' , where D is the search space
            dimension.
        """
        tf.debugging.assert_non_negative(num_samples)
        if seed is not None:  # ensure reproducibility
            tf.random.set_seed(seed)
        subspace_samples = [self._spaces[tag].sample(num_samples, seed=seed) for tag in self._tags]
        return tf.concat(subspace_samples, -1)

    def product(self, other: TaggedProductSearchSpace) -> TaggedProductSearchSpace:
        r"""
        Return the Cartesian product of the two :class:`TaggedProductSearchSpace`\ s,
        building a tree of :class:`TaggedProductSearchSpace`\ s.

        :param other: A search space of the same type as this search space.
        :return: The Cartesian product of this search space with the ``other``.
        """
        return TaggedProductSearchSpace(spaces=[self, other])

    def __eq__(self, other: object) -> bool:
        """
        :param other: A search space.
        :return: Whether the search space is identical to this one.
        """
        if not isinstance(other, TaggedProductSearchSpace):
            return NotImplemented
        return self._tags == other._tags and self._spaces == other._spaces

    def __deepcopy__(self, memo: dict[int, object]) -> TaggedProductSearchSpace:
        return self

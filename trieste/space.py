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
"""This module contains implementations of various types of search space."""
from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections import Counter
from functools import reduce
from itertools import chain
from itertools import product as itertools_product
from typing import (
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import gpflow.kernels
import numpy as np
import scipy.optimize as spo
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import check_shapes
from gpflow.keras import tf_keras
from gpflow.kernels import ActivityCondition, HierarchyNode
from typing_extensions import Protocol, runtime_checkable

from .types import TensorType
from .utils import flatten_leading_dims

SearchSpaceType = TypeVar("SearchSpaceType", bound="SearchSpace")
""" A type variable bound to :class:`SearchSpace`. """

DEFAULT_DTYPE: tf.DType = tf.float64
""" Default dtype to use when none is provided. """

EncoderFunction = Callable[[TensorType], TensorType]
""" Type alias for point encoders. These transform points from one search space to another. """


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


def _embed_constraint(
    constraint: Constraint, offset: int, part_dim: int, combined_dim: int
) -> Constraint:
    """Re-express a constraint defined on a ``part_dim``-wide sub-vector so it applies to a wider
    ``combined_dim``-wide flat vector, with the sub-vector occupying columns
    ``[offset, offset + part_dim)`` and all other columns ignored.

    Used to carry a search space's (positional) constraints through a Cartesian product, where the
    space's columns are relocated to a contiguous block of the combined flat vector.

    :param constraint: A :class:`LinearConstraint` or :class:`NonlinearConstraint` over the
        ``part_dim``-wide sub-vector.
    :param offset: Index of the sub-vector's first column in the combined vector.
    :param part_dim: Width of the sub-vector the constraint was defined on.
    :param combined_dim: Width of the combined vector.
    :return: An equivalent constraint over the combined vector.
    :raises NotImplementedError: For an unsupported constraint type.
    """
    if isinstance(constraint, LinearConstraint):
        A = tf.convert_to_tensor(constraint.A)
        if int(A.shape[-1]) != part_dim:
            raise ValueError(
                f"LinearConstraint matrix A has {int(A.shape[-1])} columns but the space it "
                f"constrains has dimension {part_dim}; global constraints must span the full "
                f"flat vector."
            )
        padded = tf.pad(A, [[0, 0], [offset, combined_dim - offset - part_dim]])
        return LinearConstraint(padded, constraint.lb, constraint.ub, constraint.keep_feasible)
    if isinstance(constraint, NonlinearConstraint):
        orig_fun = constraint._orig_fun

        def shifted_fun(
            x: TensorType, _fun: Callable = orig_fun, _o: int = offset, _w: int = part_dim
        ) -> TensorType:
            return _fun(x[..., _o : _o + _w])

        return NonlinearConstraint(
            shifted_fun, constraint.lb, constraint.ub, constraint.keep_feasible
        )
    raise NotImplementedError(
        f"Cannot embed constraint of type {type(constraint).__name__} into a wider space."
    )


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
    def has_bounds(self) -> bool:
        """Whether the search space has meaningful numerical bounds."""

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
            Subclasses that do not support a same-type product should raise
            :exc:`NotImplementedError`; :meth:`__mul__` then falls back to a
            :class:`TaggedProductSearchSpace`.
        """

    @overload
    def __mul__(self: SearchSpaceType, other: SearchSpaceType) -> SearchSpaceType: ...

    @overload
    def __mul__(  # type: ignore[overload-cannot-match]
        self: SearchSpaceType,
        other: SearchSpace,
    ) -> SearchSpace:
        # mypy complains that this is superfluous, but it seems to use it fine to infer
        # that Box * Box = Box, while Box * Discrete = SearchSpace.
        ...

    def __mul__(self, other: SearchSpace) -> SearchSpace:
        """
        :param other: A search space.
        :return: The Cartesian product of this search space with the ``other``.
            If both spaces are of the same type (and have no constraints)
            then this calls the :meth:`product` method.
            Otherwise, it generates a :class:`TaggedProductSearchSpace`.
        """
        # If the search space has any constraints, always return a tagged product search space.
        if not self.has_constraints and not other.has_constraints and isinstance(other, type(self)):
            try:
                return self.product(other)
            except NotImplementedError:
                # Subclass opts out of a same-type product; fall back to a tagged product.
                pass
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
        # By default, assume there are no constraints; can be overridden by a subclass.
        return False


class GeneralDiscreteSearchSpace(SearchSpace):
    """
    An ABC representing different types of discrete search spaces (not just numerical).
    This contains a default implementation using explicitly provided points which subclasses
    may ignore.
    """

    def __init__(self, points: TensorType):
        """
        :param points: The points that define the discrete space, with shape ('N', 'D').
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``points`` has an invalid shape.
        """

        tf.debugging.assert_shapes([(points, ("N", "D"))])
        self._points = points
        self._dimension = tf.shape(self._points)[-1]

    @property
    def points(self) -> TensorType:
        """All the points in this space."""
        return self._points

    @property
    def dimension(self) -> TensorType:
        """The number of inputs in this search space."""
        return self._dimension

    def _contains(self, value: TensorType) -> TensorType:
        comparison = tf.math.equal(self.points, tf.expand_dims(value, -2))  # [..., N, D]
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


class DiscreteSearchSpace(GeneralDiscreteSearchSpace):
    r"""
    A discrete :class:`SearchSpace` representing a finite set of :math:`D`-dimensional points in
    :math:`\mathbb{R}^D`.

    For example:

        >>> points = tf.constant([[-1.0, 0.4], [-1.0, 0.6], [0.0, 0.4]])
        >>> search_space = DiscreteSearchSpace(points)
        >>> assert tf.constant([0.0, 0.4]) in search_space
        >>> assert tf.constant([1.0, 0.5]) not in search_space

    """

    def __repr__(self) -> str:
        """"""
        return f"DiscreteSearchSpace({self._points!r})"

    @property
    def has_bounds(self) -> bool:
        return True

    @property
    def lower(self) -> TensorType:
        """The lowest value taken across all points by each search space dimension."""
        return tf.reduce_min(self.points, -2)

    @property
    def upper(self) -> TensorType:
        """The highest value taken across all points by each search space dimension."""
        return tf.reduce_max(self.points, -2)

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


class BooleanSearchSpace(DiscreteSearchSpace):
    r"""
    A 1-D :class:`DiscreteSearchSpace` restricted to :math:`\{0, 1\}`, representing a single
    Boolean indicator variable from the GDP formulation. Provides a distinct type for
    dispatch in validation and downstream consumers (e.g. GA bitflip mutation, kernel
    active/inactive checks).

    Example:

        >>> space = BooleanSearchSpace()
        >>> assert space.dimension == 1
        >>> assert tf.constant([0.0], dtype=tf.float64) in space
        >>> assert tf.constant([1.0], dtype=tf.float64) in space
        >>> assert tf.constant([2.0], dtype=tf.float64) not in space

    """

    def __init__(self, dtype: tf.DType = DEFAULT_DTYPE) -> None:
        """
        :param dtype: The dtype of the points. Defaults to :data:`DEFAULT_DTYPE`.
        """
        super().__init__(points=tf.constant([[0], [1]], dtype=dtype))

    def __repr__(self) -> str:
        """"""
        return f"BooleanSearchSpace({self.points.dtype!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BooleanSearchSpace):
            return NotImplemented
        return self.points.dtype == other.points.dtype


@runtime_checkable
class HasOneHotEncoder(Protocol):
    """A categorical search space that contains default logic for one-hot encoding."""

    @property
    @abstractmethod
    def one_hot_encoder(self) -> EncoderFunction:
        "A one-hot encoder for points in the search space."


def one_hot_encoder(space: SearchSpace) -> EncoderFunction:
    "A utility function for one-hot encoding a search space when it supports it."
    return space.one_hot_encoder if isinstance(space, HasOneHotEncoder) else lambda x: x


def cast_encoder(
    encoder: EncoderFunction,
    input_dtype: Optional[tf.DType] = None,
    output_dtype: Optional[tf.DType] = None,
) -> EncoderFunction:
    "A utility function for casting the input and/or output of an encoder."

    def cast_and_encode(x: TensorType) -> TensorType:
        if input_dtype is not None:
            x = tf.cast(x, input_dtype)
        y = encoder(x)
        if output_dtype is not None:
            y = tf.cast(y, output_dtype)
        return y

    return cast_and_encode


def one_hot_encoded_space(space: SearchSpace) -> SearchSpace:
    "A bounded search space corresponding to the one-hot encoding of the given space."

    if isinstance(space, GeneralDiscreteSearchSpace) and isinstance(space, HasOneHotEncoder):
        return DiscreteSearchSpace(space.one_hot_encoder(space.points))
    elif isinstance(space, TaggedProductSearchSpace):
        spaces = [one_hot_encoded_space(space.get_subspace(tag)) for tag in space.subspace_tags]
        return TaggedProductSearchSpace(spaces=spaces, tags=space.subspace_tags)
    elif isinstance(space, HasOneHotEncoder):
        raise NotImplementedError(f"Unsupported one-hot-encoded space {type(space)}")
    else:
        return space


class CategoricalSearchSpace(GeneralDiscreteSearchSpace, HasOneHotEncoder):
    r"""
    A categorical :class:`SearchSpace` representing a finite set :math:`\mathcal{C}` of categories,
    or a finite Cartesian product :math:`\mathcal{C}_1 \times \cdots \times \mathcal{C}_n` of
    such sets.

    For example:

        >>> CategoricalSearchSpace(5)
        CategoricalSearchSpace([('0', '1', '2', '3', '4')])
        >>> CategoricalSearchSpace(["Red", "Green", "Blue"])
        CategoricalSearchSpace([('Red', 'Green', 'Blue')])
        >>> CategoricalSearchSpace([2,3])
        CategoricalSearchSpace([('0', '1'), ('0', '1', '2')])
        >>> CategoricalSearchSpace([["R", "G", "B"], ["Y", "N"]])
        CategoricalSearchSpace([('R', 'G', 'B'), ('Y', 'N')])

    Note that internally categories are represented by numeric indices:

        >>> rgb = CategoricalSearchSpace(["Red", "Green", "Blue"])
        >>> assert tf.constant([1], dtype=tf.float64) in rgb
        >>> assert tf.constant([3], dtype=tf.float64) not in rgb
        >>> rgb.to_tags(tf.constant([[1], [0], [2]]))
        <tf.Tensor: shape=(3, 1), dtype=string, numpy=
        array([[b'Green'],
               [b'Red'],
               [b'Blue']], dtype=object)>

    """

    def __init__(
        self,
        categories: int | Sequence[int] | Sequence[str] | Sequence[Sequence[str]],
        dtype: tf.DType = DEFAULT_DTYPE,
    ):
        """
        :param categories: Number of categories or category names. Can be an array for
            multidimensional spaces.
        :param dtype: The dtype of the returned indices, either tf.float32 or tf.float64.
        """
        if isinstance(categories, int) or any(isinstance(x, str) for x in categories):
            categories = [categories]  # type: ignore[assignment]

        if not isinstance(categories, Sequence) or not (
            all(
                isinstance(x, Sequence)
                and not isinstance(x, str)
                and all(isinstance(y, str) for y in x)
                for x in categories
            )
            or all(isinstance(x, int) for x in categories)
        ):
            raise TypeError(
                f"Invalid category description {categories!r}: " "expected either numbers or names."
            )

        elif any(isinstance(x, int) for x in categories):
            category_lens: Sequence[int] = categories  # type: ignore[assignment]
            if any(x <= 0 for x in category_lens):
                raise ValueError("Numbers of categories must be positive")
            tags = [tuple(f"{i}" for i in range(n)) for n in category_lens]
        else:
            category_names: Sequence[Sequence[str]] = categories  # type: ignore[assignment]
            if any(len(ts) == 0 for ts in category_names):
                raise ValueError("Category name lists cannot be empty")
            tags = [tuple(ts) for ts in category_names]

        self._tags = tags
        self._dtype = dtype

        ranges = [tf.range(len(ts), dtype=dtype) for ts in tags]
        meshgrid = tf.meshgrid(*ranges, indexing="ij")
        points = (
            tf.reshape(tf.stack(meshgrid, axis=-1), [-1, len(tags)]) if tags else tf.zeros([0, 0])
        )

        super().__init__(points)

    def __repr__(self) -> str:
        """"""
        return f"CategoricalSearchSpace({self._tags!r})"

    @property
    def has_bounds(self) -> bool:
        return False

    @property
    def lower(self) -> TensorType:
        raise AttributeError("Categorical search spaces do not have numerical bounds")

    @property
    def upper(self) -> TensorType:
        raise AttributeError("Categorical search spaces do not have numerical bounds")

    @property
    def tags(self) -> Sequence[Sequence[str]]:
        """The tags of the categories."""
        return self._tags

    @property
    def one_hot_encoder(self) -> EncoderFunction:
        """A one-hot encoder for the numerical indices. Note that binary categories
        are left unchanged instead of adding an unnecessary second feature."""

        def binary_encoder(x: TensorType) -> TensorType:
            # no need to one-hot encode binary categories (but we should still validate)
            tf.debugging.Assert(tf.reduce_all((x == 0) | (x == 1)), [tf.constant([])])
            return x

        def encoder(x: TensorType) -> TensorType:
            flat_x, unflatten = flatten_leading_dims(x)
            tf.debugging.assert_equal(tf.shape(flat_x)[-1], len(self.tags))
            columns = tf.split(flat_x, len(self.tags), axis=1)
            encoders = [
                (
                    binary_encoder
                    if len(ts) == 2
                    else tf_keras.layers.CategoryEncoding(num_tokens=len(ts), output_mode="one_hot")
                )
                for ts in self.tags
            ]
            encoded = tf.concat(
                [
                    tf.cast(encoder(tf.reshape(column, [-1, 1])), dtype=x.dtype)
                    for encoder, column in zip(encoders, columns)
                ],
                axis=1,
            )
            return unflatten(encoded)

        return encoder

    def to_tags(self, indices: TensorType) -> TensorType:
        """
        Convert a tensor of indices (such as one returned by :meth:`sample`) to one of
        category tags.

        :param indices: A tensor of integer indices.
        :return: A tensor of string tags.
        """
        if indices.dtype.is_floating:
            if not tf.reduce_all(tf.math.equal(indices, tf.math.floor(indices))):
                raise ValueError("Non-integral indices passed to to_tags")
            indices = tf.cast(indices, dtype=tf.int32)

        def extract_tags(row: TensorType) -> TensorType:
            return tf.stack(
                [tf.gather(tf.constant(self._tags[i]), row[i]) for i in range(len(row))]
            )

        return tf.map_fn(extract_tags, indices, dtype=tf.string)

    def product(self, other: CategoricalSearchSpace) -> CategoricalSearchSpace:
        r"""
        Return the Cartesian product of the two :class:`CategoricalSearchSpace`\ s. For example:

            >>> rgb = CategoricalSearchSpace(["Red", "Green", "Blue"])
            >>> yn = CategoricalSearchSpace(["Yes", "No"])
            >>> rgb * yn
            CategoricalSearchSpace([('Red', 'Green', 'Blue'), ('Yes', 'No')])

        :param other: A :class:`CategoricalSearchSpace`.
        :return: The Cartesian product of the two :class:`CategoricalSearchSpace`\ s.
        """
        return CategoricalSearchSpace(tuple(chain(self.tags, other.tags)))

    def __eq__(self, other: object) -> bool:
        """
        :param other: A search space.
        :return: Whether the search space is identical to this one.
        """
        if not isinstance(other, CategoricalSearchSpace):
            return NotImplemented
        return self.tags == other.tags


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
    ): ...

    @overload
    def __init__(
        self,
        lower: TensorType,
        upper: TensorType,
        constraints: Optional[Sequence[Constraint]] = None,
        ctol: float | TensorType = 1e-7,
    ): ...

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
            - ``upper`` is not greater or equal to ``lower`` across all dimensions.
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

        tf.debugging.assert_less_equal(self._lower, self._upper)

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
    def has_bounds(self) -> bool:
        return True

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
            return tf.constant([], dtype=self._lower.dtype)
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
            return tf.constant([], dtype=self._lower.dtype)
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


class CollectionSearchSpace(SearchSpace):
    r"""
    An abstract :class:`SearchSpace` consisting of a collection of multiple :class:`SearchSpace`
    objects, each with a unique tag. This class provides functionality for accessing each individual
    space.

    Note that the individual spaces are not combined in any way.
    """

    def __init__(self, spaces: Sequence[SearchSpace], tags: Optional[Sequence[str]] = None):
        r"""
        Build a :class:`CollectionSearchSpace` from a list ``spaces`` of other spaces. If
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
        self._tags = tuple(tags)  # avoid accidental modification by users

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(spaces =
                {[self.get_subspace(tag) for tag in self.subspace_tags]},
                tags = {self.subspace_tags})
                """

    @property
    def has_bounds(self) -> bool:
        """Whether the search space has meaningful numerical bounds."""
        return all(self.get_subspace(tag).has_bounds for tag in self.subspace_tags)

    @property
    def subspace_lower(self) -> Sequence[TensorType]:
        """The lowest values taken by each space dimension, in the same order as specified when
        initializing the space."""
        return [self.get_subspace(tag).lower for tag in self.subspace_tags]

    @property
    def subspace_upper(self) -> Sequence[TensorType]:
        """The highest values taken by each space dimension, in the same order as specified when
        initializing the space."""
        return [self.get_subspace(tag).upper for tag in self.subspace_tags]

    @property
    def subspace_tags(self) -> tuple[str, ...]:
        """Return the names of the subspaces contained in this space."""
        return self._tags

    @property
    def subspace_dimension(self) -> Sequence[TensorType]:
        """The number of inputs in each subspace, in the same order as specified when initializing
        the space."""
        return [self.get_subspace(tag).dimension for tag in self.subspace_tags]

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

    def subspace_sample(self, num_samples: int, seed: Optional[int] = None) -> Sequence[TensorType]:
        """
        Sample randomly from the space by sampling from each subspace
        and returning the resulting samples in the same order as specified when initializing
        the space.

        :param num_samples: The number of points to sample from each subspace.
        :param seed: Optional tf.random seed.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly,
            from each search subspace with shape '[num_samples, D]' , where D is the search space
            dimension.
        """
        tf.debugging.assert_non_negative(num_samples)
        if seed is not None:  # ensure reproducibility
            tf.random.set_seed(seed)
        return [
            # ensure subspaces (which may be identical) don't all use the same seed
            self._spaces[tag].sample(num_samples, seed=None if seed is None else seed + i)
            for i, tag in enumerate(self._tags)
        ]

    def __eq__(self, other: object) -> bool:
        """
        :param other: A search space.
        :return: Whether the search space is identical to this one.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._tags == other._tags and self._spaces == other._spaces


class TaggedProductSearchSpace(CollectionSearchSpace, HasOneHotEncoder):
    r"""
    Product :class:`SearchSpace` consisting of a product of
    multiple :class:`SearchSpace`. This class provides functionality for
    accessing either the resulting combined search space or each individual space.
    This class is useful for defining mixed search spaces, for example:

        context_space = DiscreteSearchSpace(tf.constant([[-0.5, 0.5]]))
        decision_space = Box([-1, -2], [2, 3])
        mixed_space = TaggedProductSearchSpace(spaces=[context_space, decision_space])

    Note: the dtype of all the component search spaces must be the same.

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

        super().__init__(spaces, tags)
        subspace_sizes = self.subspace_dimension

        self._subspace_sizes_by_tag = dict(zip(self._tags, subspace_sizes))

        self._subspace_starting_indices = dict(
            zip(self._tags, tf.cumsum(subspace_sizes, exclusive=True))
        )

        self._dimension = tf.cast(tf.reduce_sum(subspace_sizes), dtype=tf.int32)

    @property
    @check_shapes("return: [D]")
    def lower(self) -> TensorType:
        """The lowest values taken by each space dimension, concatenated across subspaces."""
        lower_for_each_subspace = self.subspace_lower
        return (
            tf.concat(lower_for_each_subspace, axis=-1)
            if lower_for_each_subspace
            else tf.constant([], dtype=DEFAULT_DTYPE)
        )

    @property
    @check_shapes("return: [D]")
    def upper(self) -> TensorType:
        """The highest values taken by each space dimension, concatenated across subspaces."""
        upper_for_each_subspace = self.subspace_upper
        return (
            tf.concat(upper_for_each_subspace, axis=-1)
            if upper_for_each_subspace
            else tf.constant([], dtype=DEFAULT_DTYPE)
        )

    @property
    @check_shapes("return: []")
    def dimension(self) -> TensorType:
        """The number of inputs in this product search space."""
        return self._dimension

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

    @check_shapes("return: [num_samples, D]")
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
        subspace_samples = self.subspace_sample(num_samples, seed)
        return tf.concat(subspace_samples, -1)

    def product(self, other: TaggedProductSearchSpace) -> TaggedProductSearchSpace:
        r"""
        Return the Cartesian product of the two :class:`TaggedProductSearchSpace`\ s,
        building a tree of :class:`TaggedProductSearchSpace`\ s.

        :param other: A search space of the same type as this search space.
        :return: The Cartesian product of this search space with the ``other``.
        """
        return TaggedProductSearchSpace(spaces=[self, other])

    @property
    def one_hot_encoder(self) -> EncoderFunction:
        """An encoder that one-hot-encodes all subpsaces that support it (and leaves
        the other subspaces unchanged)."""

        def encoder(x: TensorType) -> TensorType:
            components = []
            for tag in self.subspace_tags:
                component = self.get_subspace_component(tag, x)
                space = self.get_subspace(tag)
                if isinstance(space, HasOneHotEncoder):
                    component = space.one_hot_encoder(component)
                components.append(component)
            return tf.concat(components, axis=-1)

        return encoder


def _reject_duplicate_tags(seq_name: str, tags: Sequence[str]) -> None:
    """Raise ``ValueError`` if ``tags`` contains duplicates (shared by the hierarchy helper and
    :class:`HierarchicalSearchSpace`)."""
    duplicates = sorted(t for t, count in Counter(tags).items() if count > 1)
    if duplicates:
        raise ValueError(f"`{seq_name}` contains duplicate tags: {duplicates}.")


def _build_tag_to_columns_map(tag_sizes: Sequence[tuple[str, int]]) -> dict[str, list[int]]:
    """Lay tags out in order along the flat vector, each occupying ``size`` consecutive columns.

    :param tag_sizes: ``(tag, dimension)`` pairs in flat-vector order.
    :return: a mapping from tag to its flat-vector column indices.
    """
    tag_to_columns: dict[str, list[int]] = {}
    cursor = 0
    for tag, size in tag_sizes:
        tag_to_columns[tag] = list(range(cursor, cursor + size))
        cursor += size
    return tag_to_columns


def hierarchy_node_from_tags(
    name: str,
    *,
    subspace_tags: Sequence[str],
    activity_condition_tags: Mapping[str, int] | None = None,
    subspaces: Mapping[str, SearchSpace],
    indicator_tags: set[str] | None = None,
) -> HierarchyNode:
    """Construct a :class:`gpflow.kernels.HierarchyNode` from tag-based inputs.

    The GPflow type is keyed on integer column indices (``feature_dims``) and an
    :class:`ActivityCondition` whose keys are also flat-vector column indices --
    the global column of the gating indicator, in the same coordinate system as
    ``feature_dims``. This helper accepts the user-facing tag form and resolves it
    against a ``subspaces`` mapping whose iteration order defines the flat-vector
    layout (one column per dimension of each subspace, in insertion order).

    :param name: Human-readable label for the node (also the kernel-association
        identifier downstream).
    :param subspace_tags: Tags of non-indicator subspaces owned by this node. Each
        tag must be a key of ``subspaces`` and must define numerical bounds (e.g.
        :class:`Box`); categorical non-indicator subspaces are not supported. Each
        subspace contributes one column per dimension, in the order it appears in
        ``subspaces``. Must not contain duplicates.
    :param activity_condition_tags: ``{indicator_tag: required_value}`` for the
        indicators that gate this node. Each key must be a key of ``subspaces``; it
        is resolved to that indicator's global flat-vector column, which is the key
        stored in :class:`gpflow.kernels.ActivityCondition`. The default of ``None``
        means the node is unconditionally active.
    :param subspaces: Tag-keyed mapping of subspaces; its iteration order
        defines the flat-vector column layout.
    :param indicator_tags: Optional cross-check: when given, every
        ``activity_condition_tags`` key must appear here (catches typos). May be
        omitted, in which case any valid ``subspaces`` tag is accepted as an
        indicator.
    :return: An assembled :class:`gpflow.kernels.HierarchyNode`. GPflow
        performs node-local validation (uniqueness of ``feature_dims``,
        non-negativity of required values, ...).
    """
    activity_condition_tags = activity_condition_tags or {}
    indicator_tags = set(indicator_tags or ())

    if not subspace_tags:
        raise ValueError(
            "`subspace_tags` must be non-empty; every node must own at least one "
            "(feature) subspace."
        )

    # Reject duplicate ``subspace_tags``: they would yield duplicate ``feature_dims``
    # (rejected by GPflow downstream, but with an opaque message). ``indicator_tags`` is a
    # set, so it cannot contain duplicates.
    _reject_duplicate_tags("subspace_tags", subspace_tags)

    # A tag cannot be both an owned (feature) subspace and an indicator.
    tag_overlap = set(subspace_tags) & set(indicator_tags)
    if tag_overlap:
        raise ValueError(
            f"`subspace_tags` and `indicator_tags` must be disjoint; overlapping tags: "
            f"{sorted(tag_overlap)}."
        )

    # Build a tag -> [columns] map from the subspaces mapping (insertion order).
    tag_to_columns = _build_tag_to_columns_map(
        [(tag, int(sub.dimension)) for tag, sub in subspaces.items()]
    )

    feature_dims: list[int] = []
    bounds_rows: list[tuple[float, float]] = []
    for stag in subspace_tags:
        if stag not in tag_to_columns:
            raise ValueError(
                f"subspace_tag '{stag}' is not a key of `subspaces` " f"{list(tag_to_columns)}."
            )
        sub = subspaces[stag]
        # Non-indicator subspaces are encoded as (lower, upper) feature bounds, so they must
        # expose numerical bounds. Box and the discrete spaces qualify; categorical subspaces
        # (``has_bounds`` is False) are rejected here with a clear message rather than letting
        # ``sub.lower`` raise an opaque AttributeError downstream.
        if not sub.has_bounds:
            raise ValueError(
                f"subspace_tag '{stag}' refers to a {type(sub).__name__} without numerical "
                f"bounds; non-indicator subspaces must define bounds (e.g. a Box or a discrete "
                f"space)."
            )
        lower = tf.cast(sub.lower, dtype=tf.float64).numpy().tolist()
        upper = tf.cast(sub.upper, dtype=tf.float64).numpy().tolist()
        for col, lo, hi in zip(tag_to_columns[stag], lower, upper):
            feature_dims.append(col)
            bounds_rows.append((float(lo), float(hi)))

    # Translate indicator-tag keys to global flat-vector columns (the same
    # coordinate system as ``feature_dims``, matching gpflow's convention) and
    # coerce values to int so K-ary categorical category indices survive without
    # bool-coercion.
    requirements: dict[int, int] = {}
    for ind_tag, required in activity_condition_tags.items():
        # An activity-condition key is an indicator by definition; resolve its column from
        # ``subspaces``. ``indicator_tags`` is an optional cross-check: when supplied, the key
        # must be one of the declared indicators (catches typos); when omitted, any valid
        # subspace tag is accepted.
        if ind_tag not in tag_to_columns:
            raise ValueError(
                f"activity_condition_tags key '{ind_tag}' is not a key of `subspaces` "
                f"{list(tag_to_columns)}."
            )
        if indicator_tags and ind_tag not in indicator_tags:
            raise ValueError(
                f"activity_condition_tags key '{ind_tag}' is not in "
                f"indicator_tags {sorted(indicator_tags)}."
            )
        if len(tag_to_columns[ind_tag]) != 1:
            raise ValueError(
                f"Indicator '{ind_tag}' must be 1-dimensional, got "
                f"{len(tag_to_columns[ind_tag])} columns."
            )
        requirements[tag_to_columns[ind_tag][0]] = int(required)

    return HierarchyNode(
        name=name,
        feature_dims=feature_dims,
        feature_bounds=tf.constant(bounds_rows, dtype=tf.float64),
        activity_condition=ActivityCondition(requirements=requirements),
    )


class HierarchicalSearchSpace(CollectionSearchSpace):
    r"""
    A :class:`SearchSpace` that extends :class:`CollectionSearchSpace` with a hierarchy
    specification describing conditional activation of subspaces.

    The hierarchy is described by a sequence of :class:`gpflow.kernels.HierarchyNode`
    objects (each carrying integer ``feature_dims`` and a
    :class:`gpflow.kernels.ActivityCondition`). Variables fall into three roles:

    - **Indicators** (:class:`BooleanSearchSpace` or dimension-1
      :class:`CategoricalSearchSpace`), declared via ``indicator_tags``.
      Boolean indicators take values in :math:`\{0, 1\}`, ``K``-ary categorical
      indicators in :math:`\{0, \ldots, K-1\}`. Indicators are always
      unconditional; their flat-vector columns must not appear in any
      ``HierarchyNode.feature_dims``.
    - **Unconditional variables** owned by a node with the default (empty)
      ``activity_condition``. Always active.
    - **Conditional variables** owned by a node with non-empty
      ``activity_condition.requirements``. Active only when the referenced
      indicators take the required values.

    Points are represented as flat vectors concatenated in the iteration order
    of ``subspaces`` (a ``dict`` preserves insertion order in Python 3.7+), the
    same convention as :class:`TaggedProductSearchSpace`.

    .. note::
        ``sample`` and ``contains`` operate on the full flat vector and do **not**
        enforce "active-branch" semantics: every subspace always contributes its
        columns regardless of the indicator values, and inactive-branch columns are
        neither masked nor ignored. This matches the flat-vector convention expected
        by the hierarchical GP kernel, which reads the indicator columns itself to
        decide which features are active.

    .. note::
        Non-indicator subspaces must define numerical bounds (e.g. :class:`Box`).
        Categorical (non-indicator) subspaces are not supported, since the hierarchy
        encodes each non-indicator dimension as a ``(lower, upper)`` bound and there
        is no one-hot handling here; constructing such a space raises ``ValueError``.

    Example::

        subspaces = {
            "x1": Box([0.0], [1.0]),
            "y1": BooleanSearchSpace(),
            "x2": Box([0.0], [5.0]),
            "x4": Box([-2.0], [2.0]),
            "x3": Box([-1.0], [1.0]),
        }
        hierarchy = [
            hierarchy_node_from_tags(
                "shared", subspace_tags=["x1"],
                subspaces=subspaces, indicator_tags=["y1"],
            ),
            hierarchy_node_from_tags(
                "branch_A", subspace_tags=["x2", "x4"],
                activity_condition_tags={"y1": 1},
                subspaces=subspaces, indicator_tags=["y1"],
            ),
            hierarchy_node_from_tags(
                "branch_B", subspace_tags=["x3"],
                activity_condition_tags={"y1": 0},
                subspaces=subspaces, indicator_tags=["y1"],
            ),
        ]
        space = HierarchicalSearchSpace(
            subspaces, hierarchy, indicator_tags=["y1"])
    """

    def __init__(
        self,
        subspaces: Mapping[str, SearchSpace],
        hierarchy: Sequence[HierarchyNode],
        indicator_tags: Optional[Sequence[str]] = None,
        constraints: Optional[Sequence[Constraint]] = None,
        ctol: float | TensorType = 1e-7,
    ) -> None:
        """
        :param subspaces: Tag-keyed mapping of subspaces; iteration order
            defines the flat-vector layout.
        :param hierarchy: A sequence of :class:`gpflow.kernels.HierarchyNode`
            objects defining the conditional structure. Every non-indicator
            flat-vector column must appear in at least one node's
            ``feature_dims``.
        :param indicator_tags: Tags corresponding to indicator subspaces; each
            tag must be a key of ``subspaces`` and the corresponding subspace
            must be either a :class:`BooleanSearchSpace` or a dimension-1
            :class:`CategoricalSearchSpace`. **Optional**: if omitted, the
            indicators are inferred as the subspaces whose columns appear as
            ``activity_condition`` keys in ``hierarchy`` (in subspace order).
            Pass it explicitly to disambiguate a Boolean intended as a plain
            feature, or to have an ungated indicator flagged as an error.
        :param constraints: Optional explicit constraints enforced on the full
            flat-vector representation, following the same ``constraints``
            contract as :class:`Box`. Consumed by :meth:`constraints_residuals`
            and :meth:`is_feasible`, so a constrained space plugs into the usual
            Bayesian optimization machinery unchanged.
        :param ctol: Tolerance used by :meth:`is_feasible` when checking that
            residuals are non-negative.
        :raises ValueError: If any validation rule is violated.
        """
        subspaces = dict(subspaces)  # decouple from caller, preserve order
        super().__init__(list(subspaces.values()), list(subspaces.keys()))
        self._hierarchy = tuple(hierarchy)
        self._indicator_value_sets: dict[str, tuple[int, ...]] = {}
        self._constraints: Sequence[Constraint] = [] if constraints is None else list(constraints)
        self._ctol = ctol

        subspace_sizes = self.subspace_dimension
        self._subspace_sizes_by_tag: dict[str, TensorType] = dict(zip(self._tags, subspace_sizes))
        self._subspace_starting_indices: dict[str, TensorType] = dict(
            zip(self._tags, tf.cumsum(subspace_sizes, exclusive=True))
        )
        self._dimension = tf.cast(tf.reduce_sum(subspace_sizes), dtype=tf.int32)

        # Pre-compute a flat tag -> [column indices] map and its inverse
        # (column index -> owning tag) so feature_dims can be translated back
        # to subspace tags by the activity/query methods.
        self._tag_to_columns = _build_tag_to_columns_map(
            [(tag, int(self._subspace_sizes_by_tag[tag])) for tag in self._tags]
        )
        self._column_to_tag = {
            col: tag for tag, cols in self._tag_to_columns.items() for col in cols
        }
        self._total_columns = sum(len(cols) for cols in self._tag_to_columns.values())

        # Resolve the indicator tags. When not given explicitly, infer them by role: an
        # indicator is a subspace whose column is referenced as an ``activity_condition`` key
        # somewhere in the hierarchy (kept in subspace order for determinism). An explicit
        # value is used as-is so callers can mark a Boolean as a plain feature or have an
        # ungated indicator rejected by ``_validate``.
        if indicator_tags is None:
            referenced_columns = {
                col for node in self._hierarchy for col in node.activity_condition.requirements
            }
            self._indicator_tags = tuple(
                tag
                for tag in self._tags
                if any(col in referenced_columns for col in self._tag_to_columns[tag])
            )
        else:
            self._indicator_tags = tuple(indicator_tags)

        self._validate()

    def _validate(self) -> None:
        all_tags = set(self.subspace_tags)

        # indicator_tags must not contain duplicates; a repeated tag would corrupt
        # the indicator column lookup used by ActivityCondition.
        _reject_duplicate_tags("indicator_tags", self._indicator_tags)

        # indicator_tags must exist and reference a BooleanSearchSpace or a 1-D
        # CategoricalSearchSpace; record each indicator's permitted value set.
        for itag in self._indicator_tags:
            if itag not in all_tags:
                raise ValueError(
                    f"Indicator tag '{itag}' is not a key of `subspaces` {sorted(all_tags)}."
                )
            sub = self.get_subspace(itag)
            if isinstance(sub, BooleanSearchSpace):
                self._indicator_value_sets[itag] = (0, 1)
            elif isinstance(sub, CategoricalSearchSpace):
                if len(sub.tags) != 1:
                    raise ValueError(
                        f"Indicator tag '{itag}' must reference a dimension-1 "
                        f"CategoricalSearchSpace, got a categorical of dimension "
                        f"{len(sub.tags)}."
                    )
                self._indicator_value_sets[itag] = tuple(range(len(sub.tags[0])))
            else:
                raise ValueError(
                    f"Indicator tag '{itag}' must reference a BooleanSearchSpace or a "
                    f"dimension-1 CategoricalSearchSpace, got {type(sub).__name__}."
                )

        # Non-indicator subspaces must expose numerical bounds: the hierarchy
        # encodes each as a (lower, upper) ``feature_bounds`` row, which requires
        # ``lower``/``upper`` to be defined. Categorical (non-indicator) subspaces
        # have no numerical bounds (``has_bounds`` is False) and no one-hot
        # handling here, so reject them with a clear error rather than letting
        # ``sub.lower``/``sub.upper`` raise an opaque ``AttributeError`` later.
        for ntag in self.non_indicator_tags:
            if not self.get_subspace(ntag).has_bounds:
                raise ValueError(
                    f"Non-indicator subspace '{ntag}' "
                    f"({type(self.get_subspace(ntag)).__name__}) has no numerical "
                    f"bounds and is not supported in a HierarchicalSearchSpace. "
                    f"Non-indicator subspaces must define numerical bounds (e.g. Box)."
                )

        # Set of flat-vector columns occupied by indicators (forbidden in
        # feature_dims) and by non-indicator subspaces (must each be covered).
        indicator_columns: set[int] = set()
        for itag in self._indicator_tags:
            indicator_columns.update(self._tag_to_columns[itag])
        non_indicator_columns: set[int] = set(range(self._total_columns)) - indicator_columns

        covered_non_indicator_columns: set[int] = set()
        used_indicator_columns: set[int] = set()

        for node in self._hierarchy:
            node_feature_dims = list(node.feature_dims)
            node_bounds = tf.convert_to_tensor(node.feature_bounds, dtype=tf.float64)

            # feature_dims must be in range and not point at indicator columns
            for col in node_feature_dims:
                if col < 0 or col >= self._total_columns:
                    raise ValueError(
                        f"HierarchyNode '{node.name}' has feature_dims column {col} "
                        f"out of range [0, {self._total_columns - 1}]."
                    )
                if col in indicator_columns:
                    raise ValueError(
                        f"HierarchyNode '{node.name}' feature_dims column {col} points "
                        f"at an indicator column."
                    )
                covered_non_indicator_columns.add(col)

            # feature_bounds rows must match the underlying subspace bounds
            for row_index, col in enumerate(node_feature_dims):
                owning_tag = self._column_to_tag[col]
                owning_sub = self.get_subspace(owning_tag)
                col_within_sub = self._tag_to_columns[owning_tag].index(col)
                expected_lower = float(
                    tf.cast(owning_sub.lower, tf.float64).numpy()[col_within_sub]
                )
                expected_upper = float(
                    tf.cast(owning_sub.upper, tf.float64).numpy()[col_within_sub]
                )
                actual_lower = float(node_bounds[row_index, 0].numpy())
                actual_upper = float(node_bounds[row_index, 1].numpy())
                if not (
                    np.isclose(expected_lower, actual_lower)
                    and np.isclose(expected_upper, actual_upper)
                ):
                    raise ValueError(
                        f"HierarchyNode '{node.name}' feature_bounds row {row_index} "
                        f"= ({actual_lower}, {actual_upper}) disagrees with the "
                        f"underlying subspace bounds ({expected_lower}, "
                        f"{expected_upper}) at column {col} (tag '{owning_tag}')."
                    )

            # ActivityCondition requirements keys are the global flat-vector
            # columns of the gating indicators (gpflow's convention, the same
            # coordinate system as feature_dims). Each must be an indicator column
            # and its required value must be in that indicator's permitted set.
            for indicator_col, required in node.activity_condition.requirements.items():
                if indicator_col not in indicator_columns:
                    raise ValueError(
                        f"HierarchyNode '{node.name}' activity_condition references "
                        f"column {indicator_col}, which is not an indicator column "
                        f"(indicators are at columns {sorted(indicator_columns)})."
                    )
                ind_tag = self._column_to_tag[indicator_col]
                permitted = self._indicator_value_sets[ind_tag]
                if int(required) not in permitted:
                    raise ValueError(
                        f"HierarchyNode '{node.name}' activity_condition required value "
                        f"{required!r} for indicator '{ind_tag}' (column "
                        f"{indicator_col}) is not in the indicator's permitted set "
                        f"{list(permitted)}."
                    )
                used_indicator_columns.add(indicator_col)

        # Every non-indicator column must be covered by some node.feature_dims
        orphan_columns = non_indicator_columns - covered_non_indicator_columns
        if orphan_columns:
            orphan_tags = sorted({self._column_to_tag[c] for c in orphan_columns})
            raise ValueError(
                f"Non-indicator subspace tags {orphan_tags} have orphan columns "
                f"{sorted(orphan_columns)} that do not appear in any "
                f"HierarchyNode.feature_dims."
            )

        # Every indicator column must appear as an activity_condition key somewhere
        unused = indicator_columns - used_indicator_columns
        if unused:
            unused_tags = sorted({self._column_to_tag[c] for c in unused})
            raise ValueError(
                f"Indicator tags {unused_tags} are unused: they do not appear as a key "
                f"in any HierarchyNode.activity_condition.requirements."
            )

    @staticmethod
    def _nodes_equal(left: Sequence[HierarchyNode], right: Sequence[HierarchyNode]) -> bool:
        """Compare two hierarchies by value. ``HierarchyNode`` is a frozen dataclass,
        but its ``feature_bounds`` is a tensor, so the auto-generated ``__eq__`` would
        compare tensors elementwise (and fail ``bool()``); compare field by field
        instead, using ``np.array_equal`` for the bounds."""
        if len(left) != len(right):
            return False
        for a, b in zip(left, right):
            if a.name != b.name:
                return False
            if list(a.feature_dims) != list(b.feature_dims):
                return False
            if dict(a.activity_condition.requirements) != dict(b.activity_condition.requirements):
                return False
            if not np.array_equal(
                tf.convert_to_tensor(a.feature_bounds, dtype=tf.float64).numpy(),
                tf.convert_to_tensor(b.feature_bounds, dtype=tf.float64).numpy(),
            ):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        """
        :param other: A search space.
        :return: Whether the search space is identical to this one, including its
            hierarchy and indicator tags (the base-class comparison only covers the
            subspaces and their tags).
        """
        if not isinstance(other, HierarchicalSearchSpace):
            return NotImplemented
        if not super().__eq__(other):
            return False
        return (
            self._indicator_tags == other._indicator_tags
            and self._constraints == other._constraints
            and self._nodes_equal(self._hierarchy, other._hierarchy)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"spaces={[self.get_subspace(tag) for tag in self.subspace_tags]}, "
            f"tags={list(self.subspace_tags)}, "
            f"hierarchy={list(self._hierarchy)}, "
            f"indicator_tags={list(self._indicator_tags)}, "
            f"constraints={self._constraints!r}, "
            f"ctol={self._ctol!r})"
        )

    @property
    def hierarchy(self) -> tuple[HierarchyNode, ...]:
        """The hierarchy specification."""
        return self._hierarchy

    @property
    def indicator_tags(self) -> tuple[str, ...]:
        """The declared indicator tags (Boolean or categorical)."""
        return self._indicator_tags

    @property
    def indicator_dims(self) -> list[int]:
        """Flat-vector column indices corresponding to ``indicator_tags``, in declared order.

        These are the columns referenced by the keys of each node's
        ``activity_condition.requirements`` (gpflow's convention). Each indicator
        subspace contributes exactly one column (dimension 1).
        """
        return [self._tag_to_columns[t][0] for t in self._indicator_tags]

    @property
    def indicator_value_sets(self) -> dict[str, tuple[int, ...]]:
        """The permitted integer-valued set for each indicator tag, as built during
        validation. ``(0, 1)`` for a :class:`BooleanSearchSpace` indicator and
        ``(0, ..., K-1)`` for a ``K``-ary :class:`CategoricalSearchSpace` indicator."""
        return dict(self._indicator_value_sets)

    @property
    def non_indicator_tags(self) -> tuple[str, ...]:
        """All subspace tags that are not indicators, in tag order."""
        indicator_set = set(self._indicator_tags)
        return tuple(t for t in self.subspace_tags if t not in indicator_set)

    @property
    def constraints(self) -> Sequence[Constraint]:
        """The explicit constraints enforced on the full flat-vector representation."""
        return self._constraints

    @property
    def ctol(self) -> float | TensorType:
        """The tolerance applied when checking the explicit constraints."""
        return self._ctol

    @property
    def has_constraints(self) -> bool:
        """Returns ``True`` if this search space has any explicit constraints specified."""
        return len(self._constraints) > 0

    def constraints_residuals(self, points: TensorType) -> TensorType:
        """
        Return residuals for all explicit constraints in this search space, evaluated on the
        full flat-vector representation.

        :param points: The points to get the residuals for, with shape ``[..., D]``.
        :return: A tensor of all the residuals with shape ``[..., C]``, where ``C`` is the
            total number of constraint residual columns.
        :raise NotImplementedError: If this search space has no constraints.
        """
        if not self._constraints:
            raise NotImplementedError(
                "No constraints to compute residuals for in this search space."
            )
        residuals = [constraint.residual(points) for constraint in self._constraints]
        return tf.concat(residuals, axis=-1)

    def is_feasible(self, points: TensorType) -> TensorType:
        """
        Check whether points satisfy the explicit constraints of this search space. Note that
        membership of the space (the conditional activity structure) is not checked here.

        :param points: The points to check, with shape ``[..., D]``.
        :return: A boolean tensor of shape ``[...]``, ``True`` where the point is feasible.
        """
        if not self.has_constraints:
            return tf.cast(tf.ones(tf.shape(points)[:-1]), dtype=bool)
        return tf.math.reduce_all(self.constraints_residuals(points) >= -self._ctol, axis=-1)

    @property
    @check_shapes("return: []")
    def dimension(self) -> TensorType:
        """The total number of dimensions across all subspaces."""
        return self._dimension

    @property
    @check_shapes("return: [D]")
    def lower(self) -> TensorType:
        """The lowest values taken by each dimension, concatenated across subspaces."""
        lower_for_each = self.subspace_lower
        return (
            tf.concat(lower_for_each, axis=-1)
            if lower_for_each
            else tf.constant([], dtype=DEFAULT_DTYPE)
        )

    @property
    @check_shapes("return: [D]")
    def upper(self) -> TensorType:
        """The highest values taken by each dimension, concatenated across subspaces."""
        upper_for_each = self.subspace_upper
        return (
            tf.concat(upper_for_each, axis=-1)
            if upper_for_each
            else tf.constant([], dtype=DEFAULT_DTYPE)
        )

    @check_shapes("return: [num_samples, D]")
    def sample(self, num_samples: int, seed: Optional[int] = None) -> TensorType:
        """Sample from the space by sampling each subspace and concatenating.

        Every subspace is sampled and contributes its columns; this does **not**
        enforce active-branch semantics (inactive-branch columns are still filled).
        See the class docstring.
        """
        subspace_samples = self.subspace_sample(num_samples, seed)
        return tf.concat(subspace_samples, -1)

    def _contains(self, value: TensorType) -> TensorType:
        # Membership requires every subspace component to be in bounds; this does
        # not enforce active-branch semantics (inactive-branch columns are still
        # checked against their subspace). See the class docstring.
        in_each_subspace = [
            self._spaces[tag].contains(self.get_subspace_component(tag, value))
            for tag in self._tags
        ]
        return tf.reduce_all(in_each_subspace, axis=0)

    def get_subspace_component(self, tag: str, values: TensorType) -> TensorType:
        """
        Extract the columns of ``values`` corresponding to a particular subspace.

        :param tag: The subspace tag.
        :param values: Points from this space, shape ``[N, D]``.
        :return: The sub-components, shape ``[N, D_sub]``.
        """
        start = self._subspace_starting_indices[tag]
        end = start + self._subspace_sizes_by_tag[tag]
        return values[..., start:end]

    def _node_subspace_tags(self, node: HierarchyNode) -> list[str]:
        """Translate a node's ``feature_dims`` to the (unique, ordered) list of
        non-indicator subspace tags it owns."""
        seen: list[str] = []
        for col in node.feature_dims:
            tag = self._column_to_tag.get(int(col))
            if tag is not None and tag not in seen:
                seen.append(tag)
        return seen

    def active_subspace_tags(self, indicator_config: Mapping[str, int]) -> list[str]:
        """
        Return the non-indicator subspace tags that are active for a given indicator
        configuration.

        :param indicator_config: A mapping ``{indicator_tag: value}`` providing a value for
            every indicator of this space. Extra keys are ignored.
        :return: List of active non-indicator subspace tags.
        :raise ValueError: If ``indicator_config`` omits any indicator.
        """
        self._require_complete_config(indicator_config)
        active: list[str] = []
        for node in self._hierarchy:
            if self._node_is_active(node, indicator_config):
                for stag in self._node_subspace_tags(node):
                    if stag not in active:
                        active.append(stag)
        return active

    def enumerate_tasks(self) -> list[dict[str, int]]:
        """
        Return every indicator configuration as the Cartesian product of each indicator's
        permitted value set.

        Both Boolean and ``K``-ary categorical indicators contribute the integer values
        ``[0, 1, ..., K-1]`` (with ``K = 2`` for Boolean indicators); the total number of
        configurations equals :math:`\\prod_k |\\mathcal{C}_k|`.

        :return: A list of ``{indicator_tag: value}`` dictionaries, one per task.
        """
        if not self._indicator_tags:
            return [{}]
        per_indicator_values: list[Sequence[int]] = [
            list(self._indicator_value_sets[t]) for t in self._indicator_tags
        ]
        return [
            dict(zip(self._indicator_tags, combo))
            for combo in itertools_product(*per_indicator_values)
        ]

    def is_active(self, tag: str, indicator_config: Mapping[str, int]) -> bool:
        """
        Check whether a non-indicator subspace is active for a given indicator configuration.

        :param tag: A non-indicator subspace tag.
        :param indicator_config: A mapping ``{indicator_tag: value}`` providing a value for
            every indicator of this space. Extra keys are ignored.
        :return: True if the subspace is active.
        :raise ValueError: If ``indicator_config`` omits any indicator.
        """
        self._require_complete_config(indicator_config)
        for node in self._hierarchy:
            if tag in self._node_subspace_tags(node) and self._node_is_active(
                node, indicator_config
            ):
                return True
        return False

    def node_for_subspace(self, tag: str) -> list[HierarchyNode]:
        """
        Return the :class:`HierarchyNode` objects whose ``feature_dims`` include any column
        belonging to the given subspace tag.

        :param tag: A non-indicator subspace tag.
        :return: List of nodes containing this tag.
        """
        return [node for node in self._hierarchy if tag in self._node_subspace_tags(node)]

    def _require_complete_config(self, indicator_config: Mapping[str, int]) -> None:
        """Reject a partial configuration: ``indicator_config`` must provide a value for every
        indicator of this space (extra keys are ignored). Catches forgotten or mistyped
        indicator keys, which would otherwise be silently treated as unsatisfied."""
        missing = [tag for tag in self._indicator_tags if tag not in indicator_config]
        if missing:
            raise ValueError(
                f"`indicator_config` must provide a value for every indicator "
                f"{list(self._indicator_tags)}; missing: {missing}."
            )

    def product(self, other: HierarchicalSearchSpace) -> HierarchicalSearchSpace:
        """
        Return a new :class:`HierarchicalSearchSpace` that is the combination of this space and
        ``other``. Tags in the two spaces must be disjoint. The hierarchy nodes are
        concatenated; both the node ``feature_dims`` and the ``activity_condition``
        requirement columns in ``other`` are shifted by ``self.dimension`` so they index the
        combined flat-vector layout.

        Global (positional) constraints are remapped into the combined layout via
        :func:`_embed_constraint`: ``self``'s constraints keep columns ``[0, D_self)`` and
        ``other``'s are shifted to ``[D_self, D_self + D_other)``. The combined space takes the
        tighter (minimum) of the two operands' constraint tolerances.

        :param other: Another :class:`HierarchicalSearchSpace`.
        :return: The combined hierarchical space.
        :raises ValueError: If the two spaces share any tags.
        """
        overlap = set(self.subspace_tags) & set(other.subspace_tags)
        if overlap:
            raise ValueError(f"Cannot combine spaces with overlapping tags: {overlap}")
        combined_subspaces: dict[str, SearchSpace] = {
            **{tag: self.get_subspace(tag) for tag in self.subspace_tags},
            **{tag: other.get_subspace(tag) for tag in other.subspace_tags},
        }
        offset = int(self.dimension)
        combined_dim = int(self.dimension) + int(other.dimension)
        constraints = [
            _embed_constraint(c, 0, int(self.dimension), combined_dim) for c in self._constraints
        ] + [
            _embed_constraint(c, offset, int(other.dimension), combined_dim)
            for c in other._constraints
        ]
        shifted_other_hierarchy: list[HierarchyNode] = []
        for node in other.hierarchy:
            shifted_other_hierarchy.append(
                HierarchyNode(
                    name=node.name,
                    feature_dims=[int(c) + offset for c in node.feature_dims],
                    feature_bounds=node.feature_bounds,
                    activity_condition=ActivityCondition(
                        # requirement keys are global indicator columns; shift them
                        # into the combined layout by the same offset as feature_dims.
                        requirements={
                            k + offset: v
                            for k, v in node.activity_condition.requirements.items()
                        }
                    ),
                )
            )
        hierarchy = list(self.hierarchy) + shifted_other_hierarchy
        indicator_tags = list(self.indicator_tags) + list(other.indicator_tags)
        return HierarchicalSearchSpace(
            combined_subspaces,
            hierarchy,
            indicator_tags,
            constraints=constraints,
            ctol=min(self.ctol, other.ctol),
        )

    def _node_is_active(self, node: HierarchyNode, indicator_config: Mapping[str, int]) -> bool:
        """Check whether a node's activity_condition.requirements (keyed by indicator global
        column) are all satisfied by ``indicator_config`` (keyed by indicator tag). Uses
        integer equality so that K-ary categorical indicators are compared exactly rather
        than via Boolean truthiness. Assumes a complete config (see
        :meth:`_require_complete_config`, enforced by the public callers)."""
        for indicator_col, required in node.activity_condition.requirements.items():
            ind_tag = self._column_to_tag[indicator_col]
            if int(indicator_config[ind_tag]) != int(required):
                return False
        return True


class TaggedMultiSearchSpace(CollectionSearchSpace):
    r"""
    A :class:`SearchSpace` made up of a collection of multiple :class:`SearchSpace` subspaces,
    each with a unique tag. All subspaces must have the same dimensionality.

    Each subspace is treated as an independent space and not combined in any way. This class
    provides functionality for accessing all the subspaces at once by using the usual search space
    methods, as well as for accessing individual subspaces.

    When accessing all subspaces at once from this class (e.g. `lower()`, `upper()`, `sample()`),
    the returned tensors have an extra dimension corresponding to the subspaces.

    This class can be useful to represent a collection of search spaces that do not interact with
    each other. For example, it is used to implement batch trust region rules in the
    :class:`BatchTrustRegion` class.
    """

    def __init__(self, spaces: Sequence[SearchSpace], tags: Optional[Sequence[str]] = None):
        r"""
        Build a :class:`TaggedMultiSearchSpace` from a list ``spaces`` of other spaces. If
        ``tags`` are provided then they form the identifiers of the subspaces, otherwise the
        subspaces are labelled numerically.

        :param spaces: A sequence of :class:`SearchSpace` objects representing the space's subspaces
        :param tags: An optional list of tags giving the unique identifiers of
            the space's subspaces.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``spaces`` has a different
            length to ``tags`` when ``tags`` is provided or if ``tags`` contains duplicates.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``spaces`` has a different
            dimension to each other.
        """

        # At least one subspace is required.
        tf.debugging.assert_greater(
            len(spaces),
            0,
            message=f"""
                At least one subspace is required but received {len(spaces)}.
                """,
        )

        tf.debugging.assert_equal(
            len({int(space.dimension) for space in spaces}),
            1,
            message=f"""
                All subspaces must have the same dimension but received
                {[int(space.dimension) for space in spaces]}.
                """,
        )

        super().__init__(spaces, tags)

    @property
    @check_shapes("return: [V, D]")
    def lower(self) -> TensorType:
        """Returns the stacked lower bounds of all the subspaces.

        :return: The lower bounds of shape [V, D], where V is the number of subspaces and D is
            the dimensionality of each subspace.
        """
        return tf.stack(self.subspace_lower, axis=0)

    @property
    @check_shapes("return: [V, D]")
    def upper(self) -> TensorType:
        """Returns the stacked upper bounds of all the subspaces.

        :return: The upper bounds of shape [V, D], where V is the number of subspaces and D is
            the dimensionality of each subspace.
        """
        return tf.stack(self.subspace_upper, axis=0)

    @property
    @check_shapes("return: []")
    def dimension(self) -> TensorType:
        """The number of inputs in this search space."""
        return self.get_subspace(self.subspace_tags[0]).dimension

    @check_shapes("return: [num_samples, V, D]")
    def sample(self, num_samples: int, seed: Optional[int] = None) -> TensorType:
        """
        Sample randomly from the space by sampling from each subspace
        and returning the resulting samples stacked along the second axis in the same order as
        specified when initializing the space.

        :param num_samples: The number of points to sample from each subspace.
        :param seed: Optional tf.random seed.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly,
            from each search subspace with shape '[num_samples, V, D]' , where V is the number of
            subspaces and D is the search space dimension.
        """
        return tf.stack(self.subspace_sample(num_samples, seed), axis=1)

    def _contains(self, value: TensorType) -> TensorType:
        """
        Return `True` if ``value`` is a member of this search space, else `False`. A point
        is a member if it is a member of any of the subspaces.

        :param value: A point or points to check for membership of this :class:`SearchSpace`.
        :return: A boolean array showing membership for each point in value.
        """
        return tf.reduce_any(
            [self.get_subspace(tag)._contains(value) for tag in self.subspace_tags], axis=0
        )

    def product(self, other: TaggedMultiSearchSpace) -> TaggedMultiSearchSpace:
        r"""
        Return a bigger collection of two :class:`TaggedMultiSearchSpace`\ s, regenerating the
        tags.

        :param other: A search space of the same type as this search space.
        :return: The product of this search space with the ``other``.
        """
        return TaggedMultiSearchSpace(
            spaces=tuple(self._spaces.values()) + tuple(other._spaces.values())
        )

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
        samples = self.sample(num_samples)  # Sample num_samples points from each subspace.
        samples = tf.reshape(samples, [-1, self.dimension])  # Flatten the samples across subspaces.
        samples = tf.random.shuffle(samples)[:num_samples]  # Randomly pick num_samples points.
        return DiscreteSearchSpace(points=samples)

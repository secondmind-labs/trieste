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

from abc import ABC, abstractmethod
from typing import Dict, List, TypeVar, Union

import tensorflow as tf
from typing_extensions import Final

from .type import TensorType
from .utils import shapes_equal

SP = TypeVar("SP", bound="SearchSpace")


class SearchSpace(ABC):
    """
    A :class:`SearchSpace` represents the domain over which an objective function is optimized.
    """

    @abstractmethod
    def sample(self, num_samples: int) -> TensorType:
        """
        :param num_samples: The number of points to sample from this search space.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly from this search space.
        """

    @abstractmethod
    def __contains__(self, value: TensorType) -> Union[bool, TensorType]:
        """
        :param value: A point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`. May return a
            scalar boolean `tf.Tensor` instead of the `bool` itself.
        :raise ValueError (or InvalidArgumentError): If ``value`` has a different dimensionality
            from this :class:`SearchSpace`.
        """

    @abstractmethod
    def __mul__(self: SP, other: SP) -> SP:
        """
        :param other: A search space of the same type as this search space.
        :return: The Cartesian product of this search space with ``other``.
        """

    def __pow__(self: SP, other: int) -> SP:
        """
        Return the Cartesian product of ``other`` instances of this search space. For example, for
        an exponent of `3`, and search space `s`, this is `s ** 3`, which is equivalent to
        `s * s * s`.

        :param other: The number of instances of this search space to multiply. Must be strictly
            positive.
        :return: The Cartesian product of ``other`` instances of this search space.
        :raise ValueError: If the exponent ``other`` is less than 1.
        """
        if other < 1:
            raise ValueError("The exponent ``other`` can only be a strictly positive integer")

        space = self
        for _ in range(other - 1):
            space *= self
        return space


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
        :raise ValueError (or InvalidArgumentError): If ``points`` has an invalid shape.
        """
        tf.debugging.assert_shapes([(points, ("N", "D"))])

        self.points: Final[TensorType] = points
        """ All the points in this space. """

    def __repr__(self) -> str:
        return f"DiscreteSearchSpace({self.points!r})"

    def __contains__(self, value: TensorType) -> Union[bool, TensorType]:
        tf.debugging.assert_shapes([(value, self.points.shape[1:])])
        return tf.reduce_any(tf.reduce_all(value == self.points, axis=1))

    def sample(self, num_samples: int) -> TensorType:
        """
        :param num_samples: The number of points to sample from this search space.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly, and without replacement,
            from this search space.
        """
        num_points = self.points.shape[0]
        if num_samples > num_points:
            raise ValueError(
                "Number of samples cannot be greater than the number of points"
                f" {num_points} in discrete search space, got {num_samples}"
            )

        return tf.random.shuffle(self.points)[:num_samples, :]

    def __mul__(self, other: DiscreteSearchSpace) -> DiscreteSearchSpace:
        r"""
        Return the Cartesian product of the two :class:`DiscreteSearchSpace`\ s. For example:

            >>> sa = DiscreteSearchSpace(tf.constant([[0, 1], [2, 3]]))
            >>> sb = DiscreteSearchSpace(tf.constant([[4, 5, 6], [7, 8, 9]]))
            >>> (sa * sb).points.numpy()
            array([[0, 1, 4, 5, 6],
                   [0, 1, 7, 8, 9],
                   [2, 3, 4, 5, 6],
                   [2, 3, 7, 8, 9]], dtype=int32)

        :param other: :class:`DiscreteSearchSpace`.
        :return: the new combined :class:`DiscreteSearchSpace`
        :raise TypeError: If the lhs and rhs :class:`DiscreteSearchSpace` points have different
            types.
        """
        if self.points.dtype is not other.points.dtype:
            return NotImplemented

        N = self.points.shape[0]
        M = other.points.shape[0]
        tile_self = tf.tile(tf.expand_dims(self.points, 1), [1, M, 1])
        tile_dss = tf.tile(tf.expand_dims(other.points, 0), [N, 1, 1])
        cartesian_product = tf.concat([tile_self, tile_dss], axis=2)

        return DiscreteSearchSpace(tf.reshape(cartesian_product, [N * M, -1]))

    def __deepcopy__(self, memo: Dict[int, object]) -> DiscreteSearchSpace:
        return self


Vector = TypeVar("Vector", TensorType, List[float])
r""" A type variable representing either a `TensorType` or `list` of `float`\ s. """


class Box(SearchSpace):
    r"""
    Continuous :class:`SearchSpace` representing a :math:`D`-dimensional box in
    :math:`\mathbb{R}^D`. Mathematically it is equivalent to the Cartesian product of :math:`D`
    closed bounded intervals in :math:`\mathbb{R}`.
    """

    def __init__(self, lower: Vector, upper: Vector):
        r"""
        If ``lower`` and ``upper`` are `list`\ s, they will be converted to tensors of dtype
        `tf.float64`.

        **Type hints:** If ``lower`` or ``upper`` is a `list`, they must both be `list`\ s.

        :param lower: The lower (inclusive) bounds of the box. Must have shape [D] for positive D,
            and if a tensor, must have float type.
        :param upper: The upper (inclusive) bounds of the box. Must have shape [D] for positive D,
            and if a tensor, must have float type.
        :raise ValueError (or InvalidArgumentError): If any of the following are true:

            - ``lower`` and ``upper`` have invalid shapes.
            - ``lower`` and ``upper`` do not have the same floating point type.
            - ``upper`` is not greater than ``lower`` across all dimensions.
        """
        tf.debugging.assert_shapes([(lower, ["D"]), (upper, ["D"])])
        tf.assert_rank(lower, 1)
        tf.assert_rank(upper, 1)

        if len(lower) == 0:
            raise ValueError(f"Bounds must have shape [D] for positive D, got {tf.shape(lower)}.")

        if isinstance(lower, list):
            lower_as_tensor = tf.cast(lower, dtype=tf.float64)
            upper_as_tensor = tf.cast(upper, dtype=tf.float64)
        else:
            lower_as_tensor = tf.convert_to_tensor(lower)
            upper_as_tensor = tf.convert_to_tensor(upper)

            tf.debugging.assert_same_float_dtype([lower_as_tensor, upper_as_tensor])

        tf.debugging.assert_less(lower_as_tensor, upper_as_tensor)

        self.lower: Final[TensorType] = lower_as_tensor
        """ The lower bounds of the box. """

        self.upper: Final[TensorType] = upper_as_tensor
        """ The upper bounds of the box. """

    def __repr__(self) -> str:
        return f"Box({self.lower!r}, {self.upper!r})"

    def __contains__(self, value: TensorType) -> Union[bool, TensorType]:
        """
        Return `True` if ``value`` is a member of this search space, else `False`. A point is a
        member if all of its coordinates lie in the closed intervals bounded by the lower and upper
        bounds.

        :param value: A point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`. May return a
            scalar boolean `tf.Tensor` instead of the `bool` itself.
        :raise ValueError (or InvalidArgumentError): If ``value`` has a different dimensionality
            from the search space.
        """
        if not shapes_equal(value, self.lower):
            raise ValueError(
                f"value must have same dimensionality as search space: {self.lower.shape},"
                f" got shape {value.shape}"
            )

        return tf.reduce_all(value >= self.lower) and tf.reduce_all(value <= self.upper)

    def sample(self, num_samples: int) -> TensorType:
        dim = tf.shape(self.lower)[-1]
        return tf.random.uniform(
            (num_samples, dim), minval=self.lower, maxval=self.upper, dtype=self.lower.dtype
        )

    def discretize(self, num_samples: int) -> DiscreteSearchSpace:
        """
        :param num_samples: The number of points in the :class:`DiscreteSearchSpace`.
        :return: A discrete search space consisting of ``num_samples`` points sampled uniformly from
            this :class:`Box`.
        """
        return DiscreteSearchSpace(points=self.sample(num_samples))

    def __mul__(self, other: Box) -> Box:
        r"""
        Return the Cartesian product of the two :class:`Box`\ es (concatenating their respective
        lower and upper bounds).

        :param box: :class:`Box`.
        :return: the new combined :class:`Box`
        :raise TypeError: If the lhs and rhs :class:`Box` bounds have different types.
        """
        if self.lower.dtype is not other.lower.dtype:
            return NotImplemented

        expanded_lower_bound = tf.concat([self.lower, other.lower], axis=-1)
        expanded_upper_bound = tf.concat([self.upper, other.upper], axis=-1)
        return Box(expanded_lower_bound, expanded_upper_bound)

    def __deepcopy__(self, memo: Dict[int, object]) -> Box:
        return self

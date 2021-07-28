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
from typing import Optional, Sequence, TypeVar, overload

import tensorflow as tf
import tensorflow_probability as tfp

from .type import TensorType
from .utils import shapes_equal

SP = TypeVar("SP", bound="SearchSpace")
""" A type variable bound to :class:`SearchSpace`. """


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
    def __contains__(self, value: TensorType) -> bool | TensorType:
        """
        :param value: A point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`. May return a
            scalar boolean `TensorType` instead of the `bool` itself.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``value`` has a different
            dimensionality from this :class:`SearchSpace`.
        """

    @abstractmethod
    def __mul__(self: SP, other: SP) -> SP:
        """
        :param other: A search space of the same type as this search space.
        :return: The Cartesian product of this search space with the ``other``.
        """

    def __pow__(self: SP, other: int) -> SP:
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

    def __repr__(self) -> str:
        """"""
        return f"DiscreteSearchSpace({self._points!r})"

    @property
    def points(self) -> TensorType:
        """All the points in this space."""
        return self._points

    def __contains__(self, value: TensorType) -> bool | TensorType:
        tf.debugging.assert_shapes([(value, self.points.shape[1:])])
        return tf.reduce_any(tf.reduce_all(value == self._points, axis=1))

    def sample(self, num_samples: int) -> TensorType:
        """
        :param num_samples: The number of points to sample from this search space.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly, and without replacement,
            from this search space.
        """
        tf.debugging.assert_less_equal(
            num_samples,
            len(self._points),
            message="Number of samples cannot be greater than the number of points in search space",
        )
        return tf.random.shuffle(self._points)[:num_samples, :]

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

    def __deepcopy__(self, memo: dict[int, object]) -> DiscreteSearchSpace:
        return self


class Box(SearchSpace):
    r"""
    Continuous :class:`SearchSpace` representing a :math:`D`-dimensional box in
    :math:`\mathbb{R}^D`. Mathematically it is equivalent to the Cartesian product of :math:`D`
    closed bounded intervals in :math:`\mathbb{R}`.
    """

    @overload
    def __init__(self, lower: Sequence[float], upper: Sequence[float]):
        ...

    @overload
    def __init__(self, lower: TensorType, upper: TensorType):
        ...

    def __init__(
        self,
        lower: Sequence[float] | TensorType,
        upper: Sequence[float] | TensorType,
    ):
        r"""
        If ``lower`` and ``upper`` are `Sequence`\ s of floats (such as lists or tuples),
        they will be converted to tensors of dtype `tf.float64`.

        :param lower: The lower (inclusive) bounds of the box. Must have shape [D] for positive D,
            and if a tensor, must have float type.
        :param upper: The upper (inclusive) bounds of the box. Must have shape [D] for positive D,
            and if a tensor, must have float type.
        :raise ValueError (or tf.errors.InvalidArgumentError): If any of the following are true:

            - ``lower`` and ``upper`` have invalid shapes.
            - ``lower`` and ``upper`` do not have the same floating point type.
            - ``upper`` is not greater than ``lower`` across all dimensions.
        """

        tf.debugging.assert_shapes([(lower, ["D"]), (upper, ["D"])])
        tf.assert_rank(lower, 1)
        tf.assert_rank(upper, 1)

        tf.debugging.assert_positive(len(lower), message="bounds cannot be empty")

        if isinstance(lower, Sequence):
            self._lower = tf.constant(lower, dtype=tf.float64)
            self._upper = tf.constant(upper, dtype=tf.float64)
        else:
            self._lower = tf.convert_to_tensor(lower)
            self._upper = tf.convert_to_tensor(upper)

            tf.debugging.assert_same_float_dtype([self._lower, self._upper])

        tf.debugging.assert_less(self._lower, self._upper)

    def __repr__(self) -> str:
        """"""
        return f"Box({self._lower!r}, {self._upper!r})"

    @property
    def lower(self) -> TensorType:
        """The lower bounds of the box."""
        return self._lower

    @property
    def upper(self) -> TensorType:
        """The upper bounds of the box."""
        return self._upper

    def __contains__(self, value: TensorType) -> bool | TensorType:
        """
        Return `True` if ``value`` is a member of this search space, else `False`. A point is a
        member if all of its coordinates lie in the closed intervals bounded by the lower and upper
        bounds.

        :param value: A point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`. May return a
            scalar boolean `TensorType` instead of the `bool` itself.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``value`` has a different
            dimensionality from the search space.
        """
        tf.debugging.assert_equal(
            shapes_equal(value, self._lower),
            True,
            message="value must have same dimensionality as search space",
        )

        return tf.reduce_all(value >= self._lower) and tf.reduce_all(value <= self._upper)

    def sample(self, num_samples: int) -> TensorType:
        """
        Sample randomly from the space.

        :param num_samples: The number of points to sample from this search space.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly, and without replacement,
            from this search space with shape '[num_samples, D]' , where D is the search space
            dimension.
        """
        tf.debugging.assert_non_negative(num_samples)

        dim = tf.shape(self._lower)[-1]
        return tf.random.uniform(
            (num_samples, dim), minval=self._lower, maxval=self._upper, dtype=self._lower.dtype
        )

    def sample_halton(self, num_samples: int, seed: Optional[int] = None) -> TensorType:
        """
        Sample from the space using a Halton sequence. The resulting samples are guaranteed to be
        diverse and are reproducible by using the same choice of ``seed``.

        :param num_samples: The number of points to sample from this search space.
        :param seed: Random seed for the halton sequence
        :return: ``num_samples`` of points, using halton sequence with shape '[num_samples, D]' ,
            where D is the search space dimension.
        """

        tf.debugging.assert_non_negative(num_samples)
        if num_samples == 0:
            return tf.constant([])
        if seed is not None:  # ensure reproducibility
            tf.random.set_seed(seed)
        dim = tf.shape(self._lower)[-1]
        return (self._upper - self._lower) * tfp.mcmc.sample_halton_sequence(
            dim=dim, num_results=num_samples, dtype=self._lower.dtype, seed=seed
        ) + self._lower

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
            skip = tf.random.uniform([1], maxval=2 ** 16, dtype=tf.int32)[0]
        dim = tf.shape(self._lower)[-1]
        return (self._upper - self._lower) * tf.math.sobol_sample(
            dim=dim, num_results=num_samples, dtype=self._lower.dtype, skip=skip
        ) + self._lower

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

    def __deepcopy__(self, memo: dict[int, object]) -> Box:
        return self

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
from abc import abstractmethod, ABC
from typing import Union

import tensorflow as tf

from .type import TensorType
from .utils import shapes_equal


class SearchSpace(ABC):
    """
    A :class:`SearchSpace` represents the domain over which an objective function is optimized.
    """

    @abstractmethod
    def sample(self, num_samples: int) -> tf.Tensor:
        """
        :param num_samples: The number of points to sample from this search space.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly from this search space.
        """

    @abstractmethod
    def __contains__(self, value: TensorType) -> Union[bool, tf.Tensor]:
        """
        :param value: A point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`. May return a
            scalar boolean `tf.Tensor` instead of the `bool` itself.
        """


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
        :param points: The points that define the discrete space.
        """
        self._points = points

    @property
    def points(self) -> TensorType:
        """ All the points in this space. """
        return self._points

    def __contains__(self, value: TensorType) -> Union[bool, tf.Tensor]:
        if tf.size(value) == tf.shape(self._points)[-1]:
            is_point_for_all_points = tf.reduce_all(value == self._points, axis=1)
            return tf.reduce_any(is_point_for_all_points)

        return False

    def sample(self, num_samples: int) -> tf.Tensor:
        """
        :param num_samples: The number of points to sample from this search space.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly, and without replacement,
            from this search space.
        """
        num_points = self._points.shape[0]
        if num_samples > num_points:
            raise ValueError(
                "Number of samples cannot be greater than the number of points"
                f" {num_points} in discrete search space, got {num_samples}"
            )

        return tf.random.shuffle(self._points)[:num_samples, :]


class Box(SearchSpace):
    r"""
    Continuous :class:`SearchSpace` representing a :math:`D`-dimensional box in
    :math:`\mathbb{R}^D`. Mathematically it is equivalent to the Cartesian product of :math:`D`
    closed bounded intervals in :math:`\mathbb{R}`.
    """

    def __init__(self, lower: TensorType, upper: TensorType):
        """
        :param lower: The lower (inclusive) bounds of the box.
        :param upper: The upper (inclusive) bounds of the box.
        :raise ValueError: If ``lower`` and ``upper`` have different shapes. Or if ``upper`` is not
            greater than ``lower`` across all dimensions.
        :raise TypeError: If ``lower`` and ``upper`` have different dtypes.
        """

        if not shapes_equal(lower, upper):
            raise ValueError(
                f"Lower and upper bounds must have the same shape, got {lower.shape} and"
                f" {upper.shape}"
            )

        if lower.dtype is not upper.dtype:
            raise TypeError(
                f"Lower and upper bounds must have the same dtype, got {lower.shape} and"
                f" {upper.shape}"
            )

        if tf.reduce_any(lower >= upper):
            raise ValueError(
                f"Upper bound must be greater that lower for all dimensions, got lower bound"
                f" {lower} and upper {upper}."
            )

        self._lower = lower
        self._upper = upper

    @property
    def lower(self) -> TensorType:
        """ The lower bounds of the box. """
        return self._lower

    @property
    def upper(self) -> TensorType:
        """ The upper bounds of the box. """
        return self._upper

    def __contains__(self, value: TensorType) -> Union[bool, tf.Tensor]:
        """
        Return `True` if ``value`` is a member of this search space, else `False`. A point is a
        member if all of its coordinates lie in the closed intervals bounded by the lower and upper
        bounds.

        :param value: A point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`. May return a
            scalar boolean `tf.Tensor` instead of the `bool` itself.
        :raise ValueError: If ``value`` has a different shape from the search space.
        """
        if not shapes_equal(value, self._lower):
            raise ValueError(
                f"Point must have the same shape as the Box bounds, got {value.shape}, expected"
                f" {self._lower.shape}"
            )

        return tf.reduce_all(value >= self._lower) and tf.reduce_all(value <= self._upper)

    def sample(self, num_samples: int) -> tf.Tensor:
        dim = tf.shape(self._lower)[-1]
        return tf.random.uniform(
            (num_samples, dim), minval=self._lower, maxval=self._upper, dtype=self._lower.dtype
        )

    def discretize(self, num_samples: int) -> DiscreteSearchSpace:
        """
        :param num_samples: The number of points in the :class:`DiscreteSearchSpace`.
        :return: A discrete search space consisting of ``num_samples`` points sampled uniformly from
            this :class:`Box`.
        """
        return DiscreteSearchSpace(points=self.sample(num_samples))

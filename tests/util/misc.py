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

import functools
from typing import FrozenSet, List, Tuple, Mapping, TypeVar, Callable, Union, cast

import tensorflow as tf

from trieste.acquisition.rule import AcquisitionRule
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.space import Box, SearchSpace
from trieste.type import QueryPoints


C = TypeVar('C', bound=Callable)
""" Type variable for callables. """


def random_seed(seed: int = 0) -> Callable[[C], C]:
    """
    :param seed: The randomness seed to use.
    :return: A decorator. The decorated function will set the TensorFlow randomness seed to `seed`
        before executing.
    """
    def decorator(f: C) -> C:
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            tf.random.set_seed(seed)
            return f(*args, **kwargs)

        return cast(C, decorated)

    return decorator


def zero_dataset() -> Dataset:
    """
    :return: A 1D input, 1D output dataset with a single entry of zeroes.
    """
    return Dataset(tf.constant([[0.]]), tf.constant([[0.]]))


def one_dimensional_range(lower: float, upper: float) -> Box:
    """
    :param lower: The box lower bound.
    :param upper:  The box upper bound.
    :return: A one-dimensional box with range given by ``lower`` and ``upper``, and bound dtype
        `tf.float32`.
    :raise ValueError: If ``lower`` is not less than ``upper``.
    """
    return Box(tf.constant([lower], dtype=tf.float32), tf.constant([upper], tf.float32))


class FixedAcquisitionRule(AcquisitionRule[None, SearchSpace]):
    """ An acquisition rule that returns the same fixed value on every step. """
    def __init__(self, query_points: QueryPoints):
        """
        :param query_points: The value to return on each step.
        """
        self._qp = query_points

    def acquire(
            self,
            search_space: SearchSpace,
            datasets: Mapping[str, Dataset],
            models: Mapping[str, ProbabilisticModel],
            state: None = None
    ) -> Tuple[QueryPoints, None]:
        """
        :param search_space: Unused.
        :param datasets: Unused.
        :param models: Unused.
        :param state: Unused.
        :return: The fixed value specified on initialisation, and `None`.
        """
        return self._qp, None


ShapeLike = Union[tf.TensorShape, Tuple[int, ...], List[int]]
""" Type alias for types that can represent tensor shapes. """


def various_shapes() -> FrozenSet[Tuple[int, ...]]:
    """
    :return: A reasonably comprehensive variety of tensor shapes.
    """
    return frozenset(
        {(), (0,), (1,), (0, 0), (1, 0), (0, 1), (3, 4), (1, 0, 3), (1, 2, 3), (1, 2, 3, 4, 5, 6)}
    )

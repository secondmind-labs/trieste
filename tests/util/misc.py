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
from typing import (
    Any,
    Callable,
    Container,
    FrozenSet,
    List,
    Mapping,
    NoReturn,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy.testing as npt
import tensorflow as tf
from typing_extensions import Final

from trieste.acquisition.rule import AcquisitionRule
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.space import Box, SearchSpace
from trieste.type import TensorType
from trieste.utils import shapes_equal

TF_DEBUGGING_ERROR_TYPES: Final[Tuple[Type[Exception], ...]] = (
    ValueError,
    tf.errors.InvalidArgumentError,
)
""" Error types thrown by TensorFlow's debugging functionality. """

C = TypeVar("C", bound=Callable)
""" Type variable bound to `typing.Callable`. """


def random_seed(f: C) -> C:
    """
    :param f: A function.
    :return: The function ``f``, but with the TensorFlow randomness seed fixed to a hardcoded value.
    """

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        tf.random.set_seed(0)
        return f(*args, **kwargs)

    return cast(C, decorated)


T = TypeVar("T")
""" Unbound type variable. """

NList = Union[
    List[T],
    List[List[T]],
    List[List[List[T]]],
    List[List[List[List[Any]]]],
]
""" Type alias for a nested list with array shape. """


def mk_dataset(query_points: NList[List[float]], observations: NList[List[float]]) -> Dataset:
    """
    :param query_points: The query points.
    :param observations: The observations.
    :return: A :class:`Dataset` containing the specified ``query_points`` and ``observations`` with
        dtype `tf.float64`.
    """
    return Dataset(tf.cast(query_points, tf.float64), tf.cast(observations, tf.float64))


def zero_dataset() -> Dataset:
    """
    :return: A 1D input, 1D output dataset with a single entry of zeroes.
    """
    return Dataset(tf.constant([[0.0]]), tf.constant([[0.0]]))


def one_dimensional_range(lower: float, upper: float) -> Box:
    """
    :param lower: The box lower bound.
    :param upper:  The box upper bound.
    :return: A one-dimensional box with range given by ``lower`` and ``upper``, and bound dtype
        `tf.float32`.
    :raise ValueError: If ``lower`` is not less than ``upper``.
    """
    return Box(tf.constant([lower], dtype=tf.float32), tf.constant([upper], tf.float32))


def raise_(*_: object, **__: object) -> NoReturn:
    """
    Raise an exception. This dummy function can be used wherever a function of any signature is
    expected but isn't intended to be used.

    :raise Exception: Always.
    """
    raise Exception


def quadratic(x: tf.Tensor) -> tf.Tensor:
    r"""
    The multi-dimensional quadratic function.

    :param x: A tensor whose last dimension is of length greater than zero.
    :return: The sum :math:`\Sigma x^2` of the squares of ``x``.
    :raise ValueError: If ``x`` is a scalar or has empty trailing dimension.
    """
    if x.shape == [] or x.shape[-1] == 0:
        raise ValueError(f"x must have non-empty trailing dimension, got shape {x.shape}")

    return tf.reduce_sum(x ** 2, axis=-1, keepdims=True)


class FixedAcquisitionRule(AcquisitionRule[None, SearchSpace]):
    """ An acquisition rule that returns the same fixed value on every step. """

    def __init__(self, query_points: TensorType):
        """
        :param query_points: The value to return on each step.
        """
        self._qp = query_points

    def __repr__(self) -> str:
        return f"FixedAcquisitionRule({self._qp!r})"

    def acquire(
        self,
        search_space: SearchSpace,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ProbabilisticModel],
        state: None = None,
    ) -> Tuple[TensorType, None]:
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


def various_shapes(*, excluding_ranks: Container[int] = ()) -> FrozenSet[Tuple[int, ...]]:
    """
    :param excluding_ranks: Ranks to exclude from the result.
    :return: A reasonably comprehensive variety of tensor shapes, where no shapes will have a rank
        in ``excluding_ranks``.
    """
    shapes = (
        {()}
        | {(0,), (1,), (3,)}
        | {(0, 0), (1, 0), (0, 1), (3, 4)}
        | {(1, 0, 3), (1, 2, 3)}
        | {(1, 2, 3, 4, 5, 6)}
    )
    return frozenset(s for s in shapes if len(s) not in excluding_ranks)


def assert_datasets_allclose(this: Dataset, that: Dataset) -> None:
    """
    Check the :attr:`query_points` in ``this`` and ``that`` have the same shape and dtype, and all
    elements are approximately equal. Also check the same for :attr:`observations`.

    :param this: A dataset.
    :param that: A dataset.
    :raise AssertionError: If any of the following are true:
        - shapes are not equal
        - dtypes are not equal
        - elements are not approximately equal.
    """
    assert shapes_equal(this.query_points, that.query_points)
    assert shapes_equal(this.observations, that.observations)

    assert this.query_points.dtype == that.query_points.dtype
    assert this.observations.dtype == that.observations.dtype

    npt.assert_allclose(this.query_points, that.query_points)
    npt.assert_allclose(this.observations, that.observations)

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
from __future__ import annotations

import functools
import os
import random
from collections.abc import Container, Mapping
from typing import (
    Any,
    Callable,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import numpy.testing as npt
import tensorflow as tf
from typing_extensions import Final

from trieste.acquisition.rule import AcquisitionRule, LocalDatasetsAcquisitionRule, SearchSpaceType
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.objectives import Branin, Hartmann6
from trieste.objectives.utils import mk_observer
from trieste.space import SearchSpace
from trieste.types import State, Tag, TensorType
from trieste.utils import shapes_equal

TF_DEBUGGING_ERROR_TYPES: Final[tuple[type[Exception], ...]] = (
    ValueError,
    tf.errors.InvalidArgumentError,
)
""" Error types thrown by TensorFlow's debugging functionality for tensor shapes. """

C = TypeVar("C", bound=Callable[..., object])
""" Type variable bound to `typing.Callable`. """


@overload
def random_seed(f_py: C, seed: int = 0) -> C: ...


@overload
def random_seed(f_py: None = None, seed: int = 0) -> Callable[[C], C]: ...


def random_seed(f_py: Optional[C] = None, seed: int = 0) -> Callable[[C], C] | C:
    """
    Decorates function ``f`` with TensorFlow, numpy and Python randomness seeds fixed to ``seed``.
    This decorator can be used without and with the ``seed`` parameter. When used with the default
    seed::

        @random_seed
        def foo():
            pass

    or::

        @random_seed()
        def foo():
            pass

    However, if ``seed`` needs to be set to a custom value parameter needs to be named::

        @random_seed(seed=1)
        def foo():
            pass

    :param f_py: A function to be decorated, used when ``seed`` parameter is not set.
    :param seed: A seed to be fixed, defaults to 0.
    """
    assert callable(f_py) or f_py is None

    def _decorator(f: C) -> C:
        """
        :param f: A function.
        :return: The function ``f``, but with TensorFlow, numpy and Python randomness seeds fixed.
        """

        @functools.wraps(f)
        def decorated(*args: Any, **kwargs: Any) -> Any:
            os.environ["PYTHONHASHSEED"] = str(seed)
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            return f(*args, **kwargs)

        return cast(C, decorated)

    if f_py is None:
        return _decorator
    else:
        return _decorator(f_py)


T = TypeVar("T")
""" Unbound type variable. """

SequenceN = Union[
    Sequence[T],
    Sequence[Sequence[T]],
    Sequence[Sequence[Sequence[T]]],
    Sequence[Sequence[Sequence[Sequence[Any]]]],
]
""" Type alias for a nested sequence (e.g. list or tuple) with array shape. """


def mk_dataset(
    query_points: SequenceN[Sequence[float]], observations: SequenceN[Sequence[float]]
) -> Dataset:
    """
    :param query_points: The query points.
    :param observations: The observations.
    :return: A :class:`Dataset` containing the specified ``query_points`` and ``observations`` with
        dtype `tf.float64`.
    """
    return Dataset(
        tf.constant(query_points, dtype=tf.float64), tf.constant(observations, dtype=tf.float64)
    )


def empty_dataset(
    query_point_shape: ShapeLike, observation_shape: ShapeLike, dtype: tf.DType = tf.float64
) -> Dataset:
    """
    :param query_point_shape: The shape of a *single* query point.
    :param observation_shape: The shape of a *single* observation.
    :param dtype: The dtype.
    :return: An empty dataset with points of the specified shapes, and dtype `tf.float64`.
    """
    qp = tf.zeros(tf.TensorShape([0]) + query_point_shape, dtype)
    obs = tf.zeros(tf.TensorShape([0]) + observation_shape, dtype)
    return Dataset(qp, obs)


def raise_exc(*_: object, **__: object) -> NoReturn:
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

    return tf.reduce_sum(x**2, axis=-1, keepdims=True)


class FixedAcquisitionRule(AcquisitionRule[TensorType, SearchSpace, ProbabilisticModel]):
    """An acquisition rule that returns the same fixed value on every step."""

    def __init__(self, query_points: SequenceN[Sequence[float]]):
        """
        :param query_points: The value to return on each step. Will be converted to a tensor with
            dtype `tf.float64`.
        """
        self._qp = tf.constant(query_points, dtype=tf.float64)

    def __repr__(self) -> str:
        return f"FixedAcquisitionRule({self._qp!r})"

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[Tag, ProbabilisticModel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        """
        :param search_space: Unused.
        :param models: Unused.
        :param datasets: Unused.
        :return: The fixed value specified on initialisation.
        """
        return self._qp


class FixedLocalAcquisitionRule(
    FixedAcquisitionRule,
    LocalDatasetsAcquisitionRule[State[Optional[int], TensorType], SearchSpace, ProbabilisticModel],
):
    """A local dataset acquisition rule that returns the same fixed value on every step and
    keeps track of a counter internal State."""

    def __init__(self, query_points: TensorType, num_local_datasets: int) -> None:
        super().__init__(query_points)
        self._num_local_datasets = num_local_datasets
        self._initialize_subspaces_calls = 0

    @property
    def num_local_datasets(self) -> int:
        return self._num_local_datasets

    def initialize_subspaces(self, search_space: SearchSpaceType) -> None:
        self._initialize_subspaces_calls += 1

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[Tag, ProbabilisticModel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        def state_func(state: int | None) -> tuple[int | None, TensorType]:
            new_state = 1 if state is None else state + 1
            return new_state, self._qp

        return state_func

    def filter_datasets(
        self, models: Mapping[Tag, ProbabilisticModel], datasets: Mapping[Tag, Dataset]
    ) -> Mapping[Tag, Dataset] | State[int | None, Mapping[Tag, Dataset]]:
        def state_func(state: int | None) -> Tuple[int | None, Mapping[Tag, Dataset]]:
            new_state = 1 if state is None else state + 1
            return new_state, datasets

        return state_func


ShapeLike = Union[tf.TensorShape, Sequence[int]]
""" Type alias for types that can represent tensor shapes. """


def various_shapes(*, excluding_ranks: Container[int] = ()) -> frozenset[tuple[int, ...]]:
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


def hartmann_6_dataset(num_query_points: int) -> Dataset:
    """
    Generate example dataset based on Hartmann 6 objective function.
    :param num_query_points: A number of samples from the objective function.
    :return: A dataset.
    """
    search_space = Hartmann6.search_space
    query_points = search_space.sample(num_query_points)

    observer = mk_observer(Hartmann6.objective)
    data = observer(query_points)

    return data


def branin_dataset(num_query_points: int) -> Dataset:
    """
    Generate example dataset based on Hartmann 6 objective function.
    :param num_query_points: A number of samples from the objective function.
    :return: A dataset.
    """
    search_space = Branin.search_space
    query_points = search_space.sample(num_query_points)

    observer = mk_observer(Branin.objective)
    data = observer(query_points)

    return data

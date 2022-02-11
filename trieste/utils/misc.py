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

from abc import ABC, abstractmethod
from time import perf_counter
from types import TracebackType
from typing import Any, Callable, Generic, Mapping, NoReturn, Optional, Tuple, Type, TypeVar

import numpy as np
import tensorflow as tf
from typing_extensions import Final, final

from ..types import TensorType

C = TypeVar("C", bound=Callable[..., object])
""" A type variable bound to `typing.Callable`. """


def jit(apply: bool = True, **optimize_kwargs: Any) -> Callable[[C], C]:
    """
    A decorator that conditionally wraps a function with `tf.function`.

    :param apply: If `True`, the decorator is equivalent to `tf.function`. If `False`, the decorator
        does nothing.
    :param optimize_kwargs: Additional arguments to `tf.function`.
    :return: The decorator.
    """

    def decorator(func: C) -> C:
        return tf.function(func, **optimize_kwargs) if apply else func

    return decorator


def shapes_equal(this: TensorType, that: TensorType) -> TensorType:
    """
    Return a scalar tensor containing: `True` if ``this`` and ``that`` have equal runtime shapes,
    else `False`.
    """
    return tf.rank(this) == tf.rank(that) and tf.reduce_all(tf.shape(this) == tf.shape(that))


def to_numpy(t: TensorType) -> np.ndarray:
    """
    :param t: An array-like object.
    :return: ``t`` as a NumPy array.
    """
    if isinstance(t, tf.Tensor):
        return t.numpy()

    return t


ResultType = TypeVar("ResultType", covariant=True)
""" An unbounded covariant type variable. """


class Result(Generic[ResultType], ABC):
    """
    Represents the result of an operation that can fail with an exception. It contains either the
    operation return value (in an :class:`Ok`), or the exception raised (in an :class:`Err`).

    To check whether instances such as

        >>> res = Ok(1)
        >>> other_res = Err(ValueError("whoops"))

    contain a value, use :attr:`is_ok` (or :attr:`is_err`)

        >>> res.is_ok
        True
        >>> other_res.is_ok
        False

    We can access the value if it :attr:`is_ok` using :meth:`unwrap`.

        >>> res.unwrap()
        1

    Trying to access the value of a failed :class:`Result`, or :class:`Err`, will raise the wrapped
    exception

        >>> other_res.unwrap()
        Traceback (most recent call last):
            ...
        ValueError: whoops

    **Note:** This class is not intended to be subclassed other than by :class:`Ok` and
    :class:`Err`.
    """

    @property
    @abstractmethod
    def is_ok(self) -> bool:
        """`True` if this :class:`Result` contains a value, else `False`."""

    @property
    def is_err(self) -> bool:
        """
        `True` if this :class:`Result` contains an error, else `False`. The opposite of
        :attr:`is_ok`.
        """
        return not self.is_ok

    @abstractmethod
    def unwrap(self) -> ResultType:
        """
        :return: The contained value, if it exists.
        :raise Exception: If there is no contained value.
        """


@final
class Ok(Result[ResultType]):
    """Wraps the result of a successful evaluation."""

    def __init__(self, value: ResultType):
        """
        :param value: The result of a successful evaluation.
        """
        self._value = value

    def __repr__(self) -> str:
        """"""
        return f"Ok({self._value!r})"

    @property
    def is_ok(self) -> bool:
        """`True` always."""
        return True

    def unwrap(self) -> ResultType:
        """
        :return: The wrapped value.
        """
        return self._value


@final
class Err(Result[NoReturn]):
    """Wraps the exception that occurred during a failed evaluation."""

    def __init__(self, exc: Exception):
        """
        :param exc: The exception that occurred.
        """
        self._exc = exc

    def __repr__(self) -> str:
        """"""
        return f"Err({self._exc!r})"

    @property
    def is_ok(self) -> bool:
        """`False` always."""
        return False

    def unwrap(self) -> NoReturn:
        """
        :raise Exception: Always. Raises the wrapped exception.
        """
        raise self._exc


class DEFAULTS:
    """Default constants used in Trieste."""

    JITTER: Final[float] = 1e-6
    """
    The default jitter, typically used to stabilise computations near singular points, such as in
    Cholesky decomposition.
    """


K = TypeVar("K")
""" An unbound type variable. """

U = TypeVar("U")
""" An unbound type variable. """

V = TypeVar("V")
""" An unbound type variable. """


def map_values(f: Callable[[U], V], mapping: Mapping[K, U]) -> Mapping[K, V]:
    """
    Apply ``f`` to each value in ``mapping`` and return the result. If ``f`` does not modify its
    argument, :func:`map_values` does not modify ``mapping``. For example:

    >>> import math
    >>> squares = {'a': 1, 'b': 4, 'c': 9}
    >>> map_values(math.sqrt, squares)['b']
    2.0
    >>> squares
    {'a': 1, 'b': 4, 'c': 9}

    :param f: The function to apply to the values in ``mapping``.
    :param mapping: A mapping.
    :return: A new mapping, whose keys are the same as ``mapping``, and values are the result of
        applying ``f`` to each value in ``mapping``.
    """
    return {k: f(u) for k, u in mapping.items()}


class Timer:
    """
    Functionality for timing chunks of code. For example:
    >>> from time import sleep
    >>> with Timer() as timer: sleep(2.0)
    >>> timer.time  # doctest: +SKIP
    2.0
    """

    def __enter__(self) -> Timer:
        self.start = perf_counter()
        return self

    def __exit__(
        self,
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.end = perf_counter()
        self.time = self.end - self.start


def flatten_leading_dims(x: TensorType) -> Tuple[TensorType, Callable[[TensorType], TensorType]]:
    """
    Flattens the leading dimensions of `x` (all but the last two dimensions), and returns a
    function that can be used to restore them (typically after first manipulating the
    flattened tensor).
    """
    x_batched_shape = tf.shape(x)
    batch_shape = x_batched_shape[:-1]
    input_shape = x_batched_shape[-1:]
    x_flat_shape = tf.concat([[-1], input_shape], axis=0)

    def unflatten(y: TensorType) -> TensorType:
        tf.debugging.assert_rank(y, 2, message="unflatten is expecting a rank two tensor.")
        y_flat_shape = tf.shape(y)
        output_shape = y_flat_shape[1:]
        y_batched_shape = tf.concat([batch_shape, output_shape], axis=0)
        y_batched = tf.reshape(y, y_batched_shape)
        tf.debugging.assert_shapes([(y, ["N", "D"]), (y_batched, [..., "M", "D"])])
        return y_batched

    return tf.reshape(x, x_flat_shape), unflatten

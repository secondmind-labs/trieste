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
from typing import Any, Callable, Mapping, Optional, Tuple, TypeVar, overload

import numpy as np
import tensorflow as tf

from ..type import TensorType

C = TypeVar("C", bound=Callable)
""" A type variable bound to `typing.Callable`. """


def jit(apply: bool = True, **optimize_kwargs) -> Callable[[C], C]:
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


def shapes_equal(this: tf.Tensor, that: tf.Tensor) -> tf.Tensor:
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


K = TypeVar("K")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


@overload
def zip_with(u: Mapping[K, U], v: Mapping[K, V]) -> Mapping[K, Tuple[U, V]]:
    ...


@overload
def zip_with(u: Mapping[K, U], v: Mapping[K, V], f: Callable[[U, V], W]) -> Mapping[K, W]:
    ...


def zip_with(
    u: Mapping[K, U], v: Mapping[K, V], f: Callable[[U, V], Any] = lambda u, v: (u, v)
) -> Mapping[K, Any]:
    """
    Zip the mappings ``u`` and ``v``, combining the values according to the function ``f``. For
    example:

    >>> zip_with({'a': 1, 'b': 2}, {'a': 3, 'b': 4})
    {'a': (1, 3), 'b': (2, 4)}
    >>> zip_with({'a': 1, 'b': 2}, {'a': 3, 'b': 4}, lambda a, b: a + b)
    {'a': 4, 'b': 6}

    :param u: A mapping.
    :param v: A mapping with the same keys as ``u``.
    :param f: A function taking one value from each of ``u`` and ``v``.
    :return: A mapping with the same keys as ``u`` and ``v``. For each key, the value is found by
        applying ``f`` to the corresponding values in ``u`` and ``v``.
    """
    if u.keys() != v.keys():
        raise ValueError(f"u and v must have the same keys, got {u.keys()} and {v.keys()}")

    return {k: f(u[k], v[k]) for k in u}

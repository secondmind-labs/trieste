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
from typing import Callable

import numpy as np
import tensorflow as tf

from ..type import TensorType


def jit(apply: bool = True, **optimize_kwargs) -> Callable[[Callable], Callable]:
    """
    A decorator that conditionally wraps a function with `tf.function`.

    :param apply: If `True`, the decorator is equivalent to `tf.function`. If `False`, the decorator
        does nothing.
    :param optimize_kwargs: Additional arguments to `tf.function`.
    :return: The decorator.
    """

    def decorator(func: Callable) -> Callable:
        if apply:
            return tf.function(func, **optimize_kwargs)
        else:
            return func

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

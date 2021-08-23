# Copyright 2021 The Trieste Contributors
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

import copy
from typing import TypeVar

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp


M = TypeVar("M", bound=tf.Module)
""" A type variable bound to :class:`tf.Module`. """


def module_deepcopy(self: M, memo: dict[int, object]) -> M:
    r"""
    This function provides a workaround for `a bug`_ in TensorFlow Probability (fixed in `version
    0.12`_) where a :class:`tf.Module` cannot be deep-copied if it has
    :class:`tfp.bijectors.Bijector` instances on it. The function can be used to directly copy an
    object ``self`` as e.g. ``module_deepcopy(self, {})``, but it is perhaps more useful as an
    implemention for :meth:`__deepcopy__` on classes, where it can be used as follows:

    .. _a bug: https://github.com/tensorflow/probability/issues/547
    .. _version 0.12: https://github.com/tensorflow/probability/releases/tag/v0.12.1

    .. testsetup:: *

        >>> import tensorflow_probability as tfp

    >>> class Foo(tf.Module):
    ...     example_bijector = tfp.bijectors.Exp()
    ...
    ...     __deepcopy__ = module_deepcopy

    Classes with this method can be deep-copied even if they contain
    :class:`tfp.bijectors.Bijector`\ s.

    :param self: The object to copy.
    :param memo: References to existing deep-copied objects (by object :func:`id`).
    :return: A deep-copy of ``self``.
    """
    gpflow.utilities.reset_cache_bijectors(self)

    new = self.__new__(type(self))
    memo[id(self)] = new

    for name, value in self.__dict__.items():
        setattr(new, name, copy.deepcopy(value, memo))

    return new

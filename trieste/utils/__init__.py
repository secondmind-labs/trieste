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
""" This package contains library utilities. """
from typing import Callable, Mapping, TypeVar

from typing_extensions import Final

from . import pareto
from .misc import DEFAULTS, Err, Ok, Result, T_co, jit, shapes_equal, to_numpy

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

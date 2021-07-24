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

"""
This module contains functions convenient for creating :class:`Observer` objects that return data
from objective functions, appropriately formatted for usage with the toolbox.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, cast, overload

from ..data import Dataset
from ..observer import MultiObserver, Observer, SingleObserver
from ..type import TensorType


@overload
def mk_observer(objective: Callable[[TensorType], TensorType]) -> SingleObserver:
    ...


@overload
def mk_observer(objective: Callable[[TensorType], TensorType], key: str) -> MultiObserver:
    ...


def mk_observer(
    objective: Callable[[TensorType], TensorType], key: Optional[str] = None
) -> Observer:
    """
    :param objective: An objective function designed to be used with a single data set and model.
    :param key: An optional key to use to access the data from the observer result.
    :return: An observer returning the data from ``objective``.
    """
    if key is not None:
        return lambda qp: {cast(str, key): Dataset(qp, objective(qp))}
    else:
        return lambda qp: Dataset(qp, objective(qp))

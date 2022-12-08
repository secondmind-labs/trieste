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
"""This module contains type aliases."""
from typing import Callable, Hashable, Tuple, TypeVar, Union

import tensorflow as tf

TensorType = Union[tf.Tensor, tf.Variable]
"""Type alias for tensor-like types."""

S = TypeVar("S")
"""Unbound type variable."""

T = TypeVar("T")
"""Unbound type variable."""

State = Callable[[S], Tuple[S, T]]
"""
A `State` produces a value of type `T`, given a state of type `S`, and in doing so can update the
state. If the state is updated, it is not updated in-place. Instead, a new state is created. This
is a referentially transparent alternative to mutable state.
"""

Tag = Hashable
"""Type alias for a tag used to label datasets and models."""

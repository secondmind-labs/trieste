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
from typing import Callable, Mapping

import pytest

from trieste.utils import K, U, V, map_values


@pytest.mark.parametrize(
    "f, mapping, expected",
    [
        (abs, {}, {}),
        (abs, {1: -1, -2: 2}, {1: 1, -2: 2}),
        (len, {"a": [1, 2, 3], "b": [4, 5]}, {"a": 3, "b": 2}),
    ],
)
def test_map_values(f: Callable[[U], V], mapping: Mapping[K, U], expected: Mapping[K, V]) -> None:
    assert map_values(f, mapping) == expected

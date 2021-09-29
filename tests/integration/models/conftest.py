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

from typing import Any, Callable

import pytest

from tests.util.trieste.utils.objectives import hartmann_6_dataset
from trieste.data import Dataset


@pytest.fixture(name="depth", params=[2, 3])
def _depth_fixture(request: Any) -> int:
    return request.param


@pytest.fixture(name="hartmann_6_dataset_function", scope="session")
def _hartmann_6_dataset_function_fixture() -> Callable[[int], Dataset]:
    return hartmann_6_dataset

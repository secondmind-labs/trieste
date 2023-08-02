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

from typing import Iterable

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from check_shapes import (
    DocstringFormat,
    ShapeCheckingState,
    get_enable_check_shapes,
    get_enable_function_call_precompute,
    get_rewrite_docstrings,
    set_enable_check_shapes,
    set_enable_function_call_precompute,
    set_rewrite_docstrings,
)


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store",
        default="no",
        choices=("yes", "no", "only"),
        help="whether to run slow tests",
    )
    parser.addoption(
        "--qhsri",
        action="store",
        default="no",
        choices=("yes", "no", "only"),
        help="whether to run qhsri tests",
    )
    parser.addoption(
        "--eager",
        action="store_true",
        default=False,
        help="whether to run all functions eagerly",
    )


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "qhsri: mark test as requiring qhsri dependencies")


def pytest_collection_modifyitems(config: Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow") == "no":
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    elif config.getoption("--runslow") == "only":
        skip_fast = pytest.mark.skip(reason="--skipfast option set")
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(skip_fast)

    if config.getoption("--qhsri") == "no":
        skip_qhsri = pytest.mark.skip(reason="need --qhsri option to run")
        for item in items:
            if "qhsri" in item.keywords:
                item.add_marker(skip_qhsri)
    if config.getoption("--qhsri") == "only":
        skip_non_qhsri = pytest.mark.skip(reason="--qhsri only option set")
        for item in items:
            if "qhsri" not in item.keywords:
                item.add_marker(skip_non_qhsri)

    if config.getoption("--eager"):
        import tensorflow as tf

        tf.config.experimental_run_functions_eagerly(True)


@pytest.fixture(autouse=True)
def enable_shape_checks() -> Iterable[None]:
    # ensure `check_shapes` is always enabled for tests
    old_enable = get_enable_check_shapes()
    old_rewrite_docstrings = get_rewrite_docstrings()
    old_function_call_precompute = get_enable_function_call_precompute()
    set_enable_check_shapes(ShapeCheckingState.ENABLED)
    set_rewrite_docstrings(DocstringFormat.SPHINX)
    set_enable_function_call_precompute(True)
    yield
    set_enable_function_call_precompute(old_function_call_precompute)
    set_rewrite_docstrings(old_rewrite_docstrings)
    set_enable_check_shapes(old_enable)

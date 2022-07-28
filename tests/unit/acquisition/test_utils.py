# Copyright 2022 The Trieste Contributors
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

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from trieste.acquisition import AcquisitionFunction
from trieste.acquisition.utils import select_nth_output, split_acquisition_function


@pytest.mark.parametrize(
    "f",
    [
        lambda x: x ** 2,
        lambda x: tf.cast(x, tf.float64),
    ],
)
@pytest.mark.parametrize(
    "x, split_size, expected_batches",
    [
        (np.zeros((0,)), 2, 1),
        (np.array([1]), 2, 1),
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), 2, 6),
        (np.array([1, 2, 3, 4]), 3, 2),
        (np.array([1, 2, 3, 4]), 4, 1),
        (np.array([1, 2, 3, 4]), 1, 4),
        (np.array([1, 2, 3, 4]), 10, 1),
    ],
)
def test_split_acquisition_function(
    f: AcquisitionFunction, x: "np.ndarray[Any, Any]", split_size: int, expected_batches: int
) -> None:

    mock_f = MagicMock()
    mock_f.side_effect = f
    batch_f = split_acquisition_function(mock_f, split_size=split_size)
    np.testing.assert_allclose(f(x), batch_f(x))
    assert expected_batches == mock_f.call_count


@pytest.mark.parametrize("split_size", [0, -1])
def test_split_acquisition_function__invalid_split_size(split_size: int) -> None:
    with pytest.raises(ValueError):
        split_acquisition_function(MagicMock(), split_size=split_size)


def test_select_nth_output() -> None:
    a = tf.random.normal([5, 6])

    assert np.all(select_nth_output(a) == a[..., 0])
    assert np.all(select_nth_output(a, 3) == a[..., 3])

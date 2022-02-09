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
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from trieste.acquisition import AcquisitionFunction
from trieste.acquisition.batch import batchify_acquisition_function


@pytest.mark.parametrize(
    "f",
    [
        lambda x: x ** 2,
        lambda x: tf.cast(x, tf.float64),
    ],
)
@pytest.mark.parametrize(
    "x, batch_size, expected_batches",
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
def test_batchify_acquisition_function(
    f: AcquisitionFunction, x: np.ndarray, batch_size: int, expected_batches: int
) -> None:

    mock_f = MagicMock()
    mock_f.side_effect = f
    batch_f = batchify_acquisition_function(mock_f, batch_size=batch_size)
    np.testing.assert_allclose(f(x), batch_f(x))
    assert expected_batches == mock_f.call_count


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batchify_acquisition_function__invalid_batch_size(batch_size: int) -> None:
    with pytest.raises(AssertionError):
        batchify_acquisition_function(MagicMock(), batch_size=batch_size)

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

from collections.abc import Iterable
from typing import Any

import pytest
import tensorflow as tf

from trieste.data import Dataset
from trieste.models.optimizer import DatasetTransformer
from trieste.types import TensorType


def _batcher_1(dataset: Dataset, batch_size: int) -> Iterable[tuple[TensorType, TensorType]]:
    ds = tf.data.Dataset.from_tensor_slices(dataset.astuple())
    ds = ds.shuffle(100)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    return iter(ds)


def _batcher_2(dataset: Dataset, batch_size: int) -> tuple[TensorType, TensorType]:
    return dataset.astuple()


@pytest.fixture(name="batcher", params=[_batcher_1, _batcher_2])
def _batcher_fixture(request: Any) -> DatasetTransformer:
    return request.param


@pytest.fixture(name="compile", params=[True, False])
def _compile_fixture(request: Any) -> bool:
    return request.param

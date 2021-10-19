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

import pytest

import gpflow

from trieste.models.optimizer import (
    BatchOptimizer,
    Optimizer,
    create_optimizer,
)


def test_create_optimizer_scipy_produces_correct_optimizer() -> None:
    optim = create_optimizer(gpflow.optimizers.Scipy(), {})
    assert isinstance(optim, Optimizer) and not isinstance(optim, BatchOptimizer)

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

from collections.abc import Callable
from typing import Any

import pytest
from gpflow.models import GPModel

from tests.util.models.gpflow.models import (
    ModelFactoryType,
    gpr_model,
    sgpr_model,
    svgp_model,
    vgp_model,
)
from trieste.models.gpflow import (
    GaussianProcessRegression,
    GPflowPredictor,
    SparseVariational,
    VariationalGaussianProcess,
)
from trieste.models.optimizer import Optimizer
from trieste.types import TensorType


@pytest.fixture(
    name="gpr_interface_factory",
    params=[
        (GaussianProcessRegression, gpr_model),
        (GaussianProcessRegression, sgpr_model),
        (VariationalGaussianProcess, vgp_model),
        (SparseVariational, svgp_model),
    ],
)
def _gpr_interface_factory(request: Any) -> ModelFactoryType:
    def model_interface_factory(
        x: TensorType, y: TensorType, optimizer: Optimizer | None = None
    ) -> tuple[GPflowPredictor, Callable[[TensorType, TensorType], GPModel]]:
        model_interface: type[GaussianProcessRegression] = request.param[0]
        base_model: GaussianProcessRegression = request.param[1](x, y)
        reference_model: Callable[[TensorType, TensorType], GPModel] = request.param[1]
        return model_interface(base_model, optimizer=optimizer), reference_model

    return model_interface_factory


@pytest.fixture(name="dim", params=[1, 10])
def _dim_fixture(request: Any) -> int:
    return request.param

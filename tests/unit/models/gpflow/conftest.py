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

from trieste.models.gpflow import (
    GaussianProcessRegression,
    VariationalGaussianProcess,
)
from tests.util.models.gpflow.models import (
    gpr_model,
    sgpr_model,
    vgp_model,
)

@pytest.fixture(
    name="gpr_interface_factory",
    params=[
        (GaussianProcessRegression, gpr_model),
        (GaussianProcessRegression, sgpr_model),
        (VariationalGaussianProcess, vgp_model),
    ],
)
def _gpr_interface_factory(
    request,
) -> Callable[[TensorType, TensorType, Optimizer | None], GaussianProcessRegression]:
    def interface_factory(
        x: TensorType, y: TensorType, optimizer: Optimizer | None = None
    ) -> GaussianProcessRegression:
        interface: type[GaussianProcessRegression] = request.param[0]
        base_model: GaussianProcessRegression = request.param[1](x, y)
        return interface(base_model, optimizer=optimizer)  # type: ignore

    return interface_factory


@pytest.fixture(name="dim", params=[1, 10])
def _dim_fixture(request):
    return request.param

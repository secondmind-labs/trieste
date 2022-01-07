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

"""
In this module, we test that we are wrapping GPflux architectures correctly, leading to the same
model.
"""

from __future__ import annotations

import gpflow
import pytest
from gpflow.models import GPR, SGPR, SVGP, VGP

from tests.util.misc import mk_dataset
from tests.util.models.gpflow.models import mock_data
from trieste.models.gpflow.architectures import build_gpr
from trieste.space import Box


@pytest.mark.parametrize("noise_free", [True, False])
@pytest.mark.parametrize("kernel_priors", [True, False])
def test_build_gpr_returns_correct_model(noise_free: bool, kernel_priors: bool) -> None:
    qp, obs = mock_data()
    data = mk_dataset(qp, obs)
    search_space = Box([0.0], [1.0]) ** qp.shape[-1]

    model = build_gpr(data, search_space, kernel_priors, noise_free)

    assert isinstance(model, GPR)
    assert model.data == (qp, obs)
    assert model.likelihood.variance.trainable == (not noise_free)
    assert isinstance(model.kernel, gpflow.kernels.Matern52)
    if kernel_priors:
        assert model.kernel.variance.prior is not None
        assert model.kernel.lengthscales.prior is not None
    else:
        assert model.kernel.variance.prior is None
        assert model.kernel.lengthscales.prior is None

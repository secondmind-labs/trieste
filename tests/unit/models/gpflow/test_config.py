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
from typing import Any, Dict

import gpflow
import numpy as np
import pytest
import tensorflow as tf
from gpflow.models import GPMC, GPR, SGPR, SVGP, VGP

from tests.util.models.gpflow.models import fnc_3x_plus_10, gpr_model
from trieste.models import TrainableProbabilisticModel
from trieste.models.gpflow import (
    GaussianProcessRegression,
    GPflowModelConfig,
    SparseVariational,
    VariationalGaussianProcess,
)
from trieste.models.optimizer import Optimizer


def test_gpflow_model_config_raises_not_supported_model_type() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    y = fnc_3x_plus_10(x)
    model_specs = {"model": GPMC((x, y), gpflow.kernels.Matern32(), gpflow.likelihoods.Gaussian())}

    with pytest.raises(NotImplementedError):
        GPflowModelConfig(**model_specs)


def test_gpflow_model_config_has_correct_supported_models() -> None:

    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model_specs = {"model": gpr_model(x, fnc_3x_plus_10(x))}
    model_config = GPflowModelConfig(**model_specs)

    models_mapping: Dict[Any, Callable[[Any, Optimizer], TrainableProbabilisticModel]] = {
        GPR: GaussianProcessRegression,
        SGPR: GaussianProcessRegression,
        VGP: VariationalGaussianProcess,
        SVGP: SparseVariational,
    }

    assert model_config.supported_models() == models_mapping


def test_gpflow_model_config_has_correct_default_optimizer() -> None:

    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model_specs = {"model": gpr_model(x, fnc_3x_plus_10(x))}
    model_config = GPflowModelConfig(**model_specs)

    default_optimizer = gpflow.optimizers.Scipy

    assert isinstance(model_config.optimizer, default_optimizer)


def test_gpflow_model_config_allows_changing_default_optimizer() -> None:

    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model_specs = {
        "model": gpr_model(x, fnc_3x_plus_10(x)),
        "optimizer": tf.optimizers.Adam(),
    }
    model_config = GPflowModelConfig(**model_specs)

    expected_optimizer = tf.optimizers.Adam

    assert isinstance(model_config.optimizer, expected_optimizer)

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

from tests.util.models.gpflow.models import GPRcopy, gpr_copy_model
from trieste.models import ModelConfig, ModelRegistry, create_model
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.optimizer import Optimizer


def test_model_config_raises_not_supported_model_type() -> None:

    model = gpr_copy_model()
    with pytest.raises(NotImplementedError):
        ModelConfig(model)


def test_model_registry_raises_on_unsupported_model() -> None:

    model = gpr_copy_model()

    with pytest.raises(ValueError):
        ModelRegistry.get_interface(model)

    with pytest.raises(ValueError):
        ModelRegistry.get_optimizer(model)


def test_model_registry_register_model() -> None:

    ModelRegistry.register_model(GPRcopy, GaussianProcessRegression, Optimizer)
    model_type = type(gpr_copy_model())

    assert ModelRegistry.get_interface(model_type) == GaussianProcessRegression
    assert ModelRegistry.get_optimizer(model_type) == Optimizer


def test_model_registry_register_model_warning() -> None:

    with pytest.warns(UserWarning) as record:
        ModelRegistry.register_model(GPRcopy, GaussianProcessRegression, Optimizer)

    assert len(record) == 1
    assert "you have now overwritten it" in record[0].message.args[0]


def test_model_config_builds_model_correctly() -> None:

    model = gpr_copy_model()

    assert isinstance(ModelConfig(model).build_model(), GaussianProcessRegression)


def test_create_model_builds_model_correctly() -> None:

    model = gpr_copy_model()

    assert isinstance(create_model(ModelConfig(model)), GaussianProcessRegression)
    assert isinstance(create_model({"model": model}), GaussianProcessRegression)
    assert isinstance(create_model(GaussianProcessRegression(model)), GaussianProcessRegression)

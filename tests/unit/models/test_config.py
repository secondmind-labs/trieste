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

from typing import Any, Tuple, Type

import gpflow
import pytest
import tensorflow as tf
from gpflow.models import GPR, SVGP

from tests.util.models.gpflow.models import gpr_model, mock_data
from trieste.models import ModelConfig, ModelRegistry, create_model
from trieste.models.gpflow import GaussianProcessRegression, SparseVariational
from trieste.models.optimizer import BatchOptimizer, Optimizer


class GPRcopy(GPR):
    """A copy of the GPR model."""


class SVGPcopy(SVGP):
    """A copy of the SVGP model."""


def gpr_copy_model() -> GPRcopy:
    return GPRcopy(mock_data(), gpflow.kernels.Matern32())


class Scipy_copy(gpflow.optimizers.Scipy):
    """A copy of the GPR model."""


class Adam_copy(tf.optimizers.Adam):
    """A copy of the SVGP model."""


def Scipy_copy_optimizer() -> Scipy_copy:
    return Scipy_copy()


def test_model_config_raises_not_supported_model_type() -> None:

    model = gpr_copy_model()
    with pytest.raises(NotImplementedError):
        ModelConfig(model)


def test_model_registry_raises_on_unsupported_model() -> None:

    model = gpr_copy_model()

    with pytest.raises(ValueError):
        ModelRegistry.get_model_wrapper(model)


# def test_model_registry_raises_on_unsupported_optimizer() -> None:

#     optimizer = gpr_copy_model()

#     with pytest.raises(ValueError):
#         ModelRegistry.get_optimizer_wrapper(optimizer)


def test_model_registry_register_model() -> None:

    ModelRegistry.register_model(GPRcopy, GaussianProcessRegression)
    model_type = type(gpr_copy_model())

    assert ModelRegistry.get_model_wrapper(model_type) == GaussianProcessRegression


def test_model_registry_register_model_warning() -> None:

    ModelRegistry.register_model(SVGPcopy, SparseVariational)

    with pytest.warns(UserWarning) as record:
        ModelRegistry.register_model(SVGPcopy, SparseVariational)

    assert len(record) == 1
    assert "you have now overwritten it" in record[0].message.args[0]


def test_model_registry_register_optimizer() -> None:

    ModelRegistry.register_optimizer(Scipy_copy, Optimizer)
    optimizer_type = type(Scipy_copy_optimizer())

    assert ModelRegistry.get_optimizer_wrapper(optimizer_type) == Optimizer


def test_model_registry_register_optimizer_warning() -> None:

    ModelRegistry.register_optimizer(Adam_copy, Optimizer)

    with pytest.warns(UserWarning) as record:
        ModelRegistry.register_optimizer(Adam_copy, Optimizer)

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


@pytest.mark.parametrize(
    "supported_optimizers",
    [(type(gpflow.optimizers.Scipy()), Optimizer)]
    + [
        (optimizer_type, BatchOptimizer)
        for optimizer_type in tf.optimizers.Optimizer.__subclasses__()
    ],
)
def test_supported_optimizers_are_correctly_registered(
    supported_optimizers: Tuple[Type[Any], Type[Optimizer]]
) -> None:

    optimizer_type, optimizer_wrapper = supported_optimizers

    assert optimizer_type in ModelRegistry.get_registered_optimizers()
    assert ModelRegistry.get_optimizer_wrapper(optimizer_type) == optimizer_wrapper


def test_config_uses_correct_optimizer_wrappers() -> None:
    data = mock_data()

    model_config = {"model": gpr_model(*data), "optimizer": gpflow.optimizers.Scipy()}
    model = create_model(model_config)
    assert not isinstance(model.optimizer, BatchOptimizer)  # type: ignore

    model_config = {"model": gpr_model(*data), "optimizer": tf.optimizers.Adam()}
    model = create_model(model_config)
    assert isinstance(model.optimizer, BatchOptimizer)  # type: ignore

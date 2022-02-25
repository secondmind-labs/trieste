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
    """A copy of the scipy optimizer."""


class Adam_copy(tf.optimizers.Adam):  # type: ignore[misc]
    """A copy of the adam optimizer."""


class DummyOptimizer:
    """A dummy optimizer."""


class DummyModel:
    """A dummy model."""


def test_model_config_raises_not_supported_model_type() -> None:

    model = gpr_copy_model()
    with pytest.raises(NotImplementedError):
        ModelConfig(model)


def test_model_registry_raises_on_unsupported_model() -> None:

    with pytest.raises(ValueError):
        ModelRegistry.get_model_wrapper(DummyModel)


def test_model_registry_raises_on_unsupported_optimizer() -> None:

    with pytest.raises(ValueError):
        ModelRegistry.get_optimizer_wrapper(DummyOptimizer)


def test_model_registry_register_model() -> None:

    ModelRegistry.register_model(GPRcopy, GaussianProcessRegression)

    assert ModelRegistry.get_model_wrapper(GPRcopy) == GaussianProcessRegression


def test_model_registry_register_model_warning() -> None:

    ModelRegistry.register_model(SVGPcopy, SparseVariational)

    with pytest.warns(UserWarning) as record:
        ModelRegistry.register_model(SVGPcopy, SparseVariational)

    assert len(record) == 1
    assert isinstance(record[0].message, Warning)
    assert "you have now overwritten it" in record[0].message.args[0]


def test_model_registry_register_optimizer() -> None:

    ModelRegistry.register_optimizer(Scipy_copy, Optimizer)

    assert ModelRegistry.get_optimizer_wrapper(Scipy_copy) == Optimizer


def test_model_registry_register_optimizer_warning() -> None:

    ModelRegistry.register_optimizer(Adam_copy, Optimizer)

    with pytest.warns(UserWarning) as record:
        ModelRegistry.register_optimizer(Adam_copy, Optimizer)

    assert len(record) == 1
    assert isinstance(record[0].message, Warning)
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
    "supported_optimizer_types",
    [
        (gpflow.optimizers.Scipy, Optimizer),
        (tf.optimizers.Optimizer, BatchOptimizer),
    ],
)
def test_supported_optimizer_types_are_correctly_registered(
    supported_optimizer_types: Tuple[Type[Any], Type[Optimizer]]
) -> None:

    optimizer_type, optimizer_wrapper = supported_optimizer_types

    assert optimizer_type in ModelRegistry.get_registered_optimizers()
    assert ModelRegistry.get_optimizer_wrapper(optimizer_type) == optimizer_wrapper


@pytest.mark.parametrize(
    "supported_optimizers",
    [
        (gpflow.optimizers.Scipy(), Optimizer),
        (tf.optimizers.Adam(), BatchOptimizer),
        (tf.optimizers.RMSprop(), BatchOptimizer),
        (tf.optimizers.SGD(), BatchOptimizer),
        (tf.optimizers.Adadelta(), BatchOptimizer),
        (tf.optimizers.Adagrad(), BatchOptimizer),
        (tf.optimizers.Adamax(), BatchOptimizer),
        (tf.optimizers.Nadam(), BatchOptimizer),
        (tf.optimizers.Ftrl(), BatchOptimizer),
    ],
)
def test_supported_optimizers_are_correctly_registered(
    supported_optimizers: Tuple[Any, Type[Optimizer]]
) -> None:

    optimizer, optimizer_wrapper = supported_optimizers

    assert ModelRegistry.get_optimizer_wrapper(type(optimizer)) == optimizer_wrapper


def test_config_uses_correct_optimizer_wrappers() -> None:
    data = mock_data()

    model_config = {"model": gpr_model(*data), "optimizer": gpflow.optimizers.Scipy()}
    model = create_model(model_config)
    assert not isinstance(model.optimizer, BatchOptimizer)  # type: ignore

    model_config = {"model": gpr_model(*data), "optimizer": tf.optimizers.Adam()}
    model = create_model(model_config)
    assert isinstance(model.optimizer, BatchOptimizer)  # type: ignore

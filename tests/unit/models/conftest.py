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
from typing import Any, Callable

import pytest
import tensorflow as tf
from gpflow.models import GPModel
from gpflux.models import DeepGP

from tests.util.models.gpflow.models import (
    ModelFactoryType,
    gpr_model,
    sgpr_model,
    svgp_model,
    vgp_model,
)
from tests.util.models.gpflux.models import simple_two_layer_dgp_model, two_layer_dgp_model
from tests.util.trieste.utils.objectives import branin_dataset, hartmann_6_dataset
from trieste.data import Dataset
from trieste.models.gpflow import (
    GaussianProcessRegression,
    GPflowPredictor,
    SparseVariational,
    VariationalGaussianProcess,
)
from trieste.models.optimizer import DatasetTransformer, Optimizer
from trieste.types import TensorType


@pytest.fixture(
    name="gpflow_interface_factory",
    params=[
        (GaussianProcessRegression, gpr_model),
        (GaussianProcessRegression, sgpr_model),
        (VariationalGaussianProcess, vgp_model),
        (SparseVariational, svgp_model),
    ],
)
def _gpflow_interface_factory(request: Any) -> ModelFactoryType:
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


def _batcher_bs_100(dataset: Dataset, batch_size: int) -> Iterable[tuple[TensorType, TensorType]]:
    ds = tf.data.Dataset.from_tensor_slices(dataset.astuple())
    ds = ds.shuffle(100)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    return iter(ds)


def _batcher_full_batch(dataset: Dataset, batch_size: int) -> tuple[TensorType, TensorType]:
    return dataset.astuple()


@pytest.fixture(name="batcher", params=[_batcher_bs_100, _batcher_full_batch])
def _batcher_fixture(request: Any) -> DatasetTransformer:
    return request.param


@pytest.fixture(name="compile", params=[True, False])
def _compile_fixture(request: Any) -> bool:
    return request.param


@pytest.fixture(name="two_layer_model", params=[two_layer_dgp_model, simple_two_layer_dgp_model])
def _two_layer_model_fixture(request: Any) -> Callable[[TensorType], DeepGP]:
    return request.param


# Teardown fixture to set keras floatx to float64 then return it to previous value at test finish
# pytest uses yield in a funny way, so we use type ignore
@pytest.fixture(name="keras_float")  # type: ignore
def _keras_float() -> None:
    current_float = tf.keras.backend.floatx()
    tf.keras.backend.set_floatx("float64")
    yield
    tf.keras.backend.set_floatx(current_float)


@pytest.fixture(name="ensemble_size", params=[2, 5])
def _ensemble_size_fixture(request):
    return request.param


@pytest.fixture(name="num_hidden_layers", params=[0, 1, 3])
def _num_hidden_layers_fixture(request):
    return request.param


@pytest.fixture(name="independent_normal", params=[False, True])
def _independent_normal_fixture(request):
    return request.param


@pytest.fixture(name="bootstrap_data", params=[False, True])
def _bootstrap_data_fixture(request):
    return request.param


@pytest.fixture(name="branin_dataset_function", scope="session")
def _branin_dataset_function_fixture() -> Callable:
    return branin_dataset


@pytest.fixture(name="hartmann_6_dataset_function", scope="session")
def _hartmann_6_dataset_function_fixture() -> Callable:
    return hartmann_6_dataset


_EXAMPLE_DATASET = [
    branin_dataset(200),
    hartmann_6_dataset(200),
]


@pytest.fixture(name="example_data", params=_EXAMPLE_DATASET, scope="session")
def _example_dataset_fixture(request):
    return request.param

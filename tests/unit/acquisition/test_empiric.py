# Copyright 2020 The Trieste Contributors
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
from typing import NoReturn

import pytest

from tests.util.misc import empty_dataset, raise_exc
from tests.util.model import QuadraticMeanAndRBFKernel
from trieste.acquisition import AcquisitionFunction
from trieste.acquisition.empiric import SingleModelEmpiric
from trieste.data import Dataset
from trieste.models import ProbabilisticModel


class _ArbitrarySingleModelEmpiric(SingleModelEmpiric[NoReturn]):
    def acquire(self, dataset: Dataset, model: ProbabilisticModel) -> AcquisitionFunction:
        return raise_exc


def test_single_model_empiric_raises_immediately_for_wrong_key() -> None:
    builder = _ArbitrarySingleModelEmpiric().using("foo")

    with pytest.raises(KeyError):
        builder.acquire({"bar": empty_dataset([1], [1])}, {"bar": QuadraticMeanAndRBFKernel()})


def test_single_model_acquisition_builder_repr_includes_class_name() -> None:
    builder = _ArbitrarySingleModelEmpiric()
    assert type(builder).__name__ in repr(builder)


def test_single_model_acquisition_builder_using_passes_on_correct_dataset_and_model() -> None:
    class SME(SingleModelEmpiric[AcquisitionFunction]):
        def acquire(self, dataset: Dataset, model: ProbabilisticModel) -> AcquisitionFunction:
            assert dataset is data["foo"]
            assert model is models["foo"]
            return raise_exc

    data = {"foo": empty_dataset([1], [1]), "bar": empty_dataset([1], [1])}
    models = {"foo": QuadraticMeanAndRBFKernel(), "bar": QuadraticMeanAndRBFKernel()}
    SME().using("foo").acquire(data, models)

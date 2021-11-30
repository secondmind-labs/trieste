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
from __future__ import annotations

from typing import Optional

import pytest

from tests.util.misc import empty_dataset, raise_exc
from tests.util.models.gpflow.models import QuadraticMeanAndRBFKernel
from trieste.acquisition import (
    AugmentedExpectedImprovement,
    BatchMonteCarloExpectedImprovement,
    ExpectedConstrainedHypervolumeImprovement,
    ExpectedConstrainedImprovement,
    ExpectedHypervolumeImprovement,
    ExpectedImprovement,
    NegativeLowerConfidenceBound,
    NegativePredictiveMean,
    PredictiveVariance,
    ProbabilityOfFeasibility,
)
from trieste.acquisition.interface import (
    AcquisitionFunction,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.types import TensorType
from trieste.utils import DEFAULTS


class _ArbitrarySingleBuilder(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        return raise_exc


class _ArbitraryGreedySingleBuilder(SingleModelGreedyAcquisitionBuilder[ProbabilisticModel]):
    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        return raise_exc


def test_single_model_acquisition_builder_raises_immediately_for_wrong_key() -> None:
    builder = _ArbitrarySingleBuilder().using("foo")

    with pytest.raises(KeyError):
        builder.prepare_acquisition_function(
            {"bar": QuadraticMeanAndRBFKernel()}, datasets={"bar": empty_dataset([1], [1])}
        )


def test_single_model_acquisition_builder_repr_includes_class_name() -> None:
    builder = _ArbitrarySingleBuilder()
    assert type(builder).__name__ in repr(builder)


def test_single_model_acquisition_builder_using_passes_on_correct_dataset_and_model() -> None:
    class Builder(SingleModelAcquisitionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            model: ProbabilisticModel,
            dataset: Optional[Dataset] = None,
        ) -> AcquisitionFunction:
            assert dataset is data["foo"]
            assert model is models["foo"]
            return raise_exc

    data = {"foo": empty_dataset([1], [1]), "bar": empty_dataset([1], [1])}
    models = {"foo": QuadraticMeanAndRBFKernel(), "bar": QuadraticMeanAndRBFKernel()}
    Builder().using("foo").prepare_acquisition_function(models, datasets=data)


def test_single_model_greedy_acquisition_builder_raises_immediately_for_wrong_key() -> None:
    builder = _ArbitraryGreedySingleBuilder().using("foo")

    with pytest.raises(KeyError):
        builder.prepare_acquisition_function(
            {"bar": QuadraticMeanAndRBFKernel()}, {"bar": empty_dataset([1], [1])}, None
        )


def test_single_model_greedy_acquisition_builder_repr_includes_class_name() -> None:
    builder = _ArbitraryGreedySingleBuilder()
    assert type(builder).__name__ in repr(builder)


@pytest.mark.parametrize(
    "function, function_repr",
    [
        (ExpectedImprovement(), "ExpectedImprovement()"),
        (AugmentedExpectedImprovement(), "AugmentedExpectedImprovement()"),
        (NegativeLowerConfidenceBound(1.96), "NegativeLowerConfidenceBound(1.96)"),
        (NegativePredictiveMean(), "NegativePredictiveMean()"),
        (ProbabilityOfFeasibility(0.5), "ProbabilityOfFeasibility(0.5)"),
        (ExpectedHypervolumeImprovement(), "ExpectedHypervolumeImprovement()"),
        (
            BatchMonteCarloExpectedImprovement(10_000),
            f"BatchMonteCarloExpectedImprovement(10000, jitter={DEFAULTS.JITTER})",
        ),
        (PredictiveVariance(), f"PredictiveVariance(jitter={DEFAULTS.JITTER})"),
    ],
)
def test_single_model_acquisition_function_builder_reprs(
    function: SingleModelAcquisitionBuilder[ProbabilisticModel], function_repr: str
) -> None:
    assert repr(function) == function_repr
    assert repr(function.using("TAG")) == f"{function_repr} using tag 'TAG'"
    assert (
        repr(ExpectedConstrainedImprovement("TAG", function.using("TAG"), 0.0))
        == f"ExpectedConstrainedImprovement('TAG', {function_repr} using tag 'TAG', 0.0)"
    )
    assert (
        repr(ExpectedConstrainedHypervolumeImprovement("TAG", function.using("TAG"), 0.0))
        == f"ExpectedConstrainedHypervolumeImprovement('TAG', {function_repr} using tag 'TAG', 0.0)"
    )

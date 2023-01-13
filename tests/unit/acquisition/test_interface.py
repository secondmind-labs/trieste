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

from typing import Iterator, List, Mapping, Optional, Tuple, cast

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
    AcquisitionFunctionBuilder,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.models.interfaces import SupportsPredictJoint
from trieste.observer import OBJECTIVE
from trieste.types import Tag, TensorType
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

    FOO: Tag = "foo"
    BAR: Tag = "bar"
    data = {FOO: empty_dataset([1], [1]), BAR: empty_dataset([1], [1])}
    models = {FOO: QuadraticMeanAndRBFKernel(), BAR: QuadraticMeanAndRBFKernel()}
    Builder().using(FOO).prepare_acquisition_function(models, datasets=data)


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
    cast(
        List[Tuple[SingleModelAcquisitionBuilder[SupportsPredictJoint]]],
        [
            (ExpectedImprovement(), "ExpectedImprovement(None)"),
            (AugmentedExpectedImprovement(), "AugmentedExpectedImprovement()"),
            (NegativeLowerConfidenceBound(1.96), "NegativeLowerConfidenceBound(1.96)"),
            (NegativePredictiveMean(), "NegativePredictiveMean()"),
            (ProbabilityOfFeasibility(0.5), "ProbabilityOfFeasibility(0.5)"),
            (
                ExpectedHypervolumeImprovement(),
                "ExpectedHypervolumeImprovement(get_reference_point)",
            ),
            (
                BatchMonteCarloExpectedImprovement(10_000),
                f"BatchMonteCarloExpectedImprovement(10000, jitter={DEFAULTS.JITTER})",
            ),
            (PredictiveVariance(), f"PredictiveVariance(jitter={DEFAULTS.JITTER})"),
        ],
    ),
)
def test_single_model_acquisition_function_builder_reprs(
    function: SingleModelAcquisitionBuilder[SupportsPredictJoint], function_repr: str
) -> None:
    assert repr(function) == function_repr
    assert repr(function.using("TAG")) == f"{function_repr} using tag 'TAG'"
    assert (
        repr(ExpectedConstrainedImprovement("TAG", function.using("TAG"), 0.0))
        == f"ExpectedConstrainedImprovement('TAG', {function_repr} using tag 'TAG', 0.0, None)"
    )
    assert (
        repr(ExpectedConstrainedHypervolumeImprovement("TAG", function.using("TAG"), 0.0))
        == f"ExpectedConstrainedHypervolumeImprovement('TAG', "
        f"{function_repr} using tag 'TAG', 0.0, get_reference_point)"
    )


class CustomDatasets(Mapping[Tag, Dataset]):
    """Custom dataset mapping to show that we can store metadata in the datasets argument."""

    def __init__(self, datasets: Mapping[Tag, Dataset], iteration_id: int):
        self.iteration_id = iteration_id
        self._datasets = dict(datasets)

    def __getitem__(self, key: Tag) -> Dataset:
        return self._datasets[key]

    def __setitem__(self, key: Tag, value: Dataset) -> None:
        self._datasets[key] = value

    def __delitem__(self, key: Tag) -> None:
        del self._datasets[key]

    def __iter__(self) -> Iterator[Tag]:
        return iter(self._datasets)

    def __len__(self) -> int:
        return len(self._datasets)


def test_custom_dataset_mapping() -> None:
    """
    Check that the datasets argument can be an arbitrary Mapping[Tag, Dataset], not just
    a dict. In particular, check that we can store metadata there and retrieve it in the
    acquisition function.
    """

    class _CustomData(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[Tag, ProbabilisticModel],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
        ) -> AcquisitionFunction:
            assert datasets is not None
            assert len(datasets) == 1
            assert set(datasets) == {OBJECTIVE}
            assert len(datasets[OBJECTIVE]) == 0
            assert "FOO" not in datasets
            assert isinstance(datasets, CustomDatasets)
            assert datasets.iteration_id == 2
            return raise_exc

    data = CustomDatasets({OBJECTIVE: empty_dataset([1], [1])}, 2)
    models = {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    _CustomData().prepare_acquisition_function(models, data)

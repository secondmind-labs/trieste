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

from collections.abc import Mapping, Sequence
from typing import Optional

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import empty_dataset, raise_exc
from tests.util.models.gpflow.models import QuadraticMeanAndRBFKernel
from trieste.acquisition import AcquisitionFunction
from trieste.acquisition.combination import Map, Product, Reducer, Sum
from trieste.acquisition.rule import AcquisitionFunctionBuilder
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.types import Tag

# tags
TAG: Tag = ""


def test_reducer_raises_for_no_builders() -> None:
    class UseFirst(Reducer[ProbabilisticModel]):
        def _reduce(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
            return inputs[0]

    with pytest.raises(tf.errors.InvalidArgumentError):
        UseFirst()


def test_reducer__repr_builders() -> None:
    class Dummy(Reducer[ProbabilisticModel]):
        _reduce = raise_exc

    class Builder(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def __init__(self, name: str):
            self._name = name

        def __repr__(self) -> str:
            return f"Builder({self._name!r})"

        def prepare_acquisition_function(
            self,
            models: Mapping[Tag, ProbabilisticModel],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
        ) -> AcquisitionFunction:
            return raise_exc

    assert repr(Dummy(Builder("foo"))) == "Dummy(Builder('foo'))"
    assert repr(Dummy(Builder("foo"), Builder("bar"))) == "Dummy(Builder('foo'), Builder('bar'))"


class _Static(AcquisitionFunctionBuilder[ProbabilisticModel]):
    def __init__(self, f: AcquisitionFunction):
        self._f = f

    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        return self._f

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models: Mapping[Tag, ProbabilisticModel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        return lambda x: function(x) + 1


def test_reducer__reduce() -> None:
    class Mean(Reducer[ProbabilisticModel]):
        def _reduce(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
            return tf.reduce_mean(inputs, axis=0)

    mean = Mean(_Static(lambda x: -2.0 * x), _Static(lambda x: 3.0 * x))
    data, models = {TAG: empty_dataset([1], [1])}, {TAG: QuadraticMeanAndRBFKernel()}
    acq = mean.prepare_acquisition_function(models, datasets=data)
    xs = tf.random.uniform([3, 5, 1], minval=-1.0)
    npt.assert_allclose(acq(xs), 0.5 * xs)


def test_sum() -> None:
    sum_ = Sum(_Static(lambda x: x), _Static(lambda x: x**2), _Static(lambda x: x**3))
    data, models = {TAG: empty_dataset([1], [1])}, {TAG: QuadraticMeanAndRBFKernel()}
    acq = sum_.prepare_acquisition_function(models, datasets=data)
    xs = tf.random.uniform([3, 5, 1], minval=-1.0)
    npt.assert_allclose(acq(xs), xs + xs**2 + xs**3)


def test_product() -> None:
    prod = Product(_Static(lambda x: x + 1), _Static(lambda x: x + 2))
    data, models = {TAG: empty_dataset([1], [1])}, {TAG: QuadraticMeanAndRBFKernel()}
    acq = prod.prepare_acquisition_function(models, datasets=data)
    xs = tf.random.uniform([3, 5, 1], minval=-1.0, dtype=tf.float64)
    npt.assert_allclose(acq(xs), (xs + 1) * (xs + 2))


def test_reducer_calls_update() -> None:
    prod = Product(_Static(lambda x: x + 1), _Static(lambda x: x + 2))
    data, models = {TAG: empty_dataset([1], [1])}, {TAG: QuadraticMeanAndRBFKernel()}
    acq = prod.prepare_acquisition_function(models, datasets=data)
    acq = prod.update_acquisition_function(acq, models, datasets=data)
    xs = tf.random.uniform([3, 5, 1], minval=-1.0, dtype=tf.float64)
    npt.assert_allclose(acq(xs), (xs + 2) * (xs + 3))


@pytest.mark.parametrize("reducer_class", [Sum, Product])
def test_sum_and_product_for_single_builder(
    reducer_class: type[Sum[ProbabilisticModel] | Product[ProbabilisticModel]],
) -> None:
    data, models = {TAG: empty_dataset([1], [1])}, {TAG: QuadraticMeanAndRBFKernel()}
    acq = reducer_class(_Static(lambda x: x**2)).prepare_acquisition_function(models, datasets=data)
    xs = tf.random.uniform([3, 5, 1], minval=-1.0)
    npt.assert_allclose(acq(xs), xs**2)


def test_map() -> None:
    prod = Map(lambda x: x + 1, _Static(lambda x: x + 2))
    data, models = {TAG: empty_dataset([1], [1])}, {TAG: QuadraticMeanAndRBFKernel()}
    acq = prod.prepare_acquisition_function(models, datasets=data)
    xs = tf.random.uniform([3, 5, 1], minval=-1.0, dtype=tf.float64)
    npt.assert_allclose(acq(xs), (xs + 3))

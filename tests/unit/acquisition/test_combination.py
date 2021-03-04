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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import pytest
import tensorflow as tf

from tests.util.misc import raise_
from tests.util.model import QuadraticMeanAndRBFKernel
from trieste.acquisition import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    ExpectedImprovement,
    NegativeLowerConfidenceBound,
    Reducer,
    expected_improvement,
    lower_confidence_bound,
)
from trieste.acquisition.combination import Product, Sum
from trieste.data import Dataset
from trieste.models import ProbabilisticModel


def test_reducer__repr_builders() -> None:
    class Acq1(AcquisitionFunctionBuilder):
        def __repr__(self) -> str:
            return "Acq1()"

        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return raise_

    class Acq2(AcquisitionFunctionBuilder):
        def __repr__(self) -> str:
            return "Acq2()"

        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return raise_

    class Foo(Reducer):
        def __repr__(self) -> str:
            return f"Foo({self._repr_builders()})"

        def _reduce(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
            return inputs[0]

    assert repr(Foo(Acq1())) == "Foo(Acq1())"
    assert repr(Foo(Acq1(), Acq2())) == "Foo(Acq1(), Acq2())"


@dataclass
class ReducerTestData:
    type_class: type[Sum | Product]
    raw_reduce_op: Callable[[Sequence], float]
    dataset: Dataset = Dataset(
        np.arange(5, dtype=np.float64).reshape(-1, 1), np.zeros(5).reshape(-1, 1)
    )
    query_point: tf.Tensor = tf.convert_to_tensor(np.array([[0.1], [0.2]]))


_sum_fn = partial(np.sum, axis=0)
_prod_fn = partial(np.prod, axis=0)
_reducers = [ReducerTestData(Sum, _sum_fn), ReducerTestData(Product, _prod_fn)]


@pytest.mark.parametrize("reducer", _reducers)
def test_reducers_on_ei(reducer):
    m = 6
    zero = tf.convert_to_tensor([0.0], dtype=tf.float64)
    model = QuadraticMeanAndRBFKernel()
    acqs = [ExpectedImprovement().using("foo") for _ in range(m)]
    acq = reducer.type_class(*acqs)
    acq_fn = acq.prepare_acquisition_function({"foo": reducer.dataset}, {"foo": model})
    individual_ei = [expected_improvement(model, zero, reducer.query_point) for _ in range(m)]
    expected = reducer.raw_reduce_op(individual_ei)
    desired = acq_fn(reducer.query_point)
    np.testing.assert_array_almost_equal(desired, expected)


@pytest.mark.parametrize("reducer", _reducers)
def test_reducers_on_lcb(reducer):
    m = 6
    beta = tf.convert_to_tensor(1.96, dtype=tf.float64)
    model = QuadraticMeanAndRBFKernel()
    acqs = [NegativeLowerConfidenceBound(beta).using("foo") for _ in range(m)]
    acq = reducer.type_class(*acqs)
    acq_fn = acq.prepare_acquisition_function({"foo": reducer.dataset}, {"foo": model})
    individual_lcb = [-lower_confidence_bound(model, beta, reducer.query_point) for _ in range(m)]
    expected = reducer.raw_reduce_op(individual_lcb)
    desired = acq_fn(reducer.query_point)
    np.testing.assert_array_almost_equal(expected, desired)


@pytest.mark.parametrize("reducer_class", [Sum, Product])
def test_reducer_fails(reducer_class):
    with pytest.raises(TypeError):
        reducer_class(1, 2, 3)

    with pytest.raises(ValueError):
        reducer_class()


class _InputIdentity(AcquisitionFunctionBuilder):
    def __init__(self, result: tf.Tensor):
        self._result = result

    def prepare_acquisition_function(
        self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        return lambda _: self._result


@pytest.mark.parametrize("combination", [(Product, _prod_fn), (Sum, _sum_fn)])
@pytest.mark.parametrize(
    "inputs",
    [
        [2.0, 3.0],
        [np.arange(5.0), np.flip(np.arange(5.0))],
        [np.arange(6.0).reshape(2, 3), np.flip(np.arange(6.0)).reshape(2, 3)],
    ],
)
def test_product_reducer_multiplies_tensors(combination, inputs):
    combination_builder, expected_fn = combination
    inputs = [np.array(i) for i in inputs]
    expected = expected_fn(inputs)
    builders = [_InputIdentity(i) for i in inputs]
    reducer = combination_builder(*builders)
    data = Dataset(tf.zeros((1, 1)), tf.zeros((1, 1)))
    prepared_fn = reducer.prepare_acquisition_function(data, QuadraticMeanAndRBFKernel())
    result = prepared_fn(tf.zeros(1))
    np.testing.assert_allclose(result, expected)

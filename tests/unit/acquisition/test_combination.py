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

from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence, Union, Type

import numpy as np
import pytest
import tensorflow as tf
from returns.primitives.hkt import Kind1

from tests.util.model import QuadraticWithUnitVariance

from trieste.acquisition.combination import Product, Sum
from trieste.acquisition.function import (
    AcquisitionFunction,
    ExpectedImprovement,
    NegativeLowerConfidenceBound,
    expected_improvement,
    lower_confidence_bound,
)
from trieste.acquisition.rule import AcquisitionFunctionBuilder
from trieste.datasets import Dataset
from trieste.models import ModelInterface
from trieste.utils.grouping import G


@dataclass
class ReducerTestData:
    type_class: Type[Union[Sum, Product]]
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
    model = QuadraticWithUnitVariance()
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
    model = QuadraticWithUnitVariance()
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


class _InputIdentity(AcquisitionFunctionBuilder[G]):
    def __init__(self, result: tf.Tensor):
        self._result = result

    def prepare_acquisition_function(
        self, datasets: Kind1[G, Dataset], models: Kind1[G, ModelInterface]
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
    prepared_fn = reducer.prepare_acquisition_function(data, QuadraticWithUnitVariance())
    result = prepared_fn(tf.zeros(1))
    np.testing.assert_allclose(result, expected)

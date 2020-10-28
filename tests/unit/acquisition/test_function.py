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
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
import numpy.testing as npt
import tensorflow as tf
import tensorflow_probability as tfp

from trieste.acquisition import SingleModelAcquisitionBuilder
from trieste.data import Dataset
from trieste.acquisition.function import (
    AcquisitionFunction,
    ExpectedImprovement,
    NegativeLowerConfidenceBound,
    ProbabilityOfFeasibility,
    expected_improvement,
    lower_confidence_bound,
    probability_of_feasibility,
)
from trieste.models import ModelInterface
from tests.util.misc import ShapeLike, various_shapes, zero_dataset
from tests.util.model import QuadraticWithUnitVariance, StaticModelInterface, StaticWithUnitVariance
from trieste.type import TensorType


class _IdentitySingleBuilder(SingleModelAcquisitionBuilder):
    def prepare_acquisition_function(
        self, dataset: Dataset, model: ModelInterface
    ) -> AcquisitionFunction:
        return lambda at: at


def test_single_builder_raises_immediately_for_wrong_key() -> None:
    builder = _IdentitySingleBuilder().using("foo")

    with pytest.raises(KeyError):
        builder.prepare_acquisition_function(
            {"bar": zero_dataset()}, {"bar": QuadraticWithUnitVariance()}
        )


def test_single_builder_repr_includes_class_name() -> None:
    assert "_IdentitySingleBuilder" in repr(_IdentitySingleBuilder())


def test_single_builder_using_passes_on_correct_dataset_and_model() -> None:
    class _Mock(SingleModelAcquisitionBuilder):
        def prepare_acquisition_function(
            self, dataset: Dataset, model: ModelInterface
        ) -> AcquisitionFunction:
            assert dataset is data["foo"]
            assert model is models["foo"]
            return lambda at: at

    builder = _Mock().using("foo")

    data = {"foo": zero_dataset(), "bar": zero_dataset()}
    models = {"foo": QuadraticWithUnitVariance(), "bar": QuadraticWithUnitVariance()}
    builder.prepare_acquisition_function(data, models)


# todo shouldn't this test that it defines eta as the best posterior mean, as opposed to the best
#  observation? does it test that? if so, change test name
@pytest.mark.parametrize('query_at', [
    tf.constant([[-2.0], [-1.5], [-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5], [2.0]])
])
def test_expected_improvement_builder_builds_expected_improvement(
        query_at: tf.Tensor
) -> None:
    dataset = Dataset(tf.constant([[-2.], [-1.], [0.], [1.], [2.]]), tf.zeros([5, 1]))
    model = QuadraticWithUnitVariance()
    builder = ExpectedImprovement()
    acq_fn = builder.prepare_acquisition_function(dataset, model)
    expected = expected_improvement(model, tf.constant([0.]), query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)


def test_expected_improvement() -> None:
    class _Model(StaticWithUnitVariance):
        def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
            def mk_range(
                leq_zero: tf.Tensor, zero_to_one: tf.Tensor, gt_one: tf.Tensor
            ) -> tf.Tensor:
                return tf.where(
                    query_points <= 0.0, leq_zero, tf.where(query_points < 1.0, zero_to_one, gt_one)
                )

            mean = mk_range(tf.constant([[10.0]]), tf.constant([[-7.0]]), tf.constant([[5.0]]))
            var = mk_range(tf.constant([[1.0]]), tf.constant([[1.0]]), tf.constant([[1.0]]))
            return mean, var

    x = tf.constant([[-1.0], [0.0], [0.5], [1.0], [2.0]])
    npt.assert_allclose(
        expected_improvement(_Model(), tf.constant([5.0]), x),
        tf.constant([[0.0], [0.0], [12.0], [12.0], [0.0]])
    )


# def test_expected_improvement() -> None:
#     def _ei(x: tf.Tensor) -> tf.Tensor:
#         n = tfp.distributions.Normal(0, 1)
#         return - x * n.cdf(-x) + n.prob(-x)
#
#     query_at = tf.constant([[-2.0], [-1.5], [-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5], [2.0]])
#     actual = expected_improvement(QuadraticWithUnitVariance(), tf.constant([0.]), query_at)
#     npt.assert_array_almost_equal(actual, _ei(query_at ** 2))


def test_negative_lower_confidence_bound_builder_builds_negative_lower_confidence_bound() -> None:
    model = QuadraticWithUnitVariance()
    beta = 1.96
    acq_fn = NegativeLowerConfidenceBound(beta).prepare_acquisition_function(
        Dataset(tf.constant([[]]), tf.constant([[]])), model
    )
    query_at = tf.constant([[-3.], [-2.], [-1.], [0.], [1.], [2.], [3.]])
    expected = - lower_confidence_bound(model, beta, query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)


@pytest.mark.parametrize('beta', [-0.1, -2.0])
def test_lower_confidence_bound_raises_for_negative_beta(beta: float) -> None:
    with pytest.raises(ValueError):
        lower_confidence_bound(MagicMock(ModelInterface), beta, tf.constant([[]]))


@pytest.mark.parametrize('beta', [0.0, 0.1, 7.8])
def test_lower_confidence_bound(beta: float) -> None:
    query_at = tf.constant([[-3.], [-2.], [-1.], [0.], [1.], [2.], [3.]])
    actual = lower_confidence_bound(QuadraticWithUnitVariance(), beta, query_at)
    npt.assert_array_almost_equal(actual, query_at ** 2 - beta)


@pytest.mark.parametrize('threshold, at, expected', [
    (0.0, tf.constant([[0.0]]), 0.5),
    # values looked up on a standard normal table
    (2.0, tf.constant([[1.0]]), 0.5 + 0.34134),
    (-0.25, tf.constant([[-0.5]]), 0.5 - 0.19146),
])
def test_probability_of_feasibility(threshold: float, at: tf.Tensor, expected: float) -> None:
    actual = probability_of_feasibility(QuadraticWithUnitVariance(), threshold, at)
    npt.assert_allclose(actual, expected, rtol=1e-4)


@pytest.mark.parametrize('at', [tf.constant([[0.0]]), tf.constant([[-3.4]]), tf.constant([[0.2]])])
@pytest.mark.parametrize('threshold', [-2.3, 0.2])
def test_probability_of_feasibility_builder_builds_pof(threshold: float, at: tf.Tensor) -> None:
    builder = ProbabilityOfFeasibility(threshold)
    acq = builder.prepare_acquisition_function(zero_dataset(), QuadraticWithUnitVariance())
    expected = probability_of_feasibility(QuadraticWithUnitVariance(), threshold, at)
    npt.assert_allclose(acq(at), expected)


@pytest.mark.parametrize('shape', various_shapes() - {()})
def test_probability_of_feasibility_raises_on_non_scalar_threshold(shape: ShapeLike) -> None:
    threshold = tf.ones(shape)
    with pytest.raises(ValueError):
        probability_of_feasibility(QuadraticWithUnitVariance(), threshold, tf.constant([[0.0]]))


@pytest.mark.parametrize('shape', [[], [0], [2]])
def test_probability_of_feasibility_raises_on_incorrect_at_shape(shape: ShapeLike) -> None:
    at = tf.ones(shape)
    with pytest.raises(ValueError):
        probability_of_feasibility(QuadraticWithUnitVariance(), 0.0, at)


@pytest.mark.parametrize('shape', various_shapes() - {()})
def test_probability_of_feasibility_builder_raises_on_non_scalar_threshold(
    shape: ShapeLike
) -> None:
    threshold = tf.ones(shape)
    with pytest.raises(ValueError):
        ProbabilityOfFeasibility(threshold)

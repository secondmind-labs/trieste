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

from typing import Mapping, Tuple
from unittest.mock import MagicMock

import pytest
import numpy.testing as npt
import tensorflow as tf
import tensorflow_probability as tfp

from trieste.acquisition import SingleModelAcquisitionBuilder
from trieste.data import Dataset
from trieste.acquisition.function import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    ExpectedConstrainedImprovement,
    ExpectedImprovement,
    NegativeLowerConfidenceBound,
    ProbabilityOfFeasibility,
    expected_improvement,
    lower_confidence_bound,
    probability_of_feasibility,
)
from trieste.models import ProbabilisticModel
from trieste.type import TensorType
from trieste.utils.objectives import branin, BRANIN_GLOBAL_MINIMUM

from tests.util.misc import ShapeLike, various_shapes, zero_dataset, random_seed
from tests.util.model import CustomMeanWithUnitVariance, QuadraticWithUnitVariance, GaussianMarginal


class _IdentitySingleBuilder(SingleModelAcquisitionBuilder):
    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
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
            self, dataset: Dataset, model: ProbabilisticModel
        ) -> AcquisitionFunction:
            assert dataset is data["foo"]
            assert model is models["foo"]
            return lambda at: at

    builder = _Mock().using("foo")

    data = {"foo": zero_dataset(), "bar": zero_dataset()}
    models = {"foo": QuadraticWithUnitVariance(), "bar": QuadraticWithUnitVariance()}
    builder.prepare_acquisition_function(data, models)


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


@random_seed()
@pytest.mark.parametrize('best', [
    tf.constant([50.0]), BRANIN_GLOBAL_MINIMUM, BRANIN_GLOBAL_MINIMUM * 1.01
])
@pytest.mark.parametrize('variance_scale, num_samples_per_point, rtol, atol', [
    (0.1, 1000, 0.01, 1e-9),
    (1.0, 50_000, 0.01, 1e-3),
    (10.0, 100_000, 0.01, 1e-2),
    (100.0, 150_000, 0.01, 1e-1),
])
def test_expected_improvement(
    variance_scale: float, num_samples_per_point: int, best: tf.Tensor, rtol: float, atol: float
) -> None:
    variance_scale = tf.constant(variance_scale, tf.float64)
    best = tf.cast(best, dtype=tf.float64)

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing='ij'), axis=-1), (-1, 2))

    class _Model(GaussianMarginal):
        kernel = tfp.math.psd_kernels.MaternFiveHalves(variance_scale, length_scale=0.25).apply

        def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
            return branin(query_points), self.kernel(query_points, query_points)[:, None]

    mean, variance = _Model().predict(xs)
    samples = tfp.distributions.Normal(mean, tf.sqrt(variance)).sample(num_samples_per_point)
    samples_improvement = tf.where(samples < best, best - samples, 0)
    ei_approx = tf.reduce_mean(samples_improvement, axis=0)

    ei = expected_improvement(_Model(), best, xs)

    npt.assert_allclose(ei, ei_approx, rtol=rtol, atol=atol)


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
        lower_confidence_bound(MagicMock(ProbabilisticModel), beta, tf.constant([[]]))


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


def test_expected_constrained_improvement_raises_for_non_scalar_min_pof() -> None:
    pof = ProbabilityOfFeasibility(0.0).using("")
    with pytest.raises(ValueError):
        ExpectedConstrainedImprovement("", pof, tf.constant([0.0]))


def test_expected_constrained_improvement_can_reproduce_expected_improvement() -> None:
    class _Certainty(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return tf.ones_like

    data = {"foo": Dataset(tf.constant([[0.5]]), tf.constant([[0.25]]))}
    models_ = {"foo": QuadraticWithUnitVariance()}

    eci = ExpectedConstrainedImprovement(
        "foo", _Certainty(), 0
    ).prepare_acquisition_function(data, models_)

    ei = ExpectedImprovement().using("foo").prepare_acquisition_function(data, models_)

    at = tf.constant([[-0.1], [1.23], [-6.78]])
    npt.assert_allclose(eci(at), ei(at))


def test_expected_constrained_improvement_is_relative_to_feasible_point() -> None:
    class _Constraint(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return lambda x: tf.cast(x >= 0, x.dtype)

    models_ = {"foo": QuadraticWithUnitVariance()}

    eci_data = {"foo": Dataset(tf.constant([[-0.2], [0.3]]), tf.constant([[0.04], [0.09]]))}
    eci = ExpectedConstrainedImprovement(
        "foo", _Constraint()
    ).prepare_acquisition_function(eci_data, models_)

    ei_data = {"foo": Dataset(tf.constant([[0.3]]), tf.constant([[0.09]]))}
    ei = ExpectedImprovement().using("foo").prepare_acquisition_function(ei_data, models_)

    npt.assert_allclose(eci(tf.constant([[0.1]])), ei(tf.constant([[0.1]])))


def test_expected_constrained_improvement_is_less_for_constrained_points() -> None:
    class _Constraint(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return lambda x: tf.cast(x >= 0, x.dtype)

    def two_global_minima(x: tf.Tensor) -> tf.Tensor:
        return x ** 4 / 4 - x ** 2 / 2

    initial_query_points = tf.constant([[- 2.0], [0.0], [1.2]])
    data = {"foo": Dataset(initial_query_points, two_global_minima(initial_query_points))}
    models_ = {"foo": CustomMeanWithUnitVariance(two_global_minima)}

    eci = ExpectedConstrainedImprovement(
        "foo", _Constraint()
    ).prepare_acquisition_function(data, models_)

    npt.assert_array_less(eci(tf.constant(- 1.0)), eci(tf.constant(1.0)))


def test_expected_constrained_improvement_raises_for_empty_data() -> None:
    class _Constraint(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return lambda x: x

    data = {"foo": Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))}
    models_ = {"foo": QuadraticWithUnitVariance()}
    builder = ExpectedConstrainedImprovement("foo", _Constraint())

    with pytest.raises(ValueError):
        builder.prepare_acquisition_function(data, models_)


def test_expected_constrained_improvement_is_constraint_when_no_feasible_points() -> None:
    class _Constraint(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return lambda x: tf.cast(tf.logical_and(0.0 <= x, x < 1.0), x.dtype)

    data = {"foo": Dataset(tf.constant([[-2.0], [1.0]]), tf.constant([[4.0], [1.0]]))}
    models_ = {"foo": QuadraticWithUnitVariance()}
    eci = ExpectedConstrainedImprovement(
        "foo", _Constraint()
    ).prepare_acquisition_function(data, models_)

    constraint_fn = _Constraint().prepare_acquisition_function(data, models_)

    xs = tf.range(-10.0, 10.0, 100)
    npt.assert_allclose(eci(xs), constraint_fn(xs))


def test_expected_constrained_improvement_min_feasibility_probability_bound_is_inclusive() -> None:
    pof = tfp.bijectors.Sigmoid().forward

    class _Constraint(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return pof

    models_ = {"foo": QuadraticWithUnitVariance()}

    data = {"foo": Dataset(tf.constant([[1.1], [2.0]]), tf.constant([[1.21], [4.0]]))}
    eci = ExpectedConstrainedImprovement(
        "foo", _Constraint(), min_feasibility_probability=pof(1.0)
    ).prepare_acquisition_function(data, models_)

    ei = ExpectedImprovement().using("foo").prepare_acquisition_function(data, models_)

    x = tf.constant([[1.5]])
    npt.assert_allclose(eci(x), ei(x) * pof(x))

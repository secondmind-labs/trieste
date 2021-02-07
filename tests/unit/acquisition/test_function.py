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

import math
from typing import Callable, Mapping, Tuple, Union

import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import (
    TF_DEBUGGING_ERROR_TYPES,
    ShapeLike,
    mk_dataset,
    quadratic,
    raise_,
    random_seed,
    various_shapes,
    zero_dataset,
)
from tests.util.model import GaussianProcess, QuadraticMeanAndRBFKernel, rbf
from trieste.acquisition import SingleModelBatchAcquisitionBuilder
from trieste.acquisition.function import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    BatchMonteCarloExpectedImprovement,
    BatchReparametrizationSampler,
    ExpectedConstrainedImprovement,
    ExpectedImprovement,
    IndependentReparametrizationSampler,
    MCIndAcquisitionFunctionBuilder,
    NegativeLowerConfidenceBound,
    ProbabilityOfFeasibility,
    SingleModelAcquisitionBuilder,
    SingleModelMCIndAcquisitionFunctionBuilder,
    expected_improvement,
    lower_confidence_bound,
    probability_of_feasibility,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.type import TensorType
from trieste.utils.objectives import BRANIN_GLOBAL_MINIMUM, branin


class _ArbitrarySingleBuilder(SingleModelAcquisitionBuilder):
    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        return raise_


class _ArbitraryBatchSingleBuilder(SingleModelBatchAcquisitionBuilder):
    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        return raise_


@pytest.mark.parametrize(
    "single_builder", [_ArbitrarySingleBuilder(), _ArbitraryBatchSingleBuilder()]
)
def test_single_builder_raises_immediately_for_wrong_key(
    single_builder: Union[SingleModelAcquisitionBuilder, SingleModelBatchAcquisitionBuilder]
) -> None:
    builder = single_builder.using("foo")

    with pytest.raises(KeyError):
        builder.prepare_acquisition_function(
            {"bar": zero_dataset()}, {"bar": QuadraticMeanAndRBFKernel()}
        )


@pytest.mark.parametrize("builder", [_ArbitrarySingleBuilder(), _ArbitraryBatchSingleBuilder()])
def test_single_builder_repr_includes_class_name(
    builder: Union[SingleModelAcquisitionBuilder, SingleModelBatchAcquisitionBuilder]
) -> None:
    assert type(builder).__name__ in repr(builder)


def _prepare_acquisition_function_assert(
    _: object, dataset: Dataset, model: ProbabilisticModel
) -> Callable[[TensorType], TensorType]:
    npt.assert_allclose(dataset.query_points, 0.0)
    _, var = model.predict(tf.constant([0.0]))
    npt.assert_allclose(var, 0.0)
    return raise_


class _MockIndBuilder(SingleModelAcquisitionBuilder):
    prepare_acquisition_function = _prepare_acquisition_function_assert


class _MockBatchBuilder(SingleModelBatchAcquisitionBuilder):
    prepare_acquisition_function = _prepare_acquisition_function_assert


@pytest.mark.parametrize("single_builder", [_MockIndBuilder(), _MockBatchBuilder()])
def test_single_builder_using_passes_on_correct_dataset_and_model(
    single_builder: Union[SingleModelAcquisitionBuilder, SingleModelBatchAcquisitionBuilder]
) -> None:
    builder = single_builder.using("foo")
    data = {"foo": mk_dataset([[0.0]], [[0.0]]), "bar": mk_dataset([[1.0]], [[1.0]])}
    models = {"foo": QuadraticMeanAndRBFKernel(0.0), "bar": QuadraticMeanAndRBFKernel(1.0)}
    builder.prepare_acquisition_function(data, models)


def test_expected_improvement_builder_builds_expected_improvement_using_best_from_model() -> None:
    dataset = Dataset(
        tf.constant([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]]),
    )
    model = QuadraticMeanAndRBFKernel()
    acq_fn = ExpectedImprovement().prepare_acquisition_function(dataset, model)
    xs = tf.linspace([-10.0], [10.0], 100)
    expected = expected_improvement(model, tf.constant([0.0]), xs)
    npt.assert_allclose(acq_fn(xs), expected)


def test_expected_improvement_builder_raises_for_empty_data() -> None:
    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))

    with pytest.raises(ValueError):
        ExpectedImprovement().prepare_acquisition_function(data, QuadraticMeanAndRBFKernel())


@random_seed
@pytest.mark.parametrize(
    "best", [tf.constant([50.0]), BRANIN_GLOBAL_MINIMUM, BRANIN_GLOBAL_MINIMUM * 1.01]
)
@pytest.mark.parametrize(
    "variance_scale, num_samples_per_point, rtol, atol",
    [
        (0.1, 1000, 0.01, 1e-9),
        (1.0, 50_000, 0.01, 1e-3),
        (10.0, 100_000, 0.01, 1e-2),
        (100.0, 150_000, 0.01, 1e-1),
    ],
)
def test_expected_improvement(
    variance_scale: float, num_samples_per_point: int, best: tf.Tensor, rtol: float, atol: float
) -> None:
    variance_scale = tf.constant(variance_scale, tf.float64)
    best = tf.cast(best, dtype=tf.float64)

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    kernel = tfp.math.psd_kernels.MaternFiveHalves(variance_scale, length_scale=0.25)
    model = GaussianProcess([branin], [kernel])

    mean, variance = model.predict(xs)
    samples = tfp.distributions.Normal(mean, tf.sqrt(variance)).sample(num_samples_per_point)
    samples_improvement = tf.where(samples < best, best - samples, 0)
    ei_approx = tf.reduce_mean(samples_improvement, axis=0)

    ei = expected_improvement(model, best, xs)

    npt.assert_allclose(ei, ei_approx, rtol=rtol, atol=atol)


def test_negative_lower_confidence_bound_builder_builds_negative_lower_confidence_bound() -> None:
    model = QuadraticMeanAndRBFKernel()
    beta = 1.96
    acq_fn = NegativeLowerConfidenceBound(beta).prepare_acquisition_function(
        Dataset(tf.zeros([0, 1]), tf.zeros([0, 1])), model
    )
    query_at = tf.linspace([-10], [10], 100)
    expected = -lower_confidence_bound(model, beta, query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)


@pytest.mark.parametrize("beta", [-0.1, -2.0])
def test_lower_confidence_bound_raises_for_negative_beta(beta: float) -> None:
    with pytest.raises(ValueError):
        lower_confidence_bound(QuadraticMeanAndRBFKernel(), beta, tf.constant([[]]))


@pytest.mark.parametrize("beta", [0.0, 0.1, 7.8])
def test_lower_confidence_bound(beta: float) -> None:
    query_at = tf.constant([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
    actual = lower_confidence_bound(QuadraticMeanAndRBFKernel(), beta, query_at)
    npt.assert_array_almost_equal(actual, query_at ** 2 - beta)


@pytest.mark.parametrize(
    "threshold, at, expected",
    [
        (0.0, tf.constant([[0.0]]), 0.5),
        # values looked up on a standard normal table
        (2.0, tf.constant([[1.0]]), 0.5 + 0.34134),
        (-0.25, tf.constant([[-0.5]]), 0.5 - 0.19146),
    ],
)
def test_probability_of_feasibility(threshold: float, at: tf.Tensor, expected: float) -> None:
    actual = probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold, at)
    npt.assert_allclose(actual, expected, rtol=1e-4)


@pytest.mark.parametrize("at", [tf.constant([[0.0]]), tf.constant([[-3.4]]), tf.constant([[0.2]])])
@pytest.mark.parametrize("threshold", [-2.3, 0.2])
def test_probability_of_feasibility_builder_builds_pof(threshold: float, at: tf.Tensor) -> None:
    builder = ProbabilityOfFeasibility(threshold)
    acq = builder.prepare_acquisition_function(zero_dataset(), QuadraticMeanAndRBFKernel())
    expected = probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold, at)
    npt.assert_allclose(acq(at), expected)


@pytest.mark.parametrize("shape", various_shapes() - {()})
def test_probability_of_feasibility_raises_on_non_scalar_threshold(shape: ShapeLike) -> None:
    threshold = tf.ones(shape)
    with pytest.raises(ValueError):
        probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold, tf.constant([[0.0]]))


@pytest.mark.parametrize("shape", [[], [0], [2]])
def test_probability_of_feasibility_raises_on_incorrect_at_shape(shape: ShapeLike) -> None:
    at = tf.ones(shape)
    with pytest.raises(ValueError):
        probability_of_feasibility(QuadraticMeanAndRBFKernel(), 0.0, at)


@pytest.mark.parametrize("shape", various_shapes() - {()})
def test_probability_of_feasibility_builder_raises_on_non_scalar_threshold(
    shape: ShapeLike,
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
    models_ = {"foo": QuadraticMeanAndRBFKernel()}

    eci = ExpectedConstrainedImprovement("foo", _Certainty(), 0).prepare_acquisition_function(
        data, models_
    )

    ei = ExpectedImprovement().using("foo").prepare_acquisition_function(data, models_)

    at = tf.constant([[-0.1], [1.23], [-6.78]])
    npt.assert_allclose(eci(at), ei(at))


def test_expected_constrained_improvement_is_relative_to_feasible_point() -> None:
    class _Constraint(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return lambda x: tf.cast(x >= 0, x.dtype)

    models_ = {"foo": QuadraticMeanAndRBFKernel()}

    eci_data = {"foo": Dataset(tf.constant([[-0.2], [0.3]]), tf.constant([[0.04], [0.09]]))}
    eci = ExpectedConstrainedImprovement("foo", _Constraint()).prepare_acquisition_function(
        eci_data, models_
    )

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

    initial_query_points = tf.constant([[-2.0], [0.0], [1.2]])
    data = {"foo": Dataset(initial_query_points, two_global_minima(initial_query_points))}
    models_ = {"foo": GaussianProcess([two_global_minima], [rbf()])}

    eci = ExpectedConstrainedImprovement("foo", _Constraint()).prepare_acquisition_function(
        data, models_
    )

    npt.assert_array_less(eci(tf.constant([-1.0])), eci(tf.constant([1.0])))


def test_expected_constrained_improvement_raises_for_empty_data() -> None:
    class _Constraint(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return raise_

    data = {"foo": Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))}
    models_ = {"foo": QuadraticMeanAndRBFKernel()}
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
    models_ = {"foo": QuadraticMeanAndRBFKernel()}
    eci = ExpectedConstrainedImprovement("foo", _Constraint()).prepare_acquisition_function(
        data, models_
    )

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

    models_ = {"foo": QuadraticMeanAndRBFKernel()}

    data = {"foo": Dataset(tf.constant([[1.1], [2.0]]), tf.constant([[1.21], [4.0]]))}
    eci = ExpectedConstrainedImprovement(
        "foo", _Constraint(), min_feasibility_probability=pof(1.0)
    ).prepare_acquisition_function(data, models_)

    ei = ExpectedImprovement().using("foo").prepare_acquisition_function(data, models_)

    x = tf.constant([[1.5]])
    npt.assert_allclose(eci(x), ei(x) * pof(x))


@pytest.mark.parametrize("sample_size", [0, -2])
def test_independent_reparametrization_sampler_raises_for_negative_sample_size(
    sample_size: int,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        IndependentReparametrizationSampler(sample_size, QuadraticMeanAndRBFKernel())


def test_independent_reparametrization_sampler_sample_raises_for_invalid_at_shape() -> None:
    sampler = IndependentReparametrizationSampler(1, QuadraticMeanAndRBFKernel())

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.constant(0))


def _assert_kolmogorov_smirnov_95(
    # fmt: off
    samples: tf.Tensor,  # [..., S]
    distribution: tfp.distributions.Distribution
    # fmt: on
) -> None:
    assert distribution.event_shape == ()
    tf.debugging.assert_shapes([(samples, [..., "S"])])

    sample_size = samples.shape[-1]
    samples_sorted = tf.sort(samples, axis=-1)  # [..., S]
    edf = tf.range(1.0, sample_size + 1, dtype=samples.dtype) / sample_size  # [S]
    expected_cdf = distribution.cdf(samples_sorted)  # [..., S]

    _95_percent_bound = 1.36 / math.sqrt(sample_size)
    assert tf.reduce_max(tf.abs(edf - expected_cdf)) < _95_percent_bound


def _dim_two_gp(mean_shift: Tuple[float, float] = (0.0, 0.0)) -> GaussianProcess:
    matern52 = tfp.math.psd_kernels.MaternFiveHalves(
        amplitude=tf.cast(2.3, tf.float64), length_scale=tf.cast(0.5, tf.float64)
    )
    return GaussianProcess(
        [lambda x: mean_shift[0] + branin(x), lambda x: mean_shift[1] + quadratic(x)],
        [matern52, rbf()],
    )


@random_seed
def test_independent_reparametrization_sampler_samples_approximate_expected_distribution() -> None:
    sample_size = 1000
    x = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)

    model = _dim_two_gp()
    samples = IndependentReparametrizationSampler(sample_size, model).sample(x)  # [N, S, L]

    assert samples.shape == [len(x), sample_size, 2]

    mean, var = model.predict(x)  # [N, L], [N, L]
    _assert_kolmogorov_smirnov_95(
        tf.linalg.matrix_transpose(samples),
        tfp.distributions.Normal(mean[..., None], tf.sqrt(var[..., None])),
    )


@random_seed
def test_independent_reparametrization_sampler_sample_is_continuous() -> None:
    sampler = IndependentReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_array_less(tf.abs(sampler.sample(xs + 1e-20) - sampler.sample(xs)), 1e-20)


def test_independent_reparametrization_sampler_sample_is_repeatable() -> None:
    sampler = IndependentReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))


@random_seed
def test_independent_reparametrization_sampler_samples_are_distinct_for_new_instances() -> None:
    sampler1 = IndependentReparametrizationSampler(100, _dim_two_gp())
    sampler2 = IndependentReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_array_less(1e-9, tf.abs(sampler2.sample(xs) - sampler1.sample(xs)))


def test_mc_ind_acquisition_function_builder_raises_for_invalid_sample_size() -> None:
    class _Acq(MCIndAcquisitionFunctionBuilder):
        def _build_with_sampler(
            self,
            datasets: Mapping[str, Dataset],
            models: Mapping[str, ProbabilisticModel],
            samplers: Mapping[str, IndependentReparametrizationSampler],
        ) -> AcquisitionFunction:
            return raise_

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        _Acq(-1)


@random_seed
def test_mc_ind_acquisition_function_builder_approximates_model_samples() -> None:
    class _Acq(MCIndAcquisitionFunctionBuilder):
        def _build_with_sampler(
            self,
            datasets: Mapping[str, Dataset],
            models: Mapping[str, ProbabilisticModel],
            samplers: Mapping[str, IndependentReparametrizationSampler],
        ) -> AcquisitionFunction:
            assert samplers.keys() == {"foo", "bar", "baz"}

            x = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)

            for key in samplers:
                samples = samplers[key].sample(x)
                mean, var = models[key].predict(x)
                _assert_kolmogorov_smirnov_95(
                    tf.linalg.matrix_transpose(samples),
                    tfp.distributions.Normal(mean[..., None], tf.sqrt(var)[..., None]),
                )

            return raise_

    data = Dataset(tf.zeros([0, 2], dtype=tf.float64), tf.zeros([0, 2], dtype=tf.float64))
    _Acq(20_000).prepare_acquisition_function(
        {"foo": data, "bar": data, "baz": data},
        {
            "foo": _dim_two_gp((0.5, 0.5)),
            "bar": _dim_two_gp((1.3, 1.3)),
            "baz": _dim_two_gp((-0.7, -0.7)),
        },
    )


def test_single_model_mc_ind_acquisition_function_builder_raises_for_invalid_sample_size() -> None:
    class _Acq(SingleModelMCIndAcquisitionFunctionBuilder):
        def _build_with_sampler(
            self,
            dataset: Dataset,
            model: ProbabilisticModel,
            sampler: IndependentReparametrizationSampler,
        ) -> AcquisitionFunction:
            return raise_

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        _Acq(-1)


@random_seed
def test_single_model_mc_ind_acquisition_function_builder_approximates_model_samples() -> None:
    class _Acq(SingleModelMCIndAcquisitionFunctionBuilder):
        def _build_with_sampler(
            self,
            dataset: Dataset,
            model: ProbabilisticModel,
            sampler: IndependentReparametrizationSampler,
        ) -> AcquisitionFunction:
            x = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
            samples = sampler.sample(x)
            mean, var = model.predict(x)
            _assert_kolmogorov_smirnov_95(
                tf.linalg.matrix_transpose(samples),
                tfp.distributions.Normal(mean[..., None], tf.sqrt(var)[..., None]),
            )
            return raise_

    data = Dataset(tf.zeros([0, 2], dtype=tf.float64), tf.zeros([0, 2], dtype=tf.float64))
    _Acq(1000).prepare_acquisition_function(data, _dim_two_gp())


@pytest.mark.parametrize("sample_size", [0, -2])
def test_batch_reparametrization_sampler_raises_for_invalid_sample_size(sample_size: int) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchReparametrizationSampler(sample_size, _dim_two_gp())


@random_seed
def test_batch_reparametrization_sampler_samples_approximate_mean_and_covariance() -> None:
    model = _dim_two_gp()
    sample_size = 10_000
    leading_dims = [3]
    batch_size = 4
    xs = tf.random.uniform(leading_dims + [batch_size, 2], maxval=1.0, dtype=tf.float64)
    samples = BatchReparametrizationSampler(sample_size, model).sample(xs)

    assert samples.shape == leading_dims + [sample_size, batch_size, 2]

    samples_mean = tf.reduce_mean(samples, axis=-3)
    samples_covariance = tf.transpose(
        tfp.stats.covariance(samples, sample_axis=-3, event_axis=-2), [0, 3, 1, 2]
    )

    model_mean, model_cov = model.predict_joint(xs)

    npt.assert_allclose(samples_mean, model_mean, rtol=0.02)
    npt.assert_allclose(samples_covariance, model_cov, rtol=0.04)


def test_batch_reparametrization_sampler_samples_are_continuous() -> None:
    sampler = BatchReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    npt.assert_array_less(tf.abs(sampler.sample(xs + 1e-20) - sampler.sample(xs)), 1e-20)


def test_batch_reparametrization_sampler_samples_are_repeatable() -> None:
    sampler = BatchReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))


@random_seed
def test_batch_reparametrization_sampler_samples_are_distinct_for_new_instances() -> None:
    model = _dim_two_gp()
    sampler1 = BatchReparametrizationSampler(100, model)
    sampler2 = BatchReparametrizationSampler(100, model)
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    npt.assert_array_less(1e-9, tf.abs(sampler2.sample(xs) - sampler1.sample(xs)))


@pytest.mark.parametrize("at", [tf.constant([0.0]), tf.constant(0.0), tf.ones([0, 1])])
def test_batch_reparametrization_sampler_sample_raises_for_invalid_at_shape(at: tf.Tensor) -> None:
    sampler = BatchReparametrizationSampler(100, QuadraticMeanAndRBFKernel())

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(at)


def test_batch_reparametrization_sampler_sample_raises_for_negative_jitter() -> None:
    sampler = BatchReparametrizationSampler(100, QuadraticMeanAndRBFKernel())

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.constant([[0.0]]), jitter=-1e-6)


def test_batch_reparametrization_sampler_sample_raises_for_inconsistent_batch_size() -> None:
    sampler = BatchReparametrizationSampler(100, QuadraticMeanAndRBFKernel())
    sampler.sample(tf.constant([[0.0], [1.0], [2.0]]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.constant([[0.0], [1.0]]))


@pytest.mark.parametrize("sample_size", [-2, 0])
def test_batch_monte_carlo_expected_improvement_raises_for_invalid_sample_size(
    sample_size: int,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMonteCarloExpectedImprovement(sample_size)


def test_batch_monte_carlo_expected_improvement_raises_for_invalid_jitter() -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMonteCarloExpectedImprovement(100, jitter=-1.0)


def test_batch_monte_carlo_expected_improvement_raises_for_empty_data() -> None:
    builder = BatchMonteCarloExpectedImprovement(100)
    data = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))
    model = QuadraticMeanAndRBFKernel()
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(data, model)


def test_batch_monte_carlo_expected_improvement_raises_for_model_with_wrong_event_shape() -> None:
    builder = BatchMonteCarloExpectedImprovement(100)
    data = mk_dataset([[0.0, 0.0]], [[0.0, 0.0]])
    model = _dim_two_gp()
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(data, model)


@random_seed
def test_batch_monte_carlo_expected_improvement_can_reproduce_ei() -> None:
    known_query_points = tf.random.uniform([5, 2], dtype=tf.float64)
    data = Dataset(known_query_points, quadratic(known_query_points))
    model = QuadraticMeanAndRBFKernel()
    batch_ei = BatchMonteCarloExpectedImprovement(10_000).prepare_acquisition_function(data, model)
    ei = ExpectedImprovement().prepare_acquisition_function(data, model)
    xs = tf.random.uniform([3, 5, 1, 2], dtype=tf.float64)
    npt.assert_allclose(batch_ei(xs), ei(tf.squeeze(xs, -2)), rtol=0.03)  # todo a little high


@random_seed
def test_batch_monte_carlo_expected_improvement() -> None:
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    model = QuadraticMeanAndRBFKernel()

    mean, cov = model.predict_joint(xs)
    mvn = tfp.distributions.MultivariateNormalFullCovariance(tf.linalg.matrix_transpose(mean), cov)
    mvn_samples = mvn.sample(10_000)
    expected = tf.reduce_mean(tf.reduce_max(tf.maximum(0.09 - mvn_samples, 0.0), axis=-1), axis=0)

    builder = BatchMonteCarloExpectedImprovement(10_000)
    acq = builder.prepare_acquisition_function(mk_dataset([[0.3], [0.5]], [[0.09], [0.25]]), model)

    npt.assert_allclose(acq(xs), expected, rtol=0.05)  # todo quite high?

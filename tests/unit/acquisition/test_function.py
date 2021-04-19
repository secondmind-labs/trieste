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

import itertools
import math
import unittest.mock
from collections.abc import Mapping

import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import (
    TF_DEBUGGING_ERROR_TYPES,
    ShapeLike,
    empty_dataset,
    mk_dataset,
    quadratic,
    raise_exc,
    random_seed,
    various_shapes,
)
from tests.util.model import GaussianProcess, QuadraticMeanAndRBFKernel, rbf
from trieste.acquisition.function import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    BatchMonteCarloExpectedImprovement,
    BatchReparametrizationSampler,
    ExpectedConstrainedImprovement,
    ExpectedHypervolumeImprovement,
    ExpectedImprovement,
    IndependentReparametrizationSampler,
    MinValueEntropySearch,
    NegativeLowerConfidenceBound,
    ProbabilityOfFeasibility,
    SingleModelAcquisitionBuilder,
    expected_hv_improvement,
    expected_improvement,
    lower_confidence_bound,
    min_value_entropy_search,
    probability_of_feasibility,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.space import Box
from trieste.type import TensorType
from trieste.utils.objectives import BRANIN_MINIMUM, branin
from trieste.utils.pareto import Pareto, get_reference_point


class _ArbitrarySingleBuilder(SingleModelAcquisitionBuilder):
    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        return raise_exc


def test_single_model_acquisition_builder_raises_immediately_for_wrong_key() -> None:
    builder = _ArbitrarySingleBuilder().using("foo")

    with pytest.raises(KeyError):
        builder.prepare_acquisition_function(
            {"bar": empty_dataset([1], [1])}, {"bar": QuadraticMeanAndRBFKernel()}
        )


def test_single_model_acquisition_builder_repr_includes_class_name() -> None:
    builder = _ArbitrarySingleBuilder()
    assert type(builder).__name__ in repr(builder)


def test_single_model_acquisition_builder_using_passes_on_correct_dataset_and_model() -> None:
    class Builder(SingleModelAcquisitionBuilder):
        def prepare_acquisition_function(
            self, dataset: Dataset, model: ProbabilisticModel
        ) -> AcquisitionFunction:
            assert dataset is data["foo"]
            assert model is models["foo"]
            return raise_exc

    data = {"foo": empty_dataset([1], [1]), "bar": empty_dataset([1], [1])}
    models = {"foo": QuadraticMeanAndRBFKernel(), "bar": QuadraticMeanAndRBFKernel()}
    Builder().using("foo").prepare_acquisition_function(data, models)


def test_expected_improvement_builder_builds_expected_improvement_using_best_from_model() -> None:
    dataset = Dataset(
        tf.constant([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        tf.constant([[4.1], [0.9], [0.1], [1.1], [3.9]]),
    )
    model = QuadraticMeanAndRBFKernel()
    acq_fn = ExpectedImprovement().prepare_acquisition_function(dataset, model)
    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    expected = expected_improvement(model, tf.constant([0.0]))(xs)
    npt.assert_allclose(acq_fn(xs), expected)


def test_expected_improvement_builder_raises_for_empty_data() -> None:
    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))

    with pytest.raises(ValueError):
        ExpectedImprovement().prepare_acquisition_function(data, QuadraticMeanAndRBFKernel())


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_expected_improvement_raises_for_invalid_batch_size(at: TensorType) -> None:
    ei = expected_improvement(QuadraticMeanAndRBFKernel(), tf.constant([1.0]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ei(at)


@random_seed
@pytest.mark.parametrize("best", [tf.constant([50.0]), BRANIN_MINIMUM, BRANIN_MINIMUM * 1.01])
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

    ei = expected_improvement(model, best)(xs[..., None, :])

    npt.assert_allclose(ei, ei_approx, rtol=rtol, atol=atol)


def test_min_value_entropy_search_builder_raises_for_empty_data() -> None:
    data = Dataset(tf.zeros([0, 1]), tf.ones([0, 1]))
    search_space = Box([0, 0], [1, 1])
    builder = MinValueEntropySearch(search_space)
    with pytest.raises(ValueError):
        builder.prepare_acquisition_function(data, QuadraticMeanAndRBFKernel())


def test_min_value_entropy_search_builder_raises_for_invalid_gumbel_sample_sizes() -> None:
    search_space = Box([0, 0], [1, 1])
    with pytest.raises(ValueError):
        MinValueEntropySearch(search_space, num_samples=-5)
    with pytest.raises(ValueError):
        MinValueEntropySearch(search_space, grid_size=-5)


@unittest.mock.patch("trieste.acquisition.function.min_value_entropy_search")
def test_min_value_entropy_search_builder_gumbel_samples(mocked_mves) -> None:
    dataset = Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))
    search_space = Box([0, 0], [1, 1])
    builder = MinValueEntropySearch(search_space)
    model = QuadraticMeanAndRBFKernel()
    builder.prepare_acquisition_function(dataset, model)
    mocked_mves.assert_called_once()

    # check that the Gumbel samples look sensible
    gumbel_samples = mocked_mves.call_args[0][1]
    query_points = builder._search_space.sample(num_samples=builder._grid_size)
    query_points = tf.concat([dataset.query_points, query_points], 0)
    fmean, _ = model.predict(query_points)
    assert max(gumbel_samples) < min(fmean)


@pytest.mark.parametrize("samples", [tf.constant([]), tf.constant([[]])])
def test_min_value_entropy_search_raises_for_gumbel_samples_with_invalid_shape(
    samples: TensorType,
) -> None:
    with pytest.raises(ValueError):
        min_value_entropy_search(QuadraticMeanAndRBFKernel(), samples)


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_min_value_entropy_search_raises_for_invalid_batch_size(at: TensorType) -> None:
    mes = min_value_entropy_search(QuadraticMeanAndRBFKernel(), tf.constant([1.0]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        mes(at)


def test_min_value_entropy_search_returns_correct_shape() -> None:
    model = QuadraticMeanAndRBFKernel()
    gumbel_samples = tf.constant([1.0])
    query_at = tf.linspace([[-10.0]], [[10.0]], 5)
    evals = min_value_entropy_search(model, gumbel_samples)(query_at)
    npt.assert_array_equal(evals.shape, tf.constant([5, 1]))


def test_min_value_entropy_search_chooses_same_as_probability_of_improvement() -> None:
    """
    When based on a single max-value sample, MES should choose the same point that probability of
    improvement would when calcualted with the max-value as its threshold (See :cite:`wang2017max`).
    """

    kernel = tfp.math.psd_kernels.MaternFiveHalves()
    model = GaussianProcess([branin], [kernel])

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    gumbel_sample = tf.constant([1.0], dtype=tf.float64)
    mes_evals = min_value_entropy_search(model, gumbel_sample)(xs[..., None, :])

    mean, variance = model.predict(xs)
    gamma = (tf.cast(gumbel_sample, dtype=mean.dtype) - mean) / tf.sqrt(variance)
    norm = tfp.distributions.Normal(tf.cast(0, dtype=mean.dtype), tf.cast(1, dtype=mean.dtype))
    pi_evals = norm.cdf(gamma)
    npt.assert_array_equal(tf.argmax(mes_evals), tf.argmax(pi_evals))


def test_negative_lower_confidence_bound_builder_builds_negative_lower_confidence_bound() -> None:
    model = QuadraticMeanAndRBFKernel()
    beta = 1.96
    acq_fn = NegativeLowerConfidenceBound(beta).prepare_acquisition_function(
        Dataset(tf.zeros([0, 1]), tf.zeros([0, 1])), model
    )
    query_at = tf.linspace([[-10]], [[10]], 100)
    expected = -lower_confidence_bound(model, beta)(query_at)
    npt.assert_array_almost_equal(acq_fn(query_at), expected)


@pytest.mark.parametrize("beta", [-0.1, -2.0])
def test_lower_confidence_bound_raises_for_negative_beta(beta: float) -> None:
    with pytest.raises(ValueError):
        lower_confidence_bound(QuadraticMeanAndRBFKernel(), beta)


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_lower_confidence_bound_raises_for_invalid_batch_size(at: TensorType) -> None:
    lcb = lower_confidence_bound(QuadraticMeanAndRBFKernel(), tf.constant(1.0))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        lcb(at)


@pytest.mark.parametrize("beta", [0.0, 0.1, 7.8])
def test_lower_confidence_bound(beta: float) -> None:
    query_at = tf.linspace([[-3]], [[3]], 10)
    actual = lower_confidence_bound(QuadraticMeanAndRBFKernel(), beta)(query_at)
    npt.assert_array_almost_equal(actual, tf.squeeze(query_at, -2) ** 2 - beta)


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
    actual = probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold)(at)
    npt.assert_allclose(actual, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "at",
    [
        tf.constant([[0.0]], tf.float64),
        tf.constant([[-3.4]], tf.float64),
        tf.constant([[0.2]], tf.float64),
    ],
)
@pytest.mark.parametrize("threshold", [-2.3, 0.2])
def test_probability_of_feasibility_builder_builds_pof(threshold: float, at: tf.Tensor) -> None:
    builder = ProbabilityOfFeasibility(threshold)
    acq = builder.prepare_acquisition_function(empty_dataset([1], [1]), QuadraticMeanAndRBFKernel())
    expected = probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold)(at)

    npt.assert_allclose(acq(at), expected)


@pytest.mark.parametrize("shape", various_shapes() - {()})
def test_probability_of_feasibility_raises_on_non_scalar_threshold(shape: ShapeLike) -> None:
    threshold = tf.ones(shape)
    with pytest.raises(ValueError):
        probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold)


@pytest.mark.parametrize("shape", [[], [0], [2], [2, 1], [1, 2, 1]])
def test_probability_of_feasibility_raises_on_invalid_at_shape(shape: ShapeLike) -> None:
    at = tf.ones(shape)
    pof = probability_of_feasibility(QuadraticMeanAndRBFKernel(), 0.0)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        pof(at)


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


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_expected_constrained_improvement_raises_for_invalid_batch_size(at: TensorType) -> None:
    pof = ProbabilityOfFeasibility(0.0).using("")
    builder = ExpectedConstrainedImprovement("", pof, tf.constant(0.0))
    initial_query_points = tf.constant([[-1.0]])
    initial_objective_function_values = tf.constant([[1.0]])
    data = {"": Dataset(initial_query_points, initial_objective_function_values)}

    eci = builder.prepare_acquisition_function(data, {"": QuadraticMeanAndRBFKernel()})

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        eci(at)


def test_expected_constrained_improvement_can_reproduce_expected_improvement() -> None:
    class _Certainty(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return lambda x: tf.ones_like(tf.squeeze(x, -2))

    data = {"foo": Dataset(tf.constant([[0.5]]), tf.constant([[0.25]]))}
    models_ = {"foo": QuadraticMeanAndRBFKernel()}

    eci = ExpectedConstrainedImprovement("foo", _Certainty(), 0).prepare_acquisition_function(
        data, models_
    )

    ei = ExpectedImprovement().using("foo").prepare_acquisition_function(data, models_)

    at = tf.constant([[[-0.1]], [[1.23]], [[-6.78]]])
    npt.assert_allclose(eci(at), ei(at))


def test_expected_constrained_improvement_is_relative_to_feasible_point() -> None:
    class _Constraint(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return lambda x: tf.cast(tf.squeeze(x, -2) >= 0, x.dtype)

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
            return lambda x: tf.cast(tf.squeeze(x, -2) >= 0, x.dtype)

    def two_global_minima(x: tf.Tensor) -> tf.Tensor:
        return x ** 4 / 4 - x ** 2 / 2

    initial_query_points = tf.constant([[-2.0], [0.0], [1.2]])
    data = {"foo": Dataset(initial_query_points, two_global_minima(initial_query_points))}
    models_ = {"foo": GaussianProcess([two_global_minima], [rbf()])}

    eci = ExpectedConstrainedImprovement("foo", _Constraint()).prepare_acquisition_function(
        data, models_
    )

    npt.assert_array_less(eci(tf.constant([[-1.0]])), eci(tf.constant([[1.0]])))


def test_expected_constrained_improvement_raises_for_empty_data() -> None:
    class _Constraint(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return raise_exc

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
            def acquisition(x: TensorType) -> TensorType:
                x_ = tf.squeeze(x, -2)
                return tf.cast(tf.logical_and(0.0 <= x_, x_ < 1.0), x.dtype)

            return acquisition

    data = {"foo": Dataset(tf.constant([[-2.0], [1.0]]), tf.constant([[4.0], [1.0]]))}
    models_ = {"foo": QuadraticMeanAndRBFKernel()}
    eci = ExpectedConstrainedImprovement("foo", _Constraint()).prepare_acquisition_function(
        data, models_
    )

    constraint_fn = _Constraint().prepare_acquisition_function(data, models_)

    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    npt.assert_allclose(eci(xs), constraint_fn(xs))


def test_expected_constrained_improvement_min_feasibility_probability_bound_is_inclusive() -> None:
    def pof(x_: TensorType) -> TensorType:
        return tfp.bijectors.Sigmoid().forward(tf.squeeze(x_, -2))

    class _Constraint(AcquisitionFunctionBuilder):
        def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
        ) -> AcquisitionFunction:
            return pof

    models_ = {"foo": QuadraticMeanAndRBFKernel()}

    data = {"foo": Dataset(tf.constant([[1.1], [2.0]]), tf.constant([[1.21], [4.0]]))}
    eci = ExpectedConstrainedImprovement(
        "foo", _Constraint(), min_feasibility_probability=tfp.bijectors.Sigmoid().forward(1.0)
    ).prepare_acquisition_function(data, models_)

    ei = ExpectedImprovement().using("foo").prepare_acquisition_function(data, models_)
    x = tf.constant([[1.5]])
    npt.assert_allclose(eci(x), ei(x) * pof(x))


def _mo_test_model(num_obj: int, *kernel_amplitudes: float | TensorType | None) -> GaussianProcess:
    means = [quadratic, lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), quadratic]
    kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(k_amp) for k_amp in kernel_amplitudes]
    return GaussianProcess(means[:num_obj], kernels[:num_obj])


def test_ehvi_builder_raises_for_empty_data() -> None:
    num_obj = 3
    dataset = empty_dataset([2], [num_obj])
    model = QuadraticMeanAndRBFKernel()

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ExpectedHypervolumeImprovement().prepare_acquisition_function(dataset, model)


def test_ehvi_builder_builds_expected_hv_improvement_using_pareto_from_model() -> None:
    num_obj = 2
    train_x = tf.constant([[-2.0], [-1.5], [-1.0], [0.0], [0.5], [1.0], [1.5], [2.0]])
    dataset = Dataset(
        train_x,
        tf.tile(
            tf.constant([[4.1], [0.9], [1.2], [0.1], [-8.8], [1.1], [2.1], [3.9]]), [1, num_obj]
        ),
    )

    model = _mo_test_model(num_obj, *[None] * num_obj)
    acq_fn = ExpectedHypervolumeImprovement().prepare_acquisition_function(dataset, model)

    model_pred_observation = model.predict(train_x)[0]
    _prt = Pareto(model_pred_observation)
    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    expected = expected_hv_improvement(model, _prt, get_reference_point(_prt.front))(xs)
    npt.assert_allclose(acq_fn(xs), expected)


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_ehvi_raises_for_invalid_batch_size(at: TensorType) -> None:
    num_obj = 2
    train_x = tf.constant([[-2.0], [-1.5], [-1.0], [0.0], [0.5], [1.0], [1.5], [2.0]])

    model = _mo_test_model(num_obj, *[None] * num_obj)
    model_pred_observation = model.predict(train_x)[0]
    _prt = Pareto(model_pred_observation)
    ehvi = expected_hv_improvement(model, _prt, get_reference_point(_prt.front))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ehvi(at)


@random_seed
@pytest.mark.parametrize(
    "input_dim, num_samples_per_point, existing_observations, obj_num, variance_scale",
    [
        pytest.param(
            1,
            100_000,
            tf.constant([[0.3, 0.2], [0.2, 0.22], [0.1, 0.25], [0.0, 0.3]]),
            2,
            1.0,
            id="1d_input_2obj_gp_var_1",
        ),
        pytest.param(
            1,
            200_000,
            tf.constant([[0.3, 0.2], [0.2, 0.22], [0.1, 0.25], [0.0, 0.3]]),
            2,
            2.0,
            id="1d_input_2obj_gp_var_2",
        ),
        pytest.param(2, 50_000, tf.constant([[0.0, 0.0]]), 2, 1.0, id="2d_input_2obj_gp_var_2"),
        pytest.param(
            3,
            50_000,
            tf.constant([[2.0, 1.0], [0.8, 3.0]]),
            2,
            1.0,
            id="3d_input_2obj_gp_var_1",
        ),
        pytest.param(
            4,
            100_000,
            tf.constant([[3.0, 2.0, 1.0], [1.1, 2.0, 3.0]]),
            3,
            1.0,
            id="4d_input_3obj_gp_var_1",
        ),
    ],
)
def test_expected_hypervolume_improvement(
    input_dim: int,
    num_samples_per_point: int,
    existing_observations: tf.Tensor,
    obj_num: int,
    variance_scale: float,
) -> None:
    # Note: the test data number grows exponentially with num of obj
    data_num_seg_per_dim = 2  # test data number per input dim
    N = data_num_seg_per_dim ** input_dim
    xs = tf.convert_to_tensor(
        list(itertools.product(*[list(tf.linspace(-1, 1, data_num_seg_per_dim))] * input_dim))
    )

    xs = tf.cast(xs, dtype=existing_observations.dtype)
    model = _mo_test_model(obj_num, *[variance_scale] * obj_num)
    mean, variance = model.predict(xs)

    predict_samples = tfp.distributions.Normal(mean, tf.sqrt(variance)).sample(
        num_samples_per_point  # [f_samples, batch_size, obj_num]
    )
    _pareto = Pareto(existing_observations)
    ref_pt = get_reference_point(_pareto.front)
    lb_points, ub_points = _pareto.hypercell_bounds(
        tf.constant([-math.inf] * ref_pt.shape[-1]), ref_pt
    )

    # calc MC approx EHVI
    splus_valid = tf.reduce_all(
        tf.tile(ub_points[tf.newaxis, :, tf.newaxis, :], [num_samples_per_point, 1, N, 1])
        > tf.expand_dims(predict_samples, axis=1),
        axis=-1,  # can predict_samples contribute to hvi in cell
    )  # [f_samples, num_cells,  B]
    splus_idx = tf.expand_dims(tf.cast(splus_valid, dtype=ub_points.dtype), -1)
    splus_lb = tf.tile(lb_points[tf.newaxis, :, tf.newaxis, :], [num_samples_per_point, 1, N, 1])
    splus_lb = tf.maximum(  # max of lower bounds and predict_samples
        splus_lb, tf.expand_dims(predict_samples, 1)
    )
    splus_ub = tf.tile(ub_points[tf.newaxis, :, tf.newaxis, :], [num_samples_per_point, 1, N, 1])
    splus = tf.concat(  # concatenate validity labels and possible improvements
        [splus_idx, splus_ub - splus_lb], axis=-1
    )

    # calculate hyper-volume improvement over the non-dominated cells
    ehvi_approx = tf.transpose(tf.reduce_sum(tf.reduce_prod(splus, axis=-1), axis=1, keepdims=True))
    ehvi_approx = tf.reduce_mean(ehvi_approx, axis=-1)  # average through mc sample

    ehvi = expected_hv_improvement(model, _pareto, ref_pt)(tf.expand_dims(xs, -2))

    npt.assert_allclose(ehvi, ehvi_approx, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("sample_size", [0, -2])
def test_independent_reparametrization_sampler_raises_for_invalid_sample_size(
    sample_size: int,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        IndependentReparametrizationSampler(sample_size, QuadraticMeanAndRBFKernel())


@pytest.mark.parametrize("shape", [[], [1], [2], [2, 3]])
def test_independent_reparametrization_sampler_sample_raises_for_invalid_at_shape(
    shape: ShapeLike,
) -> None:
    sampler = IndependentReparametrizationSampler(1, QuadraticMeanAndRBFKernel())

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.zeros(shape))


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


def _dim_two_gp(mean_shift: tuple[float, float] = (0.0, 0.0)) -> GaussianProcess:
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
    x = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)

    model = _dim_two_gp()
    samples = IndependentReparametrizationSampler(sample_size, model).sample(x)  # [N, S, 1, L]

    assert samples.shape == [len(x), sample_size, 1, 2]

    mean, var = model.predict(tf.squeeze(x, -2))  # [N, L], [N, L]
    _assert_kolmogorov_smirnov_95(
        tf.linalg.matrix_transpose(tf.squeeze(samples, -2)),
        tfp.distributions.Normal(mean[..., None], tf.sqrt(var[..., None])),
    )


@random_seed
def test_independent_reparametrization_sampler_sample_is_continuous() -> None:
    sampler = IndependentReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_array_less(tf.abs(sampler.sample(xs + 1e-20) - sampler.sample(xs)), 1e-20)


def test_independent_reparametrization_sampler_sample_is_repeatable() -> None:
    sampler = IndependentReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))


@random_seed
def test_independent_reparametrization_sampler_samples_are_distinct_for_new_instances() -> None:
    sampler1 = IndependentReparametrizationSampler(100, _dim_two_gp())
    sampler2 = IndependentReparametrizationSampler(100, _dim_two_gp())
    xs = tf.random.uniform([100, 1, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_array_less(1e-9, tf.abs(sampler2.sample(xs) - sampler1.sample(xs)))


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
    npt.assert_allclose(batch_ei(xs), ei(xs), rtol=0.03)


@random_seed
def test_batch_monte_carlo_expected_improvement() -> None:
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    model = QuadraticMeanAndRBFKernel()

    mean, cov = model.predict_joint(xs)
    mvn = tfp.distributions.MultivariateNormalFullCovariance(tf.linalg.matrix_transpose(mean), cov)
    mvn_samples = mvn.sample(10_000)
    min_predictive_mean_at_known_points = 0.09
    # fmt: off
    expected = tf.reduce_mean(tf.reduce_max(tf.maximum(
        min_predictive_mean_at_known_points - mvn_samples, 0.0
    ), axis=-1), axis=0)
    # fmt: on

    builder = BatchMonteCarloExpectedImprovement(10_000)
    acq = builder.prepare_acquisition_function(mk_dataset([[0.3], [0.5]], [[0.09], [0.25]]), model)

    npt.assert_allclose(acq(xs), expected, rtol=0.05)

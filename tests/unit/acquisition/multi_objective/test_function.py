# Copyright 2021 The Trieste Contributors
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
from typing import Mapping, Optional, cast

import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.acquisition.sampler import PseudoBatchReparametrizationSampler
from tests.util.misc import (
    TF_DEBUGGING_ERROR_TYPES,
    empty_dataset,
    quadratic,
    raise_exc,
    random_seed,
)
from tests.util.models.gpflow.models import (
    GaussianProcess,
    GaussianProcessWithSamplers,
    QuadraticMeanAndRBFKernel,
)
from trieste.acquisition import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    ProbabilityOfFeasibility,
)
from trieste.acquisition.function.multi_objective import (
    BatchMonteCarloExpectedHypervolumeImprovement,
    ExpectedConstrainedHypervolumeImprovement,
    ExpectedHypervolumeImprovement,
    batch_ehvi,
    expected_hv_improvement,
)
from trieste.acquisition.multi_objective.pareto import Pareto, get_reference_point
from trieste.acquisition.multi_objective.partition import (
    ExactPartition2dNonDominated,
    prepare_default_non_dominated_partition_bounds,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel, ReparametrizationSampler
from trieste.types import TensorType
from trieste.utils import DEFAULTS


def _mo_test_model(
    num_obj: int, *kernel_amplitudes: float | TensorType | None, with_reparam_sampler: bool = True
) -> GaussianProcess:
    means = [quadratic, lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), quadratic]
    kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(k_amp) for k_amp in kernel_amplitudes]
    if with_reparam_sampler:
        return GaussianProcessWithSamplers(means[:num_obj], kernels[:num_obj])
    else:
        return GaussianProcess(means[:num_obj], kernels[:num_obj])


def test_ehvi_builder_raises_for_empty_data() -> None:
    num_obj = 3
    dataset = empty_dataset([2], [num_obj])
    model = QuadraticMeanAndRBFKernel()

    with pytest.raises(tf.errors.InvalidArgumentError):
        ExpectedHypervolumeImprovement().prepare_acquisition_function(model, dataset=dataset)
    with pytest.raises(tf.errors.InvalidArgumentError):
        ExpectedHypervolumeImprovement().prepare_acquisition_function(model, dataset)


def test_ehvi_builder_builds_expected_hv_improvement_using_pareto_from_model() -> None:
    num_obj = 2
    train_x = tf.constant([[-2.0], [-1.5], [-1.0], [0.0], [0.5], [1.0], [1.5], [2.0]])
    dataset = Dataset(
        train_x,
        tf.tile(
            tf.constant([[4.1], [0.9], [1.2], [0.1], [-8.8], [1.1], [2.1], [3.9]]), [1, num_obj]
        ),
    )

    model = _mo_test_model(num_obj, *[10, 10] * num_obj)
    acq_fn = ExpectedHypervolumeImprovement().prepare_acquisition_function(model, dataset=dataset)

    model_pred_observation = model.predict(train_x)[0]
    _prt = Pareto(model_pred_observation)
    _partition_bounds = ExactPartition2dNonDominated(_prt.front).partition_bounds(
        tf.constant([-1e10] * 2), get_reference_point(_prt.front)
    )
    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    expected = expected_hv_improvement(model, _partition_bounds)(xs)
    npt.assert_allclose(acq_fn(xs), expected)


def test_ehvi_builder_updates_expected_hv_improvement_using_pareto_from_model() -> None:
    num_obj = 2
    train_x = tf.constant([[-2.0], [-1.5], [-1.0], [0.0], [0.5], [1.0], [1.5], [2.0]])
    dataset = Dataset(
        train_x,
        tf.tile(
            tf.constant([[4.1], [0.9], [1.2], [0.1], [-8.8], [1.1], [2.1], [3.9]]), [1, num_obj]
        ),
    )
    partial_dataset = Dataset(dataset.query_points[:4], dataset.observations[:4])
    xs = tf.linspace([[-10.0]], [[10.0]], 100)

    model = _mo_test_model(num_obj, *[10, 10] * num_obj)
    acq_fn = ExpectedHypervolumeImprovement().prepare_acquisition_function(
        model, dataset=partial_dataset
    )
    assert acq_fn.__call__._get_tracing_count() == 0  # type: ignore
    model_pred_observation = model.predict(train_x)[0]
    _prt = Pareto(model_pred_observation)
    _partition_bounds = ExactPartition2dNonDominated(_prt.front).partition_bounds(
        tf.constant([-1e10] * 2), get_reference_point(_prt.front)
    )
    expected = expected_hv_improvement(model, _partition_bounds)(xs)
    npt.assert_allclose(acq_fn(xs), expected)
    assert acq_fn.__call__._get_tracing_count() == 1  # type: ignore

    # update the acquisition function, evaluate it, and check that it hasn't been retraced
    updated_acq_fn = ExpectedHypervolumeImprovement().update_acquisition_function(
        acq_fn,
        model,
        dataset=dataset,
    )
    assert updated_acq_fn == acq_fn
    model_pred_observation = model.predict(train_x)[0]
    _prt = Pareto(model_pred_observation)
    _partition_bounds = ExactPartition2dNonDominated(_prt.front).partition_bounds(
        tf.constant([-1e10] * 2), get_reference_point(_prt.front)
    )
    expected = expected_hv_improvement(model, _partition_bounds)(xs)
    npt.assert_allclose(acq_fn(xs), expected)
    assert acq_fn.__call__._get_tracing_count() == 1  # type: ignore


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_ehvi_raises_for_invalid_batch_size(at: TensorType) -> None:
    num_obj = 2
    train_x = tf.constant([[-2.0], [-1.5], [-1.0], [0.0], [0.5], [1.0], [1.5], [2.0]])

    model = _mo_test_model(num_obj, *[None] * num_obj)
    model_pred_observation = model.predict(train_x)[0]
    _prt = Pareto(model_pred_observation)
    _partition_bounds = ExactPartition2dNonDominated(_prt.front).partition_bounds(
        tf.constant([-math.inf] * 2), get_reference_point(_prt.front)
    )
    ehvi = expected_hv_improvement(model, _partition_bounds)

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
def test_expected_hypervolume_improvement_matches_monte_carlo(
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
    lb_points, ub_points = prepare_default_non_dominated_partition_bounds(ref_pt, _pareto.front)

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

    ehvi = expected_hv_improvement(model, (lb_points, ub_points))(tf.expand_dims(xs, -2))

    npt.assert_allclose(ehvi, ehvi_approx, rtol=0.01, atol=0.01)


def test_qehvi_builder_raises_for_empty_data() -> None:
    num_obj = 3
    dataset = empty_dataset([2], [num_obj])
    model = QuadraticMeanAndRBFKernel()

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMonteCarloExpectedHypervolumeImprovement(sample_size=100).prepare_acquisition_function(
            model,
            dataset=dataset,
        )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMonteCarloExpectedHypervolumeImprovement(sample_size=100).prepare_acquisition_function(
            model,
        )


def test_batch_monte_carlo_expected_hypervolume_improvement_builder_raises_for_empty_data() -> None:
    num_obj = 3
    dataset = empty_dataset([2], [num_obj])
    model = QuadraticMeanAndRBFKernel()

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMonteCarloExpectedHypervolumeImprovement(sample_size=100).prepare_acquisition_function(
            model,
            dataset=dataset,
        )
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMonteCarloExpectedHypervolumeImprovement(sample_size=100).prepare_acquisition_function(
            model,
        )


@pytest.mark.parametrize("sample_size", [-2, 0])
def test_batch_monte_carlo_expected_hypervolume_improvement_raises_for_invalid_sample_size(
    sample_size: int,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMonteCarloExpectedHypervolumeImprovement(sample_size)


def test_batch_monte_carlo_expected_hypervolume_improvement_raises_for_invalid_jitter() -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMonteCarloExpectedHypervolumeImprovement(100, jitter=-1.0)


def test_batch_monte_carlo_ehvi_raises_for_model_without_reparam_sampler() -> None:

    model = _mo_test_model(2, *[1.0] * 2, with_reparam_sampler=False)

    training_input = tf.constant([[0.3], [0.22], [0.1], [0.35]])
    mean, _ = model.predict(training_input)  # gen prepare Pareto
    _model_based_tr_dataset = Dataset(training_input, mean)

    qehvi_builder = BatchMonteCarloExpectedHypervolumeImprovement(sample_size=10)

    with pytest.raises(ValueError):
        qehvi_builder.prepare_acquisition_function(model, dataset=_model_based_tr_dataset)


@random_seed
@pytest.mark.parametrize(
    "input_dim, num_samples_per_point, training_input, obj_num, variance_scale",
    [
        pytest.param(
            1,
            50_000,
            tf.constant([[0.3], [0.22], [0.1], [0.35]]),
            2,
            1.0,
            id="1d_input_2obj_model_var_1_q_1",
        ),
        pytest.param(
            1,
            50_000,
            tf.constant([[0.3], [0.22], [0.1], [0.35]]),
            2,
            2.0,
            id="1d_input_2obj_model_var_2_q_1",
        ),
        pytest.param(
            2,
            50_000,
            tf.constant([[0.0, 0.0], [0.2, 0.5]]),
            2,
            1.0,
            id="2d_input_2obj_model_var_1_q_1",
        ),
        pytest.param(
            3,
            25_000,
            tf.constant([[0.0, 0.0, 0.2], [-0.2, 0.5, -0.1], [0.2, -0.5, 0.2]]),
            3,
            1.0,
            id="3d_input_3obj_model_var_1_q_1",
        ),
    ],
)
def test_batch_monte_carlo_expected_hypervolume_improvement_can_reproduce_ehvi(
    input_dim: int,
    num_samples_per_point: int,
    training_input: tf.Tensor,
    obj_num: int,
    variance_scale: float,
) -> None:
    data_num_seg_per_dim = 10  # test data number per input dim

    model = _mo_test_model(obj_num, *[variance_scale] * obj_num)

    mean, _ = model.predict(training_input)  # gen prepare Pareto
    _model_based_tr_dataset = Dataset(training_input, mean)

    _model_based_pareto = Pareto(mean)
    _reference_pt = get_reference_point(_model_based_pareto.front)
    _partition_bounds = prepare_default_non_dominated_partition_bounds(
        _reference_pt, _model_based_pareto.front
    )

    qehvi_builder = BatchMonteCarloExpectedHypervolumeImprovement(sample_size=num_samples_per_point)
    qehvi_acq = qehvi_builder.prepare_acquisition_function(model, dataset=_model_based_tr_dataset)
    ehvi_acq = expected_hv_improvement(model, _partition_bounds)

    test_xs = tf.convert_to_tensor(
        list(itertools.product(*[list(tf.linspace(-1, 1, data_num_seg_per_dim))] * input_dim)),
        dtype=training_input.dtype,
    )  # [test_num, input_dim]
    test_xs = tf.expand_dims(test_xs, -2)  # add Batch dim: q=1

    npt.assert_allclose(ehvi_acq(test_xs), qehvi_acq(test_xs), rtol=1e-2, atol=1e-2)


@random_seed
@pytest.mark.parametrize(
    "test_input, obj_samples, pareto_front_obs, reference_point, expected_output",
    [
        pytest.param(
            tf.zeros(shape=(1, 2, 1)),
            tf.constant([[[-6.5, -4.5], [-7.0, -4.0]]]),
            tf.constant([[-4.0, -5.0], [-5.0, -5.0], [-8.5, -3.5], [-8.5, -3.0], [-9.0, -1.0]]),
            tf.constant([0.0, 0.0]),
            tf.constant([[1.75]]),
            id="q_2, both points contribute",
        ),
        pytest.param(
            tf.zeros(shape=(1, 2, 1)),
            tf.constant([[[-6.5, -4.5], [-6.0, -4.0]]]),
            tf.constant([[-4.0, -5.0], [-5.0, -5.0], [-8.5, -3.5], [-8.5, -3.0], [-9.0, -1.0]]),
            tf.constant([0.0, 0.0]),
            tf.constant([[1.5]]),
            id="q_2, only 1 point contributes",
        ),
        pytest.param(
            tf.zeros(shape=(1, 2, 1)),
            tf.constant([[[-2.0, -2.0], [0.0, -0.1]]]),
            tf.constant([[-4.0, -5.0], [-5.0, -5.0], [-8.5, -3.5], [-8.5, -3.0], [-9.0, -1.0]]),
            tf.constant([0.0, 0.0]),
            tf.constant([[0.0]]),
            id="q_2, neither contributes",
        ),
        pytest.param(
            tf.zeros(shape=(1, 2, 1)),
            tf.constant([[[-6.5, -4.5], [-9.0, -2.0]]]),
            tf.constant([[-4.0, -5.0], [-5.0, -5.0], [-8.5, -3.5], [-8.5, -3.0], [-9.0, -1.0]]),
            tf.constant([0.0, 0.0]),
            tf.constant([[2.0]]),
            id="obj_2_q_2, test input better than current-best first objective",
        ),
        pytest.param(
            tf.zeros(shape=(1, 2, 1)),
            tf.constant([[[-6.5, -4.5], [-6.0, -6.0]]]),
            tf.constant([[-4.0, -5.0], [-5.0, -5.0], [-8.5, -3.5], [-8.5, -3.0], [-9.0, -1.0]]),
            tf.constant([0.0, 0.0]),
            tf.constant([[8.0]]),
            id="obj_2_q_2, test input better than current best second objective",
        ),
        pytest.param(
            tf.zeros(shape=(1, 3, 1)),
            tf.constant([[[-6.5, -4.5], [-9.0, -2.0], [-7.0, -4.0]]]),
            tf.constant([[-4.0, -5.0], [-5.0, -5.0], [-8.5, -3.5], [-8.5, -3.0], [-9.0, -1.0]]),
            tf.constant([0.0, 0.0]),
            tf.constant([[2.25]]),
            id="obj_2_q_3, all points contribute",
        ),
        pytest.param(
            tf.zeros(shape=(1, 3, 1)),
            tf.constant([[[-6.5, -4.5], [-9.0, -2.0], [-7.0, -5.0]]]),
            tf.constant([[-4.0, -5.0], [-5.0, -5.0], [-8.5, -3.5], [-8.5, -3.0], [-9.0, -1.0]]),
            tf.constant([0.0, 0.0]),
            tf.constant([[3.5]]),
            id="obj_2_q_3, not all points contribute",
        ),
        pytest.param(
            tf.zeros(shape=(1, 3, 1)),
            tf.constant([[[-0.0, -4.5], [-1.0, -2.0], [-3.0, -0.0]]]),
            tf.constant([[-4.0, -5.0], [-5.0, -5.0], [-8.5, -3.5], [-8.5, -3.0], [-9.0, -1.0]]),
            tf.constant([0.0, 0.0]),
            tf.constant([[0.0]]),
            id="obj_2_q_3, none contribute",
        ),
        pytest.param(
            tf.zeros(shape=(1, 2, 1)),
            tf.constant([[[-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0]]]),
            tf.constant([[-4.0, -2.0, -3.0], [-3.0, -5.0, -1.0], [-2.0, -4.0, -2.0]]),
            tf.constant([1.0, 1.0, 1.0]),
            tf.constant([[0.0]]),
            id="obj_3_q_2, none contribute",
        ),
        pytest.param(
            tf.zeros(shape=(1, 2, 1)),
            tf.constant([[[-1.0, -2.0, -6.0], [-1.0, -3.0, -4.0]]]),
            tf.constant([[-4.0, -2.0, -3.0], [-3.0, -5.0, -1.0], [-2.0, -4.0, -2.0]]),
            tf.constant([1.0, 1.0, 1.0]),
            tf.constant([[22.0]]),
            id="obj_3_q_2, all points contribute",
        ),
        pytest.param(
            tf.zeros(shape=(1, 2, 1)),
            tf.constant(
                [[[-2.0, -3.0, -7.0], [-2.0, -4.0, -5.0]], [[-1.0, -2.0, -6.0], [-1.0, -3.0, -4.0]]]
            ),
            tf.constant([[-4.0, -2.0, -3.0], [-3.0, -5.0, -1.0], [-2.0, -4.0, -2.0]]),
            tf.constant([1.0, 1.0, 1.0]),
            tf.constant([[41.0]]),
            id="obj_3_q_2, mc sample size=2",
        ),
    ],
)
def test_batch_monte_carlo_expected_hypervolume_improvement_utility_on_specified_samples(
    test_input: TensorType,
    obj_samples: TensorType,
    pareto_front_obs: TensorType,
    reference_point: TensorType,
    expected_output: TensorType,
) -> None:
    npt.assert_allclose(
        batch_ehvi(
            cast(
                ReparametrizationSampler[ProbabilisticModel],
                PseudoBatchReparametrizationSampler(obj_samples),
            ),
            sampler_jitter=DEFAULTS.JITTER,
            partition_bounds=prepare_default_non_dominated_partition_bounds(
                reference_point, Pareto(pareto_front_obs).front
            ),
        )(test_input),
        expected_output,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_expected_constrained_hypervolume_improvement_raises_for_invalid_batch_size(
    at: TensorType,
) -> None:
    pof = ProbabilityOfFeasibility(0.0).using("")
    builder = ExpectedConstrainedHypervolumeImprovement("", pof, tf.constant(0.5))
    initial_query_points = tf.constant([[-1.0]])
    initial_objective_function_values = tf.constant([[1.0, 1.0]])
    data = {"": Dataset(initial_query_points, initial_objective_function_values)}

    echvi = builder.prepare_acquisition_function({"": QuadraticMeanAndRBFKernel()}, datasets=data)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        echvi(at)


def test_expected_constrained_hypervolume_improvement_can_reproduce_ehvi() -> None:
    num_obj = 2
    train_x = tf.constant([[-2.0], [-1.5], [-1.0], [0.0], [0.5], [1.0], [1.5], [2.0]])

    obj_model = _mo_test_model(num_obj, *[None] * num_obj)
    model_pred_observation = obj_model.predict(train_x)[0]

    class _Certainty(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> AcquisitionFunction:
            return lambda x: tf.ones_like(tf.squeeze(x, -2))

    data = {"foo": Dataset(train_x[:5], model_pred_observation[:5])}
    models_ = {"foo": obj_model}

    builder = ExpectedConstrainedHypervolumeImprovement("foo", _Certainty(), 0)
    echvi = builder.prepare_acquisition_function(models_, datasets=data)

    ehvi = (
        ExpectedHypervolumeImprovement()
        .using("foo")
        .prepare_acquisition_function(models_, datasets=data)
    )

    at = tf.constant([[[-0.1]], [[1.23]], [[-6.78]]])
    npt.assert_allclose(echvi(at), ehvi(at))

    new_data = {"foo": Dataset(train_x, model_pred_observation)}
    up_echvi = builder.update_acquisition_function(echvi, models_, datasets=new_data)
    assert up_echvi == echvi
    up_ehvi = (
        ExpectedHypervolumeImprovement()
        .using("foo")
        .prepare_acquisition_function(models_, datasets=new_data)
    )

    npt.assert_allclose(up_echvi(at), up_ehvi(at))
    assert up_echvi._get_tracing_count() == 1  # type: ignore


def test_echvi_is_constraint_when_no_feasible_points() -> None:
    class _Constraint(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> AcquisitionFunction:
            def acquisition(x: TensorType) -> TensorType:
                x_ = tf.squeeze(x, -2)
                return tf.cast(tf.logical_and(0.0 <= x_, x_ < 1.0), x.dtype)

            return acquisition

    data = {"foo": Dataset(tf.constant([[-2.0], [1.0]]), tf.constant([[4.0], [1.0]]))}
    models_ = {"foo": QuadraticMeanAndRBFKernel()}
    echvi = ExpectedConstrainedHypervolumeImprovement(
        "foo", _Constraint()
    ).prepare_acquisition_function(models_, datasets=data)

    constraint_fn = _Constraint().prepare_acquisition_function(models_, datasets=data)

    xs = tf.linspace([[-10.0]], [[10.0]], 100)
    npt.assert_allclose(echvi(xs), constraint_fn(xs))


def test_echvi_raises_for_non_scalar_min_pof() -> None:
    pof = ProbabilityOfFeasibility(0.0).using("")
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ExpectedConstrainedHypervolumeImprovement("", pof, tf.constant([0.0]))


def test_echvi_raises_for_out_of_range_min_pof() -> None:
    pof = ProbabilityOfFeasibility(0.0).using("")
    with pytest.raises(tf.errors.InvalidArgumentError):
        ExpectedConstrainedHypervolumeImprovement("", pof, 1.5)


def test_echvi_raises_for_empty_data() -> None:
    class _Constraint(AcquisitionFunctionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            models: Mapping[str, ProbabilisticModel],
            datasets: Optional[Mapping[str, Dataset]] = None,
        ) -> AcquisitionFunction:
            return raise_exc

    data = {"foo": Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))}
    models_ = {"foo": QuadraticMeanAndRBFKernel()}
    builder = ExpectedConstrainedHypervolumeImprovement("foo", _Constraint())

    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(models_, datasets=data)
    with pytest.raises(tf.errors.InvalidArgumentError):
        builder.prepare_acquisition_function(models_)

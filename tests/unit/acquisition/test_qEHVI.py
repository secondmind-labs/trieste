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
from trieste.acquisition.multiobjective.qEHVI import BatchMonteCarloHypervolumeExpectedImprovement

from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.type import TensorType
from trieste.utils.objectives import branin


# 自己写的开始:
# ------------------------------------------------
# Test Code for batch_multimodel_reparametrization_sampler
def _dim_one_gp(mean_shift: float = 0.0) -> GaussianProcess:
    matern52 = tfp.math.psd_kernels.MaternFiveHalves(
        amplitude=tf.cast(2.3, tf.float64), length_scale=tf.cast(0.5, tf.float64)
    )
    return GaussianProcess(
        [lambda x: mean_shift + branin(x)],
        [matern52],
    )


from trieste.acquisition.multiobjective.qEHVI import BatchMultiModelReparametrizationSampler


@pytest.mark.parametrize("sample_size", [0, -2])
def test_batch_multimodel_reparametrization_sampler_raises_for_invalid_sample_size(sample_size: int) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMultiModelReparametrizationSampler(sample_size, {'OBJECTIVE1': _dim_one_gp(), 'OBJECTIVE2': _dim_one_gp()})


@random_seed
def test_batch_multimodel_reparametrization_sampler_samples_approximate_mean_and_covariance() -> None:
    models = {'OBJECTIVE1': _dim_one_gp(2.5), 'OBJECTIVE2': QuadraticMeanAndRBFKernel()}
    sample_size = 10_000
    leading_dims = [3]
    batch_size = 4
    gp_input_dim = 2
    gp_output_dim = 1
    model_num = 2
    # xs = tf.random.uniform(leading_dims + [batch_size, gp_output_dim], maxval=1.0, dtype=tf.float64)
    xs = tf.random.uniform(leading_dims + [batch_size, gp_input_dim], maxval=1.0, dtype=tf.float64)
    samples = BatchMultiModelReparametrizationSampler(sample_size, models).sample(
        xs)

    assert samples.shape == leading_dims + [sample_size, batch_size, model_num * gp_output_dim]

    samples_mean = tf.reduce_mean(samples, axis=-3)
    samples_covariance = tf.transpose(
        tfp.stats.covariance(samples, sample_axis=-3, event_axis=-2), [0, 3, 1, 2]
    )

    # model_mean, model_cov = model.predict_joint(xs)
    predicts = [models[model_tag].predict_joint(xs) for model_tag in models]
    means, covs = list(zip(*predicts))
    means = tf.concat(means, axis=-1)  # [..., B, num_obj]
    covs = tf.concat(covs, axis=1)  # [num_obj, B, B]

    npt.assert_allclose(samples_mean, means, rtol=0.02)
    npt.assert_allclose(samples_covariance, covs, rtol=0.04)


def test_batch_multimodel_reparametrization_sampler_samples_are_continuous() -> None:
    models = {'OBJECTIVE1': _dim_one_gp(2.5), 'OBJECTIVE2': _dim_one_gp(5.7)}
    sampler = BatchMultiModelReparametrizationSampler(100, models)
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    npt.assert_array_less(tf.abs(sampler.sample(xs + 1e-20) - sampler.sample(xs)), 1e-20)


def test_batch_multimodel_reparametrization_sampler_samples_are_repeatable() -> None:
    models = {'OBJECTIVE1': _dim_one_gp(2.5), 'OBJECTIVE2': _dim_one_gp(5.7)}
    sampler = BatchMultiModelReparametrizationSampler(100, models)
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))


@random_seed
def test_batch_multimodel_reparametrization_sampler_samples_are_distinct_for_new_instances() -> None:
    models = {'OBJECTIVE1': _dim_one_gp(2.5), 'OBJECTIVE2': _dim_one_gp(5.7)}
    sampler1 = BatchMultiModelReparametrizationSampler(100, models)
    sampler2 = BatchMultiModelReparametrizationSampler(100, models)
    xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)
    npt.assert_array_less(1e-9, tf.abs(sampler2.sample(xs) - sampler1.sample(xs)))


@pytest.mark.parametrize("at", [tf.constant([0.0]), tf.constant(0.0), tf.ones([0, 1])])
def test_batch_reparametrization_sampler_sample_raises_for_invalid_at_shape(at: tf.Tensor) -> None:
    models = {'OBJECTIVE1': QuadraticMeanAndRBFKernel(), 'OBJECTIVE2': QuadraticMeanAndRBFKernel()}
    sampler = BatchMultiModelReparametrizationSampler(100, models)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(at)


def test_batch_reparametrization_sampler_sample_raises_for_negative_jitter() -> None:
    models = {'OBJECTIVE1': QuadraticMeanAndRBFKernel(), 'OBJECTIVE2': QuadraticMeanAndRBFKernel()}
    sampler = BatchMultiModelReparametrizationSampler(100, models)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.constant([[0.0]]), jitter=-1e-6)


def test_batch_reparametrization_sampler_sample_raises_for_inconsistent_batch_size() -> None:
    models = {'OBJECTIVE1': QuadraticMeanAndRBFKernel(), 'OBJECTIVE2': QuadraticMeanAndRBFKernel()}
    sampler = BatchMultiModelReparametrizationSampler(100, models)
    sampler.sample(tf.constant([[0.0], [1.0], [2.0]]))

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.constant([[0.0], [1.0]]))


# Test Code for batch_HEVI
@pytest.mark.parametrize("sample_size", [-2, 0])
def test_batch_monte_carlo_hypervolume_expected_improvement_raises_for_invalid_sample_size(
        sample_size: int,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMonteCarloHypervolumeExpectedImprovement(sample_size)


def test_batch_monte_carlo_expected_hypervolume_improvement_raises_for_invalid_jitter() -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMonteCarloHypervolumeExpectedImprovement(100, jitter=-1.0)


def test_batch_monte_carlo_expected_hypervolume_improvement_raises_for_empty_data() -> None:
    builder = BatchMonteCarloHypervolumeExpectedImprovement(100)
    datastes = {f'OBJECTIVE_{i + 1}': Dataset(tf.zeros([0, 2]), tf.zeros([0, 1])) for i in range(2)}
    model = {f'OBJECTIVE_{i + 1}': QuadraticMeanAndRBFKernel() for i in range(2)}
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(datastes, model)


from tests.unit.acquisition.test_function import _dim_two_gp


def test_batch_monte_carlo_expected_hypervolume_improvement_raises_for_model_with_wrong_event_shape() -> None:
    builder = BatchMonteCarloHypervolumeExpectedImprovement(100)
    datastes = {f'OBJECTIVE_{i + 1}': mk_dataset([[0.0, 0.0]], [[0.0, 0.0]]) for i in range(2)}
    models = {f'OBJECTIVE_{i + 1}': _dim_two_gp() for i in range(2)}
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(datastes, models)


# TODO:
# @random_seed
# def test_batch_monte_carlo_expected_improvement_can_reproduce_ei() -> None:
#     known_query_points = tf.random.uniform([5, 2], dtype=tf.float64)
#     data = Dataset(known_query_points, quadratic(known_query_points))
#     model = QuadraticMeanAndRBFKernel()
#     batch_ei = BatchMonteCarloExpectedImprovement(10_000).prepare_acquisition_function(data, model)
#     ei = ExpectedImprovement().prepare_acquisition_function(data, model)
#     xs = tf.random.uniform([3, 5, 1, 2], dtype=tf.float64)
#     npt.assert_allclose(batch_ei(xs), ei(tf.squeeze(xs, -2)), rtol=0.03)


# # TODO:
# @random_seed
# def test_batch_monte_carlo_expected_hypervolume_improvement() -> None:
#     num_objective = 2
#     xs = tf.random.uniform([3, 5, 7, 2], dtype=tf.float64)  # [..., , B, out_dim]
#     models = {f'OBJECTIVE{i}': QuadraticMeanAndRBFKernel() for i in range(num_objective)}
#     datasets = {f'OBJECTIVE{i}': mk_dataset([[0.3], [0.5]], [[0.09], [0.25]]) for i in range(num_objective)}
#
#     predicts = [models[model_tag].predict_joint(xs) for model_tag in models]
#     means, covs = list(zip(*predicts))
#     mvs_samples = []
#     for i in range(len(models)):
#         mean = means[i]
#         cov = covs[i]
#         mvn = tfp.distributions.MultivariateNormalFullCovariance(tf.linalg.matrix_transpose(mean), cov)
#         mvn_sample = mvn.sample(10_000)
#         mvs_samples.append(mvn_sample)
#     # TODO: Handwritten HVEI
#     expected = None
#
#     builder = BatchMonteCarloHypervolumeExpectedImprovement(10_000, q=xs.shape[-2])
#     acq = builder.prepare_acquisition_function(datasets, models)
#
#     npt.assert_allclose(acq(xs), expected, rtol=0.05)


from trieste.acquisition.multiobjective.MC_EHVI import MonteCarloHypervolumeExpectedImprovement



def _dim_one_gp_branin(mean_shift: float = 0.0) -> GaussianProcess:
    matern52 = tfp.math.psd_kernels.MaternFiveHalves(
        amplitude=tf.cast(2.3, tf.float64), length_scale=tf.cast(0.5, tf.float64)
    )
    return GaussianProcess(
        [lambda x: mean_shift + branin(x)],
        [matern52],
    )

def _dim_one_gp_invbranin(mean_shift: float = 0.0) -> GaussianProcess:
    matern52 = tfp.math.psd_kernels.MaternFiveHalves(
        amplitude=tf.cast(2.3, tf.float64), length_scale=tf.cast(0.5, tf.float64)
    )
    return GaussianProcess(
        [lambda x: mean_shift + 1/branin(x)],
        [matern52],
    )

@random_seed
def test_batch_monte_carlo_expected_hypervolume_improvement_on_single_point() -> None:
    num_objective = 2
    xs = tf.random.uniform([20, 1, 2], minval=-2, maxval=2, dtype=tf.float64)  # [..., , B, out_dim]

    # models = {f'OBJECTIVE{i}': _dim_one_gp(i) for i in range(num_objective)}
    models = {'OBJECTIVE0': _dim_one_gp_branin(), 'OBJECTIVE1': _dim_one_gp_invbranin()}
    # models = {'OBJECTIVE0': QuadraticMeanAndRBFKernel(0.1), 'OBJECTIVE1': QuadraticMeanAndRBFKernel(101.0)}
    _tr_xs = tf.random.normal(shape=(10, 2))
    datasets = {f'OBJECTIVE{i}': mk_dataset(_tr_xs, tf.random.normal(shape=(10, 1), dtype=tf.float64))
                for i in range(num_objective)}

    builder = BatchMonteCarloHypervolumeExpectedImprovement(100_00, q=1)
    acq = builder.prepare_acquisition_function(datasets, models)

    builder2 = MonteCarloHypervolumeExpectedImprovement(10000)
    acq2 = builder2.prepare_acquisition_function(datasets, models)

    npt.assert_allclose(acq(xs), acq2(tf.squeeze(xs)), rtol=0.05)

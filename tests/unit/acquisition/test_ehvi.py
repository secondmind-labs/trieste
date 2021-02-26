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

import numpy as np
import itertools
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import (
    TF_DEBUGGING_ERROR_TYPES,
    random_seed,
)
from tests.util.model import QuadraticMeanAndRBFKernel, LinearMeanAndRBFKernel
from trieste.acquisition.multiobjective.analytic import Expected_Hypervolume_Improvement, expected_hv_improvement
from trieste.data import Dataset
from trieste.utils.pareto import Pareto
from trieste.acquisition.multiobjective.function import get_nadir_point

test_models = (QuadraticMeanAndRBFKernel, LinearMeanAndRBFKernel, QuadraticMeanAndRBFKernel)


def test_expected_hypervolume_improvement_builder_raises_for_empty_data() -> None:
    num_obj = 3
    datasets = {f'OBJECTIVE{i + 1}': Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]),
    ) for i in range(num_obj)}
    models = {f'OBJECTIVE{i + 1}': QuadraticMeanAndRBFKernel() for i in range(num_obj)}

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        Expected_Hypervolume_Improvement().prepare_acquisition_function(datasets, models)


def test_expected_hypervolume_improvement_builder_builds_expected_hypervolume_improvement_using_pareto_from_model() -> None:
    num_obj = 2
    train_x = tf.constant([[-2.0], [-1.5], [-1.0], [0.0], [0.5], [1.0], [1.5], [2.0]])
    datasets = {f'OBJECTIVE{i + 1}': Dataset(
        train_x,
        tf.constant([[4.1], [0.9], [1.2], [0.1], [-8.8],  [1.1], [2.1], [3.9]]),
    ) for i in range(num_obj)}
    models = {f'OBJECTIVE{i + 1}': test_models[i]() for i in range(num_obj)}
    acq_fn = Expected_Hypervolume_Improvement().prepare_acquisition_function(datasets, models)

    model_pred_observation = tf.concat([models[model_tag].predict(train_x)[0]
                                        for model_tag in models], axis=-1)
    _prt = Pareto(Dataset(tf.zeros_like(model_pred_observation), model_pred_observation))
    xs = tf.linspace([-10.0], [10.0], 100)
    expected = expected_hv_improvement(models, xs, _prt, get_nadir_point(_prt.front))
    npt.assert_allclose(acq_fn(xs), expected)



@random_seed
@pytest.mark.parametrize(
    "input_dim, num_samples_per_point, training_observations, obj_num, rtol, atol",
    [
        (1, 50_000, tf.constant([[0.3, 0.2], [0.2, 0.22], [0.1, 0.25], [0.0, 0.3]]), 2, 0.01, 1e-2),
        (2, 50_000, tf.constant([[0.0, 0.0]]), 2, 0.01, 1e-2),
        (1, 100_000, tf.constant([[0.0, 0.0, 0.5], [0.4, -0.1, 0.2], [0.1, 0.1, 0.1], [0.3, 0.5, 0.0]]), 3, 0.01, 1e-2),
    ],
)
def test_expected_hypervolume_improvement(
    input_dim: int, num_samples_per_point: int, training_observations: tf.Tensor, obj_num: int, rtol: float, atol: float
) -> None:
    # Note: this exponentially with num of obj
    data_num_seg_per_dim = 10 # test data number per input dim
    N = data_num_seg_per_dim ** input_dim
    xs = tf.constant(list(itertools.product(*[list(np.linspace(-1, 1, data_num_seg_per_dim))] * input_dim)))

    xs = tf.cast(xs, dtype=training_observations.dtype)
    models = {f'OBJECTIVE{i}': test_models[i]() for i in range(obj_num)}

    predicts = [models[model_tag].predict(xs) for model_tag in models]
    mean, variance = (tf.concat(moment, 1) for moment in zip(*predicts))

    # [f_samples, B, L]
    predict_samples = tfp.distributions.Normal(mean, tf.sqrt(variance)).sample(num_samples_per_point)
    _pareto = Pareto(Dataset(tf.zeros_like(training_observations), training_observations))
    nadir = get_nadir_point(_pareto.front)
    lb_points, ub_points = _pareto.get_partitioned_cell_bounds(nadir)

    # calc MC EHVI
    splus_valid = tf.reduce_all(
        tf.tile(ub_points[tf.newaxis, :, tf.newaxis, :],
                [num_samples_per_point, 1, N, 1]) > tf.expand_dims(predict_samples, axis=1), axis=-1)  # num_cells x B
    splus_idx = tf.expand_dims(tf.cast(splus_valid, dtype=ub_points.dtype), -1)
    splus_lb = tf.tile(lb_points[tf.newaxis, :, tf.newaxis, :], [num_samples_per_point, 1, N, 1])
    splus_lb = tf.maximum(splus_lb, tf.expand_dims(predict_samples, 1))
    splus_ub = tf.tile(ub_points[tf.newaxis, :, tf.newaxis, :], [num_samples_per_point, 1, N, 1])  # 上界维持不变
    splus = tf.concat([splus_idx, splus_ub - splus_lb], axis=-1)

    ehvi_approx = tf.transpose(tf.reduce_sum(tf.reduce_prod(splus, axis=-1), axis=1, keepdims=True))  #
    ehvi_approx = tf.reduce_mean(ehvi_approx, axis=-1)

    ehvi = expected_hv_improvement(models, xs, _pareto, nadir)

    npt.assert_allclose(ehvi, ehvi_approx, rtol=rtol, atol=atol)
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
from math import inf
from tensorflow_probability import distributions as tfd

from tests.util.misc import (
    TF_DEBUGGING_ERROR_TYPES,
    random_seed,
)
from tests.util.model import QuadraticMeanAndRBFKernel, LinearMeanAndRBFKernel
from trieste.data import Dataset
from trieste.utils.pareto import Pareto
from trieste.acquisition.multiobjective.function import get_reference_point
from trieste.acquisition.multiobjective.analytic import Expected_Hypervolume_Improvement, expected_hv_improvement
from trieste.models.model_interfaces import ModelStack

test_models = (QuadraticMeanAndRBFKernel, LinearMeanAndRBFKernel, QuadraticMeanAndRBFKernel)


def test_expected_hypervolume_improvement_builder_raises_for_empty_data() -> None:
    num_obj = 3
    dataset = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))
    model = ModelStack(*[(QuadraticMeanAndRBFKernel(), 1) for _ in range(num_obj)])

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        Expected_Hypervolume_Improvement().prepare_acquisition_function(dataset, model)


def test_expected_hv_improvement_builder_builds_expected_hv_improvement_using_pareto_from_model() -> None:
    num_obj = 2
    train_x = tf.constant([[-2.0], [-1.5], [-1.0], [0.0], [0.5], [1.0], [1.5], [2.0]])
    dataset = Dataset(
        train_x, tf.tile(tf.constant([[4.1], [0.9], [1.2], [0.1], [-8.8], [1.1], [2.1],
                                      [3.9]]), [1, num_obj]))

    model = ModelStack(*[(test_models[_](), 1) for _ in range(num_obj)])
    acq_fn = Expected_Hypervolume_Improvement().prepare_acquisition_function(dataset, model)

    model_pred_observation = model.predict(train_x)[0]
    _prt = Pareto(model_pred_observation)
    xs = tf.linspace([-10.0], [10.0], 100)
    expected = expected_hv_improvement(model, xs, _prt, get_reference_point(_prt.front))
    npt.assert_allclose(acq_fn(xs), expected)


@random_seed
@pytest.mark.parametrize(
    "input_dim, num_samples_per_point, existing_observations, obj_num, variance_scale ,rtol, atol",
    [
        (1, 50_000, tf.constant([[0.3, 0.2], [0.2, 0.22], [0.1, 0.25], [0.0, 0.3]]), 2, 1.0, 0.01, 1e-2),
        (1, 200_000, tf.constant([[0.3, 0.2], [0.2, 0.22], [0.1, 0.25], [0.0, 0.3]]), 2, 2.0, 0.01, 1e-2),
        (2, 50_000, tf.constant([[0.0, 0.0]]), 2, 1.0, 0.01, 1e-2),
        # (1, 100_000, tf.constant([[0.0, 0.0, 0.5], [0.4, -0.1, 0.2], [0.1, 0.1, 0.1], [0.3, 0.5, 0.0]]), 3,
        #  1.0, 0.01, 1e-2),
    ],
)
def test_expected_hypervolume_improvement(
        input_dim: int, num_samples_per_point: int, existing_observations: tf.Tensor, obj_num: int,
        variance_scale: float, rtol: float, atol: float) -> None:
    # Note: the test data number grows exponentially with num of obj
    data_num_seg_per_dim = 10  # test data number per input dim
    N = data_num_seg_per_dim ** input_dim
    xs = tf.constant(list(itertools.product(*[list(np.linspace(-1, 1, data_num_seg_per_dim))] * input_dim)))

    xs = tf.cast(xs, dtype=existing_observations.dtype)
    model = ModelStack(*[(test_models[_](), 1) for _ in range(obj_num)])

    mean, variance = model.predict(xs)

    # [f_samples, B, L]
    predict_samples = tfd.Normal(mean, tf.sqrt(variance)).sample(num_samples_per_point)
    _pareto = Pareto(existing_observations)
    ref_pt = get_reference_point(_pareto.front)
    lb_points, ub_points = _pareto.get_hyper_cell_bounds(tf.constant([-inf] * ref_pt.shape[-1]), ref_pt)

    # calc MC approx EHVI
    splus_valid = tf.reduce_all(
        tf.tile(ub_points[tf.newaxis, :, tf.newaxis, :],
                [num_samples_per_point, 1, N, 1]) > tf.expand_dims(predict_samples, axis=1), axis=-1)  # num_cells x B
    splus_idx = tf.expand_dims(tf.cast(splus_valid, dtype=ub_points.dtype), -1)
    splus_lb = tf.tile(lb_points[tf.newaxis, :, tf.newaxis, :], [num_samples_per_point, 1, N, 1])
    splus_lb = tf.maximum(splus_lb, tf.expand_dims(predict_samples, 1))
    splus_ub = tf.tile(ub_points[tf.newaxis, :, tf.newaxis, :], [num_samples_per_point, 1, N, 1])
    splus = tf.concat([splus_idx, splus_ub - splus_lb], axis=-1)

    ehvi_approx = tf.transpose(tf.reduce_sum(tf.reduce_prod(splus, axis=-1), axis=1, keepdims=True))  #
    ehvi_approx = tf.reduce_mean(ehvi_approx, axis=-1)

    ehvi = expected_hv_improvement(model, xs, _pareto, ref_pt)

    npt.assert_allclose(ehvi, ehvi_approx, rtol=rtol, atol=atol)

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
from typing import Dict

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.acquisition.rule import (
    AcquisitionRule,
    EfficientGlobalOptimization,
    ThompsonSampling,
    TrustRegion,
)
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.models import GaussianProcessRegression
from trieste.space import Box
from trieste.utils.objectives import branin, BRANIN_GLOBAL_MINIMUM, BRANIN_GLOBAL_ARGMIN

from tests.util.misc import random_seed


@random_seed(1793)
@pytest.mark.parametrize('num_steps, acquisition_rule', [
    (12, EfficientGlobalOptimization()),
    (22, TrustRegion()),
    (17, ThompsonSampling(500, 3)),
])

def test_optimizer_finds_minima_of_the_branin_function(
        num_steps: int, acquisition_rule: AcquisitionRule
) -> None:
    search_space = Box(tf.constant([0.0, 0.0], tf.float64), tf.constant([1.0, 1.0], tf.float64))


    def build_model(data: Dataset) -> GaussianProcessRegression:
        variance = tf.math.reduce_variance(data.observations)
        kernel = gpflow.kernels.Matern52(variance, tf.constant([0.2, 0.2], tf.float64))
        gpr = gpflow.models.GPR((data.query_points, data.observations), kernel, noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        return GaussianProcessRegression(gpr)

    initial_qp = search_space.sample(5)
    initial_data = Dataset(initial_qp, branin(initial_qp))
    model = build_model(initial_data)

    res, _ = BayesianOptimizer(
        branin, search_space
    ).optimize(
        num_steps, initial_data, model, acquisition_rule
    )

    if res.error is not None:
        raise res.error

    dataset = res.dataset

    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]

    argmin = tf.cast(BRANIN_GLOBAL_ARGMIN, tf.float64)
    relative_argmin_err = tf.abs((best_x - argmin) / argmin)
    # these accuracies are the current best for the given number of optimization steps, which makes
    # this is a regression test
    assert tf.reduce_any(tf.reduce_all(relative_argmin_err < 0.03, axis=-1), axis=0)
    npt.assert_allclose(best_y, BRANIN_GLOBAL_MINIMUM, rtol=0.03)

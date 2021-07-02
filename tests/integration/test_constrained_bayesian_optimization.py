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
import math

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import random_seed
from trieste.acquisition.function import ExpectedConstrainedImprovement, ProbabilityOfFeasibility
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.models import GaussianProcessRegression
from trieste.space import Box
from trieste.utils import map_values


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_function_builder",
    [
        (10, ExpectedConstrainedImprovement),
    ],
)
def test_optimizer_finds_minima_of_Gardners_Simulation_1(
    num_steps: int, acquisition_function_builder
) -> None:
    """
    Test that tests the covergence of constrained BO algorithms on the
    synthetic "simulation 1" experiment of :cite:`gardner14`.
    """
    search_space = Box([0, 0], [6, 6])

    def objective(input_data):
        x, y = input_data[..., -2], input_data[..., -1]
        z = tf.cos(2.0 * x) * tf.cos(y) + tf.sin(x)
        return z[:, None]

    def constraint(input_data):
        x, y = input_data[:, -2], input_data[:, -1]
        z = tf.cos(x) * tf.cos(y) - tf.sin(x) * tf.sin(y)
        return z[:, None]

    MINIMUM = -2.0
    MINIMIZER = [math.pi * 1.5, 0.0]

    OBJECTIVE = "OBJECTIVE"
    CONSTRAINT = "CONSTRAINT"

    def observer(query_points):  # observe both objective and constraint data
        return {
            OBJECTIVE: Dataset(query_points, objective(query_points)),
            CONSTRAINT: Dataset(query_points, constraint(query_points)),
        }

    num_initial_points = 5
    initial_data = observer(search_space.sample(num_initial_points))

    def build_model(data):
        variance = tf.math.reduce_variance(data.observations)
        kernel = gpflow.kernels.Matern52(variance, tf.constant([0.2, 0.2], tf.float64))
        gpr = gpflow.models.GPR((data.query_points, data.observations), kernel, noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        return GaussianProcessRegression(gpr)

    models = map_values(build_model, initial_data)

    pof = ProbabilityOfFeasibility(threshold=0.5)
    acq = acquisition_function_builder(OBJECTIVE, pof.using(CONSTRAINT))
    rule: EfficientGlobalOptimization[Box] = EfficientGlobalOptimization(acq)

    dataset = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, models, rule)
        .try_get_final_datasets()[OBJECTIVE]
    )

    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]

    relative_minimizer_err = tf.abs(best_x - MINIMIZER)
    # these accuracies are the current best for the given number of optimization steps, which makes
    # this is a regression test

    assert tf.reduce_all(relative_minimizer_err < 0.03, axis=-1)
    npt.assert_allclose(best_y, MINIMUM, rtol=0.005)

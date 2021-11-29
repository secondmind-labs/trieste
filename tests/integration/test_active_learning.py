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

"""
Integration tests for various forms of active learning implemented in Trieste.
"""

from __future__ import annotations

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import random_seed
from trieste.acquisition.function import ExpectedFeasibility, PredictiveVariance
from trieste.acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.models import TrainableProbabilisticModel
from trieste.models.gpflow import GaussianProcessRegression
from trieste.objectives import BRANIN_SEARCH_SPACE, branin, scaled_branin
from trieste.objectives.utils import mk_observer
from trieste.space import SearchSpace
from trieste.types import TensorType


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        (80, EfficientGlobalOptimization(PredictiveVariance())),
    ],
)
def test_optimizer_learns_scaled_branin_function(
    num_steps: int, acquisition_rule: AcquisitionRule[TensorType, SearchSpace]
) -> None:
    """
    Ensure that the objective function is effecitvely learned, such that the final model
    fits well and predictions are close to actual objective values.
    """

    search_space = BRANIN_SEARCH_SPACE

    def build_model(data: Dataset) -> TrainableProbabilisticModel:
        variance = tf.math.reduce_variance(data.observations)
        kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
        prior_scale = tf.cast(1.0, dtype=tf.float64)
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.cast(-2.0, dtype=tf.float64), prior_scale
        )
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(kernel.lengthscales), prior_scale
        )
        gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
        gpflow.set_trainable(gpr.likelihood, False)

        return GaussianProcessRegression(gpr)

    num_initial_points = 20
    initial_query_points = search_space.sample_halton(num_initial_points)
    observer = mk_observer(scaled_branin)
    initial_data = observer(initial_query_points)

    model = build_model(initial_data)
    final_model = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, model, acquisition_rule)
        .try_get_final_model()
    )

    test_query_points = search_space.sample_sobol(10000 * search_space.dimension)
    test_data = observer(test_query_points)
    predicted_means, _ = final_model.model.predict_f(test_query_points)  # type: ignore

    npt.assert_allclose(predicted_means, test_data.observations, rtol=0.5)


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule, threshold",
    [
        (150, EfficientGlobalOptimization(ExpectedFeasibility(80, delta=1)), 80),
        (150, EfficientGlobalOptimization(ExpectedFeasibility(80, delta=2)), 80),
        (150, EfficientGlobalOptimization(ExpectedFeasibility(20, delta=1)), 20),
    ],
)
def test_optimizer_learns_feasibility_set_of_thresholded_branin_function(
    num_steps: int, acquisition_rule: AcquisitionRule[TensorType, SearchSpace], threshold: int
) -> None:
    """
    Ensure that the objective function is effecitvely learned, such that the final model
    fits well and predictions are close to actual objective values.
    """

    search_space = BRANIN_SEARCH_SPACE

    def build_model(data: Dataset) -> TrainableProbabilisticModel:
        variance = tf.math.reduce_variance(data.observations)
        kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
        prior_scale = tf.cast(1.0, dtype=tf.float64)
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.cast(-2.0, dtype=tf.float64), prior_scale
        )
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(kernel.lengthscales), prior_scale
        )
        gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
        gpflow.set_trainable(gpr.likelihood, False)

        return GaussianProcessRegression(gpr)

    num_initial_points = 20
    initial_query_points = search_space.sample_halton(num_initial_points)
    observer = mk_observer(branin)
    initial_data = observer(initial_query_points)

    model = build_model(initial_data)
    final_model = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, model, acquisition_rule)
        .try_get_final_model()
    )

    test_points = search_space.sample_sobol(10000 * search_space.dimension)
    prob = _excursion_probability(test_points, final_model, threshold)
    accuracy = tf.reduce_sum(prob * (1 - prob))

    assert accuracy < 0.0001


def _excursion_probability(
    x: TensorType, model: TrainableProbabilisticModel, threshold: int
) -> tfp.distributions.Distribution:
    mean, variance = model.model.predict_f(x)  # type: ignore
    normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
    t = (mean - threshold) / tf.sqrt(variance)
    return normal.cdf(t)

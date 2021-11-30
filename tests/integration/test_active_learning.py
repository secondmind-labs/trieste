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
        (50, EfficientGlobalOptimization(PredictiveVariance())),
    ],
)
def test_optimizer_learns_scaled_branin_function(
    num_steps: int, acquisition_rule: AcquisitionRule[TensorType, SearchSpace]
) -> None:
    """
    Ensure that the objective function is effectively learned, such that the final model
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

    num_initial_points = 6
    initial_query_points = search_space.sample_halton(num_initial_points)
    observer = mk_observer(scaled_branin)
    initial_data = observer(initial_query_points)

    # we set a performance criterion at 1% of the range
    # max absolute error needs to be bettter than this criterion
    test_query_points = search_space.sample_sobol(10000 * search_space.dimension)
    test_data = observer(test_query_points)
    test_range = tf.reduce_max(test_data.observations) - tf.reduce_min(test_data.observations)
    criterion = 0.01 * test_range

    # we expect a model with initial data to fail the criterion
    initial_model = build_model(initial_data)
    initial_model.optimize(initial_data)
    initial_predicted_means, _ = initial_model.model.predict_f(test_query_points)  # type: ignore
    initial_accuracy = tf.reduce_max(tf.abs(initial_predicted_means - test_data.observations))

    assert not initial_accuracy < criterion

    # after active learning the model should be much more accurate
    model = build_model(initial_data)
    final_model = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, model, acquisition_rule)
        .try_get_final_model()
    )
    final_predicted_means, _ = final_model.model.predict_f(test_query_points)  # type: ignore
    final_accuracy = tf.reduce_max(tf.abs(final_predicted_means - test_data.observations))

    assert initial_accuracy > final_accuracy
    assert final_accuracy < criterion


@random_seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_steps, acquisition_rule, threshold",
    [
        (50, EfficientGlobalOptimization(ExpectedFeasibility(80, delta=1)), 80),
        (50, EfficientGlobalOptimization(ExpectedFeasibility(80, delta=2)), 80),
        (100, EfficientGlobalOptimization(ExpectedFeasibility(20, delta=1)), 20),
    ],
)
def test_optimizer_learns_feasibility_set_of_thresholded_branin_function(
    num_steps: int, acquisition_rule: AcquisitionRule[TensorType, SearchSpace], threshold: int
) -> None:
    """
    Ensure that the feasible set is sufficiently well learned, such that the final model
    classifies with great degree of certainty whether points in the search space are in
    in the feasible set or not.
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

    num_initial_points = 6
    initial_query_points = search_space.sample_halton(num_initial_points)
    observer = mk_observer(branin)
    initial_data = observer(initial_query_points)

    # we set a performance criterion at 0.001 probability of required precision per point
    n_test = 10000 * search_space.dimension
    test_query_points = search_space.sample_sobol(n_test)
    criterion = 0.001 * (1 - 0.001) * tf.cast(n_test, tf.float64)

    # we expect a model with initial data to fail the criterion
    initial_model = build_model(initial_data)
    initial_model.optimize(initial_data)
    initial_prob = _excursion_probability(test_query_points, initial_model, threshold)
    initial_accuracy = tf.reduce_sum(initial_prob * (1 - initial_prob))

    assert not initial_accuracy < criterion

    # after active learning the model should be much more accurate
    model = build_model(initial_data)
    final_model = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, model, acquisition_rule)
        .try_get_final_model()
    )
    final_prob = _excursion_probability(test_query_points, final_model, threshold)
    final_accuracy = tf.reduce_sum(final_prob * (1 - final_prob))

    assert initial_accuracy > final_accuracy
    assert final_accuracy < criterion


def _excursion_probability(
    x: TensorType, model: TrainableProbabilisticModel, threshold: int
) -> tfp.distributions.Distribution:
    mean, variance = model.model.predict_f(x)  # type: ignore
    normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
    t = (mean - threshold) / tf.sqrt(variance)
    return normal.cdf(t)

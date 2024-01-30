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

from typing import Callable

import gpflow
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import random_seed
from trieste.acquisition import LocalPenalization
from trieste.acquisition.function import (
    BayesianActiveLearningByDisagreement,
    ExpectedFeasibility,
    IntegratedVarianceReduction,
    PredictiveVariance,
)
from trieste.acquisition.function.function import MakePositive
from trieste.acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.models import TrainableProbabilisticModel
from trieste.models.gpflow import (
    GaussianProcessRegression,
    SparseVariational,
    VariationalGaussianProcess,
    build_gpr,
)
from trieste.models.gpflow.builders import build_svgp, build_vgp_classifier
from trieste.models.interfaces import FastUpdateModel, SupportsPredictJoint
from trieste.objectives import Branin, ScaledBranin
from trieste.objectives.utils import mk_observer
from trieste.observer import Observer
from trieste.space import Box, SearchSpace
from trieste.types import TensorType


@random_seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        (50, EfficientGlobalOptimization[SearchSpace, SupportsPredictJoint](PredictiveVariance())),
        (
            70,
            EfficientGlobalOptimization(
                IntegratedVarianceReduction(ScaledBranin.search_space.sample_sobol(1000))
            ),
        ),
    ],
)
def test_optimizer_learns_scaled_branin_function(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, SupportsPredictJoint],
) -> None:
    """
    Ensure that the objective function is effectively learned, such that the final model
    fits well and predictions are close to actual objective values.
    """
    search_space = ScaledBranin.search_space
    num_initial_points = 6
    initial_query_points = search_space.sample_halton(num_initial_points)
    observer = mk_observer(ScaledBranin.objective)
    initial_data = observer(initial_query_points)

    # we set a performance criterion at 1% of the range
    # max absolute error needs to be bettter than this criterion
    test_query_points = search_space.sample_sobol(10000 * search_space.dimension)
    test_data = observer(test_query_points)
    test_range = tf.reduce_max(test_data.observations) - tf.reduce_min(test_data.observations)
    criterion = 0.02 * test_range

    # we expect a model with initial data to fail the criterion
    initial_model = GaussianProcessRegression(
        build_gpr(initial_data, search_space, likelihood_variance=1e-5)
    )
    initial_model.optimize(initial_data)
    initial_predicted_means, _ = initial_model.model.predict_f(test_query_points)
    initial_accuracy = tf.reduce_max(tf.abs(initial_predicted_means - test_data.observations))

    assert not initial_accuracy < criterion

    # after active learning the model should be much more accurate
    model = GaussianProcessRegression(
        build_gpr(initial_data, search_space, likelihood_variance=1e-5)
    )
    final_model = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, model, acquisition_rule)
        .try_get_final_model()
    )
    final_predicted_means, _ = final_model.model.predict_f(test_query_points)
    final_accuracy = tf.reduce_max(tf.abs(final_predicted_means - test_data.observations))

    assert initial_accuracy > final_accuracy
    assert final_accuracy < criterion


@random_seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_steps, acquisition_rule, threshold",
    [
        pytest.param(
            50,
            EfficientGlobalOptimization(ExpectedFeasibility(80, delta=1)),
            80,
            id="ExpectedFeasibility/80/1",
        ),
        pytest.param(
            50,
            EfficientGlobalOptimization(ExpectedFeasibility(80, delta=2)),
            80,
            id="ExpectedFeasibility/80/2",
        ),
        pytest.param(
            70,
            EfficientGlobalOptimization(ExpectedFeasibility(20, delta=1)),
            20,
            id="ExpectedFeasibility/20",
        ),
        pytest.param(
            25,
            EfficientGlobalOptimization(
                IntegratedVarianceReduction(Branin.search_space.sample_sobol(2000), 80.0),
                num_query_points=3,
            ),
            80.0,
            id="IntegratedVarianceReduction/80",
        ),
        pytest.param(
            25,
            EfficientGlobalOptimization(
                IntegratedVarianceReduction(Branin.search_space.sample_sobol(2000), [78.0, 82.0]),
                num_query_points=3,
            ),
            80.0,
            id="IntegratedVarianceReduction/[78, 82]",
        ),
        pytest.param(
            25,
            EfficientGlobalOptimization(
                LocalPenalization(
                    Branin.search_space,
                    base_acquisition_function_builder=MakePositive(
                        ExpectedFeasibility(80, delta=1)
                    ),
                ),
                num_query_points=3,
            ),
            80.0,
            id="LocalPenalization/MakePositive(ExpectedFeasibility)",
        ),
    ],
)
def test_optimizer_learns_feasibility_set_of_thresholded_branin_function(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, FastUpdateModel],
    threshold: int,
) -> None:
    """
    Ensure that the feasible set is sufficiently well learned, such that the final model
    classifies with great degree of certainty whether points in the search space are in
    in the feasible set or not.
    """
    search_space = Branin.search_space

    num_initial_points = 6
    initial_query_points = search_space.sample_halton(num_initial_points)
    observer = mk_observer(Branin.objective)
    initial_data = observer(initial_query_points)

    # we set a performance criterion at 0.001 probability of required precision per point
    # for global points and 0.01 close to the boundary
    n_global = 10000 * search_space.dimension
    n_boundary = 2000 * search_space.dimension
    global_test, boundary_test = _get_feasible_set_test_data(
        search_space, observer, n_global, n_boundary, threshold, range_pct=0.03
    )

    global_criterion = 0.001 * (1 - 0.001) * tf.cast(n_global, tf.float64)
    boundary_criterion = 0.01 * (1 - 0.01) * tf.cast(n_boundary, tf.float64)

    # we expect a model with initial data to fail the criteria
    initial_model = GaussianProcessRegression(
        build_gpr(initial_data, search_space, likelihood_variance=1e-3)
    )
    initial_model.optimize(initial_data)
    initial_accuracy_global = _get_excursion_accuracy(global_test, initial_model, threshold)
    initial_accuracy_boundary = _get_excursion_accuracy(boundary_test, initial_model, threshold)

    assert not initial_accuracy_global < global_criterion
    assert not initial_accuracy_boundary < boundary_criterion

    # after active learning the model should be much more accurate
    model = GaussianProcessRegression(
        build_gpr(initial_data, search_space, likelihood_variance=1e-3)
    )
    final_model = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, model, acquisition_rule)
        .try_get_final_model()
    )
    final_accuracy_global = _get_excursion_accuracy(global_test, final_model, threshold)
    final_accuracy_boundary = _get_excursion_accuracy(boundary_test, final_model, threshold)

    assert initial_accuracy_global > final_accuracy_global
    assert initial_accuracy_boundary > final_accuracy_boundary
    assert final_accuracy_global < global_criterion
    assert final_accuracy_boundary < boundary_criterion


def _excursion_probability(
    x: TensorType, model: TrainableProbabilisticModel, threshold: int
) -> tfp.distributions.Distribution:
    mean, variance = model.model.predict_f(x)  # type: ignore
    normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
    t = (mean - threshold) / tf.sqrt(variance)
    return normal.cdf(t)


def _get_excursion_accuracy(
    x: TensorType, model: TrainableProbabilisticModel, threshold: int
) -> float:
    prob = _excursion_probability(x, model, threshold)
    accuracy = tf.reduce_sum(prob * (1 - prob))

    return accuracy


def _get_feasible_set_test_data(
    search_space: Box,
    observer: Observer,
    n_global: int,
    n_boundary: int,
    threshold: float,
    range_pct: float = 0.01,
) -> tuple[TensorType, TensorType]:
    boundary_done = False
    global_done = False
    boundary_points = tf.constant(0, dtype=tf.float64, shape=(0, search_space.dimension))
    global_points = tf.constant(0, dtype=tf.float64, shape=(0, search_space.dimension))

    while not boundary_done and not global_done:
        test_query_points = search_space.sample(100000)
        test_data = observer(test_query_points)
        threshold_deviation = range_pct * (
            tf.reduce_max(test_data.observations)  # type: ignore
            - tf.reduce_min(test_data.observations)  # type: ignore
        )

        mask = tf.reduce_all(
            tf.concat(
                [
                    test_data.observations > threshold - threshold_deviation,  # type: ignore
                    test_data.observations < threshold + threshold_deviation,  # type: ignore
                ],
                axis=1,
            ),
            axis=1,
        )
        boundary_points = tf.concat(
            [boundary_points, tf.boolean_mask(test_query_points, mask)], axis=0
        )
        global_points = tf.concat(
            [global_points, tf.boolean_mask(test_query_points, tf.logical_not(mask))], axis=0
        )

        if boundary_points.shape[0] > n_boundary:
            boundary_done = True
        if global_points.shape[0] > n_global:
            global_done = True

    return (
        global_points[:n_global,],
        boundary_points[:n_boundary,],
    )


def vgp_classification_model(
    initial_data: Dataset, search_space: Box
) -> VariationalGaussianProcess:
    return VariationalGaussianProcess(
        build_vgp_classifier(initial_data, search_space, noise_free=True)
    )


def svgp_classification_model(initial_data: Dataset, search_space: Box) -> SparseVariational:
    return SparseVariational(build_svgp(initial_data, search_space, classification=True))


@random_seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_steps, model_builder",
    [
        (20, vgp_classification_model),
        (70, svgp_classification_model),
    ],
)
def test_bald_learner_learns_circle_function(
    num_steps: int,
    model_builder: Callable[[Dataset, Box], VariationalGaussianProcess | SparseVariational],
) -> None:
    search_space = Box([-1, -1], [1, 1])

    def circle(x: TensorType) -> TensorType:
        return tf.cast((tf.reduce_sum(tf.square(x), axis=1, keepdims=True) - 0.5) > 0, tf.float64)

    def ilink(f: TensorType) -> TensorType:
        return gpflow.likelihoods.Bernoulli().invlink(f).numpy()

    num_initial_points = 10
    initial_query_points = search_space.sample(num_initial_points)
    observer = mk_observer(circle)
    initial_data = observer(initial_query_points)

    # we set a performance criterion at 20% error
    # predictive error needs to be bettter than this criterion
    test_query_points = search_space.sample_sobol(10000 * search_space.dimension)
    test_data = observer(test_query_points)
    criterion = 0.2

    # we expect a model with initial data to fail the criterion
    initial_model = model_builder(initial_data, search_space)
    initial_model.optimize(initial_data)
    initial_predicted_means, _ = ilink(initial_model.model.predict_f(test_query_points))
    initial_error = tf.reduce_mean(tf.abs(initial_predicted_means - test_data.observations))

    assert not initial_error < criterion

    # after active learning the model should be much more accurate
    model = model_builder(initial_data, search_space)
    acq = BayesianActiveLearningByDisagreement()
    rule = EfficientGlobalOptimization(acq)  # type: ignore

    final_model = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, model, rule)
        .try_get_final_model()
    )
    final_predicted_means, _ = ilink(final_model.model.predict_f(test_query_points))
    final_error = tf.reduce_mean(tf.abs(final_predicted_means - test_data.observations))

    assert initial_error > final_error
    assert final_error < criterion

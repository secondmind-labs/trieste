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

from typing import List, Tuple, Union, cast

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import random_seed
from tests.util.models.gpflux.models import two_layer_dgp_model
from trieste.acquisition.function import (
    GIBBON,
    AcquisitionFunctionClass,
    AugmentedExpectedImprovement,
    BatchMonteCarloExpectedImprovement,
    LocalPenalizationAcquisitionFunction,
    MinValueEntropySearch,
)
from trieste.acquisition.rule import (
    AcquisitionRule,
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
    TrustRegion,
)
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.gpflux import DeepGaussianProcess
from trieste.objectives import (
    BRANIN_MINIMIZERS,
    BRANIN_SEARCH_SPACE,
    MICHALEWICZ_2_MINIMIZER,
    MICHALEWICZ_2_MINIMUM,
    MICHALEWICZ_2_SEARCH_SPACE,
    SCALED_BRANIN_MINIMUM,
    michalewicz,
    scaled_branin,
)
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box, SearchSpace
from trieste.types import State, TensorType


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    cast(
        List[
            Tuple[
                int,
                Union[
                    AcquisitionRule[TensorType, Box],
                    AcquisitionRule[State[TensorType, TrustRegion.State], Box],
                ],
            ]
        ],
        [
            (20, EfficientGlobalOptimization()),
            (25, EfficientGlobalOptimization(AugmentedExpectedImprovement().using(OBJECTIVE))),
            (
                15,
                EfficientGlobalOptimization(
                    MinValueEntropySearch(BRANIN_SEARCH_SPACE, num_fourier_features=1000).using(
                        OBJECTIVE
                    )
                ),
            ),
            (
                12,
                EfficientGlobalOptimization(
                    BatchMonteCarloExpectedImprovement(sample_size=500).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
            (
                10,
                EfficientGlobalOptimization(
                    LocalPenalizationAcquisitionFunction(
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
            (
                10,
                EfficientGlobalOptimization(
                    GIBBON(
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=2,
                ),
            ),
            (15, TrustRegion()),
            (
                15,
                TrustRegion(
                    EfficientGlobalOptimization(
                        MinValueEntropySearch(BRANIN_SEARCH_SPACE, num_fourier_features=1000).using(
                            OBJECTIVE
                        )
                    )
                ),
            ),
            (10, DiscreteThompsonSampling(500, 3)),
            (10, DiscreteThompsonSampling(500, 3, num_fourier_features=1000)),
        ],
    ),
)
def test_optimizer_finds_minima_of_the_scaled_branin_function(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace]
    | AcquisitionRule[State[TensorType, TrustRegion.State], Box],
) -> None:
    search_space = BRANIN_SEARCH_SPACE

    def build_model(data: Dataset) -> GaussianProcessRegression:
        variance = tf.math.reduce_variance(data.observations)
        kernel = gpflow.kernels.Matern52(variance, tf.constant([0.2, 0.2], tf.float64))
        scale = tf.constant(1.0, dtype=tf.float64)
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.constant(-2.0, dtype=tf.float64), scale
        )
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(kernel.lengthscales), scale
        )
        gpr = gpflow.models.GPR((data.query_points, data.observations), kernel, noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        return GaussianProcessRegression(gpr)

    initial_query_points = search_space.sample(5)
    observer = mk_observer(scaled_branin)
    initial_data = observer(initial_query_points)
    model = build_model(initial_data)

    dataset = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, model, acquisition_rule)
        .try_get_final_dataset()
    )

    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]

    relative_minimizer_err = tf.abs((best_x - BRANIN_MINIMIZERS) / BRANIN_MINIMIZERS)
    # these accuracies are the current best for the given number of optimization steps, which makes
    # this is a regression test
    assert tf.reduce_any(tf.reduce_all(relative_minimizer_err < 0.05, axis=-1), axis=0)
    npt.assert_allclose(best_y, SCALED_BRANIN_MINIMUM, rtol=0.005)

    # check that acquisition functions defined as classes aren't being retraced unnecessarily
    if isinstance(acquisition_rule, EfficientGlobalOptimization):
        acquisition_function = acquisition_rule._acquisition_function
        if isinstance(acquisition_function, AcquisitionFunctionClass):
            assert acquisition_function.__call__._get_tracing_count() == 3  # type: ignore


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        (5, DiscreteThompsonSampling(1000, 50)),
    ],
)
def test_two_layer_dgp_optimizer_finds_minima_of_michalewicz_function(
    num_steps: int, acquisition_rule: AcquisitionRule[TensorType, SearchSpace], keras_float: None
) -> None:

    search_space = MICHALEWICZ_2_SEARCH_SPACE

    def build_model(data: Dataset) -> DeepGaussianProcess:
        epochs = int(2e3)
        batch_size = 100

        dgp = two_layer_dgp_model(data.query_points)

        def scheduler(epoch: int, lr: float) -> float:
            if epoch == epochs // 2:
                return lr * 0.1
            else:
                return lr

        optimizer = tf.optimizers.Adam(0.01)
        fit_args = {
            "batch_size": batch_size,
            "epochs": epochs,
            "verbose": 0,
            "callbacks": tf.keras.callbacks.LearningRateScheduler(scheduler),
        }

        return DeepGaussianProcess(model=dgp, optimizer=optimizer, fit_args=fit_args)

    initial_query_points = search_space.sample(50)
    observer = mk_observer(michalewicz, OBJECTIVE)
    initial_data = observer(initial_query_points)
    model = build_model(initial_data[OBJECTIVE])
    dataset = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, {OBJECTIVE: model}, acquisition_rule, track_state=False)
        .try_get_final_dataset()
    )
    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]
    relative_minimizer_err = tf.abs((best_x - MICHALEWICZ_2_MINIMIZER) / MICHALEWICZ_2_MINIMIZER)

    assert tf.reduce_all(relative_minimizer_err < 0.03, axis=-1)
    npt.assert_allclose(best_y, MICHALEWICZ_2_MINIMUM, rtol=0.03)

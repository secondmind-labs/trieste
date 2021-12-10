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

from typing import List, Tuple, cast

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import random_seed
from trieste.acquisition import (
    AcquisitionFunctionClass,
    BatchMonteCarloExpectedImprovement,
    LocalPenalization,
)
from trieste.acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.objectives import (
    BRANIN_MINIMIZERS,
    BRANIN_SEARCH_SPACE,
    SCALED_BRANIN_MINIMUM,
    scaled_branin,
)
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box, DiscreteSearchSpace, TaggedProductSearchSpace
from trieste.types import TensorType


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    cast(
        List[
            Tuple[
                int,
                AcquisitionRule[TensorType, TaggedProductSearchSpace],
            ]
        ],
        [
            (15, EfficientGlobalOptimization()),
            (
                5,
                EfficientGlobalOptimization(
                    BatchMonteCarloExpectedImprovement(sample_size=500).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
            (
                5,
                EfficientGlobalOptimization(
                    LocalPenalization(
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
        ],
    ),
)
def test_optimizer_finds_minima_of_the_scaled_branin_function(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, TaggedProductSearchSpace],
) -> None:
    search_space = TaggedProductSearchSpace(
        spaces=[Box([0], [1]), DiscreteSearchSpace(tf.linspace(0, 1, 15)[:, None])],
        tags=["continuous", "discrete"],
    )

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
    assert tf.reduce_any(tf.reduce_all(relative_minimizer_err < 0.1, axis=-1), axis=0)
    npt.assert_allclose(best_y, SCALED_BRANIN_MINIMUM, rtol=0.005)

    # check that acquisition functions defined as classes aren't being retraced unnecessarily
    # They should be retraced once for the optimzier's starting grid and once for L-BFGS.
    if isinstance(acquisition_rule, EfficientGlobalOptimization):
        acquisition_function = acquisition_rule._acquisition_function
        if isinstance(acquisition_function, AcquisitionFunctionClass):
            assert acquisition_function.__call__._get_tracing_count() <= 3  # type: ignore

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
import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import random_seed
from trieste.acquisition.function import (
    GIBBON,
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
from trieste.models import GaussianProcessRegression
from trieste.observer import OBJECTIVE
from trieste.space import Box
from trieste.utils.objectives import (
    BRANIN_MINIMIZERS,
    SCALED_BRANIN_MINIMUM,
    mk_observer,
    scaled_branin,
)


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        (20, EfficientGlobalOptimization()),
        (26, EfficientGlobalOptimization(AugmentedExpectedImprovement().using(OBJECTIVE))),
        (
            15,
            EfficientGlobalOptimization(
                MinValueEntropySearch(Box([0, 0], [1, 1]), num_fourier_features=1000).using(
                    OBJECTIVE
                )
            ),
        ),
        (
            10,
            EfficientGlobalOptimization(
                BatchMonteCarloExpectedImprovement(sample_size=500).using(OBJECTIVE),
                num_query_points=3,
            ),
        ),
        (
            10,
            EfficientGlobalOptimization(
                LocalPenalizationAcquisitionFunction(
                    Box([0, 0], [1, 1]),
                ).using(OBJECTIVE),
                num_query_points=3,
            ),
        ),
        (
            10,
            EfficientGlobalOptimization(
                GIBBON(
                    Box([0, 0], [1, 1]),
                ).using(OBJECTIVE),
                num_query_points=2,
            ),
        ),
        (10, DiscreteThompsonSampling(500, 3)),
        (10, DiscreteThompsonSampling(500, 3, num_fourier_features=1000)),
    ],
)
@pytest.mark.parametrize("trust_region", [TrustRegion(), None])
def test_optimizer_finds_minima_of_the_scaled_branin_function(
    num_steps: int, acquisition_rule: AcquisitionRule, trust_region: TrustRegion | None
) -> None:
    search_space = Box([0, 0], [1, 1])

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
        .optimize(num_steps, initial_data, model, acquisition_rule, trust_region)
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

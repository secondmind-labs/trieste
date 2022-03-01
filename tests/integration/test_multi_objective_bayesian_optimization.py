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

import pytest
import tensorflow as tf

from tests.util.misc import random_seed
from trieste.acquisition import (
    HIPPO,
    BatchMonteCarloExpectedHypervolumeImprovement,
    ExpectedHypervolumeImprovement,
)
from trieste.acquisition.multi_objective.pareto import Pareto, get_reference_point
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.rule import (
    AcquisitionRule,
    AsynchronousOptimization,
    EfficientGlobalOptimization,
)
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.models.interfaces import (
    TrainableModelStack,
    TrainablePredictJointReparamModelStack,
    TrainableProbabilisticModel,
)
from trieste.objectives.multi_objectives import VLMOP2
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box
from trieste.types import TensorType


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule, convergence_threshold",
    [
        pytest.param(
            20,
            EfficientGlobalOptimization(ExpectedHypervolumeImprovement().using(OBJECTIVE)),
            -3.65,
            id="ExpectedHypervolumeImprovement",
        ),
        pytest.param(
            15,
            EfficientGlobalOptimization(
                BatchMonteCarloExpectedHypervolumeImprovement(sample_size=500).using(OBJECTIVE),
                num_query_points=2,
                optimizer=generate_continuous_optimizer(num_initial_samples=500),
            ),
            -3.44,
            id="BatchMonteCarloExpectedHypervolumeImprovement/2",
        ),
        pytest.param(
            10,
            EfficientGlobalOptimization(
                BatchMonteCarloExpectedHypervolumeImprovement(sample_size=250).using(OBJECTIVE),
                num_query_points=4,
                optimizer=generate_continuous_optimizer(num_initial_samples=500),
            ),
            -3.2095,
            id="BatchMonteCarloExpectedHypervolumeImprovement/4",
        ),
        pytest.param(
            10,
            EfficientGlobalOptimization(
                HIPPO(),
                num_query_points=4,
                optimizer=generate_continuous_optimizer(num_initial_samples=500),
            ),
            -3.2095,
            id="HIPPO/4",
        ),
        pytest.param(
            10,
            AsynchronousOptimization(
                BatchMonteCarloExpectedHypervolumeImprovement(sample_size=250).using(OBJECTIVE),
                num_query_points=4,
                optimizer=generate_continuous_optimizer(num_initial_samples=500),
            ),
            -3.2095,
            id="BatchMonteCarloExpectedHypervolumeImprovement/4",
        ),
    ],
)
def test_multi_objective_optimizer_finds_pareto_front_of_the_VLMOP2_function(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, Box, TrainableProbabilisticModel],
    convergence_threshold: float,
) -> None:
    search_space = Box([-2, -2], [2, 2])

    def build_stacked_independent_objectives_model(data: Dataset) -> TrainableModelStack:
        gprs = []
        for idx in range(2):
            single_obj_data = Dataset(
                data.query_points, tf.gather(data.observations, [idx], axis=1)
            )
            gpr = build_gpr(single_obj_data, search_space, likelihood_variance=1e-5)
            gprs.append((GaussianProcessRegression(gpr), 1))

        return TrainablePredictJointReparamModelStack(*gprs)

    observer = mk_observer(VLMOP2().objective(), OBJECTIVE)

    initial_query_points = search_space.sample(10)
    initial_data = observer(initial_query_points)

    model = build_stacked_independent_objectives_model(initial_data[OBJECTIVE])

    dataset = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, {OBJECTIVE: model}, acquisition_rule)
        .try_get_final_datasets()[OBJECTIVE]
    )

    # A small log hypervolume difference corresponds to a succesful optimization.
    ref_point = get_reference_point(dataset.observations)

    obs_hv = Pareto(dataset.observations).hypervolume_indicator(ref_point)
    ideal_pf = tf.cast(VLMOP2().gen_pareto_optimal_points(100), dtype=tf.float64)
    ideal_hv = Pareto(ideal_pf).hypervolume_indicator(ref_point)

    assert tf.math.log(ideal_hv - obs_hv) < convergence_threshold

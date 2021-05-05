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

from tests.util.misc import random_seed
from trieste.acquisition.function import  ExpectedHypervolumeImprovement
from trieste.acquisition.rule import (
    OBJECTIVE,
    AcquisitionRule,
    EfficientGlobalOptimization,
)
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.models import GaussianProcessRegression
from trieste.models.model_interfaces import ModelStack
from trieste.space import Box
from trieste.utils.multi_objectives import VLMOP2


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        (20,  EfficientGlobalOptimization(ExpectedHypervolumeImprovement().using(OBJECTIVE))),
    ],
)
def test_multi_objective_optimizer_finds_pareto_front_of_the_VLMOP2_function(
    num_steps: int, acquisition_rule: AcquisitionRule
) -> None:
    search_space = Box([-2, -2], [2, 2])

    def build_stacked_independent_objectives_model(data: Dataset) -> GaussianProcessRegression:
        gprs =[]
        for idx in range(2):
            single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
            variance = tf.math.reduce_variance(single_obj_data.observations)
            kernel = gpflow.kernels.Matern52(variance, tf.constant([0.2, 0.2], tf.float64))
            gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel, noise_variance=1e-5)
            gpflow.utilities.set_trainable(gpr.likelihood, False)
            gprs.append(GaussianProcessRegression(gpr))

        return ModelStack(*gprs)


    vlmop2 = VLMOP2().objective()
    observer = trieste.utils.objectives.mk_observer(vlmop2, OBJECTIVE)

    initial_query_points = search_space.sample(10)
    initial_data = observer(initial_query_points)
    model = build_stacked_independent_objectives_model(initial_data[OBJECTIVE])


    dataset = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, {OBJECTIVE: model}, acquisition_rule)
        .try_get_final_datasets()[OBJECTIVE]
    )



    obs_hv = Pareto(observations).hypervolume_indicator(ref_point)
    
    ideal_pf = VLMOP2().gen_pareto_optimal_points(100)
    ref_point = get_reference_point(data_observations)
    idea_hv = Pareto(tf.cast(ideal_pf, dtype=data_observations.dtype)).hypervolume_indicator(ref_point)

    log_hv_diff = math.log(idea_hv - obs_hv)


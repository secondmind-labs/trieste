# Copyright 2022 The Trieste Contributors
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


""" Runs a large number of experiments using Ray. """

import os

import ray
import tensorflow as tf

from hv.active_learning.config import ActiveLearningExperimentConfig, define_all_experiments
from hv.active_learning.experiment import run_and_save_experiment

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.kernels import Kernel
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.space import SearchSpace



from typing import Union

from trieste.acquisition import (
    ExpectedFeasibility,
    Fantasizer,
    IntegratedVarianceReduction,
    LocalPenalization,
    PredictiveVariance,
)
from trieste.acquisition.optimizer import automatic_optimizer_selector
from trieste.acquisition.rule import (
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
    RandomSampling,
)
from trieste.acquisition.utils import split_acquisition_function_calls
from trieste.space import Box

from hv.active_learning.acquisition_functions import (
    ExactThresholdSampler,
    ExpectedFeasibilitySampler,
    MaximumVariance,
    MultipleOptimismStraddle,
    ParallelContinuousThresholdSampling,
    PositiveExpectedFeasibility,
    feasGIBBON,
)
from hv.active_learning.config import ActiveLearningExperimentConfig


def _set_kernel(var: float, input_dim: int) -> Kernel:
    kernel = gpflow.kernels.Matern52(variance=var, lengthscales=0.2 * np.ones(input_dim,))
    prior_scale = tf.cast(1.0, tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(tf.cast(0.0, tf.float64), prior_scale)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        tf.math.log(kernel.lengthscales), prior_scale
    )
    return kernel


def build_model(
    data: Dataset, search_space: SearchSpace, model_type: str
) -> GaussianProcessRegression:
    """
    Return a model given some data.
    For now only return a GPR model. Future improvements will allow more models.
    """
    if model_type == "GPR_custom":
        variance = tf.math.reduce_variance(data.observations)
        kernel = _set_kernel(variance, data.query_points.shape[-1])
        gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
        gpflow.set_trainable(gpr.likelihood, False)
        return GaussianProcessRegression(gpr)
    elif model_type == "GPR":
        gpr = build_gpr(data, search_space, likelihood_variance=1e-5)
        return GaussianProcessRegression(gpr)
    else:
        raise NotImplementedError(
            f"Only config.model='GPR' is supported for now, " f"found {model_type}"
        )



def build_acquisition_rule(
    config: ActiveLearningExperimentConfig, search_space: Box
) -> Union[EfficientGlobalOptimization, DiscreteThompsonSampling, RandomSampling]:
    """
    Return the acquisition rule corresponding to config.
    """

    rule: Union[EfficientGlobalOptimization, DiscreteThompsonSampling, RandomSampling]
    optimizer = split_acquisition_function_calls(automatic_optimizer_selector, split_size=10_000)

    if config.rule == "nobatch-bichon":
        rule = EfficientGlobalOptimization(
            optimizer=optimizer, builder=ExpectedFeasibility(threshold=config.problem.threshold)
        )

    elif config.rule == "nobatch-ranjan":
        rule = EfficientGlobalOptimization(
            optimizer=optimizer,
            builder=ExpectedFeasibility(threshold=config.problem.threshold, delta=2),
        )

    elif config.rule == "lp-bichon":
        acq_lp = LocalPenalization(
            search_space,
            base_acquisition_function_builder=PositiveExpectedFeasibility(threshold=config.problem.threshold),  # type: ignore
        )
        rule = EfficientGlobalOptimization(  # type: ignore
            optimizer=optimizer, num_query_points=config.batch_size, builder=acq_lp
        )

    elif config.rule == "lp-ranjan":
        acq_lp = LocalPenalization(
            search_space,
            base_acquisition_function_builder=PositiveExpectedFeasibility(threshold=config.problem.threshold, delta=2),  # type: ignore
        )
        rule = EfficientGlobalOptimization(  # type: ignore
            optimizer=optimizer, num_query_points=config.batch_size, builder=acq_lp
        )

    elif config.rule == "kb-bichon":
        acq = Fantasizer(ExpectedFeasibility(threshold=config.problem.threshold))
        rule = EfficientGlobalOptimization(  # type: ignore
            num_query_points=config.batch_size, builder=acq
        )

    elif config.rule == "kb-ranjan":
        acq = Fantasizer(ExpectedFeasibility(threshold=config.problem.threshold, delta=2))
        rule = EfficientGlobalOptimization(  # type: ignore
            optimizer=optimizer, num_query_points=config.batch_size, builder=acq
        )

    elif config.rule == "evr":
        rule = EfficientGlobalOptimization(
            optimizer=optimizer,
            builder=IntegratedVarianceReduction(
                search_space.sample_sobol(1000), threshold=config.problem.threshold
            ),
            num_query_points=config.batch_size,
        )

    elif config.rule == "exactsampler":
        sampler = ExactThresholdSampler(threshold=config.problem.threshold)  # type: ignore
        rule = DiscreteThompsonSampling(  # type: ignore
            num_search_space_samples=5000,
            num_query_points=config.batch_size,
            thompson_sampler=sampler,  # type: ignore
        )

    elif config.rule == "ts-bichon":
        sampler = ExpectedFeasibilitySampler(threshold=config.problem.threshold)  # type: ignore
        rule = DiscreteThompsonSampling(  # type: ignore
            num_search_space_samples=5000,
            num_query_points=config.batch_size,
            thompson_sampler=sampler,  # type: ignore
        )

    elif config.rule == "ts-ranjan":
        sampler = ExpectedFeasibilitySampler(threshold=config.problem.threshold, delta=2)  # type: ignore
        rule = DiscreteThompsonSampling(  # type: ignore
            num_search_space_samples=5000,
            num_query_points=config.batch_size,
            thompson_sampler=sampler,  # type: ignore
        )

    elif config.rule == "gibbon":
        acq_gibbon = feasGIBBON(
            search_space=search_space, rescaled_repulsion=True, threshold=config.problem.threshold
        )
        rule = EfficientGlobalOptimization(
            optimizer=optimizer, builder=acq_gibbon, num_query_points=config.batch_size
        )

    elif config.rule == "predvar":
        rule = EfficientGlobalOptimization(
            optimizer=optimizer, builder=PredictiveVariance(), num_query_points=config.batch_size
        )

    elif config.rule == "maxvar":
        rule = EfficientGlobalOptimization(
            optimizer=optimizer, builder=MaximumVariance(), num_query_points=config.batch_size
        )

    elif config.rule == "random":
        rule = RandomSampling(num_query_points=config.batch_size)

    elif config.rule == "straddle":
        acq_fnc = MultipleOptimismStraddle(search_space, config.problem.threshold, 1.0)
        rule = EfficientGlobalOptimization(
            optimizer=optimizer, builder=acq_fnc, num_query_points=config.batch_size
        )

    elif config.rule == "contsampler":
        acq_fnc = ParallelContinuousThresholdSampling(config.problem.threshold)  # type: ignore
        rule = EfficientGlobalOptimization(
            optimizer=optimizer, builder=acq_fnc, num_query_points=config.batch_size
        )

    else:
        raise NotImplementedError(f"received unsupported rule name {config.rule}")

    return rule


def define_all_experiments():
    """
    Take the combination of all factors to create the set of experiments to run.
    """

    initial_budget_per_dimension = [5]
    budgets_per_dimension = [200]
    seeds = np.arange(10)  # basically the number of restarts for each problem
    problems = [
        ObjectiveVolume(Objective.HARTMANN_6, 0.10),
        ObjectiveVolume(Objective.HARTMANN_6, 0.50),
        ObjectiveVolume(Objective.ACKLEY_5, 0.10),
        ObjectiveVolume(Objective.ACKLEY_5, 0.50),
    ]
    # batch_sizes = [1, 10, 100]
    batch_sizes = [100]
    rules = [
        # "random",
        # "maxvar",
        "straddle",
        "contsampler",
        # "lp-ranjan",
        # "lp-bichon",
        # "gibbon",
        # "exactsampler",
        # "nobatch-ranjan",
        # "nobatch-bichon",
        # "evr",
        # "kb-ranjan",
        # "ts-ranjan",
        # "kb-bichon",
        # "ts-bichon",
    ]
    models = ["GPR"]
    overwrite = [False]

    all_conditions = list(
        dict_product(
            dict(
                initial_budget_per_dimension=initial_budget_per_dimension,
                budget_per_dimension=budgets_per_dimension,
                seed=seeds,
                problem_specs=problems,
                batch_size=batch_sizes,
                rule=rules,
                model_type=models,
                overwrite=overwrite,
            )
        )
    )

    # filtering out combinations that are not needed
    all_conditions[:] = [
        exp
        for exp in all_conditions
        if not (
            exp["rule"] in ["maxvar", "evr", "nobatch-ranjan", "nobatch-bichon"]
            and exp["batch_size"] != 1
        )
    ]
    all_conditions[:] = [
        exp
        for exp in all_conditions
        if not (
            exp["rule"]
            in [
                "kb-ranjan",
                "kb-bichon",
                "lp-ranjan",
                "lp-bichon",
                "gibbon",
                "ts-ranjan",
                "ts-bichon",
            ]
            and exp["batch_size"] == 1
        )
    ]

    return all_conditions



num_workers = 2

ray.init(num_cpus=num_workers)

@ray.remote
def _run_and_save_experiment_with_ray(config_args):
    try:
        exp = Experiment(config_args, True)
        exp.run()
    except:  # pylint: disable=bare-except
        config = ExperimentConfig(**config_args)
        experiment_name = config.exp_name
        print(f"Failed experiment {experiment_name}")

workers = []

experiment_list = define_all_experiments()

for exp in experiment_list:
    worker = _run_and_save_experiment_with_ray.remote(exp)
    workers.append(worker)

remaining_workers = workers

while len(remaining_workers) > 0:
    ready_workers, remaining_workers = ray.wait(workers, num_returns=1)
    print(f"Remaining workers {len(remaining_workers)}")
    workers = remaining_workers

ray.shutdown()

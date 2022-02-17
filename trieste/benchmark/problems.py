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

""" Defines the feasibility problems. """


from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Type

import tensorflow as tf

from trieste.data import Dataset
from trieste.objectives import SINGLE_OBJECTIVE_SPECS, SingleObjective
from trieste.space import Box
from trieste.types import TensorType

NUM_TEST_POINTS_PER_DIM: int = 10000
""" Default number of points for test points per dimension. """


RANGE_PCT: float = 0.01
"""
Default percentage of the observations range around the threshold within which we accept sampled
points as boundary points; relevant for the level set problem.
"""


@dataclass
class Problem:
    """
    This class stores base information we need for defining a Bayesian optimization problem.
    """
    problem_name: str
    """"""

    fun: Callable[[TensorType], TensorType]
    """"""

    search_space: SearchSpace
    """"""

    @property
    def dim(self):
        return self.search_space.dimension


@dataclass
class FunctionOptimizationProblem(Problem):
    """
    Stores all information we need to define a function optimization type of problem.
    """
    minima: TensorType
    minimizers: TensorType
    test_data: Dataset


@dataclass
class FunctionLearningProblem(Problem):
    """
    Stores all information we need to define a function learning type of problem.
    """
    test_data: Dataset


@dataclass
class LevelSetProblem(Problem):
    """
    Stores all information we need to define a level set type of problem.
    """
    threshold: float
    estimated_volume: TensorType
    global_data: Dataset
    global_feasible: TensorType
    boundary_data: Dataset
    boundary_feasible: TensorType


def get_function_optimization_problem(
    objective: SingleObjective, num_samples_per_dim: int = NUM_TEST_POINTS_PER_DIM
) -> FunctionOptimizationProblem:
    """
    Create a :class:`FunctionOptimizationProblem` from one of the existing objective functions.

    :param objective: One of the objectives implemented in Trieste :module:`~trieste.objectives`
        package.
    :param num_samples_per_dim: Number of samples for the test data used for evaluating the model.
        By default set to ``NUM_TEST_POINTS_PER_DIM``.
    :return: A fully specified function optimization problem.
    """
    if objective not in SINGLE_OBJECTIVE_SPECS:
        raise ValueError(
            f"{objective} not recognised. Should be one of {SINGLE_OBJECTIVE_SPECS.keys()}."
        )

    problem = FunctionOptimizationProblem
    problem.problem_name = objective.name
    problem.fun = SINGLE_OBJECTIVE_SPECS[objective].fun
    problem.search_space = SINGLE_OBJECTIVE_SPECS[objective].search_space
    problem.minima = SINGLE_OBJECTIVE_SPECS[objective].minima
    problem.minimizers = SINGLE_OBJECTIVE_SPECS[objective].minimizers
    problem.test_data = _get_search_space_samples(
        problem.fun, problem.search_space, int(num_samples_per_dim*problem.dim)
    )

    return problem


def get_function_learning_problem(
    objective: SingleObjective, num_samples_per_dim: int = NUM_TEST_POINTS_PER_DIM
) -> FunctionLearningProblem:
    """
    Create a :class:`FunctionLearningProblem` from one of the existing objective functions.

    :param objective: One of the objectives implemented in Trieste :module:`~trieste.objectives`
        package.
    :param num_samples_per_dim: Number of samples for the test data used for evaluating the model.
        By default set to ``NUM_TEST_POINTS_PER_DIM``.
    :return: A fully specified function learning problem.
    """
    if objective not in SINGLE_OBJECTIVE_SPECS:
        raise ValueError(
            f"{objective} not recognised. Should be one of {SINGLE_OBJECTIVE_SPECS.keys()}."
        )

    problem = FunctionLearningProblem
    problem.problem_name = objective.name
    problem.fun = SINGLE_OBJECTIVE_SPECS[objective].fun
    problem.search_space = SINGLE_OBJECTIVE_SPECS[objective].search_space
    problem.test_data = _get_search_space_samples(
        problem.fun, problem.search_space, int(num_samples_per_dim*problem.dim)
    )

    return problem


def get_level_set_problem(
    objective: Objective,
    proportion: float,
    num_global_samples_per_dim: int = NUM_TEST_POINTS_PER_DIM,
    num_boundary_samples_per_dim: int = NUM_TEST_POINTS_PER_DIM,
    range_pct: float = RANGE_PCT
) -> LevelSetProblem:
    """
    Create a LevelSetProblem from one of the existing objective functions.

    :param objective: One of the objectives implemented in Trieste :module:`~trieste.objectives`
        package.
    :param num_samples: Number of samples for the test data used for evaluating the model.
    :return: A fully specified level set learning problem.
    """
    if objective not in SINGLE_OBJECTIVE_SPECS:
        raise ValueError(
            f"{objective} not recognised. Should be one of {SINGLE_OBJECTIVE_SPECS.keys()}."
        )

    # available_thresholds = PROBLEM_THRESHOLDS.loc[
    #     (PROBLEM_THRESHOLDS[FindThresholdResult.objective.name] == objective.name),
    # ]
    # filtered_thresholds = available_thresholds.loc[
    #     (available_thresholds[FindThresholdResult.volume.name] == volume),
    # ]
    # if filtered_thresholds.shape[0] == 0:
    #     available_volumes = available_thresholds[FindThresholdResult.volume.name].values
    #     raise ValueError(f"Threshold not found. Should be one of {available_volumes}.")
    # threshold = float(filtered_thresholds[FindThresholdResult.threshold.name].values)

    problem = LevelSetProblem
    problem.problem_name = f"{objective.name}_{int(proportion*100)}"
    problem.fun = SINGLE_OBJECTIVE_SPECS[objective].fun
    problem.search_space = SINGLE_OBJECTIVE_SPECS[objective].search_space
    problem.threshold = threshold
    problem.global_data = _get_search_space_samples(
        problem.fun, problem.search_space, int(num_global_samples_per_dim*problem.dim)
    )
    problem.global_feasible = tf.cast(problem.global_data.observations < threshold, tf.float32)    
    problem.boundary_data = _get_boundary_samples(
        problem.fun,
        problem.search_space,
        threshold,
        int(num_boundary_samples_per_dim*problem.dim),
        range_pct,
    )
    problem.boundary_feasible = tf.cast(problem.boundary_data.observations < threshold, tf.float32)
    problem.estimated_volume = tf.reduce_mean(problem.global_feasible)

    return problem


def _get_search_space_samples(
    fun: Callable[[TensorType], TensorType], search_space: SearchSpace, num_samples: int
) -> Dataset:
    query_points = search_space.sample(num_samples)
    observations = fun(query_points)
    return Dataset(query_points, observations)


def _get_boundary_samples(
    fun: Callable[[TensorType], TensorType],
    search_space: SearchSpace,
    threshold: float,
    num_samples: int,
    range_pct: float,
) -> Dataset:

    # we might need several sampling iterations to accumulate enough boundary points
    boundary_done = False
    boundary_x = tf.constant(0, dtype=tf.float64, shape=(0, search_space.dimension))
    while not boundary_done:
        query_points = search_space.sample_sobol(100000)
        observations = fun(query_points)
        threshold_deviation = range_pct * (
            tf.reduce_max(observations) - tf.reduce_min(observations)
        )
        mask = tf.reduce_all(
            tf.concat(
                [
                    observations > threshold - threshold_deviation,
                    observations < threshold + threshold_deviation,
                ],
                1,
            ),
            1,
        )
        boundary_x = tf.concat([boundary_x, tf.boolean_mask(query_points, mask)], 0)
        if boundary_x.shape[0] > num_samples:
            boundary_done = True
    boundary_x = boundary_x[:num_samples,]
    boundary_y = fun(boundary_x)

    return Dataset(boundary_x, boundary_y)

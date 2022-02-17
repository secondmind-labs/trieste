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

""" Defines the set of algorithms and problems to test. """

from __future__ import annotations

import itertools
import os
from collections import namedtuple
from typing import Type

import numpy as np
from hv.active_learning.problems import ActiveLearningProblem, Objective, get_problem
from hv.config.active_learning import EXPERIMENTS_RAW_DATA

ObjectiveVolume = namedtuple("ObjectiveVolume", ("objective", "volume"))
"""
Problem is determined by the name of the objective function as stored in ``Objective`` and
a target volume, a float between 0 and 1.
"""


class ExperimentConfig:
    """
    This is a class to store all of the relevant parameters for an active learning
    experiment.
    """

    problem: Type[ActiveLearningProblem]  # the problem to solve
    rule: str  # name of the rule defined in create_acquisition_rule in acquisition_utils
    model_type: str  # type of model (only 'GPR' for now)
    seed: int  # the seed will determine the initial set of observations
    batch_size: int  # number of query points per active learning iteration
    num_initial_points: int  # size of the initial space-filling design
    budget: int  # total number of observation for AL, including the initial space-filling
    dir_name: str  # directory where the result files are written
    exp_name: str  # name tag for each result file
    results_dir: str  # path to the directory for storing the results
    overwrite: bool  # whether to overwrite the existing results or skip the experiment

    def __init__(
        self,
        problem_specs: tuple[Objective, float],
        rule: str,
        model_type: str,
        seed: int,
        initial_budget_per_dimension: int,
        batch_size: int,
        budget_per_dimension: int,
        results_dir: str = EXPERIMENTS_RAW_DATA,
        overwrite: bool = False,
    ):
        self.problem = get_problem(problem_specs)
        self.rule = rule
        self.model_type = model_type
        self.seed = seed
        self.batch_size = batch_size
        self.num_initial_points = initial_budget_per_dimension * self.problem.dim
        self.budget = budget_per_dimension * self.problem.dim
        self.results_dir = results_dir
        self.overwrite = overwrite

        self.exp_name = (
            f"problem_{self.problem.problem_name}"
            f"_rule_{self.rule}"
            f"_model_{self.model_type}"
            f"_init_{self.num_initial_points}"
            f"_budget_{self.budget}"
            f"_batch_{self.batch_size}"
            f"_seed_{self.seed}"
        )

        subdir_name = (
            f"rule_{self.rule}"
            f"_model_{self.model_type}"
            f"_init_{self.num_initial_points}"
            f"_budget_{self.budget}"
            f"_batch_{self.batch_size}"
        )

        self.dir_name = os.path.join(self.results_dir, self.problem.problem_name, subdir_name)

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

""" To run a single experiment. """

import os
import random
from time import perf_counter

import numpy as np
import pandas as pd
import tensorflow as tf
from hv.active_learning.config import ActiveLearningExperimentConfig
from hv.active_learning.metrics import compute_metrics
from hv.active_learning.model import build_model
from hv.active_learning.rule import build_acquisition_rule
from hv.config.active_learning import EXPERIMENTS_RAW_DATA
from hv.experiments import make_experiment_dir

import trieste
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.data import Dataset
from trieste.observer import OBJECTIVE
from trieste.utils import Timer
from .. import logging


summary_writer = tf.summary.create_file_writer("logs/tensorboard/experiment4")
trieste.logging.set_tensorboard_writer(summary_writer)


class Experiment:
    """"
    - uses ask tell
    - compute metrics using callables
    """
    def __init__(
        config: ExperimentConfig, save: bool = False
    ):
        self._config = ExperimentConfig(**config)
        self._save = save

    def observe(query_points: TensorType) -> Dataset:
        return Dataset(query_points, self._config.problem.fun(query_points))

    def initial_design() -> TensorType:
        search_space = trieste.space.Box(self._config.problem.lower_bounds, self._config.problem.upper_bounds)
        initial_query_points = search_space.sample_halton(self._config.num_initial_points)
        return initial_query_points

    def check_if_done() -> bool:
        return os.path.exists(
            os.path.join(self._config.results_dir, f"{self._config.experiment_name}.csv")
        )
    def get_num_iterations(data: Dataset) -> int
        num_iterations = (
            self._config.budget - data.observations.shape[0]
        ) // self._config.num_query_points
        return num_iterations

    def get_model(data: Dataset) -> TrainableProbabilisticModel:
        model = build_model(data, search_space, self._config.model_type)
        return model

    def get_rule() -> AcquisitionRule:
        acquisition_rule = build_acquisition_rule(self._config, search_space)
        return acquisition_rule

    def _run():

        # set the seed
        os.environ["PYTHONHASHSEED"] = str(self._config.seed)
        np.random.seed(self._config.seed)
        tf.random.set_seed(self._config.seed)
        random.seed(self._config.seed)

        initial_query_points = initial_design()
        initial_data = self.observe(initial_query_points)

        model = self.get_model(initial_data)
        acquisition_rule = self.get_rule()
        ask_tell = AskTellOptimizer(search_space, data, model, acquisition_rule)

        # compute metrics before beginning with optimization
        metrics_to_csv = {}
        metrics = compute_metrics(ask_tell, initial_data, self.config)
        for m, v in metrics.items():
            metrics_to_csv[m] = [v]
        metrics_to_csv["acq_step"] = [0]
        metrics_to_csv["acq_time"] = [-1.0]
        metrics_to_csv["model_time"] = [-1.0]

        if config.retrain:
            num_acq_per_loop = 10
            num_loops = num_acquisitions // num_acq_per_loop
            retrain_iteration = 

        # optimization
        num_iterations = get_num_iterations(data)
        for step in range(num_iterations):

            logging.set_step_number(step)

            if self._config.retrain:
                model = self.get_model(data)
                acquisition_rule = self.get_rule()
                ask_tell = AskTellOptimizer(search_space, data, model, acquisition_rule)

            with Timer() as acquisition_timer:
                query_points = ask_tell.ask()

            new_data = self.observe(query_points)
            with Timer() as model_fitting_timer:
                ask_tell.tell(new_data)

            data = ask_tell.try_get_final_dataset()

            metrics = compute_metrics(ask_tell, new_data, self.config)
            for m, v in metrics.items():
                metrics_to_csv[m] = metrics_to_csv[m] + [v]
            metrics_to_csv["acq_step"] = metrics_to_csv["acq_step"] + [step]
            metrics_to_csv["acq_time"] = metrics_to_csv["acq_time"] + [acquisition_timer.time]
            metrics_to_csv["model_time"] = metrics_to_csv["model_time"] + [model_fitting_timer.time]

        # record exp config
        metrics_to_csv["problem_name"] = [CONFIG.problem.problem_name] * num_iterations
        metrics_to_csv["problem_dim"] = [int(CONFIG.problem.dim)] * num_iterations
        metrics_to_csv["volume"] = [float(CONFIG.problem.estimated_volume)] * num_iterations
        metrics_to_csv["rule"] = [CONFIG.rule] * num_iterations
        metrics_to_csv["model_type"] = [CONFIG.model_type] * num_iterations
        metrics_to_csv["seed"] = [CONFIG.seed] * num_iterations
        metrics_to_csv["budget"] = [int(CONFIG.budget)] * num_iterations
        metrics_to_csv["batch_size"] = [CONFIG.batch_size] * num_iterations
        metrics_to_csv["num_initial_points"] = [int(CONFIG.num_initial_points)] * num_iterations

        return ask_tell, metrics_to_csv

    def __call__(self) -> None:

        experiment_done = self.check_if_done()

        if not experiment_done or (experiment_done and config.overwrite):
            ask_tell, metrics = self._run()

            log.INFO(f"Finished experiment {experiment_name}")
            if self._save:
                self.save(ask_tell, metrics)
            else:
                return ask_tell, metrics

        else:
            log.INFO(f"Skipping experiment {experiment_name}, already done!")
            return None

    def save(ask_tell, metrics) -> None:

        make_experiment_dir(self._config.results_dir)

        metrics_to_csv = pd.DataFrame(metrics)
        metrics_to_csv.to_csv(
            os.path.join(self._config.results_dir, f"{self._config.experiment_name}.csv"),
            index=False
        )

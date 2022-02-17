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

""" Useful functions for computing the performance metrics. """

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from hv.active_learning.config import ActiveLearningExperimentConfig
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from trieste.acquisition.rule import OBJECTIVE
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.types import TensorType


class Metric:


class SimpleRegret(Metric):


class ProbabilityOfFailure(Metric):
    """docstring for ClassName"""
    def __init__(self, arg):
        super(ClassName, self).__init__()
        self.arg = arg
        

def compute_metrics(
    ask_tell: AskTellOptimizer, config: ActiveLearningExperimentConfig
) -> Dict[str, float]:
    """
    Return metrics.
    """
    threshold = config.problem.threshold
    model = ask_tell._models[OBJECTIVE]  # pylint: disable=protected-access
    global_mean, global_variance = model.predict(config.problem.global_x)
    global_mean_feasible = tf.cast(global_mean < threshold, tf.float32)
    boundary_mean, boundary_variance = model.predict(config.problem.boundary_x)
    boundary_mean_feasible = tf.cast(boundary_mean < threshold, tf.float32)
    metrics = {}

    # classification accuracy
    metrics["cla_acc_global"] = _classification_metric(
        config.problem.global_feasible, global_mean_feasible, accuracy_score
    )
    metrics["cla_acc_boundary"] = _classification_metric(
        config.problem.boundary_feasible, boundary_mean_feasible, accuracy_score
    )

    # f1 score
    metrics["f1_global"] = _classification_metric(
        config.problem.global_feasible, global_mean_feasible, f1_score
    )
    metrics["f1_boundary"] = _classification_metric(
        config.problem.boundary_feasible, boundary_mean_feasible, f1_score
    )

    # matthews correlation
    metrics["mcc_global"] = _classification_metric(
        config.problem.global_feasible, global_mean_feasible, matthews_corrcoef
    )
    metrics["mcc_boundary"] = _classification_metric(
        config.problem.boundary_feasible, boundary_mean_feasible, matthews_corrcoef
    )

    # probability of failure
    metrics["prob_failure_error"] = _prob_failure(
        config.problem.estimated_volume, global_mean_feasible
    )

    # probability of misclassification
    metrics["prob_misclassification_global"] = _probability_of_misclassification(
        global_mean, global_variance, threshold
    )
    metrics["prob_misclassification_boundary"] = _probability_of_misclassification(
        boundary_mean, boundary_variance, threshold
    )

    # exc probability accuracy
    metrics["exc_acc_global"] = _get_excursion_accuracy(global_mean, global_variance, threshold)
    metrics["exc_acc_boundary"] = _get_excursion_accuracy(
        boundary_mean, boundary_variance, threshold
    )

    return metrics


def _classification_metric(
    y_true: TensorType, y_pred: TensorType, metric: Callable[[TensorType, TensorType], float]
) -> float:
    try:
        return float(metric(y_true, y_pred))
    except Exception:  # pylint: disable=broad-except
        return np.nan


def _prob_failure(y_true: float, y_feasible: TensorType) -> float:
    try:
        return float(tf.reduce_mean(y_feasible) - y_true)
    except Exception:  # pylint: disable=broad-except
        return np.nan


def _excursion_probability(mean: TensorType, variance: TensorType, threshold: float) -> TensorType:
    normal = tfp.distributions.Normal(tf.cast(0, mean.dtype), tf.cast(1, mean.dtype))
    t = (mean - threshold) / tf.sqrt(variance)
    return normal.cdf(t)


def _probability_of_misclassification(
    mean: TensorType, variance: TensorType, threshold: float
) -> float:
    try:
        normal = tfp.distributions.Normal(tf.cast(0, mean.dtype), tf.cast(1, mean.dtype))
        t = tf.abs(mean - threshold) / tf.sqrt(variance)
        return float(tf.reduce_mean(normal.cdf(t)))
    except Exception:  # pylint: disable=broad-except
        return np.nan


def _get_excursion_accuracy(mean: TensorType, variance: TensorType, threshold: float) -> float:
    try:
        prob = _excursion_probability(mean, variance, threshold)
        accuracy = prob * (1 - prob)
        return float(tf.reduce_mean(accuracy, axis=0))
    except Exception:  # pylint: disable=broad-except
        return np.nan



def _get_brier_score(
    y_true: TensorType, mean: TensorType, variance: TensorType, threshold: float
) -> float:
    try:
        prob = _excursion_probability(mean, variance, threshold)
        brier = tf.math.squared_difference(1 - prob, tf.cast(y_true, prob.dtype))
        breakpoint()
        brier_score_loss(y_true, 1 - prob)
        return float(tf.reduce_mean(brier, axis=0))
    except Exception:  # pylint: disable=broad-except
        return np.nan

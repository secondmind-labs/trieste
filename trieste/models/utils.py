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

"""
This module contains auxiliary objects and functions that are used by multiple model types.
"""

from __future__ import annotations

from typing import Any, Optional

import gpflow
import tensorflow as tf
from gpflow.utilities.traversal import _merge_leaf_components, leaf_components

from .. import logging
from ..data import Dataset
from ..utils.misc import get_variables
from .interfaces import ProbabilisticModel, TrainableProbabilisticModel


def write_summary_data_based_metrics(
    dataset: Dataset,
    model: ProbabilisticModel,
    prefix: str = "",
) -> None:
    """
    Logging utility for writing TensorBoard summary of various metrics for model diagnostics.

    :param dataset: The dataset to use for computing the metrics. All available data in the
        dataset will be used.
    :param model: The model to produce metrics for.
    :param prefix: The prefix to add to "accuracy" category of model summaries.
    """
    name = prefix + "accuracy"
    predict = model.predict(dataset.query_points)

    # basics
    logging.histogram(f"{name}/predict_mean", predict[0])
    logging.scalar(f"{name}/predict_mean__mean", tf.reduce_mean(predict[0]))
    logging.histogram(f"{name}/predict_variance", predict[1])
    logging.scalar(f"{name}/predict_variance__mean", tf.reduce_mean(predict[1]))
    logging.histogram(f"{name}/observations", dataset.observations)
    logging.scalar(f"{name}/observations_mean", tf.reduce_mean(dataset.observations))
    logging.scalar(f"{name}/observations_variance", tf.math.reduce_variance(dataset.observations))

    # accuracy metrics
    diffs = tf.cast(dataset.observations, predict[0].dtype) - predict[0]
    z_residuals = diffs / tf.math.sqrt(predict[1])
    logging.histogram(f"{name}/absolute_error", tf.math.abs(diffs))
    logging.histogram(f"{name}/z_residuals", z_residuals)
    logging.scalar(f"{name}/root_mean_square_error", tf.math.sqrt(tf.reduce_mean(diffs**2)))
    logging.scalar(f"{name}/mean_absolute_error", tf.reduce_mean(tf.math.abs(diffs)))
    logging.scalar(f"{name}/z_residuals_std", tf.math.reduce_std(z_residuals))

    # variance metrics
    variance_error = predict[1] - diffs**2
    logging.histogram(f"{name}/variance_error", variance_error)
    logging.scalar(
        f"{name}/root_mean_variance_error",
        tf.math.sqrt(tf.reduce_mean(variance_error**2)),
    )


def write_summary_kernel_parameters(kernel: gpflow.kernels.Kernel, prefix: str = "") -> None:
    """
    Logging utility for writing TensorBoard summary of kernel parameters. Provides useful
    diagnostics for models with a GPflow kernel. Only trainable parameters are logged.

    :param kernel: The kernel to use for computing the metrics.
    :param prefix: The prefix to add to "kernel" category of model summaries.
    """
    components = _merge_leaf_components(leaf_components(kernel))
    for k, v in components.items():
        if v.trainable:
            if tf.rank(v) == 0:
                logging.scalar(f"{prefix}kernel.{k}", v)
            elif tf.rank(v) == 1:
                for i, vi in enumerate(v):
                    logging.scalar(f"{prefix}kernel.{k}[{i}]", vi)


def write_summary_likelihood_parameters(
    likelihood: gpflow.likelihoods.Likelihood, prefix: str = ""
) -> None:
    """
    Logging utility for writing TensorBoard summary of likelihood parameters. Provides useful
    diagnostics for models with a GPflow likelihood. Only trainable parameters are logged.

    :param likelihood: The likelihood to use for computing the metrics.
    :param prefix: The prefix to add to "likelihood" category of model summaries.
    """
    likelihood_components = _merge_leaf_components(leaf_components(likelihood))
    for k, v in likelihood_components.items():
        if v.trainable:
            logging.scalar(f"{prefix}likelihood.{k}", v)


def get_module_with_variables(model: ProbabilisticModel, *dependencies: Any) -> tf.Module:
    """
    Return a fresh module with a model's variables attached, which can then be extended
    with methods and saved using tf.saved_model.

    :param model: Model to extract variables from.
    :param dependencies: Dependent objects whose variables should also be included.
    """
    module = tf.Module()
    module.saved_variables = get_variables(model)
    for dependency in dependencies:
        module.saved_variables += get_variables(dependency)
    return module


def optimize_model_and_save_result(model: TrainableProbabilisticModel, dataset: Dataset) -> None:
    """
    Optimize the model objective and save the (optimizer-specific) optimization result
    in the model object. To access it, use ``get_last_optimization_result``.

    :param dataset: The data with which to train the model.
    """
    setattr(model, "_last_optimization_result", model.optimize(dataset))


def get_last_optimization_result(model: TrainableProbabilisticModel) -> Optional[Any]:
    """
    The last saved (optimizer-specific) optimization result.
    """
    return getattr(model, "_last_optimization_result")

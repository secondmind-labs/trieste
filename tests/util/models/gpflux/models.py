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
Simple GPflux models to be used in the tests.
"""

from __future__ import annotations

from typing import Any, Dict

import gpflow
import tensorflow as tf
from gpflow.utilities import set_trainable
from gpflux.layers import GPLayer
from gpflux.models import DeepGP

from trieste.data import TensorType
from trieste.models.gpflux import DeepGaussianProcess, build_vanilla_deep_gp


def single_layer_dgp_model(x: TensorType) -> DeepGP:
    return build_vanilla_deep_gp(x, num_layers=1, num_inducing=len(x))


def two_layer_dgp_model(x: TensorType) -> DeepGP:
    return build_vanilla_deep_gp(x, num_layers=2, num_inducing=len(x))


def simple_two_layer_dgp_model(x: TensorType) -> DeepGP:
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    x_shape = x.shape[-1]
    num_data = len(x)

    Z = x.copy()
    kernel_1 = gpflow.kernels.SquaredExponential()
    inducing_variable_1 = gpflow.inducing_variables.InducingPoints(Z.copy())
    gp_layer_1 = GPLayer(
        kernel_1,
        inducing_variable_1,
        num_data=num_data,
        num_latent_gps=x_shape,
    )

    kernel_2 = gpflow.kernels.SquaredExponential()
    inducing_variable_2 = gpflow.inducing_variables.InducingPoints(Z.copy())
    gp_layer_2 = GPLayer(
        kernel_2,
        inducing_variable_2,
        num_data=num_data,
        num_latent_gps=1,
        mean_function=gpflow.mean_functions.Zero(),
    )

    return DeepGP([gp_layer_1, gp_layer_2], gpflow.likelihoods.Gaussian(0.01))


def trieste_deep_gaussian_process(
    query_points: TensorType,
    depth: int,
    num_inducing: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    fix_noise: bool = False,
) -> [DeepGaussianProcess, Dict[str, Any]]:
    dgp = build_vanilla_deep_gp(query_points, num_layers=depth, num_inducing=num_inducing)
    if fix_noise:
        dgp.likelihood_layer.likelihood.variance.assign(1e-4)
        set_trainable(dgp.likelihood_layer, False)
    optimizer = tf.optimizers.Adam(learning_rate)

    def scheduler(epoch: int, lr: float) -> float:
        if epoch == epochs // 2:
            return lr * 0.1
        else:
            return lr

    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
        "callbacks": tf.keras.callbacks.LearningRateScheduler(scheduler),
    }

    model = DeepGaussianProcess(dgp, optimizer, fit_args)

    return model, fit_args

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

from typing import Any, Dict, Tuple

import gpflow
import tensorflow as tf
from gpflow.keras import tf_keras
from gpflow.utilities import set_trainable
from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from gpflux.helpers import construct_basic_kernel
from gpflux.layers import GPLayer
from gpflux.models import DeepGP

from trieste.data import Dataset, TensorType
from trieste.models.gpflux import DeepGaussianProcess, build_vanilla_deep_gp
from trieste.models.optimizer import KerasOptimizer
from trieste.space import SearchSpace
from trieste.utils import to_numpy


def single_layer_dgp_model(x: TensorType) -> DeepGP:
    x = to_numpy(x)

    config = Config(
        num_inducing=len(x),
        inner_layer_qsqrt_factor=1e-5,
        likelihood_noise_variance=1e-2,
        whiten=True,  # whiten = False not supported yet in GPflux for this model
    )

    return build_constant_input_dim_deep_gp(X=x, num_layers=1, config=config)


def two_layer_dgp_model(x: TensorType) -> DeepGP:
    x = to_numpy(x)

    config = Config(
        num_inducing=len(x),
        inner_layer_qsqrt_factor=1e-5,
        likelihood_noise_variance=1e-2,
        whiten=True,  # whiten = False not supported yet in GPflux for this model
    )

    return build_constant_input_dim_deep_gp(X=x, num_layers=2, config=config)


def simple_two_layer_dgp_model(x: TensorType) -> DeepGP:
    x = to_numpy(x)
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


def separate_independent_kernel_two_layer_dgp_model(x: TensorType) -> DeepGP:
    x = to_numpy(x)
    x_shape = x.shape[-1]
    num_data = len(x)

    Z = x.copy()
    kernel_list = [
        gpflow.kernels.SquaredExponential(
            variance=tf.exp(tf.random.normal([], dtype=gpflow.default_float())),
            lengthscales=tf.exp(tf.random.normal([], dtype=gpflow.default_float())),
        )
        for _ in range(x_shape)
    ]
    kernel_1 = construct_basic_kernel(kernel_list)
    inducing_variable_1 = gpflow.inducing_variables.SharedIndependentInducingVariables(
        gpflow.inducing_variables.InducingPoints(Z.copy())
    )
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
    data: Dataset,
    search_space: SearchSpace,
    num_layers: int,
    num_inducing_points: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    fix_noise: bool = False,
) -> Tuple[DeepGaussianProcess, Dict[str, Any]]:
    dgp = build_vanilla_deep_gp(data, search_space, num_layers, num_inducing_points)
    if fix_noise:
        dgp.likelihood_layer.likelihood.variance.assign(1e-5)
        set_trainable(dgp.likelihood_layer, False)

    def scheduler(epoch: int, lr: float) -> float:
        if epoch == epochs // 2:
            return lr * 0.1
        else:
            return lr

    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
        "callbacks": tf_keras.callbacks.LearningRateScheduler(scheduler),
    }
    optimizer = KerasOptimizer(tf_keras.optimizers.Adam(learning_rate), fit_args)

    model = DeepGaussianProcess(dgp, optimizer)

    return model, fit_args


def two_layer_trieste_dgp(data: Dataset, search_space: SearchSpace) -> DeepGaussianProcess:
    return trieste_deep_gaussian_process(data, search_space, 2, 10, 0.01, 5, 10)[0]

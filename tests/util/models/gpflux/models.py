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

from __future__ import annotations

import gpflow
import tensorflow as tf
from gpflux.layers import GPLayer
from gpflux.models import DeepGP

from trieste.models.gpflux import build_vanilla_deep_gp


def single_layer_dgp_model(x: tf.Tensor) -> DeepGP:
    return build_vanilla_deep_gp(x, num_layers=1, num_inducing=len(x))


def two_layer_dgp_model(x: tf.Tensor) -> DeepGP:
    return build_vanilla_deep_gp(x, num_layers=2, num_inducing=len(x))


def two_layer_dgp_model_no_whitening(x: tf.Tensor) -> DeepGP:
    x = x.numpy()
    num_data = len(x)

    Z = x.copy()
    kernel_1 = gpflow.kernels.SquaredExponential()
    inducing_variable_1 = gpflow.inducing_variables.InducingPoints(Z.copy())
    gp_layer_1 = GPLayer(
        kernel_1,
        inducing_variable_1,
        num_data=num_data,
        num_latent_gps=1,
        whiten=False,
    )

    kernel_2 = gpflow.kernels.SquaredExponential()
    inducing_variable_2 = gpflow.inducing_variables.InducingPoints(Z.copy())
    gp_layer_2 = GPLayer(
        kernel_2,
        inducing_variable_2,
        num_data=num_data,
        num_latent_gps=1,
        whiten=False,
        mean_function=gpflow.mean_functions.Zero(),
    )

    return DeepGP([gp_layer_1, gp_layer_2], gpflow.likelihoods.Gaussian(0.01))


def fnc_3x_plus_10(x: tf.Tensor) -> tf.Tensor:
    return 3.0 * x + 10


def fnc_2sin_x_over_3(x: tf.Tensor) -> tf.Tensor:
    return 2.0 * tf.math.sin(x / 3.0)

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

r"""
This file contains wrappers for some implementations of basic GPflux architectures.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from gpflux.models import DeepGP

from ...types import TensorType

tf.keras.backend.set_floatx("float64")


def build_vanilla_deep_gp(
    X: TensorType,
    num_layers: int,
    num_inducing: int,
    inner_layer_sqrt_factor: float = 1e-5,
    likelihood_noise_variance: float = 1e-2,
) -> DeepGP:
    """
    Provides a wrapper around `build_constant_input_dim_deep_gp` from `gpflux.architectures`.

    :param X: input data, used to determine inducing point locations with k-means.
    :param num_layers: number of layers in deep GP.
    :param num_inducing: number of inducing points to use in each layer.
    :param inner_layer_sqrt_factor: A multiplicative factor used to rescale hidden layers
    :param likelihood_noise_variance: initial noise variance
    :return: :class:`gpflux.models.DeepGP`
    """

    # Input data to model must be np.ndarray for k-means algorithm
    if isinstance(X, tf.Tensor):
        X = X.numpy()

    # Pad X with additional random values to provide enough inducing points
    if num_inducing > len(X):
        X = np.concatenate([X, np.random.randn(num_inducing - len(X), *X.shape[1:])], 0)

    config = Config(
        num_inducing=num_inducing,
        inner_layer_qsqrt_factor=inner_layer_sqrt_factor,
        likelihood_noise_variance=likelihood_noise_variance,
        whiten=True,  # whiten = False not supported yet in GPflux for this model
    )

    return build_constant_input_dim_deep_gp(X, num_layers, config)

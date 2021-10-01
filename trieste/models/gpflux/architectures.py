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
This file contains wrappers for some implementations of basic GPflux architectures. Currently only
a `vanilla` DGP architecture, based off the `gpflux.architectures.build_constant_input_dim_deep_gp`
is provided.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf
from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from gpflux.models import DeepGP

from ...space import Box
from ...types import TensorType


def build_vanilla_deep_gp(
    query_points: TensorType,
    num_layers: int,
    num_inducing: int,
    inner_layer_sqrt_factor: float = 1e-5,
    likelihood_noise_variance: float = 1e-2,
    search_space: Optional[Box] = None,
) -> DeepGP:
    """
    Provides a wrapper around `build_constant_input_dim_deep_gp` from `gpflux.architectures`.

    :param query_points: input data, used to determine inducing point locations with k-means.
    :param num_layers: number of layers in deep GP.
    :param num_inducing: number of inducing points to use in each layer.
    :param inner_layer_sqrt_factor: A multiplicative factor used to rescale hidden layers
    :param likelihood_noise_variance: initial noise variance
    :param search_space: the search space for the Bayes Opt problem. Used for initialization of
        inducing locations if num_inducing > len(query_points)
    :return: :class:`gpflux.models.DeepGP`
    """

    # Input data to model must be np.ndarray for k-means algorithm
    if isinstance(query_points, tf.Tensor):
        query_points = query_points.numpy()

    num_data = len(query_points)

    # Pad query_points with additional random values to provide enough inducing points
    if num_inducing > len(query_points):
        if search_space is not None:
            if not isinstance(search_space, Box):
                raise ValueError(
                    f"Currently only `Box` instances are supported for `search_space`,"
                    f" received {type(search_space)}."
                )
            additional_points = search_space.sample_sobol(num_inducing - len(query_points)).numpy()
        else:
            additional_points = np.random.randn(
                num_inducing - len(query_points), *query_points.shape[1:]
            )
        query_points = np.concatenate([query_points, additional_points], 0)

    config = Config(
        num_inducing,
        inner_layer_sqrt_factor,
        likelihood_noise_variance,
        whiten=True,  # whiten = False not supported yet in GPflux for this model
    )

    model = build_constant_input_dim_deep_gp(query_points, num_layers, config)

    # If num_inducing is larger than the number of provided query points, the initialization for
    # num_data will be wrong. We therefore make sure it is set correctly.
    model.num_data = num_data
    for layer in model.f_layers:
        layer.num_data = num_data

    return model

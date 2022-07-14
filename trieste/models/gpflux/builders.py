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
This file contains builders for GPflux models supported in Trieste. We found the default
configurations used here to work well in most situation, but they should not be taken as
universally good solutions.
"""

from __future__ import annotations

from typing import Optional

import gpflow
import numpy as np
import tensorflow as tf
from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from gpflux.models import DeepGP

from ...data import Dataset
from ...space import Box, SearchSpace
from ...types import TensorType

NUM_LAYERS: int = 2
"""
Default number of layers in the deep gaussian process model.
"""


MAX_NUM_INDUCING_POINTS: int = 500
"""
Default maximum number of inducing points.
"""


NUM_INDUCING_POINTS_PER_DIM: int = 50
"""
Default number of inducing points per dimension of the search space.
"""


INNER_LAYER_SQRT_FACTOR: float = 1e-5
"""
Default value for a multiplicative factor used to rescale hidden layers.
"""


LIKELIHOOD_VARIANCE: float = 1e-3
"""
Default value for an initial noise variance in the likelihood function.
"""


def build_vanilla_deep_gp(
    data: Dataset,
    search_space: SearchSpace,
    num_layers: int = NUM_LAYERS,
    num_inducing_points: Optional[int] = None,
    inner_layer_sqrt_factor: float = INNER_LAYER_SQRT_FACTOR,
    likelihood_variance: float = LIKELIHOOD_VARIANCE,
    trainable_likelihood: bool = True,
) -> DeepGP:
    """
    Build a :class:`~gpflux.models.DeepGP` model with sensible initial parameters. We found the
    default configuration used here to work well in most situation, but it should not be taken as a
    universally good solution.

    Note that although we set all the relevant parameters to sensible values, we rely on
    ``build_constant_input_dim_deep_gp`` from :mod:`~gpflux.architectures` to build the model.

    :param data: Dataset from the initial design, used to estimate the variance of observations
        and to provide query points which are used to determine inducing point locations with
        k-means.
    :param search_space: Search space for performing Bayesian optimization. Used for initialization
        of inducing locations if ``num_inducing_points`` is larger than the amount of data.
    :param num_layers: Number of layers in deep GP. By default set to ``NUM_LAYERS``.
    :param num_inducing_points: Number of inducing points to use in each layer. If left unspecified
        (default), this number is set to either ``NUM_INDUCING_POINTS_PER_DIM``*dimensionality of
        the search space or value given by ``MAX_NUM_INDUCING_POINTS``, whichever is smaller.
    :param inner_layer_sqrt_factor: A multiplicative factor used to rescale hidden layers, see
        :class:`~gpflux.architectures.Config` for details. By default set to
        ``INNER_LAYER_SQRT_FACTOR``.
    :param likelihood_variance: Initial noise variance in the likelihood function, see
        :class:`~gpflux.architectures.Config` for details. By default set to
        ``LIKELIHOOD_VARIANCE``.
    :param trainable_likelihood: Trainable likelihood variance.
    :return: A :class:`~gpflux.models.DeepGP` model with sensible default settings.
    :raise: If non-positive ``num_layers``, ``inner_layer_sqrt_factor``, ``likelihood_variance``
        or ``num_inducing_points`` is provided.
    """
    tf.debugging.assert_positive(num_layers)
    tf.debugging.assert_positive(inner_layer_sqrt_factor)
    tf.debugging.assert_positive(likelihood_variance)

    # Input data to ``build_constant_input_dim_deep_gp`` must be np.ndarray for k-means algorithm
    query_points = data.query_points.numpy()

    empirical_mean, empirical_variance, num_data_points = _get_data_stats(data)

    if num_inducing_points is None:
        num_inducing_points = min(
            MAX_NUM_INDUCING_POINTS, NUM_INDUCING_POINTS_PER_DIM * int(search_space.dimension)
        )
    else:
        tf.debugging.assert_positive(num_inducing_points)

    # Pad query_points with additional random values to provide enough inducing points
    if num_inducing_points > len(query_points):
        if isinstance(search_space, Box):
            additional_points = search_space.sample_sobol(
                num_inducing_points - len(query_points)
            ).numpy()
        else:
            additional_points = search_space.sample(num_inducing_points - len(query_points)).numpy()
        query_points = np.concatenate([query_points, additional_points], 0)

    config = Config(
        num_inducing_points,
        inner_layer_sqrt_factor,
        likelihood_variance,
        whiten=True,  # whiten = False not supported yet in GPflux for this model
    )

    model = build_constant_input_dim_deep_gp(query_points, num_layers, config)

    model.f_layers[-1].kernel.kernel.variance.assign(empirical_variance)
    model.f_layers[-1].mean_function = gpflow.mean_functions.Constant(empirical_mean)

    gpflow.set_trainable(model.likelihood_layer.likelihood.variance, trainable_likelihood)

    # If num_inducing_points is larger than the number of provided query points, the initialization
    # for num_data_points will be wrong. We therefore make sure it is set correctly.
    model.num_data = num_data_points
    for layer in model.f_layers:
        layer.num_data = num_data_points

    return model


def _get_data_stats(data: Dataset) -> tuple[TensorType, TensorType, int]:
    empirical_variance = tf.math.reduce_variance(data.observations)
    empirical_mean = tf.math.reduce_mean(data.observations)
    num_data_points = len(data.observations)

    return empirical_mean, empirical_variance, num_data_points

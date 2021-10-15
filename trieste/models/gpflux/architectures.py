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

from typing import Optional, Tuple

import gpflow.mean_functions
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import Parameter
from gpflow.utilities.bijectors import positive
from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from gpflux.encoders import DirectlyParameterizedNormalDiag
from gpflux.helpers import (
    construct_basic_inducing_variables,
    construct_basic_kernel,
    construct_mean_function,
)
from gpflux.layers import GIGPLayer, GPLayer, LatentVariableLayer, LikelihoodLayer
from gpflux.models import DeepGP, GIDeepGP
from gpflux.types import ObservationType
from scipy.cluster.vq import kmeans2

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

    # Pad query_points with additional random values to provide enough inducing points
    if num_inducing > len(query_points):
        if search_space is not None:
            if not isinstance(search_space, Box):
                raise ValueError("Currently only `Box` instances are supported for `search_space`.")
            additional_points = search_space.sample_sobol(num_inducing - len(query_points)).numpy()
        else:
            additional_points = np.random.randn(
                num_inducing - len(query_points), *query_points.shape[1:]
            )
        query_points = np.concatenate([query_points, additional_points], 0)

    # TODO: make sure this provides the correct num_data - should be fine atm

    config = Config(
        num_inducing=num_inducing,
        inner_layer_qsqrt_factor=inner_layer_sqrt_factor,
        likelihood_noise_variance=likelihood_noise_variance,
        whiten=True,  # whiten = False not supported yet in GPflux for this model
    )

    return build_constant_input_dim_deep_gp(query_points, num_layers, config)


def build_gi_deep_gp(
    query_points: TensorType,
    num_layers: int,
    num_inducing: int,
    observations: Optional[TensorType] = None,
    inner_layer_prec_init: float = 1.,
    last_layer_prec_init: float = 10.,
    likelihood_noise_variance: float = 1e-2,
    last_layer_variance: float = 1.0,
    num_train_samples: int = 10,
    num_test_samples: int = 100,
    search_space: Optional[Box] = None,
) -> GIDeepGP:
    """
    Provides a global inducing version of `build_constant_input_dim_deep_gp` from
    `gpflux.architectures`.

    :param query_points: input data, used to determine inducing point locations with k-means.
    :param num_layers: number of layers in deep GP.
    :param num_inducing: number of inducing points to use in each layer.
    :param observations: output data, used to initialize the last layer inducing outputs
    :param inner_layer_prec_init: initialization of the scale of the inner layer precision
    :param last_layer_prec_init: initialization of the scale of the last layer precision
    :param likelihood_noise_variance: initial noise variance
    :param last_layer_variance: initial last layer kernel variance
    :param num_train_samples: number of samples for training
    :param num_test_samples: number of samples for testing
    :param search_space: the search space for the Bayes Opt problem. Used for initialization of
        inducing locations if num_inducing > len(query_points)
    :return: :class:`gpflux.models.GIDeepGP`
    """
    tf.debugging.assert_rank(
        query_points, 2, message="For this architecture, the rank of the input data must " "be 2."
    )
    num_data, input_dim = query_points.shape

    # Input data to model must be np.ndarray for k-means algorithm
    if isinstance(query_points, tf.Tensor):
        query_points = query_points.numpy()

    # Pad query_points with additional random values to provide enough inducing points
    if num_inducing > len(query_points):
        if search_space is not None:
            if not isinstance(search_space, Box):
                raise ValueError("Currently only `Box` instances are supported for `search_space`.")
            additional_points = search_space.sample_sobol(num_inducing - len(query_points)).numpy()
        else:
            additional_points = np.random.randn(
                num_inducing - len(query_points), *query_points.shape[1:]
            )
        query_points = np.concatenate([query_points, additional_points], 0)

    if observations is not None:
        tf.debugging.assert_shapes(
            observations,
            [num_data, 1],
            message="For this architecture, the output dim must be" " 1.",
        )
        if isinstance(observations, tf.Tensor):
            observations = observations.numpy()

        if num_inducing > len(query_points):
            observations = np.concatenate(
                [
                    observations,
                    np.random.randn(num_inducing - len(query_points), *observations.shape[1:]),
                ],
                0,
            )

    query_points_running = query_points

    gp_layers = []
    inducing_init, _ = kmeans2(query_points, k=num_inducing, minit="points")

    for i_layer in range(num_layers):
        is_last_layer = i_layer == num_layers - 1
        D_in = input_dim
        D_out = 1 if is_last_layer else input_dim

        if is_last_layer:
            mean_function = gpflow.mean_functions.Zero()
        else:
            mean_function = construct_mean_function(query_points_running, D_in, D_out)
            query_points_running = mean_function(query_points_running)
            if tf.is_tensor(query_points_running):
                query_points_running = query_points_running.numpy()

        layer = GIGPLayer(
            input_dim=D_in,
            num_latent_gps=D_out,
            num_data=num_data,
            num_inducing=num_inducing,
            mean_function=mean_function,
            kernel_variance_init=last_layer_variance if is_last_layer else 1.0,
            prec_init=inner_layer_prec_init if not is_last_layer else last_layer_prec_init,
            inducing_targets=None if not is_last_layer else observations,
        )
        gp_layers.append(layer)

    return GIDeepGP(
        f_layers=gp_layers,
        num_inducing=num_inducing,
        likelihood_var=likelihood_noise_variance,
        inducing_init=inducing_init.copy(),
        inducing_shape=None,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
    )


def build_latent_variable_dgp_model(
    query_points: TensorType,
    num_total_data: int,
    num_layers: int,
    num_inducing: int,
    inner_layer_sqrt_factor: float = 1e-5,
    likelihood_noise_variance: float = 1e-2,
    latent_dim: int | None = None,
    prior_std: float = 1.0,
    search_space: Optional[Box] = None,
) -> DeepGP:
    """
    Provides a DGP model with a latent variable layer in the input.

    :param query_points: input data, used to determine inducing point locations with k-means.
    :param num_total_data: total dataset size (including points to be acquired in Bayes Opt loop).
    :param num_layers: number of layers in deep GP.
    :param num_inducing: number of inducing points to use in each layer.
    :param inner_layer_sqrt_factor: A multiplicative factor used to rescale hidden layers
    :param likelihood_noise_variance: initial noise variance.
    :param latent_dim: dimension of latent variable in input layer.
    :param prior_std: initialization for latent variable prior standard deviation, which is learned
    :param search_space: the search space for the Bayes Opt problem. Used for initialization of
        inducing locations if num_inducing > len(query_points)
    :return: :class:`gpflux.models.DeepGP`
    """
    # Input data to model must be np.ndarray for k-means algorithm
    if isinstance(query_points, tf.Tensor):
        query_points = query_points.numpy()

    num_data, input_dim = query_points.shape

    # Pad query_points with additional random values to provide enough inducing points
    if num_inducing > len(query_points):
        if search_space is not None:
            if not isinstance(search_space, Box):
                raise ValueError("Currently only `Box` instances are supported for `search_space`.")
            additional_points = search_space.sample_sobol(num_inducing - len(query_points)).numpy()
        else:
            additional_points = np.random.randn(
                num_inducing - len(query_points), *query_points.shape[1:]
            )
        query_points = np.concatenate([query_points, additional_points], 0)

    if latent_dim is None:
        latent_dim = input_dim

    prior_means = np.zeros(latent_dim)
    prior_std_param = Parameter(prior_std, dtype=gpflow.default_float(), transform=positive())
    prior_std = prior_std_param * np.ones(latent_dim)
    encoder = DirectlyParameterizedNormalDiag(num_total_data, latent_dim)
    prior = tfp.distributions.MultivariateNormalDiag(prior_means, prior_std)
    lv = ModifiedLatentVariableLayer(num_data, prior, encoder)

    gp_layers = [lv]
    centroids, _ = kmeans2(query_points, k=num_inducing, minit="points")
    centroids_ext = np.concatenate(
        [
            centroids,
            prior_std * np.random.randn(num_inducing, latent_dim),
        ],
        axis=1,
    )

    def _construct_kernel(
        input_dim: int,
        latent_dim: int,
        is_first_layer: bool,
    ) -> gpflow.kernels.SquaredExponential:
        if is_first_layer:
            lengthscales = [0.05] * (input_dim - latent_dim) + [2.0] * latent_dim
        else:
            lengthscales = [2.0] * input_dim
        kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales, variance=1.0)
        return kernel

    for i_layer in range(num_layers):
        is_first_layer = i_layer == 0
        is_last_layer = i_layer == num_layers - 1
        D_in = input_dim + latent_dim if is_first_layer else input_dim
        D_out = 1 if is_last_layer else input_dim

        z_init = centroids_ext if is_first_layer else centroids
        inducing_var = construct_basic_inducing_variables(
            num_inducing=num_inducing, input_dim=D_in, share_variables=True, z_init=z_init.copy()
        )

        kernel = construct_basic_kernel(
            kernels=_construct_kernel(D_in, latent_dim, is_first_layer),
            output_dim=D_out,
            share_hyperparams=True,
        )

        if is_first_layer:
            mean_function = gpflow.mean_functions.Zero()
            q_sqrt_scaling = inner_layer_sqrt_factor
        elif is_last_layer:
            mean_function = gpflow.mean_functions.Zero()
            q_sqrt_scaling = 1e-5
        else:
            mean_function = gpflow.mean_functions.Identity()
            q_sqrt_scaling = inner_layer_sqrt_factor

        layer = GPLayer(
            kernel, inducing_var, num_data, mean_function=mean_function, name=f"gp_{i_layer}"
        )
        layer.q_sqrt.assign(layer.q_sqrt * q_sqrt_scaling)
        gp_layers.append(layer)

    likelihood = gpflow.likelihoods.Gaussian(likelihood_noise_variance)
    return DeepGP(gp_layers, LikelihoodLayer(likelihood))


class ModifiedLatentVariableLayer(LatentVariableLayer):
    """
    Modified version of :class:`gpflux.layers.LatentVariableLayer` to enable BayesOpt-specific
    capabilities, such as variable dataset size. Refer to GPflux for documentation.
    """

    def __init__(
        self,
        num_data: int,
        prior: tfp.distributions.Distribution,
        encoder: tf.keras.layers.Layer,
        compositor: Optional[tf.keras.layers.Layer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(prior, encoder, compositor, name)

        self.num_data = num_data

    def _inference_latent_samples_and_loss(
        self, layer_inputs: TensorType, observations: ObservationType, seed: Optional[int] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample latent variables during the *training* forward pass, hence requiring the
        observations. Also return the KL loss per datapoint. Modified from
        :class:`gpflux.layers.LatentVariableLayer` so that only the KLs for the latent variables
        corresponding to the current query points are added to the loss - more KLs are added as
        more query points are acquired.

        :param layer_inputs: The output of the previous layer: used for determining how many
            latent variables we need at this stage.
        :param observations: The ``[inputs, targets]``, with the shapes ``[batch_size, Din]`` and
            ``[batch_size, Dout]``, respectively.
        :param seed: A random seed for the sampling operation.
        :return: The samples and the loss-per-datapoint.
        """
        distribution_params = self.encoder(None, training=True)
        posteriors = self.distribution_class(*distribution_params, allow_nan_stats=False)
        samples = posteriors.sample(seed=seed)[: tf.shape(layer_inputs)[0]]
        local_kls = self._local_kls(posteriors)[: tf.shape(layer_inputs)[0]]
        loss_per_datapoint = tf.reduce_mean(local_kls, name="local_kls")
        return samples, loss_per_datapoint

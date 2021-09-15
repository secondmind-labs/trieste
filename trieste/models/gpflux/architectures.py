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
from gpflux.layers import GPLayer, GIGPLayer, LatentVariableLayer, LikelihoodLayer
from gpflux.models import DeepGP, GIDeepGP
from gpflux.types import ObservationType
from scipy.cluster.vq import kmeans2

from ...types import TensorType


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

    # TODO: make sure this provides the correct num_data - should be fine atm

    config = Config(
        num_inducing=num_inducing,
        inner_layer_qsqrt_factor=inner_layer_sqrt_factor,
        likelihood_noise_variance=likelihood_noise_variance,
        whiten=True,  # whiten = False not supported yet in GPflux for this model
    )

    return build_constant_input_dim_deep_gp(X, num_layers, config)


def build_gi_deep_gp(
    X: TensorType,
    num_layers: int,
    num_inducing: int,
    y: Optional[TensorType] = None,
    inner_layer_prec_init: float = .01,
    last_layer_prec_init: float = 1.,
    likelihood_noise_variance: float = 1e-2,
    last_layer_variance: float = 1.,
    num_train_samples: int = 10,
    num_test_samples: int = 100,
) -> GIDeepGP:
    """
    Provides a global inducing version of `build_constant_input_dim_deep_gp` from
    `gpflux.architectures`.

    :param X: input data, used to determine inducing point locations with k-means.
    :param num_layers: number of layers in deep GP.
    :param num_inducing: number of inducing points to use in each layer.
    :param y: output data, used to initialize the last layer inducing outputs
    :param inner_layer_prec_init: initialization of the scale of the inner layer precision
    :param last_layer_prec_init: initialization of the scale of the last layer precision
    :param likelihood_noise_variance: initial noise variance
    :param last_layer_variance: initial last layer kernel variance
    :param num_train_samples: number of samples for training
    :param num_test_samples: number of samples for testing
    :return: :class:`gpflux.models.GIDeepGP`
    """
    tf.debugging.assert_rank(X, 2, message="For this architecture, the rank of the input data must "
                                           "be 2.")
    num_data, input_dim = X.shape

    # Input data to model must be np.ndarray for k-means algorithm
    if isinstance(X, tf.Tensor):
        X = X.numpy()

    # Pad X with additional random values to provide enough inducing points
    if num_inducing > len(X):
        X = np.concatenate([X, np.random.randn(num_inducing - len(X), *X.shape[1:])], 0)

    if y is not None:
        tf.debugging.assert_shapes(y, [num_data, 1],
                                   message="For this architecture, the output dim must be"
                                           " 1.")
        if isinstance(y, tf.Tensor):
            y = y.numpy()

        if num_inducing > len(X):
            y = np.concatenate([y, np.random.randn(num_inducing - len(X), *y.shape[1:])], 0)

    inducing_init = X.copy()
    X_running = X

    gp_layers = []
    centroids, _ = kmeans2(X, k=num_inducing, minit="points")

    for i_layer in range(num_layers):
        is_last_layer = i_layer == num_layers - 1
        D_in = input_dim
        D_out = 1 if is_last_layer else input_dim

        if is_last_layer:
            mean_function = gpflow.mean_functions.Zero()
        else:
            mean_function = construct_mean_function(X_running, D_in, D_out)
            X_running = mean_function(X_running)
            if tf.is_tensor(X_running):
                X_running = X_running.numpy()

        layer = GIGPLayer(
            input_dim=D_in,
            num_latent_gps=D_out,
            num_data=num_data,
            num_inducing=num_inducing,
            mean_function=mean_function,
            kernel_variance_init=last_layer_variance if is_last_layer else 1.,
            prec_init=inner_layer_prec_init if not is_last_layer else last_layer_prec_init,
            inducing_targets=None if not is_last_layer else y,
        )
        gp_layers.append(layer)

    return GIDeepGP(
        f_layers=gp_layers,
        num_inducing=num_inducing,
        likelihood_var=likelihood_noise_variance,
        inducing_init=inducing_init,
        inducing_shape=None,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
    )


def build_latent_variable_dgp_model(
    X: TensorType,
    num_total_data: int,
    num_layers: int,
    num_inducing: int,
    inner_layer_sqrt_factor: float = 1e-5,
    likelihood_noise_variance: float = 1e-2,
    latent_dim: int | None = None,
    prior_std: float = 1.0,
) -> DeepGP:
    """
    Provides a DGP model with a latent variable layer in the input.

    :param X:
    :param num_layers:
    :param num_inducing:
    :param inner_layer_sqrt_factor:
    :param likelihood_noise_variance:
    :param latent_dim:
    :return:
    """
    # Input data to model must be np.ndarray for k-means algorithm
    if isinstance(X, tf.Tensor):
        X = X.numpy()

    num_data, input_dim = X.shape

    # Pad X with additional random values to provide enough inducing points
    if num_inducing > len(X):
        X = np.concatenate([X, np.random.randn(num_inducing - len(X), *X.shape[1:])], 0)

    if latent_dim is None:
        latent_dim = input_dim

    prior_means = np.zeros(latent_dim)
    prior_std_param = Parameter(prior_std, dtype=gpflow.default_float(), transform=positive())
    prior_std = prior_std_param * np.ones(latent_dim)
    encoder = DirectlyParameterizedNormalDiag(num_total_data, latent_dim)
    prior = tfp.distributions.MultivariateNormalDiag(prior_means, prior_std)
    lv = ModifiedLatentVariableLayer(num_data, prior, encoder)

    gp_layers = [lv]
    centroids, _ = kmeans2(X, k=num_inducing, minit="points")
    centroids_ext = np.concatenate(
        [
            centroids,
            np.random.randn(num_inducing, latent_dim),
        ],
        axis=1,
    )

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
    capabilities, such as variable dataset size.
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
        distribution_params = self.encoder(None, training=True)
        posteriors = self.distribution_class(*distribution_params, allow_nan_stats=False)
        samples = posteriors.sample(seed=seed)[: tf.shape(layer_inputs)[0]]
        local_kls = self._local_kls(posteriors)[: tf.shape(layer_inputs)[0]]
        loss_per_datapoint = tf.reduce_mean(local_kls, name="local_kls")
        return samples, loss_per_datapoint


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

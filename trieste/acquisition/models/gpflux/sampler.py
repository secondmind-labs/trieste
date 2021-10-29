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
This module is the home of the GPflux-specific samplers for use by Trieste's acquisition functions.
"""

from __future__ import annotations

import gpflow.mean_functions
import tensorflow as tf
import tensorflow_probability as tfp
from gpflux.layers import KG, IWLayer

from trieste.acquisition.sampler import Sampler
from trieste.models.gpflux import DeepKernelProcess
from trieste.types import TensorType


class DeepKernelProcessSampler(Sampler):
    r"""
    This sampler employs the *reparameterization trick* to approximate samples from a
    :class:`DeepKernelProcess`\ 's predictive distribution. This sampler is essentially an
    extension of :class:`trieste.acquisition.sampler.IndependentReparametrizationSampler` for use
    with DKP models. We assume a standard DKP model, where all but the last layer are IW layers.
    """

    def __init__(self, sample_size: int, model: DeepKernelProcess):
        """
        :param sample_size: The number of samples for each batch of points. Must be positive.
        :param model: The model to sample from. Must be a :class:`DeepKernelProcess`
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or if
            model is not a :class:`DeepGaussianProcess`.
        """
        if not isinstance(model, DeepKernelProcess):
            raise ValueError("Model must be a trieste.models.gpflux.DeepKernelProcess")

        super().__init__(sample_size, model)

        # Each element of _eps_list is essentially a lazy constant. It is declared and assigned an
        # empty tensor here, and populated on the first call to sample
        self._gamma_list = [
            tf.Variable(tf.ones([sample_size, 0], dtype=tf.float64), shape=[sample_size, None])
            for _ in range(len(model.model_gpflux.f_layers) - 1)
        ]

        self._eps_list = [
            tf.Variable(tf.ones([sample_size, 0, 0], dtype=tf.float64), shape=[sample_size, None, None])
            for _ in range(len(model.model_gpflux.f_layers) - 1)
        ]

        self._Gii_list = [
            tf.Variable(tf.ones([sample_size, 0, 0], dtype=tf.float64), shape=[sample_size, None, None])
            for _ in range(len(model.model_gpflux.f_layers) - 1)
        ]

        self._chol_dKii_list = [
            tf.Variable(tf.ones([sample_size, 0, 0], dtype=tf.float64), shape=[sample_size, None, None])
            for _ in range(len(model.model_gpflux.f_layers) - 1)
        ]

        self._u = tf.Variable(tf.ones([sample_size, 0, 0, 0], dtype=tf.float64),
                              shape=[sample_size, None, None, None])

        self._chol_Kuu = tf.Variable(tf.ones([sample_size, 0, 0, 0], dtype=tf.float64),
                                     shape=[sample_size, None, None, None])

        self._gp_eps = tf.Variable(tf.ones([sample_size, 0], dtype=tf.float64), shape=[sample_size, None])

        self._sample_size = sample_size

    def sample(self, at: TensorType) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`DeepGaussianProcessSampler` and ``at``, will produce
        the exact same samples. Calls to :meth:`sample` on *different*
        :class:`DeepGaussianProcessSampler` instances will produce different samples.
        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :return: The samples, of shape `[S, N, L]`, where `S` is the `sample_size` and `L` is
            the number of latent model dimensions.
        """
        tf.debugging.assert_equal(len(tf.shape(at)), 2)

        gamma_is_populated = tf.size(self._gamma_list[0]) != 0

        eps_is_populated = tf.size(self._eps_list[0]) != 0

        Gii_is_populated = tf.size(self._Gii_list[0]) != 0

        chol_dKii_is_populated = tf.size(self._chol_Kuu[0]) != 0

        u_is_populated = tf.size(self._u) != 0

        gp_eps_is_populated = tf.size(self._gp_eps) != 0

        samples = tf.tile(tf.expand_dims(at, 0), [self._sample_size, 1, 1])

        samples = self._model.model_gpflux._inducing_add(samples)  # type: ignore
        samples = self._model.model_gpflux.kernelizer(samples)  # type: ignore

        for i, layer in enumerate(self._model.model_gpflux.f_layers[:-1]):  # type: ignore
            if not isinstance(layer, IWLayer):
                raise ValueError(f"Layer {i} must be an IWLayer")
            K = layer.kernel_gram(samples, full_cov=False)

            dKii = layer.delta*K.ii
            dKit = layer.delta*K.it
            dktt = layer.delta*K.tt

            if not chol_dKii_is_populated:
                chol_dKii = tf.linalg.cholesky(dKii)
                self._chol_dKii_list[i].assign(chol_dKii)

            if not Gii_is_populated:
                Gii, _ = layer.Gii(dKii)
                self._Gii_list[i].assign(Gii)

            inv_Kii_kit = tf.linalg.cholesky_solve(self._chol_dKii_list[i], dKit)

            dktti = dktt - tf.reduce_sum(dKit * inv_Kii_kit, -2)
            alpha = (layer.delta + layer.P + tf.cast(tf.shape(dktt)[-1], tf.float64) + 1)/2

            if not gamma_is_populated:
                P = tfp.distributions.Gamma(alpha, 1)
                gamma_sample = tf.reshape(P.sample([self._sample_size]), [self._sample_size, 1])
                self._gamma_list[i].assign(gamma_sample)

            gtti = tf.math.reciprocal(self._gamma_list[i]/(0.5*dktti))  # type: ignore

            if not eps_is_populated:
                eps = tf.expand_dims(tf.random.normal(tf.shape(dKit)[:-1], dtype=tf.float64), -1)
                self._eps_list[i].assign(eps)

            inv_Gii_git = inv_Kii_kit + tf.linalg.triangular_solve(
                tf.linalg.adjoint(self._chol_dKii_list[i]),
                self._eps_list[i],
                lower=False
            ) * tf.sqrt(gtti)[:, None, :]
            git = self._Gii_list[i] @ inv_Gii_git

            gtt = gtti + tf.reduce_sum(git * inv_Gii_git, -2)

            samples = KG(self._Gii_list[i], git, gtt)

        layer = self._model.model_gpflux.f_layers[-1]  # type: ignore

        K = layer.kernel_gram(samples, full_cov=False)

        Kuu = K.ii
        Kuf = K.it
        Kfu = tf.linalg.adjoint(Kuf)
        Kff = K.tt

        Kuu, Kuf, Kfu = tf.expand_dims(Kuu, 1), tf.expand_dims(Kuf, 1), tf.expand_dims(Kfu, 1)

        if not u_is_populated:
            u, _, chol_Kuu = layer.sample_u(Kuu)
            self._u.assign(u)
            self._chol_Kuu.assign(chol_Kuu)

        mean, var = layer.predict(self._u, Kff, Kuf, self._chol_Kuu)

        if not gp_eps_is_populated:
            self._gp_eps.assign(
                tf.random.normal([self._sample_size, tf.shape(mean)[-1]], dtype=tf.float64)
            )

        samples = mean + tf.sqrt(var) * tf.cast(self._gp_eps[:, None, :], var.dtype)

        # Add mean function
        if not isinstance(layer.mean_function, gpflow.mean_functions.Zero):
            samples = samples + layer.mean_function.c

        return samples

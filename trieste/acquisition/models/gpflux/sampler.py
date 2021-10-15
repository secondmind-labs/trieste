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

import tensorflow as tf
from gpflux.layers import LatentVariableLayer

from trieste.acquisition.sampler import Sampler
from trieste.models import ProbabilisticModel
from trieste.models.gpflux import DeepGaussianProcess, GlobalInducingDeepGaussianProcess
from trieste.types import TensorType


class DeepGaussianProcessSampler(Sampler):
    r"""
    This sampler employs the *reparameterization trick* to approximate samples from a
    :class:`DeepGaussianProcess`\ 's predictive distribution. This sampler is essentially an
    extension of :class:`trieste.acquisition.sampler.IndependentReparametrizationSampler` for use
    with DGP models.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel):
        """
        :param sample_size: The number of samples for each batch of points. Must be positive.
        :param model: The model to sample from. Must be a :class:`DeepGaussianProcess`
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or if
            model is not a :class:`DeepGaussianProcess`.
        """
        if not isinstance(model, DeepGaussianProcess):
            raise ValueError("Model must be a trieste.models.gpflux.DeepGaussianProcess")

        super().__init__(sample_size, model)

        # Each element of _eps_list is essentially a lazy constant. It is declared and assigned an
        # empty tensor here, and populated on the first call to sample
        self._eps_list = [
            tf.Variable(tf.ones([sample_size, 0], dtype=tf.float64), shape=[sample_size, None])
            for _ in range(len(model.model_gpflux.f_layers))
        ]

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

        eps_is_populated = tf.size(self._eps_list[0]) != 0

        samples = tf.tile(tf.expand_dims(at, 0), [self._sample_size, 1, 1])
        for i, layer in enumerate(self._model.model_gpflux.f_layers):
            if isinstance(layer, LatentVariableLayer):
                if not eps_is_populated:
                    self._eps_list[i].assign(layer.prior.sample([tf.shape(samples)[:-1]]))
                samples = layer.compositor([samples, self._eps_list[i]])
                continue

            mean, var = layer.predict(samples, full_cov=False, full_output_cov=False)

            if not eps_is_populated:
                self._eps_list[i].assign(
                    tf.random.normal([self._sample_size, tf.shape(mean)[-1]], dtype=tf.float64)
                )

            samples = mean + tf.sqrt(var) * tf.cast(self._eps_list[i][:, None, :], var.dtype)

        return samples


class GlobalInducingDeepGaussianProcessSampler(Sampler):
    r"""
    This sampler employs the *reparameterization trick* to approximate samples from a
    :class:`GlobalInducingDeepGaussianProcess`\ 's predictive distribution. This sampler is essentially an
    extension of :class:`trieste.acquisition.sampler.IndependentReparametrizationSampler` for use
    with DGP models.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel):
        """
        :param sample_size: The number of samples for each batch of points. Must be positive.
        :param model: The model to sample from. Must be a :class:`DeepGaussianProcess`
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or if
            model is not a :class:`DeepGaussianProcess`.
        """
        if not isinstance(model, GlobalInducingDeepGaussianProcess):
            raise ValueError("Model must be a "
                             "trieste.models.gpflux.GlobalInducingDeepGaussianProcess")

        super().__init__(sample_size, model)

        # Each element of _eps_list is essentially a lazy constant. It is declared and assigned an
        # empty tensor here, and populated on the first call to sample
        self._eps_list = [
            tf.Variable(tf.ones([sample_size, 0], dtype=tf.float64), shape=[sample_size, None])
            for _ in range(len(model.model_gpflux.f_layers))
        ]

        self._u_list = [
            tf.Variable(tf.ones([sample_size, 0, 0, 0], dtype=tf.float64),
                        shape=[sample_size, None, None, None])
            for _ in range(len(model.model_gpflux.f_layers))
        ]

        self._chol_Kuu_list = [
            tf.Variable(tf.ones([sample_size, 0, 0, 0], dtype=tf.float64),
                        shape=[sample_size, None, None, None])
            for _ in range(len(model.model_gpflux.f_layers))
        ]

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

        u_is_populated = tf.size(self._u_list[0]) != 0

        eps_is_populated = tf.size(self._eps_list[0]) != 0

        inducing_data = self._model.model_gpflux.inducing_data
        inducing_data = tf.tile(
            tf.expand_dims(inducing_data, 0),
            [self._sample_size, 1, 1]
        )

        samples = tf.tile(tf.expand_dims(at, 0), [self._sample_size, 1, 1])
        for i, layer in enumerate(self._model.model_gpflux.f_layers):
            samples_mean = layer.mean_function(samples)
            ind_data_mean = layer.mean_function(inducing_data)

            Kuu = layer.kernel(inducing_data)
            Kuf = layer.kernel(inducing_data, samples)
            Kff = layer.kernel.K_diag(samples)

            Kuu, Kuf = tf.expand_dims(Kuu, 1), tf.expand_dims(Kuf, 1)

            if not u_is_populated:
                u, _, chol_Kuu = layer.sample_u(Kuu)
                self._u_list[i].assign(u)
                self._chol_Kuu_list[i].assign(chol_Kuu)

            mean, var = layer.predict(self._u_list[i], Kff, Kuf, self._chol_Kuu_list[i])

            if not eps_is_populated:
                self._eps_list[i].assign(
                    tf.random.normal([self._sample_size, tf.shape(mean)[-1]], dtype=tf.float64)
                )

            samples = mean + tf.sqrt(var) * tf.cast(self._eps_list[i][:, None, :], var.dtype)

            # Add mean function
            samples = samples + samples_mean
            inducing_data = tf.linalg.adjoint(tf.squeeze(self._u_list[i], -1)) + ind_data_mean

        return samples

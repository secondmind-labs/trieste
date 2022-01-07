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

import tensorflow as tf
from gpflux.layers import GPLayer, LatentVariableLayer
from gpflux.models import DeepGP
from gpflux.sampling.sample import Sample

from ...types import TensorType
from ...utils import DEFAULTS
from ..interfaces import (
    ProbabilisticModel,
    ReparametrizationSampler,
    TrajectoryFunction,
    TrajectorySampler,
)
from .interface import GPfluxPredictor


def sample_consistent_lv_layer(layer: LatentVariableLayer) -> Sample:
    class SampleLV(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            sample = layer.prior.sample()
            batch_shape = tf.shape(X)[:-1]
            sample_rank = tf.rank(sample)
            for _ in range(len(batch_shape)):
                sample = tf.expand_dims(sample, 0)
            sample = tf.tile(
                sample, tf.concat([batch_shape, tf.ones(sample_rank, dtype="int32")], -1)
            )
            return layer.compositor([X, sample])

    return SampleLV()


def sample_dgp(model: DeepGP) -> TrajectoryFunction:
    function_draws = []
    for layer in model.f_layers:
        if isinstance(layer, GPLayer):
            function_draws.append(layer.sample())
        elif isinstance(layer, LatentVariableLayer):
            function_draws.append(sample_consistent_lv_layer(layer))
        else:
            raise NotImplementedError(f"Sampling not implemented for {type(layer)}")

    class ChainedSample(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            for f in function_draws:
                X = f(X)
            return X

    return ChainedSample().__call__


class DeepGaussianProcessTrajectorySampler(TrajectorySampler):
    r"""
    This sampler provides trajectory samples from a :class:`DeepGaussianProcess`\ 's predictive
    distribution.
    """

    def __init__(self, model: ProbabilisticModel):
        """
        :param model: The model to sample from. Must be a :class:`GPfluxPredictor`
        :raise ValueError: If model is not a :class:`GPfluxPredictor`
        """
        if not isinstance(model, GPfluxPredictor):
            raise ValueError(
                f"Model must be a gpflux.interface.GPfluxPredictor, received {type(model)}"
            )

        super().__init__(model)

        self.model = model.model_gpflux

    def get_trajectory(self) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) by using the GPflux sampling
        functionality. These trajectories are differentiable with respect to the input, so can be
        used to e.g. find the minima of Thompson samples.

        :return: A trajectory function representing an approximate trajectory from the deep Gaussian
            process, taking an input of shape `[N, D]` and returning shape `[N, 1]`
        """

        return sample_dgp(self._model)


class DeepGaussianProcessReparamSampler(ReparametrizationSampler):
    r"""
    This sampler employs the *reparameterization trick* to approximate samples from a
    :class:`DeepGaussianProcess`\ 's predictive distribution.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel):
        """
        :param sample_size: The number of samples for each batch of points. Must be positive.
        :param model: The model to sample from. Must be a :class:`GPfluxPredictor`
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or if
            model is not a :class:`GPfluxPredictor`.
        """
        if not isinstance(model, GPfluxPredictor):
            raise ValueError(
                f"Model must be a gpflux.interface.GPfluxPredictor, received {type(model)}"
            )

        super().__init__(sample_size, model)

        self.model = model.model_gpflux

        # Each element of _eps_list is essentially a lazy constant. It is declared and assigned an
        # empty tensor here, and populated on the first call to sample
        self._eps_list = [
            tf.Variable(tf.ones([sample_size, 0], dtype=tf.float64), shape=[sample_size, None])
            for _ in range(len(model.f_layers))
        ]

    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`DeepGaussianProcessReparamSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`DeepGaussianProcessReparamSampler` instances will produce different samples.

        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :param jitter: The size of the jitter to use when stabilizing the Cholesky
            decomposition of the covariance matrix.
        :return: The samples, of shape `[S, N, L]`, where `S` is the `sample_size` and `L` is
            the number of latent model dimensions.
        """
        tf.debugging.assert_equal(len(tf.shape(at)), 2)

        eps_is_populated = tf.size(self._eps_list[0]) != 0

        samples = tf.tile(tf.expand_dims(at, 0), [self._sample_size, 1, 1])
        for i, layer in enumerate(self.model.f_layers):
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

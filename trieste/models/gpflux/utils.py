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

import copy
from typing import TypeVar

import gpflow
import tensorflow as tf
from gpflux.layers import GPLayer, LatentVariableLayer
from gpflux.models import DeepGP, GIDeepGP
from gpflux.sampling.sample import Sample
from gpflux.sampling.utils import draw_conditional_sample

from trieste.types import TensorType

M = TypeVar("M", bound=tf.Module)
""" A type variable bound to :class:`tf.Module`. """


def module_deepcopy(self: M, memo: dict[int, object]) -> M:
    r"""
    This function provides a workaround for `a bug`_ in TensorFlow Probability (fixed in `version
    0.12`_) where a :class:`tf.Module` cannot be deep-copied if it has
    :class:`tfp.bijectors.Bijector` instances on it. The function can be used to directly copy an
    object ``self`` as e.g. ``module_deepcopy(self, {})``, but it is perhaps more useful as an
    implemention for :meth:`__deepcopy__` on classes, where it can be used as follows:

    .. _a bug: https://github.com/tensorflow/probability/issues/547
    .. _version 0.12: https://github.com/tensorflow/probability/releases/tag/v0.12.1

    .. testsetup:: *

        >>> import tensorflow_probability as tfp

    >>> class Foo(tf.Module):
    ...     example_bijector = tfp.bijectors.Exp()
    ...
    ...     __deepcopy__ = module_deepcopy

    Classes with this method can be deep-copied even if they contain
    :class:`tfp.bijectors.Bijector`\ s.

    :param self: The object to copy.
    :param memo: References to existing deep-copied objects (by object :func:`id`).
    :return: A deep-copy of ``self``.
    """
    gpflow.utilities.reset_cache_bijectors(self)

    new = self.__new__(type(self))
    memo[id(self)] = new

    for name, value in self.__dict__.items():
        setattr(new, name, copy.deepcopy(value, memo))

    return new


def sample_consistent_lv_layer(layer: LatentVariableLayer) -> Sample:
    class SampleLV(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            sample = layer.prior.sample()
            batch_shape = tf.shape(X)[:-1]
            for _ in range(len(batch_shape)):
                sample = tf.expand_dims(sample, 0)
            sample = tf.tile(sample, batch_shape.numpy().tolist() + [1])
            return layer.compositor([X, sample])

    return SampleLV()


def sample_dgp(model: DeepGP) -> Sample:
    function_draws = []
    for layer in model.f_layers:
        if isinstance(layer, GPLayer):
            function_draws.append(layer.sample())
        elif isinstance(layer, LatentVariableLayer):
            function_draws.append(sample_consistent_lv_layer(layer))
        else:
            raise NotImplementedError(f"Sampling not implemented for {layer}")

    class ChainedSample(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            for f in function_draws:
                X = f(X)
            return X

    return ChainedSample()


def sample_gidgp(model: GIDeepGP) -> Sample:

    class GISample(Sample):
        u_list = [tf.Variable(tf.ones([1, 0, 0, 0], dtype=tf.float64),
                  shape=[1, None, None, None]) for _ in range(len(model.f_layers))]
        chol_Kuu_list = [tf.Variable(tf.ones([1, 0, 0, 0], dtype=tf.float64),
                         shape=[1, None, None, None]) for _ in range(len(model.f_layers))]
        X = [None for _ in range(len(model.f_layers))]
        f = [tf.zeros([0, layer.num_latent_gps], dtype=tf.float64) for layer in model.f_layers]
        P = [layer.num_latent_gps for layer in model.f_layers]

        def __call__(self, X_new: TensorType) -> tf.Tensor:
            X_new = tf.expand_dims(X_new, 0)
            inducing_data = tf.expand_dims(model.inducing_data, 0)

            u_is_populated = tf.size(self.u_list[0]) != 0

            for i, layer in enumerate(model.f_layers):
                N_old = tf.shape(self.f[i])[0]
                N_new = tf.shape(X_new)[1]

                mean_function = layer.mean_function(X_new)

                if self.X[i] is None:
                    self.X[i] = X_new
                else:
                    self.X[i] = tf.concat([self.X[i], X_new], axis=1)

                x = self.X[i]

                ind_data_mean = layer.mean_function(inducing_data)

                Kuu = layer.kernel(inducing_data)
                Kuf = layer.kernel(inducing_data, x)
                Kfu = tf.linalg.adjoint(Kuf)
                Kff = layer.kernel.K_diag(x)

                Kuu, Kuf, Kfu = tf.expand_dims(Kuu, 1), tf.expand_dims(Kuf, 1), tf.expand_dims(Kfu, 1)

                if not u_is_populated:
                    u, _, chol_Kuu = layer.sample_u(Kuu)
                    self.u_list[i].assign(u)
                    self.chol_Kuu_list[i].assign(chol_Kuu)

                mean, cov = layer.predict(self.u_list[i], Kff, Kuf, self.chol_Kuu_list[i],
                                          inputs=tf.concat([inducing_data, x], 1), full_cov=True)
                mean = tf.linalg.matrix_transpose(mean)  # [1, P, N_old + N_new]
                f_old = tf.linalg.matrix_transpose(self.f[i])  # [P, N_old]
                f_new = draw_conditional_sample(mean[0], tf.tile(cov, [self.P[i], 1, 1]), f_old)  # [P, N_new]
                f_new = tf.linalg.matrix_transpose(f_new)  # [N_new, P]
                self.f[i] = tf.concat([self.f[i], f_new], axis=0)  # [N_old + N_new, P]

                tf.debugging.assert_equal(tf.shape(self.f[i])[0], N_old + N_new)
                tf.debugging.assert_equal(tf.shape(f_new)[0], N_new)

                X_new = tf.expand_dims(f_new, 0) + mean_function
                inducing_data = tf.linalg.adjoint(tf.squeeze(self.u_list[i], -1)) + ind_data_mean

            return tf.squeeze(X_new, 0)

    return GISample()

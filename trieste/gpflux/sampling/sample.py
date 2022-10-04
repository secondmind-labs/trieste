#
# Copyright (c) 2021 The GPflux Contributors.
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
#
""" This module enables you to sample from (Deep) GPs using different approaches. """

import abc
from typing import Callable, Optional, Union

import tensorflow as tf

from gpflow.base import TensorType
from gpflow.conditionals import conditional
from gpflow.config import default_float, default_jitter
from gpflow.covariances import Kuf, Kuu
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel

from gpflux.feature_decomposition_kernels import KernelWithFeatureDecomposition
from gpflux.math import compute_A_inv_b
from gpflux.sampling.utils import draw_conditional_sample

from .dispatch import efficient_sample

""" A function that returns a :class:`Sample` of a GP posterior. """


class Sample(abc.ABC):
    """
    This class represents a sample from a GP that you can evaluate by using the ``__call__``
    at new locations within the support of the GP.

    Importantly, the same function draw (sample) is evaluated when calling it multiple
    times. This property is called consistency. Achieving consistency for vanilla GPs is costly
    because it scales cubically with the number of evaluation points,
    but works with any kernel. It is implemented in
    :meth:`_efficient_sample_conditional_gaussian`.
    For :class:`KernelWithFeatureDecomposition`, the more efficient approach
    following :cite:t:`wilson2020efficiently` is implemented in
    :meth:`_efficient_sample_matheron_rule`.

    See the tutorial notebooks `Efficient sampling
    <../../../../notebooks/efficient_sampling.ipynb>`_ and `Weight Space
    Approximation with Random Fourier Features
    <../../../../notebooks/weight_space_approximation.ipynb>`_ for an
    in-depth overview.
    """

    @abc.abstractmethod
    def __call__(self, X: TensorType) -> tf.Tensor:
        r"""
        Return the evaluation of the GP sample :math:`f(X)` for :math:`f \sim GP(0, k)`.

        :param X: The inputs, a tensor with the shape ``[N, D]``, where ``D`` is the
            input dimensionality.
        :return: Function values, a tensor with the shape ``[N, P]``, where ``P`` is the
            output dimensionality.
        """
        raise NotImplementedError

    def __add__(self, other: Union["Sample", Callable[[TensorType], TensorType]]) -> "Sample":
        """
        Allow for the summation of two instances that implement the ``__call__`` method.
        """
        this = self.__call__

        class AddSample(Sample):
            def __call__(self, X: TensorType) -> tf.Tensor:
                return this(X) + other(X)

        return AddSample()


@efficient_sample.register(InducingVariables, Kernel, object)
def _efficient_sample_conditional_gaussian(
    inducing_variable: InducingVariables,
    kernel: Kernel,
    q_mu: tf.Tensor,
    *,
    q_sqrt: Optional[TensorType] = None,
    whiten: bool = False,
) -> Sample:
    """
    Most costly implementation for obtaining a consistent GP sample.
    However, this method can be used for any kernel.
    """

    class SampleConditional(Sample):
        # N_old is 0 at first, we then start keeping track of past evaluation points.
        X = None  # [N_old, D]
        P = tf.shape(q_mu)[-1]  # num latent GPs
        f = tf.zeros((0, P), dtype=default_float())  # [N_old, P]

        def __call__(self, X_new: TensorType) -> tf.Tensor:
            N_old = tf.shape(self.f)[0]
            N_new = tf.shape(X_new)[0]

            if self.X is None:
                self.X = X_new
            else:
                self.X = tf.concat([self.X, X_new], axis=0)

            mean, cov = conditional(
                self.X,
                inducing_variable,
                kernel,
                q_mu,
                q_sqrt=q_sqrt,
                white=whiten,
                full_cov=True,
            )  # mean: [N_old+N_new, P], cov: [P, N_old+N_new, N_old+N_new]
            mean = tf.linalg.matrix_transpose(mean)  # [P, N_old+N_new]
            f_old = tf.linalg.matrix_transpose(self.f)  # [P, N_old]
            f_new = draw_conditional_sample(mean, cov, f_old)  # [P, N_new]
            f_new = tf.linalg.matrix_transpose(f_new)  # [N_new, P]
            self.f = tf.concat([self.f, f_new], axis=0)  # [N_old + N_new, P]

            tf.debugging.assert_equal(tf.shape(self.f), [N_old + N_new, self.P])
            tf.debugging.assert_equal(tf.shape(f_new), [N_new, self.P])

            return f_new

    return SampleConditional()


@efficient_sample.register(InducingVariables, KernelWithFeatureDecomposition, object)
def _efficient_sample_matheron_rule(
    inducing_variable: InducingVariables,
    kernel: KernelWithFeatureDecomposition,
    q_mu: tf.Tensor,
    *,
    q_sqrt: Optional[TensorType] = None,
    whiten: bool = False,
) -> Sample:
    """
    Implements the efficient sampling rule from :cite:t:`wilson2020efficiently` using
    the Matheron rule. To use this sampling scheme, the GP has to have a
    ``kernel`` of the :class:`KernelWithFeatureDecomposition` type .

    :param kernel: A kernel of the :class:`KernelWithFeatureDecomposition` type, which
        holds the covariance function and the kernel's features and
        coefficients.
    :param q_mu: A tensor with the shape ``[M, P]``.
    :param q_sqrt: A tensor with the shape ``[P, M, M]``.
    :param whiten: Determines the parameterisation of the inducing variables.
    """

    L = tf.shape(kernel.feature_coefficients)[0]  # num eigenfunctions  # noqa: F841
    M, P = tf.shape(q_mu)[0], tf.shape(q_mu)[1]  # num inducing, num output heads

    prior_weights = tf.sqrt(kernel.feature_coefficients) * tf.random.normal(
        (L, P), dtype=default_float()  # [L, 1], [L,P]
    )  # [L, P]

    u_sample_noise = tf.matmul(
        q_sqrt,
        tf.random.normal((P, M, 1), dtype=default_float()),  # [P, M, M]  # [P, M, 1]
    )  # [P, M, 1]
    Kmm = Kuu(inducing_variable, kernel, jitter=default_jitter())  # [M, M]

    tf.debugging.assert_equal(tf.shape(Kmm), [M, M])
    u_sample = q_mu + tf.linalg.matrix_transpose(u_sample_noise[..., 0])  # [M, P]

    if whiten:
        Luu = tf.linalg.cholesky(Kmm)  # [M, M]
        u_sample = tf.matmul(Luu, u_sample)  # [M, P]

    phi_Z = kernel.feature_functions(inducing_variable.Z)  # [M, L]

    weight_space_prior_Z = tf.matmul(phi_Z, prior_weights)  # [M, L]  # [L, P]  # [M, P]

    diff = u_sample - weight_space_prior_Z  # [M, P]
    v = compute_A_inv_b(Kmm, diff)  # [M, P]

    tf.debugging.assert_equal(tf.shape(v), [M, P])

    class WilsonSample(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            """
            :param X: evaluation points [N, D]
            :return: function value of sample [N, P]
            """
            N = tf.shape(X)[0]
            phi_X = kernel.feature_functions(X)  # [N, L]

            weight_space_prior_X = tf.matmul(phi_X, prior_weights)  # [N, L]  # [L, P]  # [N, P]

            Knm = tf.linalg.matrix_transpose(Kuf(inducing_variable, kernel, X))  # [N, M]

            function_space_update_X = tf.matmul(Knm, v)  # [N, M]  # [M, P]  # [N, P]

            tf.debugging.assert_equal(tf.shape(weight_space_prior_X), [N, P])
            tf.debugging.assert_equal(tf.shape(function_space_update_X), [N, P])

            return weight_space_prior_X + function_space_update_X  # [N, P]

    return WilsonSample()

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

from typing import Optional, Union

import tensorflow as tf

from gpflow.base import TensorType
from gpflow.config import default_float, default_jitter
from gpflow.covariances import Kuf, Kuu
from gpflow.inducing_variables import (
    MultioutputInducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import SeparateIndependent, SharedIndependent

from gpflux.feature_decomposition_kernels import (
    SeparateMultiOutputKernelWithFeatureDecomposition,
    SharedMultiOutputKernelWithFeatureDecomposition,
)
from gpflux.math import compute_A_inv_b

from ..dispatch import efficient_sample
from ..sample import Sample


@efficient_sample.register(
    MultioutputInducingVariables,
    (
        SharedMultiOutputKernelWithFeatureDecomposition,
        SeparateMultiOutputKernelWithFeatureDecomposition,
    ),
    object,
)
def _efficient_multi_output_sample_matheron_rule(
    inducing_variable: MultioutputInducingVariables,
    kernel: Union[
        SharedMultiOutputKernelWithFeatureDecomposition,
        SeparateMultiOutputKernelWithFeatureDecomposition,
    ],
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

    # Reshape kernel.feature_coefficients
    _feature_coefficients = tf.transpose(kernel.feature_coefficients[..., 0])  # [L,P]

    L = tf.shape(_feature_coefficients)[0]  # num eigenfunctions  # noqa: F841
    M, P = tf.shape(q_mu)[0], tf.shape(q_mu)[1]  # num inducing, num output heads

    prior_weights = tf.sqrt(_feature_coefficients) * tf.random.normal(
        (L, P), dtype=default_float()  # [L, P], [L,P]
    )  # [L, P]

    u_sample_noise = tf.matmul(
        q_sqrt,
        tf.random.normal((P, M, 1), dtype=default_float()),  # [P, M, M]  # [P, M, 1]
    )  # [P, M, 1]
    tf.debugging.assert_equal(tf.shape(u_sample_noise), [P, M, 1])

    if isinstance(kernel, SharedIndependent):
        Kmm = tf.tile(
            Kuu(inducing_variable, kernel, jitter=default_jitter())[None, ...],
            [P, 1, 1],
        )  # [P,M,M]
        tf.debugging.assert_equal(tf.shape(Kmm), [P, M, M])
    elif isinstance(kernel, SeparateIndependent):
        Kmm = Kuu(inducing_variable, kernel, jitter=default_jitter())  # [P,M,M]
        tf.debugging.assert_equal(tf.shape(Kmm), [P, M, M])
    else:
        raise ValueError(
            "kernel not supported. Must be either SharedIndependent or SeparateIndependent"
        )

    tf.debugging.assert_equal(tf.shape(Kmm), [P, M, M])

    u_sample = q_mu + tf.linalg.matrix_transpose(u_sample_noise[..., 0])  # [M, P]
    tf.debugging.assert_equal(tf.shape(u_sample), [M, P])

    if whiten:
        Luu = tf.linalg.cholesky(Kmm)  # [P,M,M]
        tf.debugging.assert_equal(tf.shape(Kmm), [P, M, M])

        u_sample = tf.transpose(
            tf.matmul(Luu, tf.transpose(u_sample)[..., None])[..., 0]  # [P, M, M]  # [P, M, 1]
        )  # [M, P]
        tf.debugging.assert_equal(tf.shape(u_sample), [M, P])

    if isinstance(inducing_variable, SeparateIndependentInducingVariables):

        _inducing_variable_list = []
        for ind_var in inducing_variable.inducing_variable_list:
            _inducing_variable_list.append(ind_var.Z)
        _inducing_variable_list = tf.stack(_inducing_variable_list, axis=0)

        phi_Z = kernel.feature_functions(_inducing_variable_list)  # [P, M, L]
        tf.debugging.assert_equal(tf.shape(phi_Z), [P, M, L])

    elif isinstance(inducing_variable, SharedIndependentInducingVariables):

        phi_Z = kernel.feature_functions(inducing_variable.inducing_variable.Z)  # [P, M, L]
        tf.debugging.assert_equal(tf.shape(phi_Z), [P, M, L])
    else:
        raise ValueError("inducing variable is not supported.")

    weight_space_prior_Z = tf.matmul(phi_Z, tf.transpose(prior_weights)[..., None])  # [P, M, 1]
    weight_space_prior_Z = tf.transpose(weight_space_prior_Z[..., 0])  # [M, P]

    diff = tf.transpose(u_sample - weight_space_prior_Z)[..., None]  # [P, M, 1]
    v = tf.transpose(compute_A_inv_b(Kmm, diff)[..., 0])  # [P, M, M]  # [P, M, 1]  # [M, P]

    tf.debugging.assert_equal(tf.shape(v), [M, P])

    class WilsonSample(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            """
            :param X: evaluation points [N, D]
            :return: function value of sample [N, P]
            """
            N = tf.shape(X)[0]
            phi_X = kernel.feature_functions(X)  # [P, N, L]

            weight_space_prior_X = tf.transpose(
                tf.matmul(phi_X, tf.transpose(prior_weights)[..., None],)[  # [P, N, L]  # [P, L, 1]
                    ..., 0
                ]
            )  # [N, P]

            Knm = tf.linalg.matrix_transpose(
                Kuf(inducing_variable, kernel, X)
            )  # [P, N, M] or [N,M]
            if isinstance(inducing_variable, SharedIndependentInducingVariables):
                Knm = tf.tile(Knm[None, ...], [P, 1, 1])
            tf.debugging.assert_equal(tf.shape(Knm), [P, N, M])
            function_space_update_X = tf.transpose(
                tf.matmul(Knm, tf.transpose(v)[..., None])[..., 0]  # [P, N, M]  # [P, M, 1]
            )  # [N, P]

            tf.debugging.assert_equal(tf.shape(weight_space_prior_X), [N, P])
            tf.debugging.assert_equal(tf.shape(function_space_update_X), [N, P])

            return weight_space_prior_X + function_space_update_X  # [N, P]

    return WilsonSample()

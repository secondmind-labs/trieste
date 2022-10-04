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

from typing import Mapping, Optional, Tuple, Type

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.base import DType, TensorType

from trieste.gpflux.layers.basis_functions.fourier_features.multioutput.random.base import (
    MultiOutputRandomFourierFeatures,
)
from gpflux.types import ShapeType

"""
Kernels supported by :class:`OrthogonalRandomFeatures`.

This random matrix sampling scheme only applies to the :class:`gpflow.kernels.SquaredExponential`
kernel.
For Matern kernels please use :class:`RandomFourierFeatures`
or :class:`RandomFourierFeaturesCosine`.
"""
ORF_SUPPORTED_KERNELS: Tuple[Type[gpflow.kernels.Stationary], ...] = (
    gpflow.kernels.SquaredExponential,
)


def _sample_chi_squared(nu: float, shape: ShapeType, dtype: DType) -> TensorType:
    """
    Draw samples from Chi-squared distribution with `nu` degrees of freedom.

    See https://mathworld.wolfram.com/Chi-SquaredDistribution.html for further
    details regarding relationship to Gamma distribution.
    """
    return tf.random.gamma(shape=shape, alpha=0.5 * nu, beta=0.5, dtype=dtype)


def _sample_chi(nu: float, shape: ShapeType, dtype: DType) -> TensorType:
    """
    Draw samples from Chi-distribution with `nu` degrees of freedom.
    """
    s = _sample_chi_squared(nu, shape, dtype)
    return tf.sqrt(s)


def _ceil_divide(a: float, b: float) -> int:
    """
    Ceiling division. Returns the smallest integer `m` s.t. `m*b >= a`.
    """
    return -np.floor_divide(-a, b)


class MultiOutputOrthogonalRandomFeatures(MultiOutputRandomFourierFeatures):
    r"""
    Orthogonal random Fourier features (ORF) :cite:p:`yu2016orthogonal` for more
    efficient and accurate kernel approximations than :class:`RandomFourierFeatures`.
    """

    def __init__(self, kernel: gpflow.kernels.Kernel, n_components: int, **kwargs: Mapping):

        if isinstance(kernel, gpflow.kernels.SeparateIndependent):
            for ker in kernel.kernels:
                assert isinstance(ker, ORF_SUPPORTED_KERNELS), "Unsupported Kernel"
        elif isinstance(kernel, gpflow.kernels.SharedIndependent):
            assert isinstance(kernel.kernel, ORF_SUPPORTED_KERNELS), "Unsupported Kernel"
        else:
            raise ValueError("kernel specified is not supported.")

        super(MultiOutputOrthogonalRandomFeatures, self).__init__(kernel, n_components, **kwargs)

    def _weights_init(self, shape: TensorType, dtype: Optional[DType] = None) -> TensorType:
        n_out, n_components, input_dim = shape  # P, M, D
        n_reps = _ceil_divide(n_components, input_dim)  # K, smallest integer s.t. K*D >= M

        W = tf.random.normal(shape=(n_out, n_reps, input_dim, input_dim), dtype=dtype)
        Q, _ = tf.linalg.qr(W)  # throw away R; shape [P, K, D, D]

        s = _sample_chi(
            nu=input_dim, shape=(n_out, n_reps, input_dim), dtype=dtype
        )  # shape [P, K, D]
        U = tf.expand_dims(s, axis=-1) * Q  # equiv: S @ Q where S = diag(s); shape [P, K, D, D]
        V = tf.reshape(U, shape=(n_out, -1, input_dim))  # shape [P, K*D, D]

        return V[:, : self.n_components, :]  # shape [P, M, D] (throw away K*D - M rows)

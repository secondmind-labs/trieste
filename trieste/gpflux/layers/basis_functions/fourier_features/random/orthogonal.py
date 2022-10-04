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

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.base import DType, TensorType
from gpflux.types import ShapeType

from trieste.gpflux.layers.basis_functions.fourier_features.random.base import RandomFourierFeatures

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


class OrthogonalRandomFeatures(RandomFourierFeatures):
    r"""
    Orthogonal random Fourier features (ORF) :cite:p:`yu2016orthogonal` for more
    efficient and accurate kernel approximations than :class:`RandomFourierFeatures`.
    """

    def __init__(self, kernel: gpflow.kernels.Kernel, n_components: int, **kwargs: Mapping):
        assert isinstance(kernel, ORF_SUPPORTED_KERNELS), "Unsupported Kernel"
        super(OrthogonalRandomFeatures, self).__init__(kernel, n_components, **kwargs)

    def _weights_init(self, shape: TensorType, dtype: Optional[DType] = None) -> TensorType:
        n_components, input_dim = shape  # M, D
        n_reps = _ceil_divide(n_components, input_dim)  # K, smallest integer s.t. K*D >= M

        W = tf.random.normal(shape=(n_reps, input_dim, input_dim), dtype=dtype)
        Q, _ = tf.linalg.qr(W)  # throw away R; shape [K, D, D]

        s = _sample_chi(nu=input_dim, shape=(n_reps, input_dim), dtype=dtype)  # shape [K, D]
        U = tf.expand_dims(s, axis=-1) * Q  # equiv: S @ Q where S = diag(s); shape [K, D, D]
        V = tf.reshape(U, shape=(-1, input_dim))  # shape [K*D, D]

        return V[: self.n_components]  # shape [M, D] (throw away K*D - M rows)

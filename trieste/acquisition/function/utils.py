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
This module contains utility functions for acquisition functions.
"""
from typing import Callable, Tuple

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from ...types import TensorType

# =============================================================================
# Multivariate Normal CDF
# =============================================================================


class MultivariateNormalCDF:
    def __init__(
        self,
        sample_size: int,
        dim: int,
        dtype: tf.DType,
        num_sobol_skip: int = 0,
    ) -> None:
        """Builds the cumulative density function of the multivariate Gaussian
        using the Genz approximation detailed in :cite:`genz2016numerical`.

        This is a Monte Carlo approximation which is more accurate than a naive
        Monte Carlo estimate of the expected improvent. In order to use
        reparametrised samples, the helper accepts a tensor of samples, and the
        callable uses these fixed samples whenever it is called.

        :param samples_size: int, number of samples to use.
        :param dim: int, dimension of the multivariate Gaussian.
        :param dtype: tf.DType, data type to use for calculations.
        :param num_sobol_skip: int, number of sobol samples to skip.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_positive(dim)

        self._S = sample_size
        self._Q = dim
        self._dtype = dtype

        self._num_sobol_skip = num_sobol_skip

    def _standard_normal_cdf_and_inverse_cdf(
        self,
        dtype: tf.DType,
    ) -> Tuple[Callable[[TensorType], TensorType], Callable[[TensorType], TensorType]]:
        """Returns two callables *Phi* and *iPhi*, which compute the cumulative
        density function and inverse cumulative density function of a standard
        univariate Gaussian.

        :param dtype: The data type to use, either tf.float32 or tf.float64.
        :returns Phi, iPhi: Cumulative and inverse cumulative density functions.
        """

        normal = tfd.Normal(
            loc=tf.zeros(shape=(), dtype=dtype),
            scale=tf.ones(shape=(), dtype=dtype),
        )
        Phi: Callable[[TensorType], TensorType] = normal.cdf
        iPhi: Callable[[TensorType], TensorType] = normal.quantile

        return Phi, iPhi

    def _get_update_indices(self, B: int, S: int, Q: int, q: int) -> TensorType:
        """Returns indices for updating a tensor using tf.tensor_scatter_nd_add,
        for use within the _mvn_cdf function, for computing the cumulative density
        function of a multivariate Gaussian. The indices *idx* returned are such
        that the following operation

            idx = get_update_indices(B, S, Q, q)
            tensor = tf.tensor_scatter_nd_add(tensor, idx, update)

        is equivalent to the numpy operation

            tensor = tensor[:, :, q] + update

        where *tensor* is a tensor of shape (B, S, Q).

        :param B: First dim. of tensor for which the indices are generated.
        :param S: Second dim. of tensor for which the indices are generated.
        :param Q: Third dim. of tensor for which the indices are generated.
        :param q: Index of tensor along fourth dim. to which the update is applied.
        """

        idxB = tf.tile(tf.range(B, dtype=tf.int32)[:, None, None], (1, S, 1))
        idxS = tf.tile(tf.range(S, dtype=tf.int32)[None, :, None], (B, 1, 1))
        idxQ = tf.tile(tf.convert_to_tensor(q)[None, None, None], (B, S, 1))

        idx = tf.concat([idxB, idxS, idxQ], axis=-1)

        return idx

    def __call__(
        self,
        x: TensorType,
        mean: TensorType,
        cov: TensorType,
        jitter: float = 1e-6,
    ) -> TensorType:
        """Computes the cumulative density function of the multivariate
        Gaussian using the Genz approximation.

        :param x: Tensor of shape (B, Q), batch of points to evaluate CDF at.
        :param mean: Tensor of shape (B, Q), batch of means.
        :param covariance: Tensor of shape (B, Q, Q), batch of covariances.
        :param jitter: float, jitter to use in the Cholesky factorisation.
        :returns mvn_cdf: Tensor of shape (B,), CDF values.
        """

        # Unpack batch size
        B = x.shape[0]
        tf.debugging.assert_positive(B)

        # Check shapes of input tensors
        tf.debugging.assert_shapes(
            [
                (x, (B, self._Q)),
                (mean, (B, self._Q)),
                (cov, (B, self._Q, self._Q)),
            ]
        )

        # Identify data type to use for all calculations
        dtype = mean.dtype

        # Compute Cholesky factors
        jitter = jitter * tf.eye(self._Q, dtype=dtype)[None, :, :]
        C = tf.linalg.cholesky(cov + jitter)  # (B, Q, Q)

        # Rename samples and limits for brevity
        w = tf.math.sobol_sample(
            dim=self._Q,
            num_results=self._S,
            dtype=self._dtype,
            skip=self._num_sobol_skip,
        )  # (S, Q)
        b = x - mean  # (B, Q)

        # Initialise transformation variables
        e = tf.zeros(shape=(B, self._S, self._Q), dtype=dtype)
        f = tf.zeros(shape=(B, self._S, self._Q), dtype=dtype)
        y = tf.zeros(shape=(B, self._S, self._Q), dtype=dtype)

        # Initialise standard normal for computing CDFs
        Phi, iPhi = self._standard_normal_cdf_and_inverse_cdf(dtype=dtype)

        # Get update indices for convenience later
        idx = self._get_update_indices(B=B, S=self._S, Q=self._Q, q=0)

        # Slice out common tensors
        b0 = b[:, None, 0]
        C0 = C[:, None, 0, 0] + 1e-12

        # Compute transformation variables at the first step
        e_update = tf.tile(Phi(b0 / C0), (1, self._S))  # (B, S)
        e = tf.tensor_scatter_nd_add(e, idx, e_update)
        f = tf.tensor_scatter_nd_add(f, idx, e_update)

        for i in tf.range(1, self._Q):
            # Update y tensor
            y_update = iPhi(1e-6 + (1 - 2e-6) * w[None, :, i - 1] * e[:, :, i - 1])
            y = tf.tensor_scatter_nd_add(y, idx, y_update)

            # Slice out common tensors
            bi = b[:, None, i]
            Ci_ = C[:, None, i, :i]
            Cii = C[:, None, i, i] + 1e-12
            yi = y[:, :, :i]

            # Compute indices to update d, e and f tensors
            idx = self._get_update_indices(B=B, S=self._S, Q=self._Q, q=i)

            # Update e tensor
            e_update = Phi((bi - tf.reduce_sum(Ci_ * yi, axis=-1)) / Cii)
            e = tf.tensor_scatter_nd_add(e, idx, e_update)

            # Update f tensor
            f_update = e[:, :, i] * f[:, :, i - 1]
            f = tf.tensor_scatter_nd_add(f, idx, f_update)

        mvn_cdf = tf.reduce_mean(f[:, :, -1], axis=-1)

        return mvn_cdf

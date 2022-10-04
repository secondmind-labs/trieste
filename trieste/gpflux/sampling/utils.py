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
"""
This module contains utilities for sampling from multivariate Gaussian distributions.
"""
import tensorflow as tf

from gpflow.base import TensorType
from gpflow.conditionals.util import sample_mvn

from gpflux.math import _cholesky_with_jitter


def draw_conditional_sample(mean: TensorType, cov: TensorType, f_old: TensorType) -> tf.Tensor:
    r"""
    Draw a sample :math:`\tilde{f}_\text{new}` from the conditional
    multivariate Gaussian :math:`p(f_\text{new} | f_\text{old})`, where the
    parameters ``mean`` and ``cov`` are the mean and covariance matrix of the
    joint multivariate Gaussian over :math:`[f_\text{old}, f_\text{new}]`.

    :param mean: A tensor with the shape ``[..., D, N+M]`` with the mean of
        ``[f_old, f_new]``. For each ``[..., D]`` this is a stacked vector of the
        form:

        .. math::

            \begin{pmatrix}
                 \operatorname{mean}(f_\text{old}) \;[N] \\
                 \operatorname{mean}(f_\text{new}) \;[M]
            \end{pmatrix}

    :param cov: A tensor with the shape ``[..., D, N+M, N+M]`` with the covariance of
        ``[f_old, f_new]``. For each ``[..., D]``, there is a 2x2 block matrix of the form:

        .. math::

            \begin{pmatrix}
                 \operatorname{cov}(f_\text{old}, f_\text{old}) \;[N, N]
                   & \operatorname{cov}(f_\text{old}, f_\text{new}) \;[N, M] \\
                 \operatorname{cov}(f_\text{new}, f_\text{old}) \;[M, N]
                   & \operatorname{cov}(f_\text{new}, f_\text{new}) \;[M, M]
            \end{pmatrix}

    :param f_old: A tensor of observations with the shape ``[..., D, N]``,
        drawn from Normal distribution with mean
        :math:`\operatorname{mean}(f_\text{old}) \;[N]`, and covariance
        :math:`\operatorname{cov}(f_\text{old}, f_\text{old}) \;[N, N]`

    :return: A sample :math:`\tilde{f}_\text{new}` from the conditional normal
        :math:`p(f_\text{new} | f_\text{old})` with the shape ``[..., D, M]``.
    """
    N, D = tf.shape(f_old)[-1], tf.shape(f_old)[-2]  # noqa: F841
    M = tf.shape(mean)[-1] - N
    cov_old = cov[..., :N, :N]  # [..., D, N, N]
    cov_new = cov[..., -M:, -M:]  # [..., D, M, M]
    cov_cross = cov[..., :N, -M:]  # [..., D, N, M]
    L_old = _cholesky_with_jitter(cov_old)  # [..., D, N, N]
    A = tf.linalg.triangular_solve(L_old, cov_cross, lower=True)  # [..., D, N, M]
    var_new = cov_new - tf.matmul(A, A, transpose_a=True)  # [..., D, M, M]
    mean_new = mean[..., -M:]  # [..., D, M]
    mean_old = mean[..., :N]  # [..., D, N]
    mean_old_diff = (f_old - mean_old)[..., None]  # [..., D, N, 1]
    AM = tf.linalg.triangular_solve(L_old, mean_old_diff)  # [..., D, N, 1]
    mean_new = mean_new + (tf.matmul(A, AM, transpose_a=True)[..., 0])  # [..., D, M]
    return sample_mvn(mean_new, var_new, full_cov=True)

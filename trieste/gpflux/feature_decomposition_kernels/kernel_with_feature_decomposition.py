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
r"""
The classes in this module encapsulate kernels :math:`k(\cdot, \cdot)` with
their features :math:`\phi_i(\cdot)` and coefficients :math:`\lambda_i` so
that:

.. math::

    k(x, x') = \sum_{i=0}^\infty \lambda_i \phi_i(x) \phi_i(x').

The kernels are used for efficient sampling. See the tutorial notebooks
`Efficient sampling <../../../../notebooks/efficient_sampling.ipynb>`_
and `Weight Space Approximation with Random Fourier Features
<../../../../notebooks/weight_space_approximation.ipynb>`_
for an in-depth overview.
"""
from typing import Optional, Union

import tensorflow as tf

import gpflow
from gpflow.base import TensorType

NoneType = type(None)


class _ApproximateKernel(gpflow.kernels.Kernel):
    r"""
    This class approximates a kernel by the finite feature decomposition:

    .. math:: k(x, x') = \sum_{i=0}^L \lambda_i \phi_i(x) \phi_i(x'),

    where :math:`\lambda_i` and :math:`\phi_i(\cdot)` are the coefficients
    and features, respectively.

    """

    def __init__(
        self,
        feature_functions: tf.keras.layers.Layer,
        feature_coefficients: TensorType,
    ):
        r"""
        :param feature_functions: A Keras layer for which the call evaluates the
            ``L`` features of the kernel :math:`\phi_i(\cdot)`. For ``X`` with the shape ``[N, D]``,
            ``feature_functions(X)`` returns a tensor with the shape ``[N, L]``.
        :param feature_coefficients: A tensor with the shape ``[L, 1]`'  with coefficients
            associated with the features, :math:`\lambda_i`.
        """
        self._feature_functions = feature_functions  # [N, L]
        self._feature_coefficients = feature_coefficients  # [L, 1]

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        """Approximate the true kernel by an inner product between feature functions."""
        phi = self._feature_functions(X)  # [N, L]
        if X2 is None:
            phi2 = phi
        else:
            phi2 = self._feature_functions(X2)  # [N2, L]

        r = tf.linalg.matmul(
            phi,
            tf.linalg.matrix_transpose(self._feature_coefficients) * phi2,
            transpose_b=True,
        )  # [N, N2]

        N1, N2 = tf.shape(phi)[0], tf.shape(phi2)[0]

        tf.debugging.assert_equal(tf.shape(r), [N1, N2])
        return r

    def K_diag(self, X: TensorType) -> tf.Tensor:
        """Approximate the true kernel by an inner product between feature functions."""
        phi_squared = self._feature_functions(X) ** 2  # [N, L]
        r = tf.reduce_sum(phi_squared * tf.transpose(self._feature_coefficients), axis=-1)  # [N,]
        N = tf.shape(X)[0] if tf.experimental.numpy.ndim(X) == 1 else tf.shape(X)[0]

        tf.debugging.assert_equal(tf.shape(r), [N])  # noqa: E231
        return r


class KernelWithFeatureDecomposition(gpflow.kernels.Kernel):
    r"""
    This class represents a kernel together with its finite feature decomposition:

    .. math:: k(x, x') = \sum_{i=0}^L \lambda_i \phi_i(x) \phi_i(x'),

    where :math:`\lambda_i` and :math:`\phi_i(\cdot)` are the coefficients and
    features, respectively.

    The decomposition can be derived from Mercer or Bochner's theorem. For example,
    feature-coefficient pairs could be eigenfunction-eigenvalue pairs (Mercer) or
    Fourier features with constant coefficients (Bochner).

    In some cases (e.g., [1]_ and [2]_) the left-hand side (that is, the
    covariance function :math:`k(\cdot, \cdot)`) is unknown and the kernel
    can only be approximated using its feature decomposition.
    In other cases (e.g., [3]_ and [4]_), both the covariance function and feature
    decomposition are available in closed form.

    .. [1]
        Solin, Arno, and Simo Särkkä. "Hilbert space methods for
        reduced-rank Gaussian process regression." Statistics and Computing
        (2020).
    .. [2]
        Borovitskiy, Viacheslav, et al. "Matérn Gaussian processes on
        Riemannian manifolds." In Advances in Neural Information Processing
        Systems (2020).
    .. [3]
        Ali Rahimi and Benjamin Recht. Random features for large-scale kernel
        machines. In Advances in Neural Information Processing Systems (2007).
    .. [4]
        Dutordoir, Vincent, Nicolas Durrande, and James Hensman. "Sparse
        Gaussian processes with spherical harmonic features." In International
        Conference on Machine Learning (2020).
    """

    def __init__(
        self,
        kernel: Union[gpflow.kernels.Kernel, NoneType],
        feature_functions: tf.keras.layers.Layer,
        feature_coefficients: TensorType,
    ):
        r"""
        :param kernel: The kernel corresponding to the feature decomposition.
            If ``None``, there is no analytical expression associated with the infinite
            sum and we approximate the kernel based on the feature decomposition.

            .. note::

                In certain cases, the analytical expression for the kernel is
                not available. In this case, passing `None` is allowed, and
                :meth:`K` and :meth:`K_diag` will be computed using the
                approximation provided by the feature decomposition.

        :param feature_functions: A Keras layer for which the call evaluates the
            ``L`` features of the kernel :math:`\phi_i(\cdot)`. For ``X`` with the shape ``[N, D]``,
            ``feature_functions(X)`` returns a tensor with the shape ``[N, L]``.
        :param feature_coefficients: A tensor with the shape ``[L, 1]`` with coefficients
            associated with the features, :math:`\lambda_i`.
        """
        super().__init__()

        if kernel is None:
            self._kernel = _ApproximateKernel(feature_functions, feature_coefficients)
        else:
            self._kernel = kernel

        self._feature_functions = feature_functions  # [N, L]
        self._feature_coefficients = feature_coefficients  # [L, 1]

        tf.ensure_shape(self._feature_coefficients, tf.TensorShape([None, 1]))

    @property
    def feature_functions(self) -> tf.keras.layers.Layer:
        r"""Return the kernel's features :math:`\phi_i(\cdot)`."""
        return self._feature_functions

    @property
    def feature_coefficients(self) -> tf.Tensor:
        r"""Return the kernel's coefficients :math:`\lambda_i`."""
        return self._feature_coefficients

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        return self._kernel.K(X, X2)

    def K_diag(self, X: TensorType) -> tf.Tensor:
        return self._kernel.K_diag(X)

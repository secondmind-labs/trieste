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

from typing import Tuple, Union

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.keras import tf_keras

from ...data import Dataset
from ...types import TensorType
from ...utils import DEFAULTS
from ..optimizer import BatchOptimizer, Optimizer
from .interface import GPflowPredictor


def assert_data_is_compatible(new_data: Dataset, existing_data: Dataset) -> None:
    """
    Checks that new data is compatible with existing data.

    :param new_data: New data.
    :param existing_data: Existing data.
    :raise ValueError: if trailing dimensions of the query point or observation differ.
    """
    if new_data.query_points.shape[-1] != existing_data.query_points.shape[-1]:
        raise ValueError(
            f"Shape {new_data.query_points.shape} of new query points is incompatible with"
            f" shape {existing_data.query_points.shape} of existing query points. Trailing"
            f" dimensions must match."
        )

    if new_data.observations.shape[-1] != existing_data.observations.shape[-1]:
        raise ValueError(
            f"Shape {new_data.observations.shape} of new observations is incompatible with"
            f" shape {existing_data.observations.shape} of existing observations. Trailing"
            f" dimensions must match."
        )


def randomize_hyperparameters(object: gpflow.Module) -> None:
    """
    Sets hyperparameters to random samples from their prior distributions or (for Sigmoid
    constraints with no priors) their constrained domains. Note that it is up to the caller
    to ensure that the prior, if defined, is compatible with the transform.

    :param object: Any gpflow Module.
    """
    for param in object.trainable_parameters:
        if param.prior is not None:
            # handle constant priors for multi-dimensional parameters
            # Use python conditionals here to avoid creating tensorflow `tf.cond` ops,
            # i.e. using `len(param.shape)` instead of `tf.rank(param)`.
            # Otherwise, tensorflow generates repeating random sequences for hyperparameters, see
            # https://github.com/tensorflow/tensorflow/issues/61912.
            if param.prior.batch_shape == param.prior.event_shape == [] and len(param.shape) == 1:
                sample = param.prior.sample(tf.shape(param))
            else:
                sample = param.prior.sample()
            if param.prior_on is gpflow.base.PriorOn.UNCONSTRAINED:
                param.unconstrained_variable.assign(sample)
            else:
                param.assign(sample)

        elif isinstance(param.bijector, tfp.bijectors.Sigmoid):
            sample = tf.random.uniform(
                param.bijector.low.shape,
                minval=param.bijector.low,
                maxval=param.bijector.high,
                dtype=param.bijector.low.dtype,
            )
            param.assign(sample)


def squeeze_hyperparameters(
    object: gpflow.Module, alpha: float = 1e-2, epsilon: float = 1e-7
) -> None:
    """
    Squeezes the parameters to be strictly inside their range defined by the Sigmoid,
    or strictly greater than the limit defined by the Shift+Softplus.
    This avoids having Inf unconstrained values when the parameters are exactly at the boundary.

    :param object: Any gpflow Module.
    :param alpha: the proportion of the range with which to squeeze for the Sigmoid case
    :param epsilon: the value with which to offset the shift for the Softplus case.
    :raise ValueError: If ``alpha`` is not in (0,1) or epsilon <= 0
    """

    if not 0 < alpha < 1:
        raise ValueError(f"squeeze factor alpha must be in (0, 1), found {alpha}")

    if epsilon <= 0:
        raise ValueError(f"offset factor epsilon must be > 0, found {epsilon}")

    for param in object.trainable_parameters:
        if isinstance(param.bijector, tfp.bijectors.Sigmoid):
            delta = (param.bijector.high - param.bijector.low) * alpha
            squeezed_param = tf.math.minimum(param, param.bijector.high - delta)
            squeezed_param = tf.math.maximum(squeezed_param, param.bijector.low + delta)
            param.assign(squeezed_param)
        elif (
            isinstance(param.bijector, tfp.bijectors.Chain)
            and len(param.bijector.bijectors) == 2
            and isinstance(param.bijector.bijectors[0], tfp.bijectors.Shift)
            and isinstance(param.bijector.bijectors[1], tfp.bijectors.Softplus)
        ):
            if isinstance(param.bijector.bijectors[0], tfp.bijectors.Shift) and isinstance(
                param.bijector.bijectors[1], tfp.bijectors.Softplus
            ):
                low = param.bijector.bijectors[0].shift
                squeezed_param = tf.math.maximum(param, low + epsilon * tf.ones_like(param))
                param.assign(squeezed_param)


def check_optimizer(optimizer: Union[BatchOptimizer, Optimizer]) -> None:
    """
    Check that the optimizer for the GPflow models is using a correct optimizer wrapper.

    Stochastic gradient descent based methods implemented in TensorFlow would not
    work properly without mini-batches and hence :class:`~trieste.models.optimizers.BatchOptimizer`
    that prepares mini-batches and calls the optimizer iteratively needs to be used. GPflow's
    :class:`~gpflow.optimizers.Scipy` optimizer on the other hand should use the non-batch wrapper
    :class:`~trieste.models.optimizers.Optimizer`.

    :param optimizer: An instance of the optimizer wrapper with the underlying optimizer.
    :raise ValueError: If :class:`~tf.optimizers.Optimizer` is not using
        :class:`~trieste.models.optimizers.BatchOptimizer` or :class:`~gpflow.optimizers.Scipy` is
        using :class:`~trieste.models.optimizers.BatchOptimizer`.
    """
    if isinstance(optimizer.optimizer, gpflow.optimizers.Scipy):
        if isinstance(optimizer, BatchOptimizer):
            raise ValueError(
                f"""
                The gpflow.optimizers.Scipy can only be used with an Optimizer wrapper,
                however received {optimizer}.
                """
            )

    if isinstance(optimizer.optimizer, tf_keras.optimizers.Optimizer):
        if not isinstance(optimizer, BatchOptimizer):
            raise ValueError(
                f"""
                The tf.optimizers.Optimizer can only be used with a BatchOptimizer wrapper,
                however received {optimizer}.
                """
            )


def _covariance_between_points_for_variational_models(
    kernel: gpflow.kernels.Kernel,
    inducing_points: TensorType,
    q_sqrt: TensorType,
    query_points_1: TensorType,
    query_points_2: TensorType,
    whiten: bool,
) -> TensorType:
    r"""
    Compute the posterior covariance between sets of query points.

    .. math:: \Sigma_{12} = K_{1x}BK_{x2} + K_{12} - K_{1x}K_{xx}^{-1}K_{x2}

    where :math:`B = K_{xx}^{-1}(q_{sqrt}q_{sqrt}^T)K_{xx}^{-1}`
    or :math:`B = L^{-1}(q_{sqrt}q_{sqrt}^T)(L^{-1})^T` if we are using
    a whitened representation in our variational approximation. Here
    :math:`L` is the Cholesky decomposition of :math:`K_{xx}`.
    See :cite:`titsias2009variational` for a derivation.

    Note that this function can also be applied to
    our :class:`VariationalGaussianProcess` models by passing in the training
    data rather than the locations of the inducing points.

    Although query_points_2 must be a rank 2 tensor, query_points_1 can
    have leading dimensions.

    :inducing points: The input locations chosen for our variational approximation.
    :q_sqrt: The Cholesky decomposition of the covariance matrix of our
        variational distribution.
    :param query_points_1: Set of query points with shape [..., A, D]
    :param query_points_2: Sets of query points with shape [B, D]
    :param whiten:  If True then use whitened representations.
    :return: Covariance matrix between the sets of query points with shape [..., L, A, B]
        (L being the number of latent GPs = number of output dimensions)
    """

    tf.debugging.assert_shapes([(query_points_1, [..., "A", "D"]), (query_points_2, ["B", "D"])])

    num_latent = q_sqrt.shape[0]

    K, Kx1, Kx2, K12 = _compute_kernel_blocks(
        kernel, inducing_points, query_points_1, query_points_2, num_latent
    )

    L = tf.linalg.cholesky(K)  # [L, M, M]
    Linv_Kx1 = tf.linalg.triangular_solve(L, Kx1)  # [..., L, M, A]
    Linv_Kx2 = tf.linalg.triangular_solve(L, Kx2)  # [..., L, M, B]

    def _leading_mul(M_1: TensorType, M_2: TensorType, transpose_a: bool) -> TensorType:
        if transpose_a:  # The einsum below is just A^T*B over the last 2 dimensions.
            return tf.einsum("...lji,ljk->...lik", M_1, M_2)
        else:  # The einsum below is just A*B^T over the last 2 dimensions.
            return tf.einsum("...lij,lkj->...lik", M_1, M_2)

    if whiten:
        first_cov_term = _leading_mul(
            _leading_mul(Linv_Kx1, q_sqrt, transpose_a=True),  # [..., L, A, M]
            _leading_mul(Linv_Kx2, q_sqrt, transpose_a=True),  # [..., L, B, M]
            transpose_a=False,
        )  # [..., L, A, B]
    else:
        Linv_qsqrt = tf.linalg.triangular_solve(L, q_sqrt)  # [L, M, M]
        first_cov_term = _leading_mul(
            _leading_mul(Linv_Kx1, Linv_qsqrt, transpose_a=True),  # [..., L, A, M]
            _leading_mul(Linv_Kx2, Linv_qsqrt, transpose_a=True),  # [..., L, B, M]
            transpose_a=False,
        )  # [..., L, A, B]

    second_cov_term = K12  # [..., L, A, B]
    third_cov_term = _leading_mul(Linv_Kx1, Linv_Kx2, transpose_a=True)  # [..., L, A, B]
    cov = first_cov_term + second_cov_term - third_cov_term  # [..., L, A, B]

    tf.debugging.assert_shapes(
        [
            (query_points_1, [..., "N", "D"]),
            (query_points_2, ["M", "D"]),
            (cov, [..., "L", "N", "M"]),
        ]
    )
    return cov


def _compute_kernel_blocks(
    kernel: gpflow.kernels.Kernel,
    inducing_points: TensorType,
    query_points_1: TensorType,
    query_points_2: TensorType,
    num_latent: int,
) -> tuple[TensorType, TensorType, TensorType, TensorType]:
    """
    Return all the prior covariances required to calculate posterior covariances for each latent
    Gaussian process, as specified by the `num_latent` input.

    This function returns the covariance between: `inducing_points` and `query_points_1`;
    `inducing_points` and `query_points_2`; `query_points_1` and `query_points_2`;
    `inducing_points` and `inducing_points`.

    The calculations are performed differently depending on the type of
    kernel (single output, separate independent multi-output or shared independent
    multi-output) and inducing variables (simple set, SharedIndependent or SeparateIndependent).

    Note that `num_latents` is only used when we use a single kernel for a multi-output model.
    """

    if isinstance(kernel, (gpflow.kernels.SharedIndependent, gpflow.kernels.SeparateIndependent)):
        if isinstance(inducing_points, list):
            K = tf.concat(
                [ker(Z)[None, ...] for ker, Z in zip(kernel.kernels, inducing_points)], axis=0
            )
            Kx1 = tf.concat(
                [
                    ker(Z, query_points_1)[None, ...]
                    for ker, Z in zip(kernel.kernels, inducing_points)
                ],
                axis=0,
            )  # [..., L, M, A]
            Kx2 = tf.concat(
                [
                    ker(Z, query_points_2)[None, ...]
                    for ker, Z in zip(kernel.kernels, inducing_points)
                ],
                axis=0,
            )  # [L, M, B]
            K12 = tf.concat(
                [ker(query_points_1, query_points_2)[None, ...] for ker in kernel.kernels], axis=0
            )  # [L, M, B]
        else:
            K = kernel(inducing_points, full_cov=True, full_output_cov=False)  # [L, M, M]
            Kx1 = kernel(
                inducing_points, query_points_1, full_cov=True, full_output_cov=False
            )  # [..., L, M, A]
            Kx2 = kernel(
                inducing_points, query_points_2, full_cov=True, full_output_cov=False
            )  # [L, M, B]
            K12 = kernel(
                query_points_1, query_points_2, full_cov=True, full_output_cov=False
            )  # [..., L, A, B]
    else:  # simple calculations for the single output case
        K = kernel(inducing_points)  # [M, M]
        Kx1 = kernel(inducing_points, query_points_1)  # [..., M, A]
        Kx2 = kernel(inducing_points, query_points_2)  # [M, B]
        K12 = kernel(query_points_1, query_points_2)  # [..., A, B]

    if len(tf.shape(K)) == 2:  # if single kernel then repeat for all latent dimensions
        K = tf.repeat(tf.expand_dims(K, -3), num_latent, axis=-3)
        Kx1 = tf.repeat(tf.expand_dims(Kx1, -3), num_latent, axis=-3)
        Kx2 = tf.repeat(tf.expand_dims(Kx2, -3), num_latent, axis=-3)
        K12 = tf.repeat(tf.expand_dims(K12, -3), num_latent, axis=-3)
    elif len(tf.shape(K)) > 3:
        raise NotImplementedError(
            "Covariance between points is not supported for kernels of type {type(kernel)}."
        )

    tf.debugging.assert_shapes(
        [
            (K, ["L", "M", "M"]),
            (Kx1, ["L", "M", "A"]),
            (Kx2, ["L", "M", "B"]),
            (K12, ["L", "A", "B"]),
        ]
    )

    return K, Kx1, Kx2, K12


def _whiten_points(
    model: GPflowPredictor, inducing_points: TensorType
) -> Tuple[TensorType, TensorType]:
    """
    GPFlow's VGP and SVGP can use whitened representation, i.e.
    q_mu and q_sqrt parametrize q(v), and u = f(X) = L v, where L = cholesky(K(X, X))
    Hence we need to back-transform from f_mu and f_cov to obtain the updated
    new_q_mu and new_q_sqrt.

    :param model: The whitened model.
    :para inducing_points: The new inducing point locations.
    :return: The updated q_mu and q_sqrt with shapes [N, L] and [L, N, N], respectively.
    """

    f_mu, f_cov = model.model.predict_f(inducing_points, full_cov=True)  # [N, L], [L, N, N]
    f_mu -= model.model.mean_function(inducing_points)
    Knn = model.get_kernel()(inducing_points, full_cov=True)  # [N, N]
    jitter_mat = DEFAULTS.JITTER * tf.eye(tf.shape(inducing_points)[0], dtype=Knn.dtype)
    Lnn = tf.linalg.cholesky(Knn + jitter_mat)  # [N, N]
    new_q_mu = tf.linalg.triangular_solve(Lnn, f_mu)  # [N, L]
    tmp = tf.linalg.triangular_solve(Lnn[None], f_cov)  # [L, N, N], L⁻¹ f_cov
    S_v = tf.linalg.triangular_solve(Lnn[None], tf.linalg.matrix_transpose(tmp))  # [L, N, N]
    new_q_sqrt = tf.linalg.cholesky(S_v + jitter_mat)  # [L, N, N]

    return new_q_mu, new_q_sqrt

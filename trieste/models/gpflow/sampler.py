# Copyright 2020 The Trieste Contributors
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
This module is the home of the sampling functionality required by Trieste's
GPflow wrappers.
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

try:
    from gpflux.layers.basis_functions.fourier_features import RandomFourierFeaturesCosine as RFF
except (ModuleNotFoundError, ImportError):
    # temporary support for gpflux 0.2.3
    from gpflux.layers.basis_functions import RandomFourierFeatures as RFF

from typing_extensions import Protocol, runtime_checkable

from ...types import TensorType
from ...utils import DEFAULTS, flatten_leading_dims
from ..interfaces import (
    ProbabilisticModel,
    ReparametrizationSampler,
    SupportsGetKernel,
    SupportsGetObservationNoise,
    SupportsInternalData,
    SupportsPredictJoint,
    TrajectoryFunction,
    TrajectoryFunctionClass,
    TrajectorySampler,
)


class IndependentReparametrizationSampler(ReparametrizationSampler[ProbabilisticModel]):
    r"""
    This sampler employs the *reparameterization trick* to approximate samples from a
    :class:`ProbabilisticModel`\ 's predictive distribution as

    .. math:: x \mapsto \mu(x) + \epsilon \sigma(x)

    where :math:`\epsilon \sim \mathcal N (0, 1)` is constant for a given sampler, thus ensuring
    samples form a continuous curve.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel):
        """
        :param sample_size: The number of samples to take at each point. Must be positive.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        super().__init__(sample_size, model)

        # _eps is essentially a lazy constant. It is declared and assigned an empty tensor here, and
        # populated on the first call to sample
        self._eps = tf.Variable(
            tf.ones([sample_size, 0], dtype=tf.float64), shape=[sample_size, None]
        )  # [S, 0]

    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`IndependentReparametrizationSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`IndependentReparametrizationSampler` instances will produce different samples.

        :param at: Where to sample the predictive distribution, with shape `[..., 1, D]`, for points
            of dimension `D`.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :return: The samples, of shape `[..., S, 1, L]`, where `S` is the `sample_size` and `L` is
            the number of latent model dimensions.
        :raise ValueError (or InvalidArgumentError): If ``at`` has an invalid shape or ``jitter``
            is negative.
        """
        tf.debugging.assert_shapes([(at, [..., 1, None])])
        tf.debugging.assert_greater_equal(jitter, 0.0)

        mean, var = self._model.predict(at[..., None, :, :])  # [..., 1, 1, L], [..., 1, 1, L]
        var = var + jitter

        if not self._initialized:
            self._eps.assign(
                tf.random.normal([self._sample_size, tf.shape(mean)[-1]], dtype=tf.float64)
            )  # [S, L]
            self._initialized.assign(True)

        return mean + tf.sqrt(var) * tf.cast(self._eps[:, None, :], var.dtype)  # [..., S, 1, L]


class BatchReparametrizationSampler(ReparametrizationSampler[SupportsPredictJoint]):
    r"""
    This sampler employs the *reparameterization trick* to approximate batches of samples from a
    :class:`ProbabilisticModel`\ 's predictive joint distribution as

    .. math:: x \mapsto \mu(x) + \epsilon L(x)

    where :math:`L` is the Cholesky factor s.t. :math:`LL^T` is the covariance, and
    :math:`\epsilon \sim \mathcal N (0, 1)` is constant for a given sampler, thus ensuring samples
    form a continuous curve.
    """

    def __init__(self, sample_size: int, model: SupportsPredictJoint):
        """
        :param sample_size: The number of samples for each batch of points. Must be positive.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        super().__init__(sample_size, model)
        if not isinstance(model, SupportsPredictJoint):
            raise NotImplementedError(
                f"BatchReparametrizationSampler only works with models that support "
                f"predict_joint; received {model.__repr__()}"
            )

        # _eps is essentially a lazy constant. It is declared and assigned an empty tensor here, and
        # populated on the first call to sample
        self._eps = tf.Variable(
            tf.ones([0, 0, sample_size], dtype=tf.float64), shape=[None, None, sample_size]
        )  # [0, 0, S]

    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`BatchReparametrizationSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`BatchReparametrizationSampler` instances will produce different samples.

        :param at: Batches of query points at which to sample the predictive distribution, with
            shape `[..., B, D]`, for batches of size `B` of points of dimension `D`. Must have a
            consistent batch size across all calls to :meth:`sample` for any given
            :class:`BatchReparametrizationSampler`.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :return: The samples, of shape `[..., S, B, L]`, where `S` is the `sample_size`, `B` the
            number of points per batch, and `L` the dimension of the model's predictive
            distribution.
        :raise ValueError (or InvalidArgumentError): If any of the following are true:
            - ``at`` is a scalar.
            - The batch size `B` of ``at`` is not positive.
            - The batch size `B` of ``at`` differs from that of previous calls.
            - ``jitter`` is negative.
        """
        tf.debugging.assert_rank_at_least(at, 2)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        batch_size = at.shape[-2]

        tf.debugging.assert_positive(batch_size)

        if self._initialized:
            tf.debugging.assert_equal(
                batch_size,
                tf.shape(self._eps)[-2],
                f"{type(self).__name__} requires a fixed batch size. Got batch size {batch_size}"
                f" but previous batch size was {tf.shape(self._eps)[-2]}.",
            )

        mean, cov = self._model.predict_joint(at)  # [..., B, L], [..., L, B, B]

        if not self._initialized:
            self._eps.assign(
                tf.random.normal(
                    [tf.shape(mean)[-1], batch_size, self._sample_size], dtype=tf.float64
                )  # [L, B, S]
            )
            self._initialized.assign(True)

        identity = tf.eye(batch_size, dtype=cov.dtype)  # [B, B]
        cov_cholesky = tf.linalg.cholesky(cov + jitter * identity)  # [..., L, B, B]

        variance_contribution = cov_cholesky @ tf.cast(self._eps, cov.dtype)  # [..., L, B, S]

        leading_indices = tf.range(tf.rank(variance_contribution) - 3)
        absolute_trailing_indices = [-1, -2, -3] + tf.rank(variance_contribution)
        new_order = tf.concat([leading_indices, absolute_trailing_indices], axis=0)

        return mean[..., None, :, :] + tf.transpose(variance_contribution, new_order)


@runtime_checkable
class SupportsGetKernelObservationNoiseInternalData(
    SupportsGetKernel, SupportsGetObservationNoise, SupportsInternalData, Protocol
):
    """
    A probabilistic model that supports get_kernel, get_observation noise
    and get_internal_data.
    """

    pass


class RandomFourierFeatureTrajectorySampler(
    TrajectorySampler[SupportsGetKernelObservationNoiseInternalData]
):
    r"""
    This class builds functions that approximate a trajectory sampled from an underlying Gaussian
    process model. For tractibility, the Gaussian process is approximated with a Bayesian
    Linear model across a set of features sampled from the Fourier feature decomposition of
    the model's kernel. See :cite:`hernandez2014predictive` for details.

    Achieving consistency (ensuring that the same sample draw for all evalutions of a particular
    trajectory function) for exact sample draws from a GP is prohibitively costly because it scales
    cubically with the number of query points. However, finite feature representations can be
    evaluated with constant cost regardless of the required number of queries.

    In particular, we approximate the Gaussian processes' posterior samples as the finite feature
    approximation

    .. math:: \hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i

    where :math:`\phi_i` are m Fourier features and :math:`\theta_i` are
    feature weights sampled from a posterior distribution that depends on the feature values at the
    model's datapoints.

    Our implementation follows :cite:`hernandez2014predictive`, with our calculations
    differing slightly depending on properties of the problem. In particular,  we used different
    calculation strategies depending on the number of considered features m and the number
    of data points n.

    If :math:`m<n` then we follow Appendix A of :cite:`hernandez2014predictive` and calculate the
    posterior distribution for :math:`\theta` following their Bayesian linear regression motivation,
    i.e. the computation revolves around an O(m^3)  inversion of a design matrix.

    If :math:`n<m` then we use the kernel trick to recast computation to revolve around an O(n^3)
    inversion of a gram matrix. As well as being more efficient in early BO
    steps (where :math:`n<m`), this second computation method allows much larger choices
    of m (as required to approximate very flexible kernels).
    """

    def __init__(
        self,
        model: SupportsGetKernelObservationNoiseInternalData,
        num_features: int = 1000,
    ):
        """
        :param sample_size: The desired number of samples.
        :param model: The model to sample from.
        :param num_features: The number of features used to approximate the kernel. We use a default
            of 1000 as it typically perfoms well for a wide range of kernels. Note that very smooth
            kernels (e.g. RBF) can be well-approximated with fewer features.
        :raise ValueError: If ``dataset`` is empty.
        """

        super().__init__(model)

        if not isinstance(model, SupportsGetKernelObservationNoiseInternalData):
            raise NotImplementedError(
                f"RandomFourierFeatureTrajectorySampler only works with models that support "
                f"get_kernel, get_observation_noise and get_internal_data; "
                f"but received {model.__repr__()}."
            )

        self._model = model

        tf.debugging.assert_positive(num_features)
        self._num_features = num_features  # m

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}(
        {self._model!r},
        {self._num_features!r})
        """

    def _build_theta_posterior(self) -> tfp.distributions.Distribution:
        # Calculate the posterior of theta (the feature weights) for the specified feature function.
        dataset = self._model.get_internal_data()
        num_data = tf.shape(dataset.query_points)[0]  # n
        if (
            self._num_features < num_data
        ):  # if m < n  then calculate posterior in design space (an m*m matrix inversion)
            return self._prepare_theta_posterior_in_design_space()
        else:  # if n <= m  then calculate posterior in gram space (an n*n matrix inversion)
            return self._prepare_theta_posterior_in_gram_space()

    def _prepare_theta_posterior_in_design_space(self) -> tfp.distributions.MultivariateNormalTriL:
        r"""
        Calculate the posterior of theta (the feature weights) in the design space. This
        distribution is a Gaussian

        .. math:: \theta \sim N(D^{-1}\Phi^Ty,D^{-1}\sigma^2)

        where the [m,m] design matrix :math:`D=(\Phi^T\Phi + \sigma^2I_m)` is defined for
        the [n,m] matrix of feature evaluations across the training data :math:`\Phi`
        and observation noise variance :math:`\sigma^2`.
        """
        dataset = self._model.get_internal_data()
        phi = self._feature_functions(dataset.query_points)  # [n, m]
        D = tf.matmul(phi, phi, transpose_a=True)  # [m, m]
        s = self._model.get_observation_noise() * tf.eye(self._num_features, dtype=phi.dtype)
        L = tf.linalg.cholesky(D + s)
        D_inv = tf.linalg.cholesky_solve(L, tf.eye(self._num_features, dtype=phi.dtype))

        theta_posterior_mean = tf.matmul(
            D_inv, tf.matmul(phi, dataset.observations, transpose_a=True)
        )[
            :, 0
        ]  # [m,]
        theta_posterior_chol_covariance = tf.linalg.cholesky(
            D_inv * self._model.get_observation_noise()
        )  # [m, m]

        return tfp.distributions.MultivariateNormalTriL(
            theta_posterior_mean, theta_posterior_chol_covariance
        )

    def _prepare_theta_posterior_in_gram_space(self) -> tfp.distributions.MultivariateNormalTriL:
        r"""
        Calculate the posterior of theta (the feature weights) in the gram space.

         .. math:: \theta \sim N(\Phi^TG^{-1}y,I_m - \Phi^TG^{-1}\Phi)

        where the [n,n] gram matrix :math:`G=(\Phi\Phi^T + \sigma^2I_n)` is defined for the [n,m]
        matrix of feature evaluations across the training data :math:`\Phi` and
        observation noise variance :math:`\sigma^2`.
        """
        dataset = self._model.get_internal_data()
        num_data = tf.shape(dataset.query_points)[0]  # n
        phi = self._feature_functions(dataset.query_points)  # [n, m]
        G = tf.matmul(phi, phi, transpose_b=True)  # [n, n]
        s = self._model.get_observation_noise() * tf.eye(num_data, dtype=phi.dtype)
        L = tf.linalg.cholesky(G + s)
        L_inv_phi = tf.linalg.triangular_solve(L, phi)  # [n, m]
        L_inv_y = tf.linalg.triangular_solve(L, dataset.observations)  # [n, 1]

        theta_posterior_mean = tf.tensordot(tf.transpose(L_inv_phi), L_inv_y, [[-1], [-2]])[
            :, 0
        ]  # [m,]
        theta_posterior_covariance = tf.eye(self._num_features, dtype=phi.dtype) - tf.tensordot(
            tf.transpose(L_inv_phi), L_inv_phi, [[-1], [-2]]
        )  # [m, m]
        theta_posterior_chol_covariance = tf.linalg.cholesky(theta_posterior_covariance)  # [m, m]

        return tfp.distributions.MultivariateNormalTriL(
            theta_posterior_mean, theta_posterior_chol_covariance
        )

    def get_trajectory(self) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) by sampling weights
        and evaluating the feature functions.

        :return: A trajectory function representing an approximate trajectory from the Gaussian
            process, taking an input of shape `[N, D]` and returning shape `[N, 1]`
        """

        data_dtype = self._model.get_internal_data().query_points.dtype

        self._feature_functions = RFF(
            self._model.get_kernel(), self._num_features, dtype=data_dtype
        )  # prep feature functions at data

        self._theta_posterior = self._build_theta_posterior()  # prep feature weight distribution

        self._feature_functions.b = tf.Variable(
            self._feature_functions.b
        )  # store bias as a variable to allow in place updating
        self._feature_functions.W = tf.Variable(
            self._feature_functions.W
        )  # store weights as a variable to allow in place updating

        return fourier_feature_trajectory(
            feature_functions=self._feature_functions,
            weight_distribution=self._theta_posterior,
        )

    def update_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        Efficiently update a :const:`TrajectoryFunction` to reflect an update in its
        underlying :class:`ProbabilisticModel` and resample accordingly.

        For a :class:`RandomFourierFeatureTrajectorySampler`, updating the sampler
        corresponds to resampling the feature functions (taking into account any
        changed kernel parameters) and recalculating the weight distribution.

        :param trajectory: The trajectory function to be resampled.
        :return: The new resampled trajectory function.
        """
        tf.debugging.Assert(isinstance(trajectory, fourier_feature_trajectory), [])

        if not hasattr(self._feature_functions, "_bias_init"):
            # maintain support for gpflux 0.2.3 (but with retracing)
            return self.get_trajectory()

        bias_shape = tf.shape(self._feature_functions.b)
        bias_dtype = self._feature_functions.b.dtype
        weight_shape = tf.shape(self._feature_functions.W)
        weight_dtype = self._feature_functions.W.dtype
        self._feature_functions.b.assign(  # resample feature function's bias
            self._feature_functions._bias_init(bias_shape, dtype=bias_dtype)
        )
        self._feature_functions.W.assign(  # resample feature function's weights
            self._feature_functions._weights_init(weight_shape, dtype=weight_dtype)
        )

        self._theta_posterior = self._build_theta_posterior()  # recalculate weight distribution

        trajectory.update(weight_distribution=self._theta_posterior)  # type: ignore

        return trajectory  # return trajectory with updated features and weight distribution

    def resample_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        Efficiently resample a :const:`TrajectoryFunction` in-place to avoid function retracing
        with every new sample.

        :param trajectory: The trajectory function to be resampled.
        :return: The new resampled trajectory function.
        """
        tf.debugging.Assert(isinstance(trajectory, fourier_feature_trajectory), [])
        trajectory.resample()  # type: ignore
        return trajectory  # return trajectory with resampled weights


class fourier_feature_trajectory(TrajectoryFunctionClass):
    r"""
    An approximate sample from a Gaussian processes' posterior samples represented as a
    finite weighted sum of features.

    A trajectory is given by

    .. math:: \hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i

    where :math:`\phi_i` are m feature functions and :math:`\theta_i` are
    feature weights sampled from a posterior distribution.

    The number of trajectories (i.e. batch size) is determined from the first call of the
    trajectory. In order to change the batch size, a new :class:`TrajectoryFunction` must be built.
    """

    def __init__(
        self,
        feature_functions: RFF,
        weight_distribution: tfp.distributions.MultivariateNormalTriL,
    ):
        """
        :param feature_functions: Set of feature function.
        :param weight_distribution: Distribution from which feature weights are to be sampled.
        """
        self._feature_functions = feature_functions
        self._weight_distribution = weight_distribution
        self._initialized = tf.Variable(False)

        num_features = tf.shape(weight_distribution.mean())[0]  # m
        self._theta_sample = tf.Variable(  # dummy init to be updated before trajectory evaluation
            tf.ones([0, num_features], dtype=tf.float64), shape=[None, num_features]
        )  # [0, m]
        self._batch_size = tf.Variable(
            0, dtype=tf.int32
        )  # dummy init to be updated before trajectory evaluation

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:  # [N, B, d] -> [N, B]
        """Call trajectory function."""

        if not self._initialized:  # work out desired batch size from input
            self._batch_size.assign(tf.shape(x)[-2])  # B
            self.resample()  # sample B feature weights
            self._initialized.assign(True)

        tf.debugging.assert_equal(
            tf.shape(x)[-2],
            self._batch_size.value(),
            message="""
            This trajectory only supports batch sizes of {self._batch_size}}.
            If you wish to change the batch size you must get a new trajectory
            by calling the get_trajectory method of the trajectory sampler.
            """,
        )

        flat_x, unflatten = flatten_leading_dims(x)  # [N*B, d]
        flattened_feature_evaluations = self._feature_functions(flat_x)  # [N*B, m]
        feature_evaluations = unflatten(flattened_feature_evaluations)  # [N, B, m]
        return tf.reduce_sum(feature_evaluations * self._theta_sample, -1)  # [N, B]

    def resample(self) -> None:
        """
        Efficiently resample in-place without retracing.
        """
        self._theta_sample.assign(
            self._weight_distribution.sample(self._batch_size)
        )  # resample weights

    def update(self, weight_distribution: tfp.distributions.MultivariateNormalTriL) -> None:
        """
        Efficiently update the trajectory with a new weight distribution and resample its weights.

        :param weight_distribution: new distribution from which feature weights are to be sampled.
        """
        self._weight_distribution = weight_distribution  # update weight distribution.
        self._theta_sample.assign(
            self._weight_distribution.sample(self._batch_size)
        )  # resample weights

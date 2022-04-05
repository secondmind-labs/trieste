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
from typing import List, Tuple, Callable, Optional

try:
    from gpflux.layers.basis_functions.fourier_features import RandomFourierFeaturesCosine as RFF
except (ModuleNotFoundError, ImportError):
    # temporary support for gpflux 0.2.3
    from gpflux.layers.basis_functions import RandomFourierFeatures as RFF

import tensorflow as tf
from gpflow.inducing_variables import InducingPoints
from gpflow.covariances import Kuu, Kuf
from gpflow.kernels import Kernel
from gpflow.config import default_float, default_jitter
from gpflow.inducing_variables import InducingVariables
from gpflux.math import compute_A_inv_b
from gpflux.layers import GPLayer, LatentVariableLayer
from gpflux.models import DeepGP
from gpflux.sampling.sample import Sample

from ...types import TensorType
from ...utils import DEFAULTS
from ..interfaces import ReparametrizationSampler, TrajectoryFunction, TrajectorySampler, TrajectoryFunctionClass
from .interface import GPfluxPredictor


def sample_consistent_lv_layer(layer: LatentVariableLayer) -> Sample:
    r"""
    Returns a :class:`~gpflux.sampling.sample.Sample` object which allows for consistent sampling
    (i.e. function samples) from a given :class:`~gpflux.layers.LatentVariableLayer`.

    :param layer: The GPflux latent variable layer to obtain samples from.
    :return: The GPflux sampling object which can be called to obtain consistent samples.
    """

    class SampleLV(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            sample = layer.prior.sample()
            batch_shape = tf.shape(X)[:-1]
            sample_rank = tf.rank(sample)
            for _ in range(len(batch_shape)):
                sample = tf.expand_dims(sample, 0)
            sample = tf.tile(
                sample, tf.concat([batch_shape, tf.ones(sample_rank, dtype="int32")], -1)
            )
            return layer.compositor([X, sample])

    return SampleLV()


def sample_dgp(model: DeepGP) -> TrajectoryFunction:
    r"""
    Builds a :class:`TrajectoryFunction` that can be called for a :class:`~gpflux.models.DeepGP`,
    which will give consistent (i.e. function) samples from a deep GP model.

    :param model: The GPflux deep GP model to sample from.
    :return: The trajectory function that gives a consistent sample function from the model.
    """
    function_draws = []
    for layer in model.f_layers:
        if isinstance(layer, GPLayer):
            function_draws.append(layer.sample())
        elif isinstance(layer, LatentVariableLayer):
            function_draws.append(sample_consistent_lv_layer(layer))
        else:
            raise NotImplementedError(f"Sampling not implemented for {type(layer)}")

    class ChainedSample(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            for f in function_draws:
                X = f(X)
            return X

    return ChainedSample().__call__


class DeepGaussianProcessTrajectorySampler(TrajectorySampler[GPfluxPredictor]):
    r"""
    This sampler provides trajectory samples from a :class:`GPfluxPredictor`\ 's predictive
    distribution, for :class:`GPfluxPredictor`\s with an underlying
    :class:`~gpflux.models.DeepGP` model.
    """

    def __init__(self, model: GPfluxPredictor):
        """
        :param model: The model to sample from.
        :raise ValueError: If the model is not a :class:`GPfluxPredictor`, or its underlying
            ``model_gpflux`` is not a :class:`~gpflux.models.DeepGP`.
        """
        if not isinstance(model, GPfluxPredictor):
            raise ValueError(
                f"Model must be a gpflux.interface.GPfluxPredictor, received {type(model)}"
            )

        super().__init__(model)

        self._model_gpflux = model.model_gpflux

        if not isinstance(self._model_gpflux, DeepGP):
            raise ValueError(
                f"GPflux model must be a gpflux.models.DeepGP, received {type(self._model_gpflux)}"
            )

    def get_trajectory(self) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) by using the GPflux sampling
        functionality. These trajectories are differentiable with respect to the input, so can be
        used to e.g. find the minima of Thompson samples.

        :return: A trajectory function representing an approximate trajectory from the deep Gaussian
            process, taking an input of shape `[N, D]` and returning shape `[N, 1]`.
        """

        return sample_dgp(self._model_gpflux)


class DeepGaussianProcessReparamSampler(ReparametrizationSampler[GPfluxPredictor]):
    r"""
    This sampler employs the *reparameterization trick* to approximate samples from a
    :class:`GPfluxPredictor`\ 's predictive distribution, when the :class:`GPfluxPredictor` has
    an underlying :class:`~gpflux.models.DeepGP`.
    """

    def __init__(self, sample_size: int, model: GPfluxPredictor):
        """
        :param sample_size: The number of samples for each batch of points. Must be positive.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, if the
            model is not a :class:`GPfluxPredictor`, of if its underlying ``model_gpflux`` is not a
            :class:`~gpflux.models.DeepGP`.
        """
        if not isinstance(model, GPfluxPredictor):
            raise ValueError(
                f"Model must be a gpflux.interface.GPfluxPredictor, received {type(model)}"
            )

        super().__init__(sample_size, model)

        self._model_gpflux = model.model_gpflux

        if not isinstance(self._model_gpflux, DeepGP):
            raise ValueError(
                f"GPflux model must be a gpflux.models.DeepGP, received {type(self._model_gpflux)}"
            )

        # Each element of _eps_list is essentially a lazy constant. It is declared and assigned an
        # empty tensor here, and populated on the first call to sample
        self._eps_list = [
            tf.Variable(tf.ones([sample_size, 0], dtype=tf.float64), shape=[sample_size, None])
            for _ in range(len(self._model_gpflux.f_layers))
        ]

    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`DeepGaussianProcessReparamSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`DeepGaussianProcessReparamSampler` instances will produce different samples.

        :param at: Where to sample the predictive distribution, with shape `[N, D]`, for points
            of dimension `D`.
        :param jitter: The size of the jitter to use when stabilizing the Cholesky
            decomposition of the covariance matrix.
        :return: The samples, of shape `[S, N, L]`, where `S` is the `sample_size` and `L` is
            the number of latent model dimensions.
        :raise ValueError (or InvalidArgumentError): If ``at`` has an invalid shape or ``jitter``
            is negative.
        """
        tf.debugging.assert_equal(len(tf.shape(at)), 2)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        samples = tf.tile(tf.expand_dims(at, 0), [self._sample_size, 1, 1])
        for i, layer in enumerate(self._model_gpflux.f_layers):
            if isinstance(layer, LatentVariableLayer):
                if not self._initialized:
                    self._eps_list[i].assign(layer.prior.sample([tf.shape(samples)[:-1]]))
                samples = layer.compositor([samples, self._eps_list[i]])
                continue

            mean, var = layer.predict(samples, full_cov=False, full_output_cov=False)

            if not self._initialized:
                self._eps_list[i].assign(
                    tf.random.normal([self._sample_size, tf.shape(mean)[-1]], dtype=tf.float64)
                )

            samples = mean + tf.sqrt(var) * tf.cast(self._eps_list[i][:, None, :], var.dtype)

        if not self._initialized:
            self._initialized.assign(True)

        return samples


def _efficient_sample_matheron_rule(
    inducing_variable: InducingVariables,
    kernel: Kernel,
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

    prior_weights = tf.sqrt(kernel.feature_coefficients) * tf.random.normal(
        tf.shape(kernel.feature_coefficients), dtype=default_float()
    )  # [L, 1]

    M, P = tf.shape(q_mu)[0], tf.shape(q_mu)[1]  # num inducing, num output heads
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
    weight_space_prior_Z = phi_Z @ prior_weights  # [M, 1]
    diff = u_sample - weight_space_prior_Z  # [M, P] -- using implicit broadcasting
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
            weight_space_prior_X = phi_X @ prior_weights  # [N, 1]
            Knm = tf.linalg.matrix_transpose(Kuf(inducing_variable, kernel, X))  # [N, M]
            function_space_update_X = Knm @ v  # [N, P]

            tf.debugging.assert_equal(tf.shape(weight_space_prior_X), [N, 1])
            tf.debugging.assert_equal(tf.shape(function_space_update_X), [N, P])

            return weight_space_prior_X + function_space_update_X  # [N, P]

    return WilsonSample()


# class DeepGaussianProcessDecoupledFeatureFunctions:
#     def __init__(self, model: DeepGP, n_components: int):
#         self._fourier_features_list = []
#         self._canonical_features_list = []
#
#         for i, layer in enumerate(model.f_layers):
#             if isinstance(layer, LatentVariableLayer):
#                 raise ValueError("LatentVariableLayer is not currently supported with decoupled"
#                                  "trajectory sampling")
#
#             if isinstance(layer.inducing_variable, InducingPoints):
#                 inducing_variable = layer.inducing_variable
#             else:
#                 inducing_variable = layer.inducing_variable.inducing_variable
#             fourier_features_layer = RFF(layer.kernel, n_components, dtype=tf.float64)
#             dummy_X = inducing_variable.Z[0:1, :]
#             fourier_features_layer.__call__(dummy_X)
#             fourier_features_layer.b = tf.Variable(fourier_features_layer.b)
#             fourier_features_layer.W = tf.Variable(fourier_features_layer.W)
#             self._fourier_features_list.append(fourier_features_layer)
#
#             canonical_features_function = lambda x: tf.linalg.matrix_transpose(
#                 layer.kernel.K(inducing_variable.Z, x)
#             )
#             self._canonical_features_list.append(canonical_features_function)
#
#     def resample(self) -> None:
#         """
#         Resample weights and biases
#         """
#
#         for features in self._fourier_features_list:
#             if not hasattr(features, "_bias_init"):
#                 features.b.assign(features._sample_bias(tf.shape(features.b), dtype=features._dtype))
#                 features.W.assign(features._sample_weights(tf.shape(features.W), dtype=features._dtype))
#             else:
#                 features.b.assign(features._bias_init(tf.shape(features.b), dtype=features._dtype))
#                 features.W.assign(features._weight_init(tf.shape(features.W), dtype=features._dtype))
#
#     def __call__(self) -> Tuple[List, List]:
#         return self._fourier_features_list, self._canonical_features_list
#
#
# class DeepGaussianProcessDecoupledTrajectorySampler(TrajectorySampler[GPfluxPredictor]):
#     """
#     This sampler provides approximate trajectory samples using decoupled sampling (i.e. Matheron's
#     rule).
#     """
#     def __init__(
#         self,
#         model: GPfluxPredictor,
#         num_features: int = 1000,
#     ):
#         if not isinstance(model, GPfluxPredictor):
#             raise ValueError(
#                 f"Model must be a gpflux.interface.GPfluxPredictor, received {type(model)}"
#             )
#         super().__init__(model)
#         tf.debugging.assert_positive(num_features)
#         self._num_features = num_features
#         self._model_gpflux = model.model_gpflux
#         self._feature_functions = DeepGaussianProcessDecoupledFeatureFunctions(self._model_gpflux,
#                                                                                num_features)
#
#     def __repr__(self) -> str:
#         """"""
#         return f"""{self.__class__.__name__}(
#         {self._model!r},
#         {self._feature_functions!r})
#         """
#
#     def get_trajectory(self) -> TrajectoryFunction:
#         weight_sampler = self._prepare_weight_sampler()
#
#         return feature_decomposition_trajectory(
#             feature_functions=self._feature_functions(),
#             weight_sampler=weight_sampler
#         )
#
#     def _prepare_weight_sampler(self) -> Callable[[int], TensorType]:
#
#         def weight_sampler(batch_size: int) -> Tuple[TensorType, TensorType]:
#
#
#
# class feature_decomposition_trajectory(TrajectoryFunctionClass):
#     r"""
#     An approximate sample from a Gaussian processes' posterior samples represented as a
#     finite weighted sum of features.
#
#     A trajectory is given by
#
#     .. math:: \hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i
#
#     where :math:`\phi_i` are m feature functions and :math:`\theta_i` are
#     feature weights sampled from a posterior distribution.
#
#     The number of trajectories (i.e. batch size) is determined from the first call of the
#     trajectory. In order to change the batch size, a new :class:`TrajectoryFunction` must be built.
#     """
#
#     def __init__(
#         self,
#         feature_functions: Callable[[TensorType], TensorType],
#         weight_sampler: Callable[[int], TensorType],
#     ):
#         """
#         :param feature_functions: Set of feature function.
#         :param weight_sampler: New sampler that generates feature weight samples.
#         """
#         self._feature_functions = feature_functions
#         self._weight_sampler = weight_sampler
#         self._initialized = tf.Variable(False)
#
#         self._weights_sample = tf.Variable(  # dummy init to be updated before trajectory evaluation
#             tf.ones([0, 0], dtype=tf.float64), shape=[None, None]
#         )
#
#         self._batch_size = tf.Variable(
#             0, dtype=tf.int32
#         )  # dummy init to be updated before trajectory evaluation
#
#     @tf.function
#     def __call__(self, x: TensorType) -> TensorType:  # [N, B, d] -> [N, B]
#         """Call trajectory function."""
#
#         if not self._initialized:  # work out desired batch size from input
#             self._batch_size.assign(tf.shape(x)[-2])  # B
#             self.resample()  # sample B feature weights
#             self._initialized.assign(True)
#
#         tf.debugging.assert_equal(
#             tf.shape(x)[-2],
#             self._batch_size.value(),
#             message="""
#             This trajectory only supports batch sizes of {self._batch_size}}.
#             If you wish to change the batch size you must get a new trajectory
#             by calling the get_trajectory method of the trajectory sampler.
#             """,
#         )
#
#         flat_x, unflatten = flatten_leading_dims(x)  # [N*B, d]
#         flattened_feature_evaluations = self._feature_functions(flat_x)  # [N*B, m]
#         feature_evaluations = unflatten(flattened_feature_evaluations)  # [N, B, m]
#
#         return tf.reduce_sum(feature_evaluations * self._weights_sample, -1)  # [N, B]
#
#     def resample(self) -> None:
#         """
#         Efficiently resample in-place without retracing.
#         """
#         self._weights_sample.assign(  # [B, m]
#             self._weight_sampler(self._batch_size)
#         )  # resample weights
#
#     def update(self, weight_sampler: Callable[[int], TensorType]) -> None:
#         """
#         Efficiently update the trajectory with a new weight distribution and resample its weights.
#
#         :param weight_sampler: New sampler that generates feature weight samples.
#         """
#         self._weight_sampler = weight_sampler  # update weight sampler
#         self.resample()  # resample weights

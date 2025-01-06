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

from abc import ABC
from typing import Callable, cast

import gpflow.kernels
import tensorflow as tf
from gpflow.inducing_variables import InducingPoints
from gpflux.layers import GPLayer, LatentVariableLayer
from gpflux.layers.basis_functions.fourier_features import RandomFourierFeaturesCosine
from gpflux.math import compute_A_inv_b
from gpflux.models import DeepGP

from ...types import TensorType
from ...utils import DEFAULTS, flatten_leading_dims
from ..interfaces import (
    ReparametrizationSampler,
    TrajectoryFunction,
    TrajectoryFunctionClass,
    TrajectorySampler,
)
from .interface import GPfluxPredictor


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

        if not isinstance(self._model_gpflux, DeepGP):
            raise ValueError(
                f"GPflux model must be a gpflux.models.DeepGP, received {type(self._model_gpflux)}"
            )

        # Each element of _eps_list is essentially a lazy constant. It is declared and assigned an
        # empty tensor here, and populated on the first call to sample
        self._eps_list = [
            tf.Variable(
                tf.ones([sample_size, 0], dtype=self._model_gpflux.targets.dtype),
                shape=[sample_size, None],
            )
            for _ in range(len(self._model_gpflux.f_layers))
        ]
        self._encode = lambda x: model.encode(x)

    @property
    def _model_gpflux(self) -> tf.Module:
        return self._model.model_gpflux

    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`DeepGaussianProcessReparamSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`DeepGaussianProcessReparamSampler` instances will produce different samples.

        :param at: Where to sample the predictive distribution, with shape `[..., 1, D]`, for points
            of dimension `D`.
        :param jitter: The size of the jitter to use when stabilizing the Cholesky
            decomposition of the covariance matrix.
        :return: The samples, of shape `[..., S, 1, L]`, where `S` is the `sample_size` and `L` is
            the number of latent model dimensions.
        :raise ValueError (or InvalidArgumentError): If ``at`` has an invalid shape or ``jitter``
            is negative.
        """
        tf.debugging.assert_shapes([(at, [..., 1, None])])
        tf.debugging.assert_greater_equal(jitter, 0.0)

        samples = tf.repeat(
            self._encode(at[..., None, :, :]), self._sample_size, axis=-3
        )  # [..., S, 1, D]
        for i, layer in enumerate(self._model_gpflux.f_layers):
            if isinstance(layer, LatentVariableLayer):
                if not self._initialized:
                    self._eps_list[i].assign(layer.prior.sample([tf.shape(samples)[:-1]]))
                samples = layer.compositor([samples, self._eps_list[i]])
                continue

            mean, var = layer.predict(samples, full_cov=False, full_output_cov=False)
            var = var + jitter

            if not self._initialized:
                self._eps_list[i].assign(
                    tf.random.normal([self._sample_size, tf.shape(mean)[-1]], dtype=tf.float64)
                )  # [S, L]

            samples = mean + tf.sqrt(var) * self._eps_list[i][:, None, :]

        if not self._initialized:
            self._initialized.assign(True)

        return samples  # [..., S, 1, L]


class DeepGaussianProcessDecoupledTrajectorySampler(TrajectorySampler[GPfluxPredictor]):
    r"""
    This sampler employs decoupled sampling (see :cite:`wilson2020efficiently`) to build functions
    that approximate a trajectory sampled from an underlying deep Gaussian process model. In
    particular, this sampler provides trajectory functions for :class:`GPfluxPredictor`\s with
    underlying :class:`~gpflux.models.DeepGP` models by using a feature decomposition using both
    random Fourier features and canonical features centered at inducing point locations. This allows
    for cheap approximate trajectory samples, as opposed to exact trajectory sampling, which scales
    cubically in the number of query points.
    """

    def __init__(
        self,
        model: GPfluxPredictor,
        num_features: int = 1000,
    ):
        """
        :param model: The model to sample from.
        :param num_features: The number of random Fourier features to use.
        :raise ValueError (or InvalidArgumentError): If the model is not a :class:`GPfluxPredictor`,
            or its underlying ``model_gpflux`` is not a :class:`~gpflux.models.DeepGP`, or
            ``num_features`` is not positive.
        """
        if not isinstance(model, GPfluxPredictor):
            raise ValueError(
                f"Model must be a gpflux.interface.GPfluxPredictor, received {type(model)}"
            )
        if not isinstance(model.model_gpflux, DeepGP):
            raise ValueError(
                f"GPflux model must be a gpflux.models.DeepGP, received {type(model.model_gpflux)}"
            )

        super().__init__(model)
        tf.debugging.assert_positive(num_features)
        self._num_features = num_features

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}(
        {self._model!r},
        {self._num_features!r})
        """

    def get_trajectory(self) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) from the deep GP model.

        :return: A trajectory function representing an approximate trajectory from the deep GP,
            taking an input of shape `[N, B, D]` and returning shape `[N, B, L]`.
        """

        return dgp_feature_decomposition_trajectory(self._model, self._num_features)

    def update_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        Efficiently update a :const:`TrajectoryFunction` to reflect an update in its underlying
        :class:`ProbabilisticModel` and resample accordingly.

        :param trajectory: The trajectory function to be updated and resampled.
        :return: The updated and resampled trajectory function.
        :raise InvalidArgumentError: If ``trajectory`` is not a
            :class:`dgp_feature_decomposition_trajectory`
        """

        tf.debugging.Assert(
            isinstance(trajectory, dgp_feature_decomposition_trajectory), [tf.constant([])]
        )

        cast(dgp_feature_decomposition_trajectory, trajectory).update()
        return trajectory

    def resample_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        Efficiently resample a :const:`TrajectoryFunction` in-place to avoid function retracing
        with every new sample.

        :param trajectory: The trajectory function to be resampled.
        :return: The new resampled trajectory function.
        :raise InvalidArgumentError: If ``trajectory`` is not a
            :class:`dgp_feature_decomposition_trajectory`
        """
        tf.debugging.Assert(
            isinstance(trajectory, dgp_feature_decomposition_trajectory), [tf.constant([])]
        )
        cast(dgp_feature_decomposition_trajectory, trajectory).resample()
        return trajectory


class DeepGaussianProcessDecoupledLayer(ABC):
    """
    Layer that samples an approximate decoupled trajectory for a GPflux
    :class:`~gpflux.layers.GPLayer` using Matheron's rule (:cite:`wilson2020efficiently`). Note
    that the only multi-output kernel that is supported is a
    :class:`~gpflow.kernels.SharedIndependent` kernel.
    """

    def __init__(
        self,
        model: GPfluxPredictor,
        layer_number: int,
        num_features: int = 1000,
    ):
        """
        :param model: The model to sample from.
        :param layer_number: The index of the layer that we wish to sample from.
        :param num_features: The number of features to use in the random feature approximation.
        :raise ValueError (or InvalidArgumentError): If the layer is not a
            :class:`~gpflux.layers.GPLayer`, the layer's kernel is not supported, or if
            ``num_features`` is not positive.
        """
        self._model = model
        self._layer_number = layer_number
        layer = self._layer

        if not isinstance(layer, GPLayer):
            raise ValueError(
                f"Layers other than gpflux.layers.GPLayer are not currently supported, received"
                f"{type(layer)}"
            )

        if isinstance(
            layer.inducing_variable, gpflow.inducing_variables.SeparateIndependentInducingVariables
        ):
            raise ValueError(
                "SeparateIndependentInducingVariables are not currently supported for decoupled "
                "sampling."
            )

        tf.debugging.assert_positive(num_features)
        self._num_features = num_features

        self._kernel = layer.kernel

        self._feature_functions = ResampleableDecoupledDeepGaussianProcessFeatureFunctions(
            layer, num_features
        )

        self._weight_sampler = self._prepare_weight_sampler()

        self._initialized = tf.Variable(False)

        self._weights_sample = tf.Variable(
            tf.ones([0, 0, 0], dtype=tf.float64), shape=[None, None, None]
        )

        self._batch_size = tf.Variable(0, dtype=tf.int32)

    @property
    def _layer(self) -> GPLayer:
        return self._model.model_gpflux.f_layers[self._layer_number]

    def __call__(self, x: TensorType) -> TensorType:  # [N, B, D] -> [N, B, P]
        """
        Evaluate trajectory function for layer at input.

        :param x: Input location with shape `[N, B, D]`, where `N` is the number of points, `B` is
            the batch dimension, and `D` is the input dimensionality.
        :return: Trajectory for the layer evaluated at the input, with shape `[N, B, P]`, where `P`
            is the number of latent GPs in the layer.
        :raise InvalidArgumentError: If the provided batch size does not match with the layer's
            batch size.
        """
        if not self._initialized:
            self._batch_size.assign(tf.shape(x)[-2])
            self.resample()
            self._initialized.assign(True)

        tf.debugging.assert_equal(
            tf.shape(x)[-2],
            self._batch_size.value(),
            message=f"""
            This trajectory only supports batch sizes of {self._batch_size}.
            If you wish to change the batch size you must get a new trajectory
            by calling the get_trajectory method of the trajectory sampler.
            """,
        )

        flat_x, unflatten = flatten_leading_dims(x)
        flattened_feature_evaluations = self._feature_functions(
            flat_x
        )  # [P, N, L + M] or [N, L + M]
        if self._feature_functions.is_multioutput:
            flattened_feature_evaluations = tf.transpose(
                flattened_feature_evaluations, perm=[1, 2, 0]
            )
            feature_evaluations = unflatten(flattened_feature_evaluations)  # [N, B, L + M, P]
        else:
            feature_evaluations = unflatten(flattened_feature_evaluations)[
                ..., None
            ]  # [N, B, L + M, 1]

        return tf.reduce_sum(
            feature_evaluations * self._weights_sample, -2
        ) + self._layer.mean_function(
            x
        )  # [N, B, P]

    def resample(self) -> None:
        """
        Efficiently resample in-place without retracing.
        """
        self._weights_sample.assign(self._weight_sampler(self._batch_size))

    def update(self) -> None:
        """
        Efficiently update the trajectory with a new weight distribution and resample its weights.
        """
        self._feature_functions.resample()
        self._weight_sampler = self._prepare_weight_sampler()
        self.resample()

    def _prepare_weight_sampler(self) -> Callable[[int], TensorType]:  # [B] -> [B, L+M, P]
        """
        Prepare the sampler function that provides samples of the feature weights for both the
        RFF and canonical feature functions, i.e. we return a function that takes in a batch size
        `B` and returns `B` samples for the weights of each of the `L` RFF features and `M`
        canonical features for `P` outputs.
        """

        if isinstance(self._layer.inducing_variable, InducingPoints):
            inducing_points = self._layer.inducing_variable.Z  # [M, D]
        else:
            inducing_points = self._layer.inducing_variable.inducing_variable.Z  # [M, D]

        q_mu = self._layer.q_mu  # [M, P]
        q_sqrt = self._layer.q_sqrt  # [P, M, M]
        if self._feature_functions.is_multioutput:
            Kmm = self._kernel.K(
                inducing_points, inducing_points, full_output_cov=False
            )  # [P, M, M]
        else:
            Kmm = self._kernel.K(inducing_points, inducing_points)  # [M, M]
        Kmm += tf.eye(tf.shape(inducing_points)[0], dtype=Kmm.dtype) * DEFAULTS.JITTER
        whiten = self._layer.whiten
        M, P = tf.shape(q_mu)[0], tf.shape(q_mu)[1]

        tf.debugging.assert_shapes(
            [
                (inducing_points, ["M", "D"]),
                (q_mu, ["M", "P"]),
                (q_sqrt, ["P", "M", "M"]),
            ]
        )

        def weight_sampler(batch_size: int) -> TensorType:
            prior_weights = tf.random.normal(
                [batch_size, P, self._num_features, 1], dtype=tf.float64
            )  # [B, P, L, 1]

            u_noise_sample = tf.matmul(
                q_sqrt,  # [P, M, M]
                tf.random.normal([batch_size, P, M, 1], dtype=tf.float64),  # [B, P, M, 1]
            )  # [B, P, M, 1]
            u_sample = tf.linalg.matrix_transpose(q_mu)[..., None] + u_noise_sample  # [B, P, M, 1]

            if whiten:
                Luu = tf.linalg.cholesky(Kmm)  # [M, M] or [P, M, M]
                u_sample = tf.matmul(Luu, u_sample)  # [B, P, M, 1]

            phi_Z = self._feature_functions(inducing_points)[
                ..., : self._num_features
            ]  # [M, L] or [P, M, L]
            weight_space_prior_Z = phi_Z @ prior_weights  # [B, P, M, 1]

            diff = u_sample - weight_space_prior_Z  # [B, P, M, 1]
            v = compute_A_inv_b(Kmm, diff)  # [B, P, M, 1]

            return tf.transpose(
                tf.concat([prior_weights, v], axis=2)[..., 0], perm=[0, 2, 1]
            )  # [B, L + M, P]

        return weight_sampler


class ResampleableDecoupledDeepGaussianProcessFeatureFunctions(RandomFourierFeaturesCosine):
    """
    A wrapper around GPflux's random Fourier feature function that allows for efficient in-place
    updating when generating new decompositions. In addition to providing Fourier features,
    this class concatenates a layer's Fourier feature expansion with evaluations of the canonical
    basis functions.
    """

    def __init__(self, layer: GPLayer, n_components: int):
        """
        :param layer: The layer that will be approximated by the feature functions.
        :param n_components: The number of features.
        :raise ValueError: If the layer is not a :class:`~gpflux.layers.GPLayer`.
        """
        if not isinstance(layer, GPLayer):
            raise ValueError(
                f"ResampleableDecoupledDeepGaussianProcessFeatureFunctions currently only work with"
                f"gpflux.layers.GPLayer layers, received {type(layer)} instead"
            )

        self._kernel = layer.kernel

        self._n_components = n_components
        super().__init__(self._kernel, self._n_components, dtype=tf.float64)

        if isinstance(layer.inducing_variable, InducingPoints):
            inducing_points = layer.inducing_variable.Z
        else:
            inducing_points = layer.inducing_variable.inducing_variable.Z

        if self.is_multioutput:
            self._canonical_feature_functions = lambda x: tf.linalg.matrix_transpose(
                self._kernel.K(inducing_points, x, full_output_cov=False)
            )
        else:
            self._canonical_feature_functions = lambda x: tf.linalg.matrix_transpose(
                self._kernel.K(inducing_points, x)
            )

        dummy_X = inducing_points[0:1, :]

        self.__call__(dummy_X)
        self.b: TensorType = tf.Variable(self.b)
        self.W: TensorType = tf.Variable(self.W)

    def resample(self) -> None:
        """
        Resample weights and biases.
        """
        if not hasattr(self, "_bias_init"):
            self.b.assign(self._sample_bias(tf.shape(self.b), dtype=self._dtype))
            self.W.assign(self._sample_weights(tf.shape(self.W), dtype=self._dtype))
        else:
            self.b.assign(self._bias_init(tf.shape(self.b), dtype=self._dtype))
            self.W.assign(self._weights_init(tf.shape(self.W), dtype=self._dtype))

    def __call__(self, x: TensorType) -> TensorType:  # [N, D] -> [N, L + M] or [P, N, L + M]
        """
        Evaluate and combine prior basis functions and canonical basic functions at the input.
        """
        fourier_feature_eval = super().__call__(x)  # [N, L] or [P, N, L]
        canonical_feature_eval = self._canonical_feature_functions(x)  # [P, N, M] or [N, M]
        return tf.concat([fourier_feature_eval, canonical_feature_eval], axis=-1)  # [P, N, L + M]


class dgp_feature_decomposition_trajectory(TrajectoryFunctionClass):
    r"""
    An approximate sample from a deep Gaussian process's posterior, where the samples are
    represented as a finite weighted sum of features. This class essentially takes a list of
    :class:`DeepGaussianProcessDecoupledLayer`\s and iterates through them to sample, update and
    resample.
    """

    def __init__(self, model: GPfluxPredictor, num_features: int):
        """
        :param model: The model to sample from.
        :param num_features: The number of random Fourier features to use.
        """
        self._sampling_layers = [
            DeepGaussianProcessDecoupledLayer(model, i, num_features)
            for i in range(len(model.model_gpflux.f_layers))
        ]

        self._encode = lambda x: model.encode(x)

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        """
        Call trajectory function by looping through layers.

        :param x: Input location with shape `[N, B, D]`, where `N` is the number of points, `B` is
            the batch dimension, and `D` is the input dimensionality.
        :return: Trajectory samples with shape `[N, B, L]`, where `L` is the number of outputs.
        """
        x = self._encode(x)
        for layer in self._sampling_layers:
            x = layer(x)
        return x

    def update(self) -> None:
        """Update the layers with new features and weights."""
        for layer in self._sampling_layers:
            layer.update()

    def resample(self) -> None:
        """Resample the layer weights."""
        for layer in self._sampling_layers:
            layer.resample()

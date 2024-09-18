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

from typing import Optional

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
from gpflow.conditionals.util import sample_mvn
from gpflow.keras import tf_keras
from gpflux.helpers import construct_basic_inducing_variables, construct_basic_kernel
from gpflux.layers import GPLayer
from gpflux.models import DeepGP

from tests.util.misc import random_seed
from trieste.data import Dataset
from trieste.models.gpflux import GPfluxPredictor
from trieste.space import CategoricalSearchSpace, EncoderFunction, one_hot_encoder
from trieste.types import TensorType


class _QuadraticPredictor(GPfluxPredictor):
    def __init__(
        self,
        optimizer: tf_keras.optimizers.Optimizer | None = None,
        likelihood: gpflow.likelihoods.Likelihood = gpflow.likelihoods.Gaussian(0.01),
        encoder: EncoderFunction | None = None,
    ):
        super().__init__(optimizer=optimizer, encoder=encoder)

        if optimizer is None:
            self._optimizer = tf_keras.optimizers.Adam()
        else:
            self._optimizer = optimizer
        self._model_gpflux = _QuadraticGPModel(likelihood=likelihood)

        self._model_keras = self._model_gpflux.as_training_model()

    @property
    def model_gpflux(self) -> DeepGP:
        return self._model_gpflux

    @property
    def model_keras(self) -> tf_keras.Model:
        return self._model_keras

    @property
    def optimizer(self) -> tf_keras.optimizers.Optimizer:
        return self._optimizer

    def sample_encoded(self, query_points: TensorType, num_samples: int) -> TensorType:
        # Taken from GPflow implementation of `GPModel.predict_f_samples` in gpflow.models.model
        mean, cov = self._model_gpflux.predict_f(query_points, full_cov=True)
        mean_for_sample = tf.linalg.adjoint(mean)
        samples = sample_mvn(mean_for_sample, cov, True, num_samples=num_samples)
        samples = tf.linalg.adjoint(samples)
        return samples

    def update(self, dataset: Dataset) -> None:
        return

    def log(self, dataset: Optional[Dataset] = None) -> None:
        return


class _QuadraticGPModel(DeepGP):
    def __init__(
        self, likelihood: gpflow.likelihoods.Likelihood = gpflow.likelihoods.Gaussian(0.01)
    ) -> None:
        kernel = construct_basic_kernel(
            gpflow.kernels.SquaredExponential(), output_dim=1, share_hyperparams=True
        )
        inducing_var = construct_basic_inducing_variables(
            num_inducing=5,
            input_dim=1,
            share_variables=True,
            z_init=tf.random.normal([5, 1], dtype=gpflow.default_float()),
        )

        gp_layer = GPLayer(kernel, inducing_var, 10)

        super().__init__(
            [gp_layer],  # not actually used
            likelihood,
        )

    def predict_f(
        self, Xnew: tf.Tensor, full_cov: bool = False, full_output_cov: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        assert not full_output_cov, "Test utility not implemented for full output covariance"
        mean = tf.reduce_sum(Xnew**2, axis=1, keepdims=True)
        *leading, x_samples, y_dims = mean.shape
        var_shape = [*leading, y_dims, x_samples, x_samples] if full_cov else mean.shape
        return mean, tf.ones(var_shape, dtype=mean.dtype)


def test_gpflux_predictor_predict() -> None:
    model = _QuadraticPredictor()
    mean, variance = model.predict(tf.constant([[2.5]], gpflow.default_float()))
    assert mean.shape == [1, 1]
    assert variance.shape == [1, 1]
    npt.assert_allclose(mean, [[6.25]], rtol=0.01)
    npt.assert_allclose(variance, [[1.0]], rtol=0.01)


@random_seed
def test_gpflux_predictor_sample() -> None:
    model = _QuadraticPredictor()
    num_samples = 20_000
    samples = model.sample(tf.constant([[2.5]], gpflow.default_float()), num_samples)

    assert samples.shape == [num_samples, 1, 1]

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2)

    linear_error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, [[6.25]], rtol=linear_error)
    npt.assert_allclose(sample_variance, 1.0, rtol=2 * linear_error)


def test_gpflux_predictor_sample_0_samples() -> None:
    samples = _QuadraticPredictor().sample(tf.constant([[50.0]], gpflow.default_float()), 0)
    assert samples.shape == (0, 1, 1)


def test_gpflux_predictor_get_observation_noise() -> None:
    noise_var = 0.1
    likelihood = gpflow.likelihoods.Gaussian(noise_var)
    model = _QuadraticPredictor(likelihood=likelihood)

    npt.assert_allclose(model.get_observation_noise(), noise_var)


def test_gpflux_predictor_get_observation_noise_raises_for_non_gaussian_likelihood() -> None:
    likelihood = gpflow.likelihoods.StudentT()
    model = _QuadraticPredictor(likelihood=likelihood)

    with pytest.raises(NotImplementedError):
        model.get_observation_noise()


def test_gpflux_categorical_predict() -> None:
    search_space = CategoricalSearchSpace(["Red", "Green", "Blue"])
    query_points = search_space.sample(10)
    model = _QuadraticPredictor(encoder=one_hot_encoder(search_space))
    mean, variance = model.predict(query_points)
    assert mean.shape == [10, 1]
    assert variance.shape == [10, 1]
    npt.assert_allclose(mean, [[1.0]] * 10, rtol=0.01)
    npt.assert_allclose(variance, [[1.0]] * 10, rtol=0.01)

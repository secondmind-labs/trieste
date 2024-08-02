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

from typing import Any, Optional

import gpflow
import numpy.testing as npt
import tensorflow as tf
from gpflow.models import GPModel

from tests.util.misc import random_seed
from trieste.data import Dataset
from trieste.models.gpflow import BatchReparametrizationSampler, GPflowPredictor
from trieste.space import CategoricalSearchSpace, one_hot_encoder


class _QuadraticPredictor(GPflowPredictor):
    @property
    def model(self) -> GPModel:
        return _QuadraticGPModel()

    def optimize_encoded(self, dataset: Dataset) -> None:
        self.optimizer.optimize(self.model, dataset)

    def update_encoded(self, dataset: Dataset) -> None:
        return

    def log(self, dataset: Optional[Dataset] = None) -> None:
        return


class _QuadraticGPModel(GPModel):
    def __init__(self) -> None:
        super().__init__(
            gpflow.kernels.Polynomial(2),  # not actually used
            gpflow.likelihoods.Gaussian(),
            num_latent_gps=1,
        )

    def predict_f(
        self, Xnew: tf.Tensor, full_cov: bool = False, full_output_cov: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        assert not full_output_cov, "Test utility not implemented for full output covariance"
        mean = tf.reduce_sum(Xnew**2, axis=1, keepdims=True)
        *leading, x_samples, y_dims = mean.shape
        var_shape = [*leading, y_dims, x_samples, x_samples] if full_cov else mean.shape
        return mean, tf.ones(var_shape, dtype=mean.dtype)

    def maximum_log_likelihood_objective(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        raise NotImplementedError


def test_gpflow_predictor_predict() -> None:
    model = _QuadraticPredictor()
    mean, variance = model.predict(tf.constant([[2.5]], gpflow.default_float()))
    assert mean.shape == [1, 1]
    assert variance.shape == [1, 1]
    npt.assert_allclose(mean, [[6.25]], rtol=0.01)
    npt.assert_allclose(variance, [[1.0]], rtol=0.01)


@random_seed
def test_gpflow_predictor_sample() -> None:
    model = _QuadraticPredictor()
    num_samples = 20_000
    samples = model.sample(tf.constant([[2.5]], gpflow.default_float()), num_samples)

    assert samples.shape == [num_samples, 1, 1]

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2)

    linear_error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, [[6.25]], rtol=linear_error)
    npt.assert_allclose(sample_variance, 1.0, rtol=2 * linear_error)


def test_gpflow_predictor_sample_0_samples() -> None:
    samples = _QuadraticPredictor().sample(tf.constant([[50.0]], gpflow.default_float()), 0)
    assert samples.shape == (0, 1, 1)


def test_gpflow_reparam_sampler_returns_a_param_sampler() -> None:
    sampler = _QuadraticPredictor().reparam_sampler(10)
    assert isinstance(sampler, BatchReparametrizationSampler)
    assert sampler._sample_size == 10


def test_gpflow_reparam_sampler_returns_reparam_sampler_with_correct_samples() -> None:
    num_samples = 20_000
    sampler = _QuadraticPredictor().reparam_sampler(num_samples)

    samples = sampler.sample(tf.constant([[2.5]], gpflow.default_float()))

    assert samples.shape == [num_samples, 1, 1]

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2)

    linear_error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, [[6.25]], rtol=linear_error)
    npt.assert_allclose(sample_variance, 1.0, rtol=2 * linear_error)


def test_gpflow_categorical_predict() -> None:
    search_space = CategoricalSearchSpace(["Red", "Green", "Blue"])
    query_points = search_space.sample(10)
    model = _QuadraticPredictor(encoder=one_hot_encoder(search_space))
    mean, variance = model.predict(query_points)
    assert mean.shape == [10, 1]
    assert variance.shape == [10, 1]
    npt.assert_allclose(mean, [[1.0]] * 10, rtol=0.01)
    npt.assert_allclose(variance, [[1.0]] * 10, rtol=0.01)

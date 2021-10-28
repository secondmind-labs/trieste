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
from __future__ import annotations

from typing import Callable

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf

from gpflux.models import DeepGP
from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, ShapeLike, random_seed
from tests.util.models.gpflow.models import QuadraticMeanAndRBFKernel
from tests.util.models.gpflux.models import two_layer_dgp_model
from trieste.acquisition.models import DeepGaussianProcessSampler
from trieste.data import Dataset
from trieste.models.gpflux import DeepGaussianProcess
from trieste.types import TensorType


@pytest.mark.parametrize("sample_size", [0, -2])
def test_deep_gaussian_process_sampler_raises_for_invalid_sample_size(
    sample_size: int, keras_float: None
) -> None:
    x = tf.constant([[0.0]], dtype=gpflow.default_float())
    dgp = two_layer_dgp_model(x)
    model = DeepGaussianProcess(dgp)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        DeepGaussianProcessSampler(sample_size, model)


def test_deep_gaussian_process_sampler_raises_for_invalid_model() -> None:
    with pytest.raises(ValueError, match="Model must be .*"):
        DeepGaussianProcessSampler(10, QuadraticMeanAndRBFKernel())  # type: ignore


@pytest.mark.parametrize("shape", [[], [1], [2], [2, 3, 4]])
def test_deep_gaussian_process_sampler_sample_raises_for_invalid_at_shape(
    shape: ShapeLike,
    keras_float: None,
) -> None:
    x = tf.constant([[0.0]], dtype=gpflow.default_float())
    dgp = two_layer_dgp_model(x)
    model = DeepGaussianProcess(dgp)
    sampler = DeepGaussianProcessSampler(1, model)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.zeros(shape))


@random_seed
def test_deep_gaussian_process_sampler_samples_approximate_expected_distribution(
    two_layer_model: Callable[[TensorType], DeepGP],
    keras_float: None,
) -> None:
    sample_size = 1000
    x = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    y = tf.random.normal([100, 1], dtype=tf.float64)
    dataset = Dataset(x, y)

    dgp = two_layer_model(x)
    model = DeepGaussianProcess(dgp)
    model.optimize(dataset)

    samples = DeepGaussianProcessSampler(sample_size, model).sample(x)  # [S, N, L]

    assert samples.shape == [sample_size, len(x), 1]

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2)

    num_samples = 50
    means = []
    vars = []
    for _ in range(num_samples):
        Fmean_sample, Fvar_sample = model.predict(x)
        means.append(Fmean_sample)
        vars.append(Fvar_sample)
    ref_mean = tf.reduce_mean(tf.stack(means), axis=0)
    ref_variance = tf.reduce_mean(tf.stack(vars) + tf.stack(means) ** 2, axis=0) - ref_mean ** 2

    error = 1 / tf.sqrt(tf.cast(num_samples, tf.float32))
    npt.assert_allclose(sample_mean, ref_mean, atol=2 * error)
    npt.assert_allclose(sample_variance, ref_variance, atol=4 * error)


@random_seed
def test_deep_gaussian_process_sampler_sample_is_continuous(
    two_layer_model: Callable[[TensorType], DeepGP],
    keras_float: None,
) -> None:
    x = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    y = tf.random.normal([100, 1], dtype=tf.float64)
    dataset = Dataset(x, y)

    dgp = two_layer_model(x)
    model = DeepGaussianProcess(dgp)
    model.optimize(dataset)

    sampler = DeepGaussianProcessSampler(100, model)
    xs = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_array_less(tf.abs(sampler.sample(xs + 1e-20) - sampler.sample(xs)), 1e-20)


def test_deep_gaussian_process_sampler_sample_is_repeatable(
    two_layer_model: Callable[[TensorType], DeepGP],
    keras_float: None,
) -> None:
    x = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    y = tf.random.normal([100, 1], dtype=tf.float64)
    dataset = Dataset(x, y)

    dgp = two_layer_model(x)
    model = DeepGaussianProcess(dgp)
    model.optimize(dataset)

    sampler = DeepGaussianProcessSampler(100, model)
    xs = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))


@random_seed
def test_deep_gaussian_process_sampler_samples_are_distinct_for_new_instances(
    two_layer_model: Callable[[TensorType], DeepGP],
    keras_float: None,
) -> None:
    x = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    y = tf.random.normal([100, 1], dtype=tf.float64)
    dataset = Dataset(x, y)

    dgp = two_layer_model(x)
    model = DeepGaussianProcess(dgp)
    model.optimize(dataset)

    sampler1 = DeepGaussianProcessSampler(100, model)
    sampler2 = DeepGaussianProcessSampler(100, model)

    xs = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_array_less(1e-9, tf.abs(sampler2.sample(xs) - sampler1.sample(xs)))

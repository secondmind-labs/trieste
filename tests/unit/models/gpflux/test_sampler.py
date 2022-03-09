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
In this module, we test the *behaviour* of Trieste models against reference GPflux models (thus
implicitly assuming the latter are correct).
*NOTE:* Where GPflux models are used as the underlying model in an Trieste model, we should
*not* test that the underlying model is used in any particular way. To do so would break
encapsulation. For example, we should *not* test that methods on the GPflux models are called
(except in the rare case that such behaviour is an explicitly documented behaviour of the
Trieste model).
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy.testing as npt
import pytest
import tensorflow as tf
from gpflux.models import DeepGP

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, ShapeLike, mk_dataset, quadratic, random_seed
from tests.util.models.gpflow.models import QuadraticMeanAndRBFKernel
from tests.util.models.gpflux.models import two_layer_trieste_dgp
from trieste.data import Dataset
from trieste.models.gpflux import DeepGaussianProcess
from trieste.models.gpflux.sampler import DeepGaussianProcessReparamSampler
from trieste.space import Box
from trieste.types import TensorType


@pytest.mark.parametrize("sample_size", [0, -2])
def test_deep_gaussian_process_sampler_raises_for_invalid_sample_size(
    sample_size: int, keras_float: None
) -> None:
    search_space = Box([0.0], [1.0]) ** 4
    x = search_space.sample(10)
    data = mk_dataset(x, quadratic(x))

    dgp = two_layer_trieste_dgp(data, search_space)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        DeepGaussianProcessReparamSampler(sample_size, dgp)


def test_deep_gaussian_process_sampler_raises_for_invalid_model() -> None:
    with pytest.raises(ValueError, match="Model must be .*"):
        DeepGaussianProcessReparamSampler(10, QuadraticMeanAndRBFKernel())  # type: ignore


@pytest.mark.parametrize("shape", [[], [1], [2], [2, 3, 4]])
def test_deep_gaussian_process_sampler_sample_raises_for_invalid_at_shape(
    shape: ShapeLike,
    keras_float: None,
) -> None:
    search_space = Box([0.0], [1.0])
    x = search_space.sample(10)
    data = mk_dataset(x, quadratic(x))

    dgp = two_layer_trieste_dgp(data, search_space)
    sampler = DeepGaussianProcessReparamSampler(1, dgp)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        sampler.sample(tf.zeros(shape))


def _build_dataset_and_train_deep_gp(
    two_layer_model: Callable[[TensorType], DeepGP]
) -> Tuple[Dataset, DeepGaussianProcess]:
    x = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    y = tf.random.normal([100, 1], dtype=tf.float64)
    dataset = Dataset(x, y)

    dgp = two_layer_model(x)
    model = DeepGaussianProcess(dgp)
    model.optimize(dataset)

    return dataset, model


@random_seed
def test_deep_gaussian_process_sampler_samples_approximate_expected_distribution(
    two_layer_model: Callable[[TensorType], DeepGP],
    keras_float: None,
) -> None:
    sample_size = 1000
    dataset, model = _build_dataset_and_train_deep_gp(two_layer_model)

    samples = DeepGaussianProcessReparamSampler(sample_size, model).sample(
        dataset.query_points
    )  # [S, N, L]

    assert samples.shape == [sample_size, len(dataset.query_points), 1]

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.reduce_mean((samples - sample_mean) ** 2)

    num_samples = 50
    means = []
    vars = []
    for _ in range(num_samples):
        Fmean_sample, Fvar_sample = model.predict(dataset.query_points)
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
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    sampler = DeepGaussianProcessReparamSampler(100, model)
    xs = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_array_less(tf.abs(sampler.sample(xs + 1e-20) - sampler.sample(xs)), 1e-20)


def test_deep_gaussian_process_sampler_sample_is_repeatable(
    two_layer_model: Callable[[TensorType], DeepGP],
    keras_float: None,
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    sampler = DeepGaussianProcessReparamSampler(100, model)
    xs = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_allclose(sampler.sample(xs), sampler.sample(xs))


@random_seed
def test_deep_gaussian_process_sampler_samples_are_distinct_for_new_instances(
    two_layer_model: Callable[[TensorType], DeepGP],
    keras_float: None,
) -> None:
    _, model = _build_dataset_and_train_deep_gp(two_layer_model)

    sampler1 = DeepGaussianProcessReparamSampler(100, model)
    sampler2 = DeepGaussianProcessReparamSampler(100, model)

    xs = tf.random.uniform([100, 2], minval=-10.0, maxval=10.0, dtype=tf.float64)
    npt.assert_array_less(1e-9, tf.abs(sampler2.sample(xs) - sampler1.sample(xs)))

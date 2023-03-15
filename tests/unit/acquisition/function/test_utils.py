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

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from trieste.acquisition.function.utils import MultivariateNormalCDF
from trieste.types import TensorType

tfd = tfp.distributions


@pytest.mark.parametrize("num_sobol", [-10, -1, 0])
@pytest.mark.parametrize("dim", [2, 3, 5])
def test_make_mvn_cdf_raises_exception_for_incorrect_sample_size(
    num_sobol: int,
    dim: int,
) -> None:
    # Set data type and jitter
    dtype = tf.float64

    with pytest.raises(tf.errors.InvalidArgumentError):
        MultivariateNormalCDF(sample_size=num_sobol, dim=dim, dtype=dtype)


@pytest.mark.parametrize("num_sobol", [1, 10, 100])
@pytest.mark.parametrize("dim", [-10, -1, 0])
def test_make_mvn_cdf_raises_exception_for_incorrect_dimension(
    num_sobol: int,
    dim: int,
) -> None:
    # Set data type and jitter
    dtype = tf.float64

    with pytest.raises(tf.errors.InvalidArgumentError):
        MultivariateNormalCDF(sample_size=num_sobol, dim=dim, dtype=dtype)


def test_make_mvn_cdf_raises_exception_for_incorrect_batch_size(
    num_sobol: int = 100,
    dim: int = 5,
) -> None:
    # Set data type and jitter
    dtype = tf.float64

    # Set x, mean and covariance
    x = tf.zeros((0, dim), dtype=dtype)
    mean = tf.zeros((0, dim), dtype=dtype)
    cov = tf.eye(dim, dtype=dtype)[None, :, :][:0, :, :]

    with pytest.raises(tf.errors.InvalidArgumentError):
        MultivariateNormalCDF(sample_size=num_sobol, dim=dim, dtype=dtype)(x=x, mean=mean, cov=cov)


@pytest.mark.parametrize("num_sobol", [200])
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_make_genz_cdf_matches_naive_monte_carlo_on_random_tasks(
    num_sobol: int,
    dim: int,
    batch_size: int,
) -> None:
    def mc_mvn_cdf(
        x: TensorType,
        mean: TensorType,
        cov: TensorType,
        num_samples: int = int(1e6),
    ) -> TensorType:
        # Define multivariate normal
        normal = tfd.MultivariateNormalTriL(
            loc=mean,
            scale_tril=tf.linalg.cholesky(cov),
        )

        # Draw samples
        samples = normal.sample(sample_shape=[num_samples])

        # Check shapes of input tensors
        tf.debugging.assert_shapes(
            [
                (x, ("B", "Q")),
                (mean, ("B", "Q")),
                (cov, ("B", "Q", "Q")),
                (samples, ("S", "B", "Q")),
            ]
        )

        # Compute Monte Carlo estimate
        indicator = tf.reduce_all(tf.math.less(samples, x[None, ...]), axis=-1)
        mc_mvn_cdf = tf.reduce_mean(tf.cast(indicator, tf.float64), axis=0)

        return mc_mvn_cdf

    # Seed sampling for reproducible testing
    tf.random.set_seed(0)

    # Set data type and jitter
    dtype = tf.float64
    jitter = 1e-6

    # Draw x randomly
    x = tf.random.normal((batch_size, dim), dtype=dtype) / dim**0.5

    # Draw mean randomly
    mean = tf.random.normal((batch_size, dim), dtype=dtype) / dim**0.5

    # Draw covariance randomly
    cov = tf.random.normal((batch_size, dim, dim), dtype=dtype) / dim**0.5
    cov = tf.matmul(cov, cov, transpose_a=True) + jitter * tf.eye(dim, dtype=dtype)[None, :, :]

    # Set up Genz approximation and direct Monte Carlo estimate
    genz_cdf = MultivariateNormalCDF(sample_size=num_sobol, dim=dim, dtype=dtype)(
        x=x, mean=mean, cov=cov
    )
    mc_cdf = mc_mvn_cdf(x=x, mean=mean, cov=cov)

    # Check that the Genz and direct Monte Carlo estimates agree
    tf.debugging.assert_near(mc_cdf, genz_cdf, rtol=3e-1)

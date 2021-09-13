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

import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import quadratic, random_seed
from tests.util.models.gpflow.models import GaussianProcess


def _example_gaussian_process() -> GaussianProcess:
    return GaussianProcess(
        [quadratic, lambda x: quadratic(x) / 5.0],
        [
            tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1.6, length_scale=1.0),
            tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1.6, length_scale=2.0),
        ],
    )


# fmt: off
@pytest.mark.parametrize("xs, expected_mean, expected_cov", [
    (
        tf.constant([[0.0, -1.0], [-2.0, 3.0], [4.0, 5.0]]),
        tf.constant([[1.0, 0.2], [13.0, 2.6], [41.0, 8.2]]),
        2.56 / tf.exp([
            [[0.0, 10.0, 26.0], [10.0, 0.0, 20.0], [26.0, 20.0, 0.0]],
            [[0.0, 2.5, 6.5], [2.5, 0.0, 5.0], [6.5, 5.0, 0.0]],
        ])
    ),
    (
        tf.constant([
            [[0.0, -1.0], [-2.0, 3.0], [4.0, 5.0]],
            [[-3.0, 2.0], [4.0, 3.0], [-4.0, 6.0]],
        ]),
        tf.constant([[
            [1.0, 0.2], [13.0, 2.6], [41.0, 8.2]],
            [[13.0, 2.6], [25.0, 5.0], [52.0, 10.4]]
        ]),
        2.56 / tf.exp([
            [
                [[0.0, 10.0, 26.0], [10.0, 0.0, 20.0], [26.0, 20.0, 0.0]],
                [[0.0, 2.5, 6.5], [2.5, 0.0, 5.0], [6.5, 5.0, 0.0]],
            ],
            [
                [[0.0, 25.0, 8.5], [25.0, 0.0, 36.5], [8.5, 36.5, 0.0]],
                [[0.0, 6.25, 2.125], [6.25, 0.0, 9.125], [2.125, 9.125, 0.0]],
            ]
        ])
    )
])
# fmt: on
def test_gaussian_process_predict_joint(
    xs: tf.Tensor, expected_mean: tf.Tensor, expected_cov: tf.Tensor
) -> None:
    mean, cov = _example_gaussian_process().predict_joint(xs)
    npt.assert_allclose(mean, expected_mean)
    npt.assert_allclose(cov, expected_cov, rtol=2e-6)


# fmt: off
@pytest.mark.parametrize("xs, expected_mean, expected_var", [
    (
        tf.constant([[0.0, -1.0], [-2.0, 3.0], [4.0, 5.0]]),
        tf.constant([[1.0, 0.2], [13.0, 2.6], [41.0, 8.2]]),
        tf.fill([3, 2], 2.56)
    ),
    (
        tf.constant([
            [[0.0, -1.0], [-2.0, 3.0], [4.0, 5.0]],
            [[-3.0, 2.0], [4.0, 3.0], [-4.0, 6.0]],
        ]),
        tf.constant([
            [[1.0, 0.2], [13.0, 2.6], [41.0, 8.2]],
            [[13.0, 2.6], [25.0, 5.0], [52.0, 10.4]]
        ]),
        tf.fill([2, 3, 2], 2.56)
    )
])
# fmt: on
def test_gaussian_process_predict(
    xs: tf.Tensor, expected_mean: tf.Tensor, expected_var: tf.Tensor
) -> None:
    mean, var = _example_gaussian_process().predict(xs)
    npt.assert_allclose(mean, expected_mean)
    npt.assert_allclose(var, expected_var)


@random_seed
def test_gaussian_process_sample() -> None:
    # fmt: off
    samples = _example_gaussian_process().sample(tf.constant([
        [[0.0, -1.0], [-2.0, 3.0], [4.0, 5.0]],
        [[-3.0, 2.0], [4.0, 3.0], [-4.0, 6.0]],
    ]), 10_000)
    npt.assert_allclose(tf.reduce_mean(samples, axis=-3), [
        [[1.0, 0.2], [13.0, 2.6], [41.0, 8.2]],
        [[13.0, 2.6], [25.0, 5.0], [52.0, 10.4]]
    ], rtol=0.02)
    variance = tf.math.reduce_variance(samples, axis=-3)
    npt.assert_allclose(variance, tf.fill([2, 3, 2], 2.56), rtol=0.02)
    # fmt: on


def test_gaussian_process_get_observation_noise() -> None:
    noise = _example_gaussian_process().get_observation_noise()
    npt.assert_equal(noise, tf.constant(1.0))


def test_gaussian_process_covariance_between_points() -> None:
    x = tf.reshape(tf.linspace(0.0, 5.0, 4), (-1, 1))
    x = tf.cast(x, dtype=tf.float32)

    query_points_1 = tf.concat([0.5 * x, 0.5 * x], 0)
    query_points_2 = tf.concat([2 * x, 2 * x, 2 * x], 0)

    all_query_points = tf.concat([query_points_1, query_points_2], 0)
    _, predictive_covariance = _example_gaussian_process().predict_joint(all_query_points)
    expected_covariance = predictive_covariance[:, :8, 8:]

    actual_covariance = _example_gaussian_process().covariance_between_points(
        query_points_1, query_points_2
    )

    npt.assert_allclose(expected_covariance, actual_covariance, atol=1e-5)

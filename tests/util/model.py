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

from abc import ABC
from typing import Tuple

import numpy.testing as npt
import tensorflow as tf
import tensorflow_probability as tfp

from trieste.data import Dataset
from trieste.models import ModelInterface
from trieste.type import QueryPoints, ObserverEvaluations, TensorType
from tests.util.misc import random_seed


class StaticModelInterface(ModelInterface, ABC):
    def update(self, dataset: Dataset) -> None:
        pass

    def optimize(self) -> None:
        pass


class GaussianMarginal(StaticModelInterface, ABC):
    def sample(self, query_points: QueryPoints, num_samples: int) -> ObserverEvaluations:
        mean, var = self.predict(query_points)
        return tfp.distributions.Normal(mean, var).sample(num_samples)


class QuadraticWithUnitVariance(GaussianMarginal):
    r""" An untrainable model hardcoded to the function :math:`y = \sum x^2` with unit variance. """
    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        mean = tf.reduce_sum(query_points ** 2, axis=1, keepdims=True)
        return mean, tf.ones_like(mean)


def test_quadratic_with_unit_variance() -> None:
    model = QuadraticWithUnitVariance()
    mean, var = model.predict(tf.constant([[0., 1.], [2., 3.], [4., 5.]]))
    npt.assert_array_almost_equal(mean, tf.constant([[1.], [13.], [41.]]))
    npt.assert_array_almost_equal(var, tf.constant([[1.], [1.], [1.]]))


@random_seed()
def test_guassian_marginal_sample() -> None:
    class _Sum(GaussianMarginal):
        def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
            mean = tf.reduce_sum(query_points, axis=1, keepdims=True)
            return mean, tf.ones_like(mean)

    samples = _Sum().sample(tf.constant([[0., 1.], [2., 3.], [4., 5.]]), 100_000)

    samples_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.sqrt(tf.reduce_mean((samples - samples_mean) ** 2, axis=0))

    assert samples.shape == [100_000, 3, 1]
    npt.assert_allclose(samples_mean, [[1.], [5.], [9.]], rtol=1e-2)
    npt.assert_allclose(sample_variance, [[1.], [1.], [1.]], rtol=1e-2)

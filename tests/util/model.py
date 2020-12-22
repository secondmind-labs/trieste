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
from typing import Callable, Tuple

import numpy.testing as npt
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import random_seed
from trieste.data import Dataset
from trieste.models import TrainableProbabilisticModel
from trieste.type import ObserverEvaluations, QueryPoints, TensorType


class PseudoTrainableProbModel(TrainableProbabilisticModel, ABC):
    """ A model that does nothing on :meth:`update` and :meth:`optimize`. """

    def update(self, dataset: Dataset) -> None:
        pass

    def optimize(self, dataset: Dataset) -> None:
        pass


class GaussianMarginal(PseudoTrainableProbModel, ABC):
    """ A probabilistic model with a Gaussian marginal distribution at each point. """

    def sample(self, query_points: QueryPoints, num_samples: int) -> ObserverEvaluations:
        mean, var = self.predict(query_points)
        return tfp.distributions.Normal(mean, var).sample(num_samples)


class CustomMeanWithUnitVariance(GaussianMarginal):
    def __init__(self, f: Callable[[tf.Tensor], tf.Tensor]):
        self._f = f

    def __repr__(self) -> str:
        return f"CustomMeanWithUnitVariance({self._f!r})"

    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        mean = self._f(query_points)
        return mean, tf.ones_like(mean)


class QuadraticWithUnitVariance(GaussianMarginal):
    r"""
    A probabilistic model with mean :math:`x \mapsto \sum x^2`, unit variance, and Gaussian
    marginal distribution.
    """

    def __init__(self):
        self._model = CustomMeanWithUnitVariance(
            lambda x: tf.reduce_sum(x ** 2, axis=-1, keepdims=True)
        )

    def __repr__(self) -> str:
        return "QuadraticWithUnitVariance()"

    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        return self._model.predict(query_points)


def test_quadratic_with_unit_variance() -> None:
    model = QuadraticWithUnitVariance()
    mean, var = model.predict(tf.constant([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
    npt.assert_array_almost_equal(mean, tf.constant([[1.0], [13.0], [41.0]]))
    npt.assert_array_almost_equal(var, tf.constant([[1.0], [1.0], [1.0]]))


@random_seed
def test_gaussian_marginal_sample() -> None:
    class _Sum(GaussianMarginal):
        def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
            mean = tf.reduce_sum(query_points, axis=1, keepdims=True)
            return mean, tf.ones_like(mean)

    samples = _Sum().sample(tf.constant([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]), 100_000)

    samples_mean = tf.reduce_mean(samples, axis=0)
    sample_variance = tf.sqrt(tf.reduce_mean((samples - samples_mean) ** 2, axis=0))

    assert samples.shape == [100_000, 3, 1]
    npt.assert_allclose(samples_mean, [[1.0], [5.0], [9.0]], rtol=1e-2)
    npt.assert_allclose(sample_variance, [[1.0], [1.0], [1.0]], rtol=1e-2)

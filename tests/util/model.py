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

from abc import ABC
from collections.abc import Callable, Sequence

import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import SequenceN, quadratic
from trieste.data import Dataset
from trieste.models import ProbabilisticModel, TrainableProbabilisticModel
from trieste.type import TensorType


def rbf() -> tfp.math.psd_kernels.ExponentiatedQuadratic:
    """
    :return: A :class:`tfp.math.psd_kernels.ExponentiatedQuadratic` with default arguments.
    """
    return tfp.math.psd_kernels.ExponentiatedQuadratic()


class PseudoTrainableProbModel(TrainableProbabilisticModel, ABC):
    """ A model that does nothing on :meth:`update` and :meth:`optimize`. """

    def update(self, dataset: Dataset) -> None:
        pass

    def optimize(self, dataset: Dataset) -> None:
        pass


class GaussianMarginal(ProbabilisticModel, ABC):
    """ A probabilistic model with Gaussian marginal distribution. Assumes events of shape [N]. """

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        mean, var = self.predict(query_points)
        samples = tfp.distributions.Normal(mean, tf.sqrt(var)).sample(num_samples)
        dim_order = tf.range(tf.rank(samples))
        return tf.transpose(samples, tf.concat([dim_order[1:-2], [0], dim_order[-2:]], -1))


class GaussianProcess(GaussianMarginal, ProbabilisticModel):
    """ A (static) Gaussian process over a vector random variable. """

    def __init__(
        self,
        mean_functions: Sequence[Callable[[TensorType], TensorType]],
        kernels: Sequence[tfp.math.psd_kernels.PositiveSemidefiniteKernel],
    ):
        super().__init__()
        self._mean_functions = mean_functions
        self._kernels = kernels

    def __repr__(self) -> str:
        return f"GaussianProcess({self._mean_functions!r}, {self._kernels!r})"

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        mean, cov = self.predict_joint(query_points[..., None, :])
        return tf.squeeze(mean, -2), tf.squeeze(cov, [-2, -1])

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        means = [f(query_points) for f in self._mean_functions]
        covs = [k.tensor(query_points, query_points, 1, 1)[..., None, :, :] for k in self._kernels]
        return tf.concat(means, axis=-1), tf.concat(covs, axis=-3)


class QuadraticMeanAndRBFKernel(GaussianProcess):
    r""" A Gaussian process with scalar quadratic mean and RBF kernel. """

    def __init__(
        self,
        *,
        x_shift: float | SequenceN[float] | TensorType = 0,
        kernel_amplitude: float | TensorType | None = None,
    ):
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(kernel_amplitude)
        super().__init__([lambda x: quadratic(x - x_shift)], [kernel])

    def __repr__(self) -> str:
        return "QuadraticMeanAndRBFKernel()"

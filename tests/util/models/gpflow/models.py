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

from collections.abc import Callable, Sequence

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import GPR, SGPR, SVGP, VGP, GPModel
from typing_extensions import Protocol

from tests.util.misc import SequenceN, quadratic
from trieste.data import Dataset
from trieste.models import (
    ProbabilisticModel,
    ReparametrizationSampler,
    TrainableProbabilisticModel,
    TrajectorySampler,
)
from trieste.models.gpflow import (
    BatchReparametrizationSampler,
    GPflowPredictor,
    RandomFourierFeatureTrajectorySampler,
)
from trieste.models.gpflow.interface import SupportsCovarianceBetweenPoints
from trieste.models.interfaces import SupportsGetKernel, SupportsGetObservationNoise
from trieste.models.optimizer import Optimizer
from trieste.types import TensorType


def rbf() -> tfp.math.psd_kernels.ExponentiatedQuadratic:
    """
    :return: A :class:`tfp.math.psd_kernels.ExponentiatedQuadratic` with default arguments.
    """
    return tfp.math.psd_kernels.ExponentiatedQuadratic()


class PseudoTrainableProbModel(TrainableProbabilisticModel, Protocol):
    """A model that does nothing on :meth:`update` and :meth:`optimize`."""

    def update(self, dataset: Dataset) -> None:
        pass

    def optimize(self, dataset: Dataset) -> None:
        pass


class GaussianMarginal(ProbabilisticModel):
    """A probabilistic model with Gaussian marginal distribution. Assumes events of shape [N]."""

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        mean, var = self.predict(query_points)
        samples = tfp.distributions.Normal(mean, tf.sqrt(var)).sample(num_samples)
        dim_order = tf.range(tf.rank(samples))
        return tf.transpose(samples, tf.concat([dim_order[1:-2], [0], dim_order[-2:]], -1))


class GaussianProcess(
    GaussianMarginal, SupportsCovarianceBetweenPoints, SupportsGetObservationNoise
):
    """A (static) Gaussian process over a vector random variable."""

    def __init__(
        self,
        mean_functions: Sequence[Callable[[TensorType], TensorType]],
        kernels: Sequence[tfp.math.psd_kernels.PositiveSemidefiniteKernel],
        noise_variance: float = 1.0,
    ):
        super().__init__()
        self._mean_functions = mean_functions
        self._kernels = kernels
        self._noise_variance = noise_variance

    def __repr__(self) -> str:
        return f"GaussianProcess({self._mean_functions!r}, {self._kernels!r})"

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        mean, cov = self.predict_joint(query_points[..., None, :])
        return tf.squeeze(mean, -2), tf.squeeze(cov, [-2, -1])

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        means = [f(query_points) for f in self._mean_functions]
        covs = [k.tensor(query_points, query_points, 1, 1)[..., None, :, :] for k in self._kernels]
        return tf.concat(means, axis=-1), tf.concat(covs, axis=-3)

    def get_observation_noise(self) -> TensorType:
        return tf.constant(self._noise_variance)

    def covariance_between_points(
        self, query_points_1: TensorType, query_points_2: TensorType
    ) -> TensorType:
        covs = [
            k.tensor(query_points_1, query_points_2, 1, 1)[..., None, :, :] for k in self._kernels
        ]
        return tf.concat(covs, axis=-3)


class GaussianProcessWithSamplers(GaussianProcess):
    """A (static) Gaussian process over a vector random variable with a reparam sampler"""

    def reparam_sampler(
        self, num_samples: int
    ) -> ReparametrizationSampler[GaussianProcessWithSamplers]:
        return BatchReparametrizationSampler(num_samples, self)


class QuadraticMeanAndRBFKernel(GaussianProcess, SupportsGetKernel, SupportsGetObservationNoise):
    r"""A Gaussian process with scalar quadratic mean and RBF kernel."""

    def __init__(
        self,
        *,
        x_shift: float | SequenceN[float] | TensorType = 0,
        kernel_amplitude: float | TensorType | None = None,
        noise_variance: float = 1.0,
    ):
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(kernel_amplitude)
        super().__init__([lambda x: quadratic(x - x_shift)], [self.kernel], noise_variance)

    def __repr__(self) -> str:
        return "QuadraticMeanAndRBFKernel()"

    def get_kernel(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        return self.kernel


def mock_data() -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.constant([[1.1], [2.2], [3.3], [4.4]], gpflow.default_float()),
        tf.constant([[1.2], [3.4], [5.6], [7.8]], gpflow.default_float()),
    )


class QuadraticMeanAndRBFKernelWithSamplers(QuadraticMeanAndRBFKernel):
    r"""
    A Gaussian process with scalar quadratic mean, an RBF kernel and
    a trajectory_sampler and reparam_sampler methods.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        x_shift: float | SequenceN[float] | TensorType = 0,
        kernel_amplitude: float | TensorType | None = None,
        noise_variance: float = 1.0,
    ):
        super().__init__(
            x_shift=x_shift, kernel_amplitude=kernel_amplitude, noise_variance=noise_variance
        )
        self._dataset = dataset

    def trajectory_sampler(self) -> TrajectorySampler[QuadraticMeanAndRBFKernelWithSamplers]:
        return RandomFourierFeatureTrajectorySampler(self, self._dataset, 100)

    def reparam_sampler(
        self, num_samples: int
    ) -> ReparametrizationSampler[QuadraticMeanAndRBFKernelWithSamplers]:
        return BatchReparametrizationSampler(num_samples, self)


class ModelFactoryType(Protocol):
    def __call__(
        self, x: TensorType, y: TensorType, optimizer: Optimizer | None = None
    ) -> tuple[GPflowPredictor, Callable[[TensorType, TensorType], GPModel]]:
        pass


def gpr_model(x: tf.Tensor, y: tf.Tensor) -> GPR:
    return GPR((x, y), gpflow.kernels.Matern32())


def sgpr_model(x: tf.Tensor, y: tf.Tensor) -> SGPR:
    return SGPR((x, y), gpflow.kernels.Matern32(), x[:2])


def svgp_model(x: tf.Tensor, y: tf.Tensor) -> SVGP:
    return SVGP(gpflow.kernels.Matern32(), gpflow.likelihoods.Gaussian(), x[:2], num_data=len(x))


def vgp_model(x: tf.Tensor, y: tf.Tensor) -> VGP:
    likelihood = gpflow.likelihoods.Gaussian()
    kernel = gpflow.kernels.Matern32()
    m = VGP((x, y), kernel, likelihood)
    return m


def vgp_matern_model(x: tf.Tensor, y: tf.Tensor) -> VGP:
    likelihood = gpflow.likelihoods.Gaussian()
    kernel = gpflow.kernels.Matern32(lengthscales=0.2)
    m = VGP((x, y), kernel, likelihood)
    return m

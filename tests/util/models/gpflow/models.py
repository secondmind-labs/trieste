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
from typing import Optional

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import inherit_check_shapes
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
    IndependentReparametrizationSampler,
    RandomFourierFeatureTrajectorySampler,
)
from trieste.models.gpflow.interface import SupportsCovarianceBetweenPoints
from trieste.models.interfaces import (
    HasReparamSampler,
    HasTrajectorySampler,
    SupportsCovarianceWithTopFidelity,
    SupportsGetKernel,
    SupportsGetObservationNoise,
    SupportsPredictJoint,
    SupportsPredictY,
)
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
        return

    def optimize(self, dataset: Dataset) -> None:
        return


class GaussianMarginal(ProbabilisticModel):
    """A probabilistic model with Gaussian marginal distribution. Assumes events of shape [N]."""

    @inherit_check_shapes
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
        self._mean_functions = mean_functions
        self._kernels = kernels
        self._noise_variance = noise_variance

    def __repr__(self) -> str:
        return f"GaussianProcess({self._mean_functions!r}, {self._kernels!r})"

    @inherit_check_shapes
    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        mean, cov = self.predict_joint(query_points[..., None, :])
        return tf.squeeze(mean, -2), tf.squeeze(cov, [-2, -1])

    @inherit_check_shapes
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

    def log(self, dataset: Optional[Dataset] = None) -> None:
        return


class GaussianProcessWithoutNoise(GaussianMarginal, SupportsPredictJoint, HasReparamSampler):
    """A (static) Gaussian process over a vector random variable with independent reparam sampler
    but without noise variance."""

    def __init__(
        self,
        mean_functions: Sequence[Callable[[TensorType], TensorType]],
        kernels: Sequence[tfp.math.psd_kernels.PositiveSemidefiniteKernel],
    ):
        self._mean_functions = mean_functions
        self._kernels = kernels

    def __repr__(self) -> str:
        return f"GaussianProcessWithoutNoise({self._mean_functions!r}, {self._kernels!r})"

    @inherit_check_shapes
    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        mean, cov = self.predict_joint(query_points[..., None, :])
        return tf.squeeze(mean, -2), tf.squeeze(cov, [-2, -1])

    @inherit_check_shapes
    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        means = [f(query_points) for f in self._mean_functions]
        covs = [k.tensor(query_points, query_points, 1, 1)[..., None, :, :] for k in self._kernels]
        return tf.concat(means, axis=-1), tf.concat(covs, axis=-3)

    def covariance_between_points(
        self, query_points_1: TensorType, query_points_2: TensorType
    ) -> TensorType:
        covs = [
            k.tensor(query_points_1, query_points_2, 1, 1)[..., None, :, :] for k in self._kernels
        ]
        return tf.concat(covs, axis=-3)

    def reparam_sampler(
        self: GaussianProcessWithoutNoise, num_samples: int
    ) -> ReparametrizationSampler[GaussianProcessWithoutNoise]:
        return IndependentReparametrizationSampler(num_samples, self)

    def log(self, dataset: Optional[Dataset] = None) -> None:
        return


class GaussianProcessWithSamplers(GaussianProcess, HasReparamSampler):
    """A (static) Gaussian process over a vector random variable with independent reparam sampler"""

    def reparam_sampler(
        self, num_samples: int
    ) -> ReparametrizationSampler[GaussianProcessWithSamplers]:
        return IndependentReparametrizationSampler(num_samples, self)


class GaussianProcessWithBatchSamplers(GaussianProcess, HasReparamSampler):
    """A (static) Gaussian process over a vector random variable with a batch reparam sampler"""

    def reparam_sampler(
        self, num_samples: int
    ) -> ReparametrizationSampler[GaussianProcessWithBatchSamplers]:
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
        self.mean_function = lambda x: quadratic(x - tf.cast(x_shift, dtype=x.dtype))
        super().__init__([self.mean_function], [self.kernel], noise_variance)

    def __repr__(self) -> str:
        return "QuadraticMeanAndRBFKernel()"

    def get_kernel(self) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
        return self.kernel

    def get_mean_function(self) -> Callable[[TensorType], TensorType]:
        return self.mean_function


def mock_data() -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.constant([[1.1], [2.2], [3.3], [4.4]], gpflow.default_float()),
        tf.constant([[1.2], [3.4], [5.6], [7.8]], gpflow.default_float()),
    )


class QuadraticMeanAndRBFKernelWithSamplers(
    QuadraticMeanAndRBFKernel, HasTrajectorySampler, HasReparamSampler, TrainableProbabilisticModel
):
    r"""
    A Gaussian process with scalar quadratic mean, an RBF kernel and
    trajectory_sampler and reparam_sampler methods.
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

        self._dataset = (  # mimic that when our models store data, it is as variables
            tf.Variable(
                dataset.query_points, trainable=False, shape=[None, *dataset.query_points.shape[1:]]
            ),
            tf.Variable(
                dataset.observations, trainable=False, shape=[None, *dataset.observations.shape[1:]]
            ),
        )

    def trajectory_sampler(self) -> TrajectorySampler[QuadraticMeanAndRBFKernelWithSamplers]:
        return RandomFourierFeatureTrajectorySampler(self, 100)

    def reparam_sampler(
        self, num_samples: int
    ) -> ReparametrizationSampler[QuadraticMeanAndRBFKernelWithSamplers]:
        return IndependentReparametrizationSampler(num_samples, self)

    def get_internal_data(self) -> Dataset:
        return Dataset(self._dataset[0], self._dataset[1])

    def update(self, dataset: Dataset) -> None:
        self._dataset[0].assign(dataset.query_points)
        self._dataset[1].assign(dataset.observations)

    def optimize(self, dataset: Dataset) -> None:
        pass


class MultiFidelityQuadraticMeanAndRBFKernel(
    QuadraticMeanAndRBFKernel, SupportsPredictY, SupportsCovarianceWithTopFidelity
):
    r"""
    A Gaussian process with scalar quadratic mean, an RBF kernel and
    trajectory_sampler and reparam_sampler methods.
    """

    def __init__(
        self,
        *,
        x_shift: float | SequenceN[float] | TensorType = 0,
        kernel_amplitude: float | TensorType | None = None,
        noise_variance: float = 1.0,
    ):
        super().__init__(
            x_shift=x_shift, kernel_amplitude=kernel_amplitude, noise_variance=noise_variance
        )

    @property
    def num_fidelities(self) -> int:
        return 5

    def covariance_with_top_fidelity(self, x: TensorType) -> TensorType:
        mean, _ = self.predict(x)
        return tf.ones_like(mean, dtype=mean.dtype)  # dummy covariances of correct shape

    @inherit_check_shapes
    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        fmean, fvar = self.predict(query_points)
        yvar = fvar + tf.constant(1.0, dtype=fmean.dtype)  # dummy noise variance
        return fmean, yvar


class MultiFidelityQuadraticMeanAndRBFKernelWithSamplers(
    QuadraticMeanAndRBFKernelWithSamplers, SupportsPredictY, SupportsCovarianceWithTopFidelity
):
    r"""
    A Gaussian process with scalar quadratic mean, an RBF kernel and
    trajectory_sampler and reparam_sampler methods.
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
            dataset,
            x_shift=x_shift,
            kernel_amplitude=kernel_amplitude,
            noise_variance=noise_variance,
        )

    @property
    def num_fidelities(self) -> int:
        return 5

    def covariance_with_top_fidelity(self, x: TensorType) -> TensorType:
        mean, _ = self.predict(x)
        return tf.ones_like(mean, dtype=mean.dtype)  # dummy covariances of correct shape

    @inherit_check_shapes
    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        fmean, fvar = self.predict(query_points)
        yvar = fvar + tf.constant(1.0, dtype=fmean.dtype)  # dummy noise variance
        return fmean, yvar


class QuadraticMeanAndRBFKernelWithBatchSamplers(
    QuadraticMeanAndRBFKernel, HasTrajectorySampler, HasReparamSampler
):
    r"""
    A Gaussian process with scalar quadratic mean, an RBF kernel and
    trajectory_sampler and batch reparam_sampler methods.
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
        self._dataset = (  # mimic that when our models store data, it is as variables
            tf.Variable(
                dataset.query_points, trainable=False, shape=[None, *dataset.query_points.shape[1:]]
            ),
            tf.Variable(
                dataset.observations, trainable=False, shape=[None, *dataset.observations.shape[1:]]
            ),
        )

    def trajectory_sampler(self) -> TrajectorySampler[QuadraticMeanAndRBFKernelWithBatchSamplers]:
        return RandomFourierFeatureTrajectorySampler(self, 100)

    def reparam_sampler(
        self, num_samples: int
    ) -> ReparametrizationSampler[QuadraticMeanAndRBFKernelWithBatchSamplers]:
        return BatchReparametrizationSampler(num_samples, self)

    def get_internal_data(self) -> Dataset:
        return Dataset(self._dataset[0], self._dataset[1])

    def update(self, dataset: Dataset) -> None:
        self._dataset[0].assign(dataset.query_points)
        self._dataset[1].assign(dataset.observations)


class ModelFactoryType(Protocol):
    def __call__(
        self, x: TensorType, y: TensorType, optimizer: Optimizer | None = None
    ) -> tuple[GPflowPredictor, Callable[[TensorType, TensorType], GPModel]]:
        pass


def gpr_model(x: tf.Tensor, y: tf.Tensor) -> GPR:
    return GPR((x, y), gpflow.kernels.Matern32())


def sgpr_model(x: tf.Tensor, y: tf.Tensor, num_latent_gps: int = 1) -> SGPR:
    return SGPR((x, y), gpflow.kernels.Matern32(), x[:2], num_latent_gps=num_latent_gps)


def svgp_model(x: tf.Tensor, y: tf.Tensor, num_latent_gps: int = 1) -> SVGP:
    return SVGP(
        gpflow.kernels.Matern32(),
        gpflow.likelihoods.Gaussian(),
        x[:2],
        num_data=len(x),
        num_latent_gps=num_latent_gps,
    )


def quadratic_mean_rbf_kernel_model(dataset: Dataset) -> QuadraticMeanAndRBFKernelWithSamplers:
    model = QuadraticMeanAndRBFKernelWithSamplers(
        noise_variance=tf.constant(0.9, dtype=tf.float64), dataset=dataset
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    return model


def svgp_model_with_mean(
    x: tf.Tensor, y: tf.Tensor, whiten: bool, num_inducing_points: int, num_latent_gps: int = 1
) -> SVGP:
    mean_function = gpflow.mean_functions.Linear(
        A=0.37 * np.ones((1, 1), dtype=gpflow.default_float()),
        b=0.19 * np.ones((1,), dtype=gpflow.default_float()),
    )
    q_mu = np.random.randn(num_inducing_points, 1)
    q_sqrt = np.tril(np.random.randn(1, num_inducing_points, num_inducing_points))
    m = SVGP(
        gpflow.kernels.Matern32(variance=0.91),
        gpflow.likelihoods.Gaussian(variance=0.23),
        x[:num_inducing_points],
        num_data=len(x),
        num_latent_gps=num_latent_gps,
        mean_function=mean_function,
        whiten=whiten,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
    )
    gpflow.set_trainable(mean_function, False)
    gpflow.set_trainable(m.inducing_variable, False)
    return m


def vgp_model(x: tf.Tensor, y: tf.Tensor, num_latent_gps: int = 1) -> VGP:
    likelihood = gpflow.likelihoods.Gaussian()
    kernel = gpflow.kernels.Matern32()
    m = VGP((x, y), kernel, likelihood, num_latent_gps=num_latent_gps)
    return m


def vgp_matern_model(x: tf.Tensor, y: tf.Tensor) -> VGP:
    likelihood = gpflow.likelihoods.Gaussian()
    kernel = gpflow.kernels.Matern32(lengthscales=0.2)
    m = VGP((x, y), kernel, likelihood)
    return m


def svgp_model_by_type(
    x: tf.Tensor,
    type: str,
    whiten: bool,
    num_inducing_points: int = 3,
    noise_var: Optional[float] = None,
    mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
) -> SVGP:
    num_latent_gps = 2
    ker1 = gpflow.kernels.Matern32(variance=0.8, lengthscales=0.2)
    ker2 = gpflow.kernels.Matern52(variance=0.3, lengthscales=0.7)

    if type == "shared+shared":
        kernel = gpflow.kernels.SharedIndependent(ker1, output_dim=2)
        iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(x[:num_inducing_points])
        )
    elif type == "separate+shared":
        kernel = gpflow.kernels.SeparateIndependent([ker1, ker2])
        iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(x[:num_inducing_points])
        )
    elif type == "separate+separate":
        kernel = gpflow.kernels.SeparateIndependent([ker1, ker2])
        Zs = [
            x[(num_inducing_points * i) : (num_inducing_points * i + num_inducing_points)]
            for i in range(2)
        ]
        iv_list = [gpflow.inducing_variables.InducingPoints(Z) for Z in Zs]
        iv = gpflow.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    else:
        if "single" in type:
            num_latent_gps = 1
        kernel = ker1
        iv = x[:num_inducing_points]

    return SVGP(
        kernel,
        gpflow.likelihoods.Gaussian(noise_var),
        iv,
        num_data=len(x),
        num_latent_gps=num_latent_gps,
        whiten=whiten,
        mean_function=mean_function,
    )


def two_output_sgpr_model(x: tf.Tensor, y: tf.Tensor, type: str = "separate+separate") -> SGPR:
    ker1 = gpflow.kernels.Matern32()
    ker2 = gpflow.kernels.Matern52()

    if type == "shared+shared":
        kernel = gpflow.kernels.SharedIndependent(ker1, output_dim=2)
        iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(x[:3])
        )
    elif type == "separate+shared":
        kernel = gpflow.kernels.SeparateIndependent([ker1, ker2])
        iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(x[:3])
        )
    elif type == "separate+separate":
        kernel = gpflow.kernels.SeparateIndependent([ker1, ker2])
        Zs = [x[(3 * i) : (3 * i + 3)] for i in range(2)]
        iv_list = [gpflow.inducing_variables.InducingPoints(Z) for Z in Zs]
        iv = gpflow.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    else:
        kernel = ker1
        iv = x[:3]

    return SGPR((x, y), kernel, iv, num_latent_gps=2)


def vgp_model_bernoulli(x: tf.Tensor, y: tf.Tensor) -> VGP:
    likelihood = gpflow.likelihoods.Bernoulli()
    kernel = gpflow.kernels.Matern32(lengthscales=0.2)
    m = VGP((x, y), kernel, likelihood)
    return m

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

from abc import ABC, abstractmethod
from typing import Optional

import gpflow
import tensorflow as tf
from gpflow.models import GPModel
from gpflow.posteriors import BasePosterior, PrecomputeCacheType
from typing_extensions import Protocol

from ... import logging
from ...data import Dataset
from ...types import TensorType
from ..interfaces import (
    HasReparamSampler,
    ReparametrizationSampler,
    SupportsGetKernel,
    SupportsGetObservationNoise,
    SupportsPredictJoint,
)
from ..optimizer import Optimizer
from .sampler import BatchReparametrizationSampler


class GPflowPredictor(
    SupportsPredictJoint, SupportsGetKernel, SupportsGetObservationNoise, HasReparamSampler, ABC
):
    """
    A trainable wrapper for a GPflow Gaussian process model.

    As Bayesian optimization requires a large number of sequential predictions (i.e. when maximizing
    acquisition functions), rather than calling the model directly at prediction time we instead
    call the posterior objects built by these models. These posterior objects store the
    pre-computed Gram matrices, which can be reused to allow faster subsequent predictions. However,
    note that these posterior objects need to be updated whenever the underlying model is changed
    (i.e. after calling :meth:`update` or :meth:`optimize`).
    """

    def __init__(self, optimizer: Optimizer | None = None):
        """
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        """
        if optimizer is None:
            optimizer = Optimizer(gpflow.optimizers.Scipy())

        self._optimizer = optimizer
        self._posterior: Optional[BasePosterior] = None

    def update_posterior_cache(self) -> None:
        """Update posterior objects. This needs to be called if the underlying model is changed
        after predictions have already been made."""
        if self._posterior is not None:
            self._posterior.update_cache()

    @property
    def optimizer(self) -> Optimizer:
        """The optimizer with which to train the model."""
        return self._optimizer

    @property
    @abstractmethod
    def model(self) -> GPModel:
        """The underlying GPflow model."""

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        if self._posterior is None and hasattr(self.model, "posterior"):
            self._posterior = self.model.posterior(PrecomputeCacheType.VARIABLE)
        return (self._posterior or self.model).predict_f(query_points)

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        if self._posterior is None and hasattr(self.model, "posterior"):
            self._posterior = self.model.posterior(PrecomputeCacheType.VARIABLE)
        return (self._posterior or self.model).predict_f(query_points, full_cov=True)

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        return self.model.predict_f_samples(query_points, num_samples)

    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.model.predict_y(query_points)

    def get_kernel(self) -> gpflow.kernels.Kernel:
        """
        Return the kernel of the model.

        :return: The kernel.
        """
        return self.model.kernel

    def get_observation_noise(self) -> TensorType:
        """
        Return the variance of observation noise for homoscedastic likelihoods.

        :return: The observation noise.
        :raise NotImplementedError: If the model does not have a homoscedastic likelihood.
        """
        try:
            noise_variance = self.model.likelihood.variance
        except AttributeError:
            raise NotImplementedError(f"Model {self!r} does not have scalar observation noise")

        return noise_variance

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the model with the specified `dataset`.

        :param dataset: The data with which to optimize the `model`.
        """
        self.optimizer.optimize(self.model, dataset)

    def log(self) -> None:
        """
        Log model-specific information at a given optimization step.
        """
        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                logging.scalar("kernel.variance", self.get_kernel().variance)
                lengthscales = self.get_kernel().lengthscales
                if tf.rank(lengthscales) == 0:
                    logging.scalar("kernel.lengthscale", lengthscales)
                elif tf.rank(lengthscales) == 1:
                    for i, lengthscale in enumerate(lengthscales):
                        logging.scalar(f"kernel.lengthscale[{i}]", lengthscale)

    def reparam_sampler(self, num_samples: int) -> ReparametrizationSampler[GPflowPredictor]:
        """
        Return a reparametrization sampler providing `num_samples` samples.

        :return: The reparametrization sampler.
        """
        return BatchReparametrizationSampler(num_samples, self)


class SupportsCovarianceBetweenPoints(SupportsPredictJoint, Protocol):
    """A probabilistic model that supports covariance_between_points."""

    @abstractmethod
    def covariance_between_points(
        self, query_points_1: TensorType, query_points_2: TensorType
    ) -> TensorType:
        r"""
        Compute the posterior covariance between sets of query points.

        .. math:: \Sigma_{12} = K_{12} - K_{x1}(K_{xx} + \sigma^2 I)^{-1}K_{x2}

        Note that query_points_2 must be a rank 2 tensor, but query_points_1 can
        have leading dimensions.

        :param query_points_1: Set of query points with shape [..., N, D]
        :param query_points_2: Sets of query points with shape [M, D]
        :return: Covariance matrix between the sets of query points with shape [..., L, N, M]
            (L being the number of latent GPs = number of output dimensions)
        """
        raise NotImplementedError

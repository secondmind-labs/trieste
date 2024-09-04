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
from typing import Any, Optional

import gpflow
import tensorflow as tf
from gpflow.models import GPModel
from gpflow.posteriors import BasePosterior, PrecomputeCacheType
from typing_extensions import Protocol, final

from ... import logging
from ...data import Dataset
from ...space import EncoderFunction
from ...types import TensorType
from ..interfaces import (
    EncodedProbabilisticModel,
    EncodedSupportsPredictJoint,
    EncodedSupportsPredictY,
    EncodedTrainableProbabilisticModel,
    HasReparamSampler,
    ReparametrizationSampler,
    SupportsGetKernel,
    SupportsGetObservationNoise,
    SupportsPredictJoint,
)
from ..optimizer import Optimizer
from ..utils import (
    write_summary_data_based_metrics,
    write_summary_kernel_parameters,
    write_summary_likelihood_parameters,
)
from .sampler import BatchReparametrizationSampler


class GPflowPredictor(
    EncodedSupportsPredictJoint,
    SupportsGetKernel,
    SupportsGetObservationNoise,
    EncodedSupportsPredictY,
    HasReparamSampler,
    EncodedTrainableProbabilisticModel,
    EncodedProbabilisticModel,
    ABC,
):
    """A trainable wrapper for a GPflow Gaussian process model."""

    def __init__(self, optimizer: Optimizer | None = None, encoder: EncoderFunction | None = None):
        """
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        :param encoder: Optional encoder with which to transform query points before
            generating predictions.
        """
        if optimizer is None:
            optimizer = Optimizer(gpflow.optimizers.Scipy(), compile=True)

        self._optimizer = optimizer
        self._encoder = encoder
        self._posterior: Optional[BasePosterior] = None

    @property
    def encoder(self) -> EncoderFunction | None:
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: EncoderFunction | None) -> None:
        self._encoder = encoder

    @property
    def optimizer(self) -> Optimizer:
        """The optimizer with which to train the model."""
        return self._optimizer

    def create_posterior_cache(self) -> None:
        """
        Create a posterior cache for fast sequential predictions.  Note that this must happen
        at initialisation and *after* we ensure the model data is variable. Furthermore,
        the cache must be updated whenever the underlying model is changed.
        """
        self._posterior = self.model.posterior(PrecomputeCacheType.VARIABLE)

    def _ensure_variable_model_data(self) -> None:
        """Ensure GPflow data, which is normally stored in Tensors, is instead stored in
        dynamically shaped Variables. Override this as required."""

    def __setstate__(self, state: dict[str, Any]) -> None:
        # when unpickling we may need to regenerate the posterior cache
        self.__dict__.update(state)
        self._ensure_variable_model_data()
        if self._posterior is not None:
            self.create_posterior_cache()

    def update_posterior_cache(self) -> None:
        """Update the posterior cache. This needs to be called whenever the underlying model
        is changed."""
        if self._posterior is not None:
            self._posterior.update_cache()

    @property
    @abstractmethod
    def model(self) -> GPModel:
        """The underlying GPflow model."""

    def predict_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        mean, cov = (self._posterior or self.model).predict_f(query_points)
        # posterior predict can return negative variance values [cf GPFlow issue #1813]
        if self._posterior is not None:
            cov = tf.clip_by_value(cov, 1e-12, cov.dtype.max)
        return mean, cov

    def predict_joint_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        mean, cov = (self._posterior or self.model).predict_f(query_points, full_cov=True)
        # posterior predict can return negative variance values [cf GPFlow issue #1813]
        if self._posterior is not None:
            cov = tf.linalg.set_diag(
                cov, tf.clip_by_value(tf.linalg.diag_part(cov), 1e-12, cov.dtype.max)
            )
        return mean, cov

    def sample_encoded(self, query_points: TensorType, num_samples: int) -> TensorType:
        return self.model.predict_f_samples(query_points, num_samples)

    def predict_y_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.model.predict_y(query_points)

    def get_kernel(self) -> gpflow.kernels.Kernel:
        """
        Return the kernel of the model.

        :return: The kernel.
        """
        return self.model.kernel

    def get_mean_function(self) -> gpflow.mean_functions.MeanFunction:
        """
        Return the mean function of the model.

        :return: The mean function.
        """
        return self.model.mean_function

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

    def log(self, dataset: Optional[Dataset] = None) -> None:
        """
        Log model training information at a given optimization step to the Tensorboard.
        We log kernel and likelihood parameters. We also log several training data based metrics,
        such as root mean square error between predictions and observations and several others.

        :param dataset: Optional data that can be used to log additional data-based model summaries.
        """
        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                write_summary_kernel_parameters(self.get_kernel())
                write_summary_likelihood_parameters(self.model.likelihood)
                if dataset:
                    write_summary_data_based_metrics(
                        dataset=dataset, model=self, prefix="training_"
                    )

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


class EncodedSupportsCovarianceBetweenPoints(
    EncodedProbabilisticModel, SupportsCovarianceBetweenPoints
):
    @abstractmethod
    def covariance_between_points_encoded(
        self, query_points_1: TensorType, query_points_2: TensorType
    ) -> TensorType:
        """Implementation of covariance_between_points on encoded query points."""

    @final
    def covariance_between_points(
        self, query_points_1: TensorType, query_points_2: TensorType
    ) -> TensorType:
        return self.covariance_between_points_encoded(
            self.encode(query_points_1), self.encode(query_points_2)
        )

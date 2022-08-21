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

import tensorflow as tf
import tensorflow_probability as tfp
from typing_extensions import Protocol, runtime_checkable

from ...types import TensorType
from ..interfaces import ProbabilisticModel
from ..optimizer import KerasOptimizer
from ... import logging


class KerasPredictor(ProbabilisticModel, ABC):
    """
    This is an interface for trainable wrappers of TensorFlow and Keras neural network models.
    """

    def __init__(self, optimizer: Optional[KerasOptimizer] = None):
        """
        :param optimizer: The optimizer wrapper containing the optimizer with which to train the
            model and arguments for the wrapper and the optimizer. The optimizer must
            be an instance of a :class:`~tf.optimizers.Optimizer`. Defaults to
            :class:`~tf.optimizers.Adam` optimizer with default parameters.
        :raise ValueError: If the optimizer is not an instance of :class:`~tf.optimizers.Optimizer`.
        """
        if optimizer is None:
            optimizer = KerasOptimizer(tf.optimizers.Adam())
        self._optimizer = optimizer

        if not isinstance(optimizer.optimizer, tf.optimizers.Optimizer):
            raise ValueError(
                f"Optimizer for `KerasPredictor` models must be an instance of a "
                f"`tf.optimizers.Optimizer`, received {type(optimizer.optimizer)} instead."
            )

    @property
    @abstractmethod
    def model(self) -> tf.keras.Model:
        """The compiled Keras model."""
        raise NotImplementedError

    @property
    def optimizer(self) -> KerasOptimizer:
        """The optimizer wrapper for training the model."""
        return self._optimizer

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.model.predict(query_points)

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        raise NotImplementedError(
            """
            KerasPredictor does not implement sampling. Acquisition
            functions relying on it cannot be used with this class by default. Certain
            types of neural networks might be able to generate samples and
            such subclasses should overwrite this method.
            """
        )

    def log(self, dataset: Optional[Dataset] = None) -> None:
        """
        Log basic model training information at a given optimization step. Note that epoch
        losses and metrics cannot be recorded exactly, only summary statistics. Best approximation
        available is to log histograms of epoch losses and metrics.

        :param dataset: Optional data that can be used to log additional data-based model summaries.
        """
        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                logging.scalar("num_epochs", len(self.model.history.epoch))
                for k, v in self.model.history.history.items():
                    logging.histogram(f"epoch_{k}", v)
                    logging.scalar(f"{k}_final", v[-1])
                    logging.scalar(f"{k}_diff", v[0] - v[-1])
                    logging.scalar(f"{k}_min", tf.reduce_min(v))
                    logging.scalar(f"{k}_max", tf.reduce_max(v))
                if dataset:
                    empirical_variance = tf.reduce_variance(dataset.observations)
                    predict_variance = model.predict(dataset.query_points)[1]
                    breakpoint()
                    logging.histogram("predict_variance", predict_variance)
                    logging.scalar("model_variance_mean", tf.reduce_mean(predict_variance))
                    logging.scalar("empirical_variance", empirical_variance)


@runtime_checkable
class DeepEnsembleModel(ProbabilisticModel, Protocol):
    """
    This is an interface for deep ensemble type of model, primarily for usage by trajectory
    samplers, to avoid circular imports. These models can act as probabilistic models
    by deriving estimates of epistemic uncertainty from the diversity of predictions made by
    individual models in the ensemble.
    """

    @property
    @abstractmethod
    def ensemble_size(self) -> int:
        """
        Returns the size of the ensemble, that is, the number of base learners or individual
        models in the ensemble.
        """
        raise NotImplementedError

    @abstractmethod
    def ensemble_distributions(
        self, query_points: TensorType
    ) -> tuple[tfp.distributions.Distribution, ...]:
        """
        Return distributions for each member of the ensemble. Type of the output will depend on the
        subclass, it might be a predicted value or a distribution.

        :param query_points: The points at which to return outputs.
        :return: The outputs for the observations at the specified ``query_points`` for each member
            of the ensemble.
        """
        raise NotImplementedError

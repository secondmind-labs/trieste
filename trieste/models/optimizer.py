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
r""" This module contains model optimizers. """
from __future__ import annotations

from dataclasses import dataclass, field
from functools import singledispatch
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import gpflow
import scipy
import tensorflow as tf
from gpflow.models import ExternalDataTrainingLossMixin, InternalDataTrainingLossMixin

from ..data import Dataset
from ..type import TensorType
from ..utils import jit

TrainingData = Union[Tuple[TensorType, TensorType], Iterable[Tuple[TensorType, TensorType]]]
""" Type alias for a batch, or batches, of training data. """

DatasetTransformer = Callable[[Dataset, Optional[int]], TrainingData]
"""
Type alias for a function that converts a :class:`~trieste.data.Dataset` to batches of training
data.
"""

LossClosure = Callable[[], TensorType]
""" Type alias for a loss closure, typically used in optimization. """

OptimizeResult = Union[scipy.optimize.OptimizeResult, None]
"""
Optimization result. For scipy optimizer it is :class:`~scipy.optimize.OptimizeResult`.
TensorFlow optimizer doesn't return any result.
"""


@dataclass
class Optimizer:
    """Optimizer for training models with all the training data at once."""

    optimizer: gpflow.optimizers.Scipy | tf.optimizers.Optimizer
    """ The underlying optimizer to use. """

    minimize_args: dict[str, Any] = field(default_factory=lambda: {})
    """ The keyword arguments to pass to the :meth:`minimize` method of the :attr:`optimizer`. """

    compile: bool = False
    """ If `True`, the optimization process will be compiled with :func:`tf.function`. """

    def create_loss(self, model: tf.Module, dataset: Dataset) -> LossClosure:
        """
        Build a loss function for the specified `model` with the `dataset`.

        :param model: The model to build a loss function for.
        :param dataset: The data with which to build the loss function.
        :return: The loss function.
        """
        x = tf.convert_to_tensor(dataset.query_points)
        y = tf.convert_to_tensor(dataset.observations)
        data = (x, y)
        return create_loss_function(model, data, self.compile)

    def optimize(self, model: tf.Module, dataset: Dataset) -> OptimizeResult:
        """
        Optimize the specified `model` with the `dataset`.

        :param model: The model to optimize.
        :param dataset: The data with which to optimize the `model`.
        :return: The return value of the optimizer's :meth:`minimize` method.
        """
        loss_fn = self.create_loss(model, dataset)
        variables = model.trainable_variables
        return self.optimizer.minimize(loss_fn, variables, **self.minimize_args)


@dataclass
class TFOptimizer(Optimizer):
    """Optimizer for training models with mini-batches of training data."""

    max_iter: int = 100
    """ The number of iterations over which to optimize the model. """

    batch_size: int | None = None
    """ The size of the mini-batches. """

    dataset_builder: DatasetTransformer | None = None
    """ A mapping from :class:`~trieste.observer.Observer` data to mini-batches. """

    def create_loss(self, model: tf.Module, dataset: Dataset) -> LossClosure:
        def creator_fn(data: TrainingData) -> LossClosure:
            return create_loss_function(model, data, self.compile)

        if self.dataset_builder is None and self.batch_size is None:
            x = tf.convert_to_tensor(dataset.query_points)
            y = tf.convert_to_tensor(dataset.observations)
            return creator_fn((x, y))
        elif self.dataset_builder is None:
            return creator_fn(
                iter(
                    tf.data.Dataset.from_tensor_slices(dataset.astuple())
                    .shuffle(len(dataset))
                    .batch(self.batch_size)
                    .prefetch(tf.data.experimental.AUTOTUNE)
                    .repeat()
                )
            )

        return creator_fn(self.dataset_builder(dataset, self.batch_size))

    def optimize(self, model: tf.Module, dataset: Dataset) -> None:
        """
        Optimize the specified `model` with the `dataset`.

        :param model: The model to optimize.
        :param dataset: The data with which to optimize the `model`.
        """
        loss_fn = self.create_loss(model, dataset)
        variables = model.trainable_variables

        @jit(apply=self.compile)
        def train_fn() -> None:
            self.optimizer.minimize(loss_fn, variables, **self.minimize_args)

        for _ in range(self.max_iter):
            train_fn()


@singledispatch
def create_optimizer(
    optimizer: gpflow.optimizers.Scipy | tf.optimizers.Optimizer,
    optimizer_args: Dict[str, Any],
) -> Optimizer:
    """
    Generic function for creating a :class:`Optimizer` wrapper from a specified
    `optimizer` and `optimizer_args`. The implementations depends on the type of the
    underlying optimizer.

    :param optimizer: The optimizer with which to train the model.
    :param optimizer_args: The keyword arguments to pass to the optimizer wrapper..
    :return: The :class:`Optimizer` wrapper.
    """


@create_optimizer.register
def _create_tf_optimizer(
    optimizer: tf.optimizers.Optimizer,
    optimizer_args: Dict[str, Any],
) -> Optimizer:
    return TFOptimizer(optimizer, **optimizer_args)


@create_optimizer.register
def _create_scipy_optimizer(
    optimizer: gpflow.optimizers.Scipy,
    optimizer_args: Dict[str, Any],
) -> Optimizer:
    return Optimizer(optimizer, **optimizer_args)


@singledispatch
def create_loss_function(model, dataset: TrainingData, compile: bool = False) -> LossClosure:
    """
    Generic function for building a loss function for a specified `model` and `dataset`.
    The implementations depends on the type of the model.

    :param model: The model to build a loss function for.
    :param dataset: The data with which to build the loss function.
    :param compile: Whether to compile with :func:`tf.function`.
    :return: The loss function.
    """
    raise NotImplementedError(f"Unknown model {model} passed for loss function extraction")


@create_loss_function.register
def _create_loss_function_internal(
    model: InternalDataTrainingLossMixin,
    data: TrainingData,
    compile: bool = False,
) -> LossClosure:
    return model.training_loss_closure(compile=compile)


@create_loss_function.register
def _create_loss_function_external(
    model: ExternalDataTrainingLossMixin,
    data: TrainingData,
    compile: bool = False,
) -> LossClosure:
    return model.training_loss_closure(data, compile=compile)

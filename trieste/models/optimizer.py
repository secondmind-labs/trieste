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
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import gpflow
import scipy
import tensorflow as tf
from gpflow.models import ExternalDataTrainingLossMixin, InternalDataTrainingLossMixin

from ..data import Dataset
from ..utils import jit

Batches = Union[Tuple[tf.Tensor, tf.Tensor], Iterable[Tuple[tf.Tensor, tf.Tensor]]]
""" Type alias for a batch, or batches, of training data. """

DatasetTransformer = Callable[[Dataset, Optional[int]], Batches]
"""
Type alias for a function that converts a :class:`~trieste.data.Dataset` to batches of training
data.
"""

LossClosure = Callable[[], tf.Tensor]
""" Type alias for a loss closure, typically used in optimization. """

OptimizeResult = Union[scipy.optimize.OptimizeResult, None]
"""
Optimization result. For scipy optimizer it is :class:`~scipy.optimize.OptimizeResult`.
TensorFlow optimizer doesn't return any result.
"""


@dataclass
class Optimizer:
    """ Optimizer for training models with all the training data at once. """

    optimizer: Union[gpflow.optimizers.Scipy, tf.optimizers.Optimizer]
    """ The underlying optimizer to use. """

    minimize_kwargs: Dict[str, Any] = field(default_factory=lambda: {})
    """ The keyword arguments to pass to the :meth:`minimize` method of the :attr:`optimizer`. """

    compile: bool = False
    """ If `True`, the optimization process will be compiled with :func:`tf.function`. """

    def create_loss(self, model: tf.Module, dataset: Dataset) -> LossClosure:
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
        return self.optimizer.minimize(loss_fn, variables, **self.minimize_kwargs)


@dataclass
class TFOptimizer(Optimizer):
    """ Optimizer for training models with mini-batches of training data. """

    max_iter: int = 100
    """ The number of iterations over which to optimize the model. """

    batch_size: Optional[int] = None
    """ The size of the mini-batches. """

    dataset_builder: Optional[DatasetTransformer] = None
    """ A mapping from `~trieste.observer.Observer` data to mini-batches. """

    def create_loss(self, model: tf.Module, dataset: Dataset) -> LossClosure:
        def creator_fn(data: Batches) -> LossClosure:
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
            self.optimizer.minimize(loss_fn, variables, **self.minimize_kwargs)

        for _ in range(self.max_iter):
            train_fn()


@singledispatch
def create_optimizer(
    optimizer: Union[gpflow.optimizers.Scipy, tf.optimizers.Optimizer],
    optimizer_kwargs: Dict[str, Any]
) -> Optimizer:
    pass


@create_optimizer.register
def _create_tf_optimizer(
    optimizer: tf.optimizers.Optimizer,
    optimizer_kwargs: Dict[str, Any],
) -> Optimizer:
    return TFOptimizer(optimizer, **optimizer_kwargs)


@create_optimizer.register
def _create_scipy_optimizer(
    optimizer: gpflow.optimizers.Scipy,
    optimizer_kwargs: Dict[str, Any],
) -> Optimizer:
    return Optimizer(optimizer, **optimizer_kwargs)


@singledispatch
def create_loss_function(model, dataset: Batches, compile: bool = False) -> LossClosure:
    raise NotImplementedError(f"Unknown model {model} passed for loss function extraction")


@create_loss_function.register
def _create_loss_function_internal(
    model: InternalDataTrainingLossMixin,
    data: Batches,
    compile: bool = False,
) -> LossClosure:
    return model.training_loss_closure(compile=compile)


@create_loss_function.register
def _create_loss_function_external(
    model: ExternalDataTrainingLossMixin,
    data: Batches,
    compile: bool = False,
) -> LossClosure:
    return model.training_loss_closure(data, compile=compile)

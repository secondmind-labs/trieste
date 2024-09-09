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

r"""
This module contains common optimizers based on :class:`~tf.optimizers.Optimizer` that can be used
with models. Specific models can also sub-class these optimizers or implement their own, and should
register their loss functions using a :func:`create_loss_function`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import scipy
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.keras import tf_keras

from ..data import Dataset
from ..types import TensorType
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
Optimization result. TensorFlow optimizer doesn't return any result. For scipy optimizer that is
also commonly used, it is :class:`~scipy.optimize.OptimizeResult`.
"""


@dataclass
class Optimizer:
    """Optimizer for training models with all the training data at once."""

    optimizer: Any
    """
    The underlying optimizer to use. For example, one of the subclasses of
    :class:`~tensorflow.optimizers.Optimizer` could be used. Note that we use a flexible type `Any`
    to allow for various optimizers that specific models might need to use.
    """

    minimize_args: dict[str, Any] = field(default_factory=lambda: {})
    """ The keyword arguments to pass to the :meth:`minimize` method of the :attr:`optimizer`. """

    compile: bool = False
    """ If `True`, the optimization process will be compiled with :func:`~tf.function`. """

    def create_loss(self, model: tf.Module, dataset: Dataset) -> LossClosure:
        """
        Build a loss function for the specified `model` with the `dataset` using a
        :func:`create_loss_function`.

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
class BatchOptimizer(Optimizer):
    """Optimizer for training models with mini-batches of training data."""

    max_iter: int = 100
    """ The number of iterations over which to optimize the model. """

    batch_size: int = 100
    """ The size of the mini-batches. """

    dataset_builder: DatasetTransformer | None = None
    """ A mapping from :class:`~trieste.observer.Observer` data to mini-batches. """

    def create_loss(self, model: tf.Module, dataset: Dataset) -> LossClosure:
        """
        Build a loss function for the specified `model` with the `dataset`.

        :param model: The model to build a loss function for.
        :param dataset: The data with which to build the loss function.
        :return: The loss function.
        """

        def creator_fn(data: TrainingData) -> LossClosure:
            return create_loss_function(model, data, self.compile)

        if self.dataset_builder is None:
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

    def __deepcopy__(self, memo: dict[int, object]) -> BatchOptimizer:
        # workaround for https://github.com/tensorflow/tensorflow/issues/58973
        # (keras optimizers not being deepcopyable in TF 2.11 and 2.12)
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if (
                k == "optimizer"
                and isinstance(v, tf_keras.optimizers.Optimizer)
                and hasattr(v, "_distribution_strategy")
            ):
                # avoid copying distribution strategy: reuse it instead
                strategy = v._distribution_strategy
                v._distribution_strategy = None
                try:
                    setattr(result, k, copy.deepcopy(v, memo))
                finally:
                    v._distribution_strategy = strategy
                result.optimizer._distribution_strategy = strategy
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


@dataclass
class KerasOptimizer:
    """Optimizer wrapper for training models implemented with Keras."""

    optimizer: tf_keras.optimizers.Optimizer
    """ The underlying optimizer to use for training the model. """

    fit_args: dict[str, Any] = field(default_factory=lambda: {})
    """
    The keyword arguments to pass to the ``fit`` method of a :class:`~tf.keras.Model` instance.
    See https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
    arguments in the dictionary.
    """

    loss: Optional[
        Union[
            tf_keras.losses.Loss, Callable[[TensorType, tfp.distributions.Distribution], TensorType]
        ]
    ] = None
    """ Optional loss function for training the model. """

    metrics: Optional[list[tf_keras.metrics.Metric]] = None
    """ Optional metrics for monitoring the performance of the network. """

    def __deepcopy__(self, memo: dict[int, object]) -> KerasOptimizer:
        # workaround for https://github.com/tensorflow/tensorflow/issues/58973
        # (keras optimizers not being deepcopyable in TF 2.11 and 2.12)
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "optimizer" and hasattr(v, "_distribution_strategy"):
                # avoid copying distribution strategy: reuse it instead
                strategy = v._distribution_strategy
                v._distribution_strategy = None
                try:
                    setattr(result, k, copy.deepcopy(v, memo))
                finally:
                    v._distribution_strategy = strategy
                result.optimizer._distribution_strategy = strategy
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


@singledispatch
def create_loss_function(model: Any, dataset: TrainingData, compile: bool = False) -> LossClosure:
    """
    Generic function for building a loss function for a specified `model` and `dataset`.
    The implementations depends on the type of the model, which should use this function as a
    decorator together with its register method to make a model-specific loss function available.

    :param model: The model to build a loss function for.
    :param dataset: The data with which to build the loss function.
    :param compile: Whether to compile with :func:`tf.function`.
    :return: The loss function.
    """
    raise NotImplementedError(f"Unknown model {model} passed for loss function extraction")

from typing import Any, Dict, Optional, Callable, Union, Iterable, Tuple
from functools import singledispatch
from dataclasses import dataclass, field
import tensorflow as tf
import gpflow
import scipy

from gpflow.models import ExternalDataTrainingLossMixin, InternalDataTrainingLossMixin
from ..utils import jit
from ..data import Dataset


DatasetTransformer = Callable[[Dataset, Optional[int]], Union[Iterable, Tuple]]
"""
Type alias for a function that maps a dataset from a :class:`~optiflow.datasets.Dataset`
and generates an iterator over the transformation.
"""

LossClosure = Callable[[], tf.Tensor]
"""
Type alias for a loss closure that is used generally for optimizing.
"""

OptimizeResult = Union[scipy.optimize.OptimizeResult, None]
"""
Optimization result. For scipy optimizer it is :class:`~scipy.optimize.OptmizeResult`.
TensorFlow opitimizer doesn't return any result.
"""


@dataclass
class Optimizer:
    optimizer: Union[gpflow.optimizers.Scipy, tf.optimizers.Optimizer]
    minimize_args: Dict[str, Any] = field(default_factory=lambda: {})
    compile: bool = False

    def create_loss(self, model: tf.Module, dataset: Dataset) -> LossClosure:
        x = tf.convert_to_tensor(dataset.query_points)
        y = tf.convert_to_tensor(dataset.observations)
        data = (x, y)
        return create_loss_function(model, data, self.compile)

    def optimize(self, model: tf.Module, dataset: Dataset) -> OptimizeResult:
        loss_fn = self.create_loss(model, dataset)
        variables = model.trainable_variables
        return self.optimizer.minimize(loss_fn, variables, **self.minimize_args)


@dataclass
class TFOptimizer(Optimizer):
    max_iter: int = 100
    batch_size: Optional[int] = None
    dataset_builder: Optional[DatasetTransformer] = None

    def create_loss(self, model: tf.Module, dataset: Dataset) -> LossClosure:
        def creator_fn(data: Union[Tuple, Iterable]):
            return create_loss_function(model, data, self.compile)

        if self.dataset_builder is None and self.batch_size is None:
            x = tf.convert_to_tensor(dataset.query_points)
            y = tf.convert_to_tensor(dataset.observations)
            return creator_fn((x, y))
        elif self.dataset_builder is None:
            d = tf.data.Dataset.from_tensor_slices((dataset.query_points, dataset.observations))
            size = len(dataset)
            data = iter(
                d.shuffle(size)
                .batch(self.batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
                .repeat()
            )
            return creator_fn(data)

        data = self.dataset_builder(dataset, self.batch_size)
        return creator_fn(data)

    def optimize(self, model: tf.Module, dataset: Dataset) -> None:
        loss_fn = self.create_loss(model, dataset)
        variables = model.trainable_variables

        @jit(apply=self.compile)
        def train_fn():
            self.optimizer.minimize(loss_fn, variables, **self.minimize_args)

        for _ in range(self.max_iter):
            train_fn()


@singledispatch
def create_optimizer(optimizer, optimizer_args: Dict[str, Any]) -> Optimizer:
    pass


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


Data = Union[Tuple[tf.Tensor, tf.Tensor], Iterable[Tuple[tf.Tensor, tf.Tensor]]]


@singledispatch
def create_loss_function(model, dataset: Data, compile: Optional[bool] = False) -> LossClosure:
    pass


@create_loss_function.register
def _create_loss_function_internal(
    model: InternalDataTrainingLossMixin,
    data: Data,
    compile: Optional[bool] = False,
) -> LossClosure:
    return model.training_loss_closure(compile=compile)


@create_loss_function.register
def _create_loss_function_external(
    model: ExternalDataTrainingLossMixin,
    data: Data,
    compile: Optional[bool] = False,
) -> LossClosure:
    return model.training_loss_closure(data, compile=compile)

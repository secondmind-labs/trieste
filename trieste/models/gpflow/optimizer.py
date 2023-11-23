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
This module registers the GPflow specific loss functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import tensorflow as tf
from gpflow.models import ExternalDataTrainingLossMixin, InternalDataTrainingLossMixin
from tensorflow.python.data.ops.iterator_ops import OwnedIterator as DatasetOwnedIterator

from gpflow.base import Module
from gpflow.optimizers import Scipy
from .utils import get_parameter_bounds
from ..optimizer import LossClosure, Optimizer, TrainingData, create_loss_function
from ...data import Dataset


@dataclass
class ScipyOptimizer(Optimizer):
    optimizer: Any = field(default_factory=Scipy)
    minimize_args: dict[str, Any] = field(default_factory=lambda: {})
    compile: bool = True

    def optimize(self, model: Module, dataset: Dataset) -> None:
        # TODO: Check for edge-cases. For example, if Scipy filters out unused
        #  variables then `bounds` may be indexed incorrectly...
        loss_fn = self.create_loss(model, dataset)
        bounds = []
        variables = []
        for param in model.trainable_parameters:
            bound = get_parameter_bounds(param, unconstrained=True)
            bounds.append(tf.transpose(tf.reshape(bound, (2, -1))))
            variables.append(param.unconstrained_variable)

        for var in model.trainable_variables:
            if var not in variables:
                variables.append(var)
                if bounds:
                    bound = tf.tile(
                        tf.cast([[-float("inf")], [float("inf")]], var.dtype),
                        (tf.size(var), 1)
                    )
                    bounds.append(bound)
                    variables.append(var)

        return self.optimizer.minimize(
            closure=loss_fn,
            variables=variables,
            bounds=tf.concat(bounds, axis=0) if bounds else None,  # n x 2
            **self.minimize_args,
        )


@create_loss_function.register
def _create_loss_function_internal(
    model: InternalDataTrainingLossMixin,
    data: TrainingData,
    compile: bool = False,
) -> LossClosure:
    return model.training_loss_closure(compile=compile)


class _TrainingLossClosureBuilder:
    # A cached, compiled training loss closure builder to avoid having to generate a new
    # closure each time. Stored in a separate class, so we can avoid pickling it.

    def __init__(self) -> None:
        self.closure_builder: Optional[Callable[[TrainingData], LossClosure]] = None

    def __getstate__(self) -> dict[str, Any]:
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.closure_builder = None


@create_loss_function.register
def _create_loss_function_external(
    model: ExternalDataTrainingLossMixin,
    data: TrainingData,
    compile: bool = False,
) -> LossClosure:
    if not compile:
        return model.training_loss_closure(data, compile=False)

    # when compiling, we want to avoid generating a new closure every optimization step
    # instead we compile and save a single function that can handle the dynamic data shape
    X, Y = next(data) if isinstance(data, DatasetOwnedIterator) else data

    if not hasattr(model, "_training_loss_closure_builder"):
        setattr(model, "_training_loss_closure_builder", _TrainingLossClosureBuilder())

    builder: _TrainingLossClosureBuilder = getattr(model, "_training_loss_closure_builder")
    if builder.closure_builder is None:
        shape_spec = (
            data.element_spec
            if isinstance(data, DatasetOwnedIterator)
            else (
                tf.TensorSpec([None, *X.shape[1:]], dtype=X.dtype),
                tf.TensorSpec([None, *Y.shape[1:]], dtype=Y.dtype),
            )
        )

        @tf.function(input_signature=shape_spec)
        def training_loss_builder(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            return model.training_loss((x, y))

        def closure_builder(data: TrainingData) -> LossClosure:
            x, y = next(data) if isinstance(data, DatasetOwnedIterator) else data

            def compiled_closure() -> tf.Tensor:
                return training_loss_builder(x, y)

            return compiled_closure

        builder.closure_builder = closure_builder

    return builder.closure_builder((X, Y))

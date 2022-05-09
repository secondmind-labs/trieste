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

import tensorflow as tf
from gpflow.models import ExternalDataTrainingLossMixin, InternalDataTrainingLossMixin

from ..optimizer import LossClosure, TrainingData, create_loss_function


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

    if not compile:

        def closure() -> tf.Tensor:
            return model.training_loss(data)

        return closure

    if not hasattr(model, "compiled_training_loss_closure_builder"):
        X, Y = data
        shape_spec = (
            tf.TensorSpec([None, *X.shape[1:]], dtype=X.dtype),
            tf.TensorSpec([None, *Y.shape[1:]], dtype=Y.dtype),
        )

        def training_loss_builder(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            return tf.function(model.training_loss((x, y)), input_signature=shape_spec)

        def closure_builder(data: TrainingData) -> LossClosure:
            x, y = data

            def compiled_closure() -> tf.Tensor:
                return training_loss_builder(x, y)

            return compiled_closure

        setattr(model, "compiled_training_loss_closure_builder", closure_builder)

    return getattr(model, "compiled_training_loss_closure_builder")(data)

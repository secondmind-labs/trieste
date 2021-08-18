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

import tensorflow as tf
from gpflux.models import DeepGP
from tensorflow.python.data.ops.iterator_ops import OwnedIterator as DatasetOwnedIterator

from ..optimizer import create_loss_function, LossClosure, TrainingData


@create_loss_function.register
def _create_loss_function_gpflux(
    model: DeepGP,
    data: TrainingData,
    compile: bool = False,
) -> LossClosure:
    elbo = model.elbo

    if isinstance(data, DatasetOwnedIterator):
        if compile:
            input_signature = [data.element_spec]
            elbo = tf.function(elbo, input_signature=input_signature)

        def closure():
            batch = next(data)
            return -elbo(batch)  # type: ignore

    else:

        def closure():
            return -elbo(data)

        if compile:
            closure = tf.function(closure)

    return closure

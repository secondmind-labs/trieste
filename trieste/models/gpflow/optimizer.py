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
    return model.training_loss_closure(data, compile=compile)

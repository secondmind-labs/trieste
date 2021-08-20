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

from typing import Any, Dict

import gpflow
from gpflow.models import ExternalDataTrainingLossMixin, InternalDataTrainingLossMixin

from ..optimizer import LossClosure, Optimizer, TrainingData, create_loss_function, create_optimizer


@create_optimizer.register
def _create_scipy_optimizer(
    optimizer: gpflow.optimizers.Scipy,
    optimizer_args: Dict[str, Any],
) -> Optimizer:
    return Optimizer(optimizer, **optimizer_args)


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

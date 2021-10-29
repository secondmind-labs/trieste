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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import tensorflow as tf
from gpflux.models import DeepGP, DeepIWP

from ..config import ModelConfig
from ..interfaces import TrainableProbabilisticModel
from .models import DeepGaussianProcess, DeepKernelProcess


@dataclass(frozen=True)
class GPfluxModelConfig(ModelConfig):
    """
    Specification for building a GPflux instance of
    :class:`~trieste.models.TrainableProbabilisticModel`. Note that `optimizer_args` are not used
    for GPflux models.
    """

    def supported_models(
        self,
    ) -> dict[Any, Callable[[Any, tf.optimizers.Optimizer], TrainableProbabilisticModel]]:
        models_mapping: dict[
            Any, Callable[[Any, tf.optimizers.Optimizer], TrainableProbabilisticModel]
        ] = {
            DeepGP: DeepGaussianProcess,
            DeepIWP: DeepKernelProcess
        }
        return models_mapping

    def create_optimizer(self) -> tf.optimizers.Optimizer:
        return self.optimizer

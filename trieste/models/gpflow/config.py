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

import gpflow
from gpflow.models import GPR, SGPR, VGP

from ..config import ModelConfig
from ..interfaces import TrainableProbabilisticModel
from ..optimizer import Optimizer
from .models import GaussianProcessRegression, VariationalGaussianProcess


@dataclass(frozen=True)
class GPflowModelConfig(ModelConfig):
    """
    Specification for building a GPflow instance of
    :class:`~trieste.models.TrainableProbabilisticModel`.
    """

    def default_optimizer(self) -> Callable:
        return gpflow.optimizers.Scipy()

    def supported_models(
        self,
    ) -> dict[Any, Callable[[Any, Optimizer], TrainableProbabilisticModel]]:
        models_mapping = {
            GPR: GaussianProcessRegression,
            SGPR: GaussianProcessRegression,
            VGP: VariationalGaussianProcess,
        }
        return models_mapping

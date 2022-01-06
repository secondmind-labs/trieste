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

from typing import Any, Dict, Type

from gpflux.models import DeepGP

from ..config import ModelRegistry
from ..interfaces import TrainableProbabilisticModel
from .models import DeepGaussianProcess

# Here we list all the GPflux models currently supported by model interfaces
# and optimizers, and register them for usage with ModelConfig.
_SUPPORTED_MODELS: Dict[Type[Any], Type[TrainableProbabilisticModel]] = {
    DeepGP: DeepGaussianProcess,
}
for model_type, model_wrapper in _SUPPORTED_MODELS.items():
    ModelRegistry.register_model(model_type, model_wrapper)

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
"""
This package contains the primary interface for probabilistic models,
:class:`ProbabilisticModel`, and its trainable counterpart :class:`TrainableProbabilisticModel`. It
also contains a number of :class:`TrainableProbabilisticModel` wrappers for GPflow models, as well
as tooling for creating :class:`TrainableProbabilisticModel`\ s from config.
"""
from .config import ModelConfig, ModelSpec, create_model
from .model_interfaces import (
    ProbabilisticModel,
    TrainableProbabilisticModel,
    CustomTrainable,
    GPflowPredictor,
    GaussianProcessRegression,
    Batcher,
    SparseVariational,
    VariationalGaussianProcess,
    supported_models,
)

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
This package contains the primary interface for trainable probabilistic models,
:class:`ModelInterface`, as well as a number of implementations of :class:`ModelInterface` that wrap
GPflow models, and tooling for creating :class:`ModelInterface`\ s from config.
"""
from .config import ModelConfig, ModelSpec, create_model_interface
from .model_interfaces import (
    ModelInterface,
    TrainableModelInterface,
    GPflowPredictor,
    GaussianProcessRegression,
    Batcher,
    SparseVariational,
    VariationalGaussianProcess,
    supported_models,
)

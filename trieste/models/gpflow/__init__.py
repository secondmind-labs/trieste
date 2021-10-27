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

r"""
This package contains the primary interface for Gaussian process models. It also contains a
number of :class:`TrainableProbabilisticModel` wrappers for GPflow-based models.
"""

from . import config, optimizer
from .interface import GPflowPredictor
from .models import GaussianProcessRegression, SparseVariational, VariationalGaussianProcess
from .utils import M, assert_data_is_compatible, randomize_hyperparameters, squeeze_hyperparameters

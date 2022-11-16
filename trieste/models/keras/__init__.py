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

"""
This package contains the primary interface for deep neural network models. It also contains a
number of :class:`TrainableProbabilisticModel` wrappers for neural network models. Note that
currently copying/saving models is not supported, so when
:class:`~trieste.bayesian_optimizer.BayesianOptimizer` is used ``track_state`` should be set
to `False`.
"""

from .architectures import GaussianNetwork, KerasEnsemble, KerasEnsembleNetwork
from .builders import build_keras_ensemble
from .interface import DeepEnsembleModel, KerasPredictor
from .models import DeepEnsemble
from .sampler import DeepEnsembleTrajectorySampler, deep_ensemble_trajectory
from .utils import get_tensor_spec_from_data, negative_log_likelihood, sample_with_replacement

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
This package contains the primary interface for deep Gaussian process models. It also contains a
number of :class:`TrainableProbabilisticModel` wrappers for GPflux-based models. Note that currently
copying/saving models is not supported, so in a Bayes Opt loop `track_state` should be set False.
Note as well that `tf.keras.backend.set_floatx()` should be used to set the desired float type,
consistent with the GPflow float type being used.
"""

from . import config
from .architectures import build_vanilla_deep_gp
from .interface import GPfluxPredictor
from .models import DeepGaussianProcess

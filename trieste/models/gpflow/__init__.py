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

from . import optimizer
from .builders import (
    build_gpr,
    build_multifidelity_autoregressive_models,
    build_multifidelity_nonlinear_autoregressive_models,
    build_sgpr,
    build_svgp,
    build_vgp_classifier,
)
from .inducing_point_selectors import (
    ConditionalImprovementReduction,
    ConditionalVarianceReduction,
    InducingPointSelector,
    KMeansInducingPointSelector,
    RandomSubSampleInducingPointSelector,
    UniformInducingPointSelector,
)
from .interface import GPflowPredictor
from .models import (
    GaussianProcessRegression,
    MultifidelityAutoregressive,
    MultifidelityNonlinearAutoregressive,
    SparseGaussianProcessRegression,
    SparseVariational,
    VariationalGaussianProcess,
)
from .sampler import (
    BatchReparametrizationSampler,
    DecoupledTrajectorySampler,
    IndependentReparametrizationSampler,
    RandomFourierFeatureTrajectorySampler,
    feature_decomposition_trajectory,
)
from .utils import (
    assert_data_is_compatible,
    check_optimizer,
    randomize_hyperparameters,
    squeeze_hyperparameters,
)

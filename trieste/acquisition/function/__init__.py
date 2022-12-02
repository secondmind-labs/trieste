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
""" This folder contains single-objective optimization functions. """
from .active_learning import (
    BayesianActiveLearningByDisagreement,
    ExpectedFeasibility,
    IntegratedVarianceReduction,
    PredictiveVariance,
    bayesian_active_learning_by_disagreement,
    bichon_ranjan_criterion,
    integrated_variance_reduction,
    predictive_variance,
)
from .continuous_thompson_sampling import (
    GreedyContinuousThompsonSampling,
    ParallelContinuousThompsonSampling,
)
from .entropy import (
    GIBBON,
    MinValueEntropySearch,
    gibbon_quality_term,
    gibbon_repulsion_term,
    min_value_entropy_search,
)
from .function import (
    AugmentedExpectedImprovement,
    BatchExpectedImprovement,
    BatchMonteCarloExpectedImprovement,
    ExpectedConstrainedImprovement,
    ExpectedImprovement,
    FastConstraintsFeasibility,
    MakePositive,
    MonteCarloAugmentedExpectedImprovement,
    MonteCarloExpectedImprovement,
    MultipleOptimismNegativeLowerConfidenceBound,
    NegativeLowerConfidenceBound,
    NegativePredictiveMean,
    ProbabilityOfFeasibility,
    ProbabilityOfImprovement,
    augmented_expected_improvement,
    batch_expected_improvement,
    expected_improvement,
    fast_constraints_feasibility,
    lower_confidence_bound,
    multiple_optimism_lower_confidence_bound,
    probability_below_threshold,
)
from .greedy_batch import Fantasizer, LocalPenalization, hard_local_penalizer, soft_local_penalizer
from .multi_objective import (
    HIPPO,
    BatchMonteCarloExpectedHypervolumeImprovement,
    ExpectedConstrainedHypervolumeImprovement,
    ExpectedHypervolumeImprovement,
    batch_ehvi,
    expected_hv_improvement,
)
from .utils import MultivariateNormalCDF

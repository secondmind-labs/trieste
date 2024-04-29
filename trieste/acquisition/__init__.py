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
r"""
The acquisition process aims to find the optimal point(s) at which to next evaluate the objective
function, with the aim of minimising it. The functionality in this package implements that process.
It typically uses the current observations of the objective function, or a posterior over those
observations.

In this library, the acquisition rule is the central object of the API, while acquisition functions
are supplementary. This is because some acquisition rules, such as Thompson sampling,
do not require an acquisition function. This contrasts with other libraries which may consider
the acquisition function as the central component of this process and assume Efficient Global
Optimization (EGO) for the acquisition rule.

This package contains acquisition rules, as implementations of
:class:`~trieste.acquisition.rule.AcquisitionRule`, and acquisition functions. It also contains
:class:`AcquisitionFunctionBuilder`\ s which provide a common interface for the rules to build
acquisition functions.

Acquisition rules in this library that use acquisition functions, such as
:class:`EfficientGlobalOptimization`, *maximize* these functions. This defines the sign the
acquisition function should take. Additionally, acquisition functions and builders in this library
are designed to minimize the objective function. For example, we do not provide an implementation of
UCB.
"""
from . import optimizer, rule
from .combination import Product, Reducer, Sum
from .function import (
    GIBBON,
    HIPPO,
    AugmentedExpectedImprovement,
    BatchExpectedImprovement,
    BatchMonteCarloExpectedHypervolumeImprovement,
    BatchMonteCarloExpectedImprovement,
    BayesianActiveLearningByDisagreement,
    ExpectedConstrainedHypervolumeImprovement,
    ExpectedConstrainedImprovement,
    ExpectedFeasibility,
    ExpectedHypervolumeImprovement,
    ExpectedImprovement,
    Fantasizer,
    FastConstraintsFeasibility,
    GreedyContinuousThompsonSampling,
    IntegratedVarianceReduction,
    LocalPenalization,
    MakePositive,
    MinValueEntropySearch,
    MonteCarloAugmentedExpectedImprovement,
    MonteCarloExpectedImprovement,
    MultipleOptimismNegativeLowerConfidenceBound,
    NegativeLowerConfidenceBound,
    NegativePredictiveMean,
    ParallelContinuousThompsonSampling,
    PredictiveVariance,
    ProbabilityOfFeasibility,
    augmented_expected_improvement,
    batch_ehvi,
    batch_expected_improvement,
    bayesian_active_learning_by_disagreement,
    bichon_ranjan_criterion,
    expected_hv_improvement,
    expected_improvement,
    fast_constraints_feasibility,
    gibbon_quality_term,
    gibbon_repulsion_term,
    hard_local_penalizer,
    integrated_variance_reduction,
    lower_confidence_bound,
    min_value_entropy_search,
    multiple_optimism_lower_confidence_bound,
    predictive_variance,
    probability_below_threshold,
    soft_local_penalizer,
)
from .interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    AcquisitionFunctionClass,
    GreedyAcquisitionFunctionBuilder,
    PenalizationFunction,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
    SingleModelVectorizedAcquisitionBuilder,
    UpdatablePenalizationFunction,
    VectorizedAcquisitionFunctionBuilder,
)
from .rule import (
    AcquisitionRule,
    AsynchronousGreedy,
    AsynchronousOptimization,
    BatchHypervolumeSharpeRatioIndicator,
    BatchTrustRegion,
    BatchTrustRegionBox,
    BatchTrustRegionProduct,
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
    FixedPointTrustRegionDiscrete,
    HypercubeTrustRegion,
    LocalDatasetsAcquisitionRule,
    RandomSampling,
    SingleObjectiveTrustRegionBox,
    SingleObjectiveTrustRegionDiscrete,
    TREGOBox,
    TURBOBox,
    UpdatableSearchSpace,
    UpdatableTrustRegion,
    UpdatableTrustRegionBox,
    UpdatableTrustRegionDiscrete,
    UpdatableTrustRegionProduct,
)
from .sampler import (
    ExactThompsonSampler,
    GumbelSampler,
    ThompsonSampler,
    ThompsonSamplerFromTrajectory,
)

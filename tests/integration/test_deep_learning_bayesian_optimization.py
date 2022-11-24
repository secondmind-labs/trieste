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

import pytest

from tests.integration.test_bayesian_optimization import _test_optimizer_finds_minimum
from tests.util.misc import random_seed
from trieste.acquisition import (
    GreedyContinuousThompsonSampling,
    MonteCarloAugmentedExpectedImprovement,
    MonteCarloExpectedImprovement,
    ParallelContinuousThompsonSampling,
)
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.rule import (
    AcquisitionRule,
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
)
from trieste.acquisition.sampler import ThompsonSamplerFromTrajectory
from trieste.models.gpflux import DeepGaussianProcess
from trieste.models.keras import DeepEnsemble
from trieste.space import SearchSpace
from trieste.types import TensorType


@random_seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        pytest.param(25, DiscreteThompsonSampling(1000, 8), id="DiscreteThompsonSampling"),
        pytest.param(
            25,
            EfficientGlobalOptimization(
                ParallelContinuousThompsonSampling(),
                num_query_points=3,
            ),
            id="ParallelContinuousThompsonSampling",
        ),
        pytest.param(
            12,
            EfficientGlobalOptimization(
                GreedyContinuousThompsonSampling(),
                num_query_points=4,
            ),
            id="GreedyContinuousThompsonSampling",
            marks=pytest.mark.skip(reason="too fragile"),
        ),
    ],
)
def test_bayesian_optimizer_with_dgp_finds_minima_of_scaled_branin(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, DeepGaussianProcess],
) -> None:
    _test_optimizer_finds_minimum(
        DeepGaussianProcess, num_steps, acquisition_rule, optimize_branin=True
    )


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        pytest.param(5, DiscreteThompsonSampling(1000, 1), id="DiscreteThompsonSampling"),
        pytest.param(
            5,
            EfficientGlobalOptimization(
                MonteCarloExpectedImprovement(int(1e2)), generate_continuous_optimizer(100)
            ),
            id="MonteCarloExpectedImprovement",
        ),
        pytest.param(
            5,
            EfficientGlobalOptimization(
                MonteCarloAugmentedExpectedImprovement(int(1e2)), generate_continuous_optimizer(100)
            ),
            id="MonteCarloAugmentedExpectedImprovement",
        ),
        pytest.param(
            2,
            EfficientGlobalOptimization(
                ParallelContinuousThompsonSampling(),
                num_query_points=5,
            ),
            id="ParallelContinuousThompsonSampling",
        ),
        pytest.param(
            2,
            EfficientGlobalOptimization(
                GreedyContinuousThompsonSampling(),
                num_query_points=5,
            ),
            id="GreedyContinuousThompsonSampling",
        ),
    ],
)
def test_bayesian_optimizer_with_dgp_finds_minima_of_simple_quadratic(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, DeepGaussianProcess],
) -> None:
    _test_optimizer_finds_minimum(DeepGaussianProcess, num_steps, acquisition_rule)


@random_seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        pytest.param(
            60,
            EfficientGlobalOptimization(),
            id="EfficientGlobalOptimization",
            marks=pytest.mark.skip(reason="too fragile"),
        ),
        pytest.param(
            30,
            EfficientGlobalOptimization(
                ParallelContinuousThompsonSampling(),
                num_query_points=4,
            ),
            id="ParallelContinuousThompsonSampling",
        ),
    ],
)
def test_bayesian_optimizer_with_deep_ensemble_finds_minima_of_scaled_branin(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, DeepEnsemble],
) -> None:
    _test_optimizer_finds_minimum(
        DeepEnsemble,
        num_steps,
        acquisition_rule,
        optimize_branin=True,
        model_args={"bootstrap": True, "diversify": False},
    )


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        pytest.param(5, EfficientGlobalOptimization(), id="EfficientGlobalOptimization"),
        pytest.param(10, DiscreteThompsonSampling(1000, 1), id="DiscreteThompsonSampling"),
        pytest.param(
            5,
            DiscreteThompsonSampling(1000, 1, thompson_sampler=ThompsonSamplerFromTrajectory()),
            id="DiscreteThompsonSampling/ThompsonSamplerFromTrajectory",
        ),
    ],
)
def test_bayesian_optimizer_with_deep_ensemble_finds_minima_of_simple_quadratic(
    num_steps: int, acquisition_rule: AcquisitionRule[TensorType, SearchSpace, DeepEnsemble]
) -> None:
    _test_optimizer_finds_minimum(
        DeepEnsemble,
        num_steps,
        acquisition_rule,
    )


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        pytest.param(
            5,
            EfficientGlobalOptimization(
                ParallelContinuousThompsonSampling(),
                num_query_points=3,
            ),
            id="ParallelContinuousThompsonSampling",
        ),
    ],
)
def test_bayesian_optimizer_with_PCTS_and_deep_ensemble_finds_minima_of_simple_quadratic(
    num_steps: int, acquisition_rule: AcquisitionRule[TensorType, SearchSpace, DeepEnsemble]
) -> None:
    _test_optimizer_finds_minimum(
        DeepEnsemble,
        num_steps,
        acquisition_rule,
        model_args={"diversify": False},
    )
    _test_optimizer_finds_minimum(
        DeepEnsemble,
        num_steps,
        acquisition_rule,
        model_args={"diversify": True},
    )

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

import dataclasses
from typing import cast

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import random_seed
from trieste.acquisition import (
    AcquisitionFunctionClass,
    BatchMonteCarloExpectedImprovement,
    LocalPenalization,
    ParallelContinuousThompsonSampling,
)
from trieste.acquisition.rule import (
    AcquisitionRule,
    BatchTrustRegionProduct,
    EfficientGlobalOptimization,
    FixedPointTrustRegionDiscrete,
    SingleObjectiveTrustRegionBox,
    SingleObjectiveTrustRegionDiscrete,
    UpdatableTrustRegionProduct,
)
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.models import TrainableProbabilisticModel
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.objectives import ScaledBranin, SingleObjectiveTestProblem
from trieste.objectives.single_objectives import scaled_branin
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import (
    Box,
    CategoricalSearchSpace,
    DiscreteSearchSpace,
    EncoderFunction,
    TaggedProductSearchSpace,
    one_hot_encoder,
)
from trieste.types import TensorType


def _get_mixed_search_space() -> TaggedProductSearchSpace:
    # The discrete space is defined by a set of 10 points that are equally spaced, ensuring that
    # the three Branin minimizers (of dimension 0) are included in this set. The continuous
    # dimension is defined by the interval [0, 1].
    # We observe that the first and third minimizers are equidistant from the middle minimizer, so
    # we choose the discretization points to be equally spaced around the middle minimizer.
    minimizers0 = ScaledBranin.minimizers[:, 0]
    step = (minimizers0[1] - minimizers0[0]) / 4
    points = np.concatenate(
        [
            # Equally spaced points to the left of the middle minimizer. Skip the last point as it
            # is the same as the first point in the next array.
            np.flip(np.arange(minimizers0[1], 0.0, -step))[:-1],
            # Equally spaced points to the right of the middle minimizer.
            np.arange(minimizers0[1], 1.0, step),
        ]
    )
    discrete_space = DiscreteSearchSpace(points[:, None])
    continuous_space = Box([0], [1])
    return TaggedProductSearchSpace(
        spaces=[discrete_space, continuous_space],
        tags=["discrete", "continuous"],
    )


mixed_search_space = _get_mixed_search_space()


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        pytest.param(25, EfficientGlobalOptimization(), id="EfficientGlobalOptimization"),
        pytest.param(
            5,
            EfficientGlobalOptimization(
                BatchMonteCarloExpectedImprovement(sample_size=500).using(OBJECTIVE),
                num_query_points=3,
            ),
            id="BatchMonteCarloExpectedImprovement",
        ),
        pytest.param(
            8,
            EfficientGlobalOptimization(
                LocalPenalization(
                    ScaledBranin.search_space,
                ).using(OBJECTIVE),
                num_query_points=3,
            ),
            id="LocalPenalization",
        ),
        pytest.param(
            8,
            BatchTrustRegionProduct(
                [
                    UpdatableTrustRegionProduct(
                        [
                            FixedPointTrustRegionDiscrete(
                                cast(
                                    DiscreteSearchSpace, mixed_search_space.get_subspace("discrete")
                                )
                            ),
                            SingleObjectiveTrustRegionBox(
                                cast(Box, mixed_search_space.get_subspace("continuous"))
                            ),
                        ],
                        tags=mixed_search_space.subspace_tags,
                    )
                    for _ in range(10)
                ],
                EfficientGlobalOptimization(
                    ParallelContinuousThompsonSampling(),
                    # Use a large batch to ensure discrete init finds a good point.
                    # We are using a fixed point trust region for the discrete space, so
                    # the init point is randomly chosen and then never updated.
                    num_query_points=10,
                ),
            ),
            id="TrustRegionSingleObjectiveFixed",
        ),
        pytest.param(
            8,
            BatchTrustRegionProduct(
                [
                    UpdatableTrustRegionProduct(
                        [
                            SingleObjectiveTrustRegionDiscrete(
                                cast(
                                    DiscreteSearchSpace, mixed_search_space.get_subspace("discrete")
                                )
                            ),
                            SingleObjectiveTrustRegionBox(
                                cast(Box, mixed_search_space.get_subspace("continuous"))
                            ),
                        ],
                        tags=mixed_search_space.subspace_tags,
                    )
                    for _ in range(3)
                ],
                EfficientGlobalOptimization(
                    ParallelContinuousThompsonSampling(),
                    num_query_points=3,
                ),
            ),
            id="TrustRegionSingleObjective",
        ),
    ],
)
def test_optimizer_finds_minima_of_the_scaled_branin_function(
    num_steps: int,
    acquisition_rule: AcquisitionRule[
        TensorType, TaggedProductSearchSpace, TrainableProbabilisticModel
    ],
) -> None:
    mixed_branin = cast(SingleObjectiveTestProblem[TaggedProductSearchSpace], ScaledBranin)
    _test_optimizer_finds_problem_minima(
        dataclasses.replace(mixed_branin, search_space=mixed_search_space),
        num_steps,
        acquisition_rule,
    )


def _test_optimizer_finds_problem_minima(
    problem: SingleObjectiveTestProblem[TaggedProductSearchSpace],
    num_steps: int,
    acquisition_rule: AcquisitionRule[
        TensorType, TaggedProductSearchSpace, TrainableProbabilisticModel
    ],
    encoder: EncoderFunction | None = None,
) -> None:
    initial_query_points = problem.search_space.sample(5)
    observer = mk_observer(problem.objective)
    initial_data = observer(initial_query_points)
    model = GaussianProcessRegression(
        build_gpr(initial_data, problem.search_space, likelihood_variance=1e-8),
        encoder=encoder,
    )

    dataset = (
        BayesianOptimizer(observer, problem.search_space)
        .optimize(num_steps, initial_data, model, acquisition_rule)
        .try_get_final_dataset()
    )

    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]

    relative_minimizer_err = tf.abs((best_x - problem.minimizers) / problem.minimizers)
    # these accuracies are the current best for the given number of optimization steps, which makes
    # this is a regression test
    assert tf.reduce_any(tf.reduce_all(relative_minimizer_err < 0.1, axis=-1), axis=0)
    npt.assert_allclose(best_y, ScaledBranin.minimum, rtol=0.005)

    # check that acquisition functions defined as classes aren't being retraced unnecessarily
    # They should be retraced once for the optimzier's starting grid and once for L-BFGS.
    if isinstance(acquisition_rule, EfficientGlobalOptimization):
        acquisition_function = acquisition_rule._acquisition_function
        if isinstance(acquisition_function, AcquisitionFunctionClass):
            assert acquisition_function.__call__._get_tracing_count() <= 4  # type: ignore


def categorical_scaled_branin(
    categories_to_points: TensorType,
) -> SingleObjectiveTestProblem[TaggedProductSearchSpace]:
    """
    Generate a Scaled Branin test problem defined on the product of a categorical space and a
    continuous space, with categories mapped to points using the given 1D tensor.
    """
    categorical_space = CategoricalSearchSpace([str(float(v)) for v in categories_to_points])
    continuous_space = Box([0], [1])
    search_space = TaggedProductSearchSpace(
        spaces=[categorical_space, continuous_space],
        tags=["categorical", "continuous"],
    )

    def objective(x: TensorType) -> TensorType:
        points = tf.gather(categories_to_points, tf.cast(x[..., 0], tf.int32))
        x_mapped = tf.concat([tf.expand_dims(points, -1), x[..., 1:]], axis=-1)
        return scaled_branin(x_mapped)

    minimizer_indices = []
    for minimizer0 in ScaledBranin.minimizers[..., 0]:
        indices = tf.where(tf.equal(categories_to_points, minimizer0))
        minimizer_indices.append(indices[0][0])
    category_indices = tf.expand_dims(tf.convert_to_tensor(minimizer_indices, dtype=tf.float64), -1)
    minimizers = tf.concat([category_indices, ScaledBranin.minimizers[..., 1:]], axis=-1)

    return SingleObjectiveTestProblem(
        name="Categorical scaled Branin",
        objective=objective,
        search_space=search_space,
        minimizers=minimizers,
        minimum=ScaledBranin.minimum,
    )


@random_seed
def _get_categorical_problem() -> SingleObjectiveTestProblem[TaggedProductSearchSpace]:
    # a categorical scaled branin problem with 6 categories mapping to 3 random points
    # plus the 3 minimizer points (to guarantee that the minimum is present)
    points = tf.concat(
        [tf.random.uniform([3], dtype=tf.float64), ScaledBranin.minimizers[..., 0]], 0
    )
    return categorical_scaled_branin(tf.random.shuffle(points))


cat_problem = _get_categorical_problem()


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        pytest.param(25, EfficientGlobalOptimization(), id="EfficientGlobalOptimization"),
        pytest.param(
            8,
            BatchTrustRegionProduct(
                [
                    UpdatableTrustRegionProduct(
                        [
                            SingleObjectiveTrustRegionDiscrete(
                                cast(
                                    CategoricalSearchSpace,
                                    cat_problem.search_space.get_subspace("categorical"),
                                )
                            ),
                            SingleObjectiveTrustRegionBox(
                                cast(Box, cat_problem.search_space.get_subspace("continuous"))
                            ),
                        ],
                        tags=cat_problem.search_space.subspace_tags,
                    )
                    for _ in range(3)
                ],
                EfficientGlobalOptimization(
                    ParallelContinuousThompsonSampling(),
                    num_query_points=3,
                ),
            ),
            id="TrustRegionSingleObjective",
        ),
    ],
)
def test_optimizer_finds_minima_of_the_categorical_scaled_branin_function(
    num_steps: int,
    acquisition_rule: AcquisitionRule[
        TensorType, TaggedProductSearchSpace, TrainableProbabilisticModel
    ],
) -> None:
    # model uses one-hot encoding for the categorical inputs
    _test_optimizer_finds_problem_minima(
        cat_problem,
        num_steps,
        acquisition_rule,
        encoder=one_hot_encoder(cat_problem.search_space),
    )

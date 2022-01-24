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

import tempfile
from typing import Any, List, Mapping, Optional, Tuple, Union

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import random_seed
from tests.util.models.gpflux.models import two_layer_dgp_model
from trieste.acquisition import (
    GIBBON,
    AcquisitionFunctionClass,
    AugmentedExpectedImprovement,
    BatchMonteCarloExpectedImprovement,
    Fantasizer,
    LocalPenalization,
    MinValueEntropySearch,
    MultipleOptimismNegativeLowerConfidenceBound,
)
from trieste.acquisition.rule import (
    AcquisitionRule,
    AsynchronousGreedy,
    AsynchronousOptimization,
    AsynchronousRuleState,
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
    TrustRegion,
)
from trieste.acquisition.sampler import ThompsonSamplerFromTrajectory
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.logging import tensorboard_writer
from trieste.models.gpflow import (
    GaussianProcessRegression,
    GPflowPredictor,
    SparseVariational,
    VariationalGaussianProcess,
)
from trieste.models.gpflux import DeepGaussianProcess, GPfluxPredictor
from trieste.models.optimizer import BatchOptimizer
from trieste.objectives import (
    BRANIN_MINIMIZERS,
    BRANIN_SEARCH_SPACE,
    MICHALEWICZ_2_MINIMIZER,
    MICHALEWICZ_2_MINIMUM,
    SCALED_BRANIN_MINIMUM,
    SIMPLE_QUADRATIC_MINIMIZER,
    SIMPLE_QUADRATIC_MINIMUM,
    michalewicz,
    scaled_branin,
    simple_quadratic,
)
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box, SearchSpace
from trieste.types import State, TensorType


# Optimizer parameters for testing against the branin function.
# We also use these for a quicker test against a simple quadratic function
# (regenerating is necessary as some of the acquisition rules are stateful).
def OPTIMIZER_PARAMS() -> Tuple[
    str,
    List[
        Tuple[
            int,
            Union[
                AcquisitionRule[TensorType, Box, GPflowPredictor],
                AcquisitionRule[
                    State[
                        TensorType,
                        Union[AsynchronousRuleState, TrustRegion.State],
                    ],
                    Box,
                    GPflowPredictor,
                ],
            ],
        ]
    ],
]:
    return (
        "num_steps, acquisition_rule",
        [
            (20, EfficientGlobalOptimization()),
            (25, EfficientGlobalOptimization(AugmentedExpectedImprovement().using(OBJECTIVE))),
            (
                22,
                EfficientGlobalOptimization(
                    MinValueEntropySearch(
                        BRANIN_SEARCH_SPACE,
                        min_value_sampler=ThompsonSamplerFromTrajectory(sample_min_value=True),
                    ).using(OBJECTIVE)
                ),
            ),
            (
                12,
                EfficientGlobalOptimization(
                    BatchMonteCarloExpectedImprovement(sample_size=500).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
            (12, AsynchronousOptimization(num_query_points=3)),
            (
                10,
                EfficientGlobalOptimization(
                    LocalPenalization(
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
            (
                10,
                AsynchronousGreedy(
                    LocalPenalization(
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
            (
                10,
                EfficientGlobalOptimization(
                    GIBBON(  # type: ignore[arg-type]  # (only supported by GPR models)
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=2,
                ),
            ),
            (
                10,
                EfficientGlobalOptimization(
                    MultipleOptimismNegativeLowerConfidenceBound(
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=3,
                ),
            ),
            (15, TrustRegion()),
            (
                15,
                TrustRegion(
                    EfficientGlobalOptimization(
                        MinValueEntropySearch(
                            BRANIN_SEARCH_SPACE,
                        ).using(OBJECTIVE)
                    )
                ),
            ),
            (10, DiscreteThompsonSampling(500, 3)),
            (
                15,
                DiscreteThompsonSampling(500, 3, thompson_sampler=ThompsonSamplerFromTrajectory()),
            ),
            (
                15,
                EfficientGlobalOptimization(
                    Fantasizer(),  # type: ignore[arg-type]  # (only supported by GPR models)
                    num_query_points=3,
                ),
            ),
        ],
    )


@random_seed
@pytest.mark.slow  # to run this, add --runslow yes to the pytest command
@pytest.mark.parametrize(*OPTIMIZER_PARAMS())
def test_optimizer_finds_minima_of_the_scaled_branin_function(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, GPflowPredictor]
    | AcquisitionRule[
        State[TensorType, AsynchronousRuleState | TrustRegion.State], Box, GPflowPredictor
    ],
) -> None:
    _test_optimizer_finds_minimum(num_steps, acquisition_rule, optimize_branin=True)


@random_seed
@pytest.mark.parametrize(*OPTIMIZER_PARAMS())
def test_optimizer_finds_minima_of_simple_quadratic(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, GPflowPredictor]
    | AcquisitionRule[
        State[TensorType, AsynchronousRuleState | TrustRegion.State], Box, GPflowPredictor
    ],
) -> None:
    # for speed reasons we sometimes test with a simple quadratic defined on the same search space
    # branin; currently assume that every rule should be able to solve this in 5 steps
    _test_optimizer_finds_minimum(min(num_steps, 5), acquisition_rule)


@random_seed
@pytest.mark.parametrize("use_natgrads", [False, True])
def test_optimizer_with_vgp_model(use_natgrads: bool) -> None:
    # regression test for [#406]; use natgrads doesn't work well as a model for the objective
    # so don't bother checking the results, just that it doesn't crash
    acquisition_rule: AcquisitionRule[
        TensorType, SearchSpace, GPflowPredictor
    ] = EfficientGlobalOptimization()
    _test_optimizer_finds_minimum(
        None if use_natgrads else 5,
        acquisition_rule,
        model_type="VGP",
        model_args={"use_natgrads": use_natgrads},
    )


@random_seed
def test_optimizer_with_svgp_model() -> None:
    acquisition_rule: AcquisitionRule[
        TensorType, SearchSpace, GPflowPredictor
    ] = EfficientGlobalOptimization()
    _test_optimizer_finds_minimum(5, acquisition_rule, model_type="SVGP")


def _test_optimizer_finds_minimum(
    num_steps: Optional[int],
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, GPflowPredictor]
    | AcquisitionRule[
        State[TensorType, AsynchronousRuleState | TrustRegion.State], Box, GPflowPredictor
    ],
    optimize_branin: bool = False,
    model_type: str = "GPR",  # in Python 3.8+ this could be Literal["GPR", "VGP", "SVGP"]
    model_args: Optional[Mapping[str, Any]] = None,
) -> None:

    model_args = model_args or {}
    search_space = BRANIN_SEARCH_SPACE

    def build_model(data: Dataset) -> GPflowPredictor:
        assert model_args is not None

        variance = tf.math.reduce_variance(data.observations)
        kernel = gpflow.kernels.Matern52(variance, tf.constant([0.2, 0.2], tf.float64))
        scale = tf.constant(1.0, dtype=tf.float64)
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.constant(-2.0, dtype=tf.float64), scale
        )
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(kernel.lengthscales), scale
        )

        if model_type == "GPR":
            gpr = gpflow.models.GPR(
                (data.query_points, data.observations), kernel, noise_variance=1e-5
            )
            gpflow.utilities.set_trainable(gpr.likelihood, False)
            return GaussianProcessRegression(gpr, **model_args)
        elif model_type == "VGP":
            likelihood = gpflow.likelihoods.Gaussian(1e-3)
            vgp = gpflow.models.VGP(initial_data.astuple(), kernel, likelihood)
            gpflow.utilities.set_trainable(vgp.likelihood, False)
            return VariationalGaussianProcess(vgp, **model_args)
        elif model_type == "SVGP":
            Z = search_space.sample_sobol(20)  # Initialize diverse inducing locations
            svgp = gpflow.models.SVGP(
                kernel,
                gpflow.likelihoods.Gaussian(variance=1e-5),
                Z,
                num_data=len(data.observations),
            )
            gpflow.utilities.set_trainable(svgp.likelihood, False)
            return SparseVariational(svgp, BatchOptimizer(tf.optimizers.Adam(0.1)))
        else:
            raise ValueError(f"Unsupported model_type '{model_type}'")

    initial_query_points = search_space.sample(5)
    observer = mk_observer(scaled_branin if optimize_branin else simple_quadratic)
    initial_data = observer(initial_query_points)
    model = build_model(initial_data)

    with tempfile.TemporaryDirectory() as tmpdirname:
        summary_writer = tf.summary.create_file_writer(tmpdirname)
        with tensorboard_writer(summary_writer):

            dataset = (
                BayesianOptimizer(observer, search_space)  # type: ignore
                .optimize(num_steps or 2, initial_data, model, acquisition_rule)
                .try_get_final_dataset()
            )

            arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

            best_y = dataset.observations[arg_min_idx]
            best_x = dataset.query_points[arg_min_idx]

            if num_steps is None:
                # this test is just being run to check for crashes, not performance
                pass
            elif optimize_branin:
                relative_minimizer_err = tf.abs((best_x - BRANIN_MINIMIZERS) / BRANIN_MINIMIZERS)
                # these accuracies are the current best for the given number of optimization
                # steps, which makes this is a regression test
                assert tf.reduce_any(tf.reduce_all(relative_minimizer_err < 0.05, axis=-1), axis=0)
                npt.assert_allclose(best_y, SCALED_BRANIN_MINIMUM, rtol=0.005)
            else:
                absolute_minimizer_err = tf.abs(best_x - SIMPLE_QUADRATIC_MINIMIZER)
                assert tf.reduce_any(tf.reduce_all(absolute_minimizer_err < 0.05, axis=-1), axis=0)
                npt.assert_allclose(best_y, SIMPLE_QUADRATIC_MINIMUM, rtol=0.05)

            # check that acquisition functions defined as classes aren't retraced unnecessarily
            # They should be retraced once for the optimzier's starting grid, L-BFGS, and logging.
            if isinstance(acquisition_rule, EfficientGlobalOptimization):
                acq_function = acquisition_rule._acquisition_function
                if isinstance(acq_function, AcquisitionFunctionClass):
                    assert acq_function.__call__._get_tracing_count() == 3  # type: ignore


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        (1, DiscreteThompsonSampling(1000, 50)),
    ],
)
def test_two_layer_dgp_optimizer_finds_minima_of_michalewicz_function(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, GPfluxPredictor],
    keras_float: None,
) -> None:

    # this unit test fails sometimes for
    # normal search space used with MICHALEWICZ function
    # so for stability we reduce its size here
    search_space = Box(MICHALEWICZ_2_MINIMIZER[0] - 0.5, MICHALEWICZ_2_MINIMIZER[0] + 0.5)

    def build_model(data: Dataset) -> DeepGaussianProcess:
        epochs = int(4e2)
        batch_size = 100

        dgp = two_layer_dgp_model(data.query_points)

        def scheduler(epoch: int, lr: float) -> float:
            if epoch == epochs // 2:
                return lr * 0.1
            else:
                return lr

        fit_args = {
            "batch_size": batch_size,
            "epochs": epochs,
            "verbose": 0,
            "callbacks": tf.keras.callbacks.LearningRateScheduler(scheduler),
        }
        optimizer = BatchOptimizer(tf.optimizers.Adam(0.01), fit_args)

        return DeepGaussianProcess(model=dgp, optimizer=optimizer)

    initial_query_points = search_space.sample_sobol(20)
    observer = mk_observer(michalewicz, OBJECTIVE)
    initial_data = observer(initial_query_points)
    model = build_model(initial_data[OBJECTIVE])
    dataset = (
        BayesianOptimizer(observer, search_space)
        .optimize(num_steps, initial_data, {OBJECTIVE: model}, acquisition_rule, track_state=False)
        .try_get_final_dataset()
    )
    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))

    best_y = dataset.observations[arg_min_idx]
    best_x = dataset.query_points[arg_min_idx]
    relative_minimizer_err = tf.abs((best_x - MICHALEWICZ_2_MINIMIZER) / MICHALEWICZ_2_MINIMIZER)

    assert tf.reduce_all(relative_minimizer_err < 0.03, axis=-1)
    npt.assert_allclose(best_y, MICHALEWICZ_2_MINIMUM, rtol=0.03)

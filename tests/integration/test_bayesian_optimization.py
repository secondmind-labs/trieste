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

from tests.util.misc import random_seed
from tests.util.models.gpflux.models import two_layer_dgp_model
from trieste.acquisition import (
    GIBBON,
    AcquisitionFunctionClass,
    AugmentedExpectedImprovement,
    BatchMonteCarloExpectedImprovement,
    Fantasizer,
    GreedyContinuousThompsonSampling,
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
from trieste.logging import tensorboard_writer
from trieste.models.gpflow import (
    GaussianProcessRegression,
    GPflowPredictor,
    SparseVariational,
    VariationalGaussianProcess,
    build_gpr,
    build_svgp,
)
from trieste.models.gpflux import DeepGaussianProcess
from trieste.models.keras import DeepEnsemble, build_vanilla_keras_ensemble, negative_log_likelihood
from trieste.models.optimizer import BatchOptimizer, KerasOptimizer
from trieste.objectives import (
    BRANIN_MINIMIZERS,
    BRANIN_SEARCH_SPACE,
    SCALED_BRANIN_MINIMUM,
    SIMPLE_QUADRATIC_MINIMIZER,
    SIMPLE_QUADRATIC_MINIMUM,
    SIMPLE_QUADRATIC_SEARCH_SPACE,
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
# The various   # type: ignore[arg-type]  are for rules that are only supported by GPR models.
def GPR_OPTIMIZER_PARAMS() -> Tuple[
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
            (30, EfficientGlobalOptimization(AugmentedExpectedImprovement().using(OBJECTIVE))),
            (
                24,
                EfficientGlobalOptimization(
                    MinValueEntropySearch(  # type: ignore[arg-type]
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
                    GIBBON(  # type: ignore[arg-type]
                        BRANIN_SEARCH_SPACE,
                    ).using(OBJECTIVE),
                    num_query_points=2,
                ),
            ),
            (
                20,
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
                20,
                DiscreteThompsonSampling(
                    500,
                    3,
                    thompson_sampler=ThompsonSamplerFromTrajectory(),  # type: ignore[arg-type]
                ),
            ),
            (
                15,
                EfficientGlobalOptimization(
                    Fantasizer(),  # type: ignore[arg-type]
                    num_query_points=3,
                ),
            ),
            (
                10,
                EfficientGlobalOptimization(
                    GreedyContinuousThompsonSampling(),  # type: ignore[arg-type]
                    num_query_points=5,
                ),
            ),
        ],
    )


@random_seed
@pytest.mark.slow  # to run this, add --runslow yes to the pytest command
@pytest.mark.parametrize(*GPR_OPTIMIZER_PARAMS())
def test_bayesian_optimizer_with_gpr_finds_minima_of_scaled_branin(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, GPflowPredictor]
    | AcquisitionRule[
        State[TensorType, AsynchronousRuleState | TrustRegion.State], Box, GPflowPredictor
    ],
) -> None:
    _test_optimizer_finds_minimum(num_steps, acquisition_rule, optimize_branin=True)


@random_seed
@pytest.mark.parametrize(*GPR_OPTIMIZER_PARAMS())
def test_bayesian_optimizer_with_gpr_finds_minima_of_simple_quadratic(
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
def test_bayesian_optimizer_with_vgp_finds_minima_of_simple_quadratic(use_natgrads: bool) -> None:
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
@pytest.mark.slow
def test_bayesian_optimizer_with_svgp_finds_minima_of_scaled_branin() -> None:
    acquisition_rule: AcquisitionRule[
        TensorType, SearchSpace, GPflowPredictor
    ] = EfficientGlobalOptimization()
    _test_optimizer_finds_minimum(
        50,
        acquisition_rule,
        optimize_branin=True,
        model_type="SVGP",
        model_args={"optimizer": BatchOptimizer(tf.optimizers.Adam(0.01))},
    )


@random_seed
def test_bayesian_optimizer_with_svgp_finds_minima_of_simple_quadratic() -> None:
    acquisition_rule: AcquisitionRule[
        TensorType, SearchSpace, GPflowPredictor
    ] = EfficientGlobalOptimization()
    _test_optimizer_finds_minimum(
        5,
        acquisition_rule,
        model_type="SVGP",
        model_args={"optimizer": BatchOptimizer(tf.optimizers.Adam(0.1))},
    )


@random_seed
@pytest.mark.slow
@pytest.mark.parametrize("num_steps, acquisition_rule", [(25, DiscreteThompsonSampling(1000, 8))])
def test_bayesian_optimizer_with_dgp_finds_minima_of_scaled_branin(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, GPflowPredictor],
    keras_float: None,
) -> None:
    _test_optimizer_finds_minimum(
        num_steps, acquisition_rule, optimize_branin=True, model_type="DGP"
    )


@random_seed
@pytest.mark.parametrize("num_steps, acquisition_rule", [(5, DiscreteThompsonSampling(1000, 1))])
def test_bayesian_optimizer_with_dgp_finds_minima_of_simple_quadratic(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, GPflowPredictor],
    keras_float: None,
) -> None:
    _test_optimizer_finds_minimum(num_steps, acquisition_rule, model_type="DGP")


@random_seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        (90, EfficientGlobalOptimization()),
        (30, DiscreteThompsonSampling(500, 3)),
        (
            30,
            DiscreteThompsonSampling(1000, 3, thompson_sampler=ThompsonSamplerFromTrajectory()),
        ),
    ],
)
def test_bayesian_optimizer_with_deep_ensemble_finds_minima_of_scaled_branin(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, GPflowPredictor],
) -> None:
    _test_optimizer_finds_minimum(
        num_steps,
        acquisition_rule,
        optimize_branin=True,
        model_type="DE",
        model_args={"bootstrap": True},
    )


@random_seed
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        (5, EfficientGlobalOptimization()),
        (5, DiscreteThompsonSampling(500, 1)),
        (
            5,
            DiscreteThompsonSampling(500, 1, thompson_sampler=ThompsonSamplerFromTrajectory()),
        ),
    ],
)
def test_bayesian_optimizer_with_deep_ensemble_finds_minima_of_simple_quadratic(
    num_steps: int, acquisition_rule: AcquisitionRule[TensorType, SearchSpace, GPflowPredictor]
) -> None:
    _test_optimizer_finds_minimum(num_steps, acquisition_rule, model_type="DE")


def _test_optimizer_finds_minimum(
    num_steps: Optional[int],
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, GPflowPredictor]
    | AcquisitionRule[
        State[TensorType, AsynchronousRuleState | TrustRegion.State], Box, GPflowPredictor
    ],
    optimize_branin: bool = False,
    model_type: str = "GPR",  # in Python 3.8+ a Literal["GPR", "VGP", "SVGP", "DGP", "DE"]?
    model_args: Optional[Mapping[str, Any]] = None,
) -> None:
    model_args = model_args or {}
    track_state = True

    if optimize_branin:
        search_space = BRANIN_SEARCH_SPACE
        minimizers = BRANIN_MINIMIZERS
        minima = SCALED_BRANIN_MINIMUM
        rtol_level = 0.005
        num_initial_query_points = 5
    else:
        search_space = SIMPLE_QUADRATIC_SEARCH_SPACE
        minimizers = SIMPLE_QUADRATIC_MINIMIZER
        minima = SIMPLE_QUADRATIC_MINIMUM
        rtol_level = 0.05
        num_initial_query_points = 10
    if model_type in ["SVGP", "DGP", "DE"]:
        num_initial_query_points = 20

    initial_query_points = search_space.sample(num_initial_query_points)
    observer = mk_observer(scaled_branin if optimize_branin else simple_quadratic)
    initial_data = observer(initial_query_points)

    if model_type == "GPR":
        if "LocalPenalization" in acquisition_rule.__repr__():
            likelihood_variance = 1e-3
        else:
            likelihood_variance = 1e-5
        gpr = build_gpr(initial_data, search_space, likelihood_variance=likelihood_variance)
        model = GaussianProcessRegression(gpr, **model_args)

    elif model_type == "VGP":
        empirical_variance = tf.math.reduce_variance(initial_data.observations)
        kernel = gpflow.kernels.Matern52(variance=empirical_variance, lengthscales=[0.2, 0.2])
        likelihood = gpflow.likelihoods.Gaussian(1e-3)
        vgp = gpflow.models.VGP(initial_data.astuple(), kernel, likelihood)
        gpflow.utilities.set_trainable(vgp.likelihood, False)
        model = VariationalGaussianProcess(vgp, **model_args)  # type: ignore

    elif model_type == "SVGP":
        gpr = build_svgp(initial_data, search_space)
        model = SparseVariational(gpr, **model_args)  # type: ignore

    elif model_type == "DGP":
        track_state = False
        epochs = 400
        batch_size = 100
        dgp = two_layer_dgp_model(initial_data.query_points)

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
        dgp_optimizer = BatchOptimizer(tf.optimizers.Adam(0.01), fit_args)
        model = DeepGaussianProcess(dgp, dgp_optimizer)  # type: ignore

    elif model_type == "DE":
        track_state = False
        keras_ensemble = build_vanilla_keras_ensemble(initial_data, 5, 3, 25)
        fit_args = {
            "batch_size": 20,
            "epochs": 1000,
            "callbacks": [
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss", patience=25, restore_best_weights=True
                )
            ],
            "verbose": 0,
        }
        de_optimizer = KerasOptimizer(
            tf.keras.optimizers.Adam(0.001), negative_log_likelihood, fit_args
        )
        model = DeepEnsemble(keras_ensemble, de_optimizer, **model_args)  # type: ignore

    else:
        raise ValueError(f"Unsupported model_type '{model_type}'")

    with tempfile.TemporaryDirectory() as tmpdirname:
        summary_writer = tf.summary.create_file_writer(tmpdirname)
        with tensorboard_writer(summary_writer):

            dataset = (
                BayesianOptimizer(observer, search_space)
                .optimize(
                    num_steps or 2, initial_data, model, acquisition_rule, track_state=track_state
                )
                .try_get_final_dataset()
            )

            arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
            best_y = dataset.observations[arg_min_idx]
            best_x = dataset.query_points[arg_min_idx]

            if num_steps is None:
                # this test is just being run to check for crashes, not performance
                pass
            else:
                minimizer_err = tf.abs((best_x - minimizers) / minimizers)
                # these accuracies are the current best for the given number of optimization
                # steps, which makes this is a regression test
                assert tf.reduce_any(tf.reduce_all(minimizer_err < 0.05, axis=-1), axis=0)
                npt.assert_allclose(best_y, minima, rtol=rtol_level)

            # check that acquisition functions defined as classes aren't retraced unnecessarily
            # They should be retraced once for the optimzier's starting grid, L-BFGS, and logging.
            if isinstance(acquisition_rule, EfficientGlobalOptimization):
                acq_function = acquisition_rule._acquisition_function
                if isinstance(acq_function, AcquisitionFunctionClass):
                    assert acq_function.__call__._get_tracing_count() == 3  # type: ignore

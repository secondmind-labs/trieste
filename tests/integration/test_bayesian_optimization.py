# Copyright 2021 The Trieste Contrib_fnutors
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
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple, Type, Union, cast
from unittest.mock import patch

import dill
import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
from _pytest.mark import ParameterSet
from gpflow.keras import tf_keras

from tests.util.misc import random_seed
from trieste.acquisition import (
    GIBBON,
    AcquisitionFunctionClass,
    AugmentedExpectedImprovement,
    BatchExpectedImprovement,
    BatchMonteCarloExpectedImprovement,
    Fantasizer,
    GreedyAcquisitionFunctionBuilder,
    GreedyContinuousThompsonSampling,
    LocalPenalization,
    MinValueEntropySearch,
    MonteCarloAugmentedExpectedImprovement,
    MonteCarloExpectedImprovement,
    MultipleOptimismNegativeLowerConfidenceBound,
    ParallelContinuousThompsonSampling,
)
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.rule import (
    AcquisitionRule,
    AsynchronousGreedy,
    AsynchronousOptimization,
    AsynchronousRuleState,
    BatchHypervolumeSharpeRatioIndicator,
    BatchTrustRegionBox,
    BatchTrustRegionState,
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
    SingleObjectiveTrustRegionBox,
    TREGOBox,
    TURBOBox,
    UpdatableTrustRegionBox,
)
from trieste.acquisition.sampler import ThompsonSamplerFromTrajectory
from trieste.acquisition.utils import copy_to_local_models
from trieste.bayesian_optimizer import (
    BayesianOptimizer,
    FrozenRecord,
    OptimizationResult,
    TrainableProbabilisticModelType,
    stop_at_minimum,
)
from trieste.logging import tensorboard_writer
from trieste.models import TrainableProbabilisticModel, TrajectoryFunctionClass
from trieste.models.gpflow import (
    ConditionalImprovementReduction,
    GaussianProcessRegression,
    GPflowPredictor,
    SparseGaussianProcessRegression,
    SparseVariational,
    VariationalGaussianProcess,
    build_gpr,
    build_sgpr,
    build_svgp,
)
from trieste.models.gpflux import DeepGaussianProcess, build_vanilla_deep_gp
from trieste.models.keras import DeepEnsemble, build_keras_ensemble
from trieste.models.optimizer import KerasOptimizer, Optimizer
from trieste.objectives import ScaledBranin, SimpleQuadratic
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box, SearchSpace
from trieste.types import State, TensorType

try:
    import pymoo
except ImportError:  # pragma: no cover (tested but not by coverage)
    pymoo = None


# Optimizer parameters for testing GPR against the branin function.
# We also use these for a quicker test against a simple quadratic function
# (regenerating is necessary as some of the acquisition rules are stateful).
def GPR_OPTIMIZER_PARAMS() -> Tuple[str, List[ParameterSet]]:
    return (
        "num_steps, acquisition_rule",
        [
            pytest.param(20, EfficientGlobalOptimization(), id="EfficientGlobalOptimization"),
            pytest.param(
                30,
                EfficientGlobalOptimization(AugmentedExpectedImprovement().using(OBJECTIVE)),
                id="AugmentedExpectedImprovement",
            ),
            pytest.param(
                20,
                EfficientGlobalOptimization(
                    MonteCarloExpectedImprovement(int(1e3)).using(OBJECTIVE),
                    generate_continuous_optimizer(100),
                ),
                id="MonteCarloExpectedImprovement",
            ),
            pytest.param(
                24,
                EfficientGlobalOptimization(
                    MinValueEntropySearch(
                        ScaledBranin.search_space,
                        min_value_sampler=ThompsonSamplerFromTrajectory(sample_min_value=True),
                    ).using(OBJECTIVE)
                ),
                id="MinValueEntropySearch",
            ),
            pytest.param(
                12,
                EfficientGlobalOptimization(
                    BatchExpectedImprovement(sample_size=100).using(OBJECTIVE),
                    num_query_points=3,
                ),
                id="BatchExpectedImprovement",
            ),
            pytest.param(
                12,
                EfficientGlobalOptimization(
                    BatchMonteCarloExpectedImprovement(sample_size=500).using(OBJECTIVE),
                    num_query_points=3,
                ),
                id="BatchMonteCarloExpectedImprovement",
            ),
            pytest.param(
                12, AsynchronousOptimization(num_query_points=3), id="AsynchronousOptimization"
            ),
            pytest.param(
                15,
                EfficientGlobalOptimization(
                    LocalPenalization(
                        ScaledBranin.search_space,
                    ).using(OBJECTIVE),
                    num_query_points=3,
                ),
                id="LocalPenalization",
            ),
            pytest.param(
                15,
                AsynchronousGreedy(
                    LocalPenalization(
                        ScaledBranin.search_space,
                    ).using(OBJECTIVE),
                    num_query_points=3,
                ),
                id="LocalPenalization/AsynchronousGreedy",
            ),
            pytest.param(
                10,
                EfficientGlobalOptimization(
                    GIBBON(
                        ScaledBranin.search_space,
                    ).using(OBJECTIVE),
                    num_query_points=2,
                ),
                id="GIBBON",
            ),
            pytest.param(
                25,
                EfficientGlobalOptimization(
                    MultipleOptimismNegativeLowerConfidenceBound(
                        ScaledBranin.search_space,
                    ).using(OBJECTIVE),
                    num_query_points=3,
                ),
                id="MultipleOptimismNegativeLowerConfidenceBound",
            ),
            pytest.param(
                20,
                BatchTrustRegionBox(TREGOBox(ScaledBranin.search_space)),
                id="TREGO",
            ),
            pytest.param(
                15,
                BatchTrustRegionBox(
                    TREGOBox(ScaledBranin.search_space),
                    EfficientGlobalOptimization(
                        MinValueEntropySearch(
                            ScaledBranin.search_space,
                        ).using(OBJECTIVE)
                    ),
                ),
                id="TREGO/MinValueEntropySearch",
            ),
            pytest.param(
                20,
                BatchTrustRegionBox(
                    [TREGOBox(ScaledBranin.search_space) for _ in range(3)],
                    EfficientGlobalOptimization(
                        ParallelContinuousThompsonSampling(),
                        num_query_points=3,
                    ),
                ),
                id="TREGO/ParallelContinuousThompsonSampling",
            ),
            pytest.param(
                10,
                BatchTrustRegionBox(
                    TURBOBox(ScaledBranin.search_space),
                    DiscreteThompsonSampling(500, 3),
                ),
                id="Turbo",
            ),
            pytest.param(
                10,
                BatchTrustRegionBox(
                    [SingleObjectiveTrustRegionBox(ScaledBranin.search_space) for _ in range(3)],
                    EfficientGlobalOptimization(
                        ParallelContinuousThompsonSampling(),
                        num_query_points=3,
                    ),
                ),
                id="BatchTrustRegionBox",
            ),
            pytest.param(
                10,
                (
                    BatchTrustRegionBox(
                        [
                            SingleObjectiveTrustRegionBox(ScaledBranin.search_space)
                            for _ in range(3)
                        ],
                        EfficientGlobalOptimization(
                            ParallelContinuousThompsonSampling(),
                            num_query_points=2,
                        ),
                    ),
                    3,
                ),
                id="BatchTrustRegionBox/LocalModels",
            ),
            pytest.param(15, DiscreteThompsonSampling(500, 5), id="DiscreteThompsonSampling"),
            pytest.param(
                15,
                EfficientGlobalOptimization(
                    Fantasizer(),
                    num_query_points=3,
                ),
                id="Fantasizer",
            ),
            pytest.param(
                10,
                EfficientGlobalOptimization(
                    GreedyContinuousThompsonSampling(),
                    num_query_points=5,
                ),
                id="GreedyContinuousThompsonSampling",
            ),
            pytest.param(
                10,
                EfficientGlobalOptimization(
                    ParallelContinuousThompsonSampling(),
                    num_query_points=5,
                ),
                id="ParallelContinuousThompsonSampling",
            ),
            pytest.param(
                20,
                BatchHypervolumeSharpeRatioIndicator() if pymoo else None,
                id="BatchHypevolumeSharpeRatioIndicator",
                marks=pytest.mark.qhsri,
            ),
        ],
    )


AcquisitionRuleType = Union[
    AcquisitionRule[TensorType, SearchSpace, TrainableProbabilisticModelType],
    AcquisitionRule[
        State[
            TensorType, Union[AsynchronousRuleState, BatchTrustRegionState[UpdatableTrustRegionBox]]
        ],
        Box,
        TrainableProbabilisticModelType,
    ],
]


@random_seed
@pytest.mark.slow  # to run this, add --runslow yes to the pytest command
@pytest.mark.parametrize(*GPR_OPTIMIZER_PARAMS())
def test_bayesian_optimizer_with_gpr_finds_minima_of_scaled_branin(
    num_steps: int,
    acquisition_rule: (
        AcquisitionRuleType[GaussianProcessRegression]
        | Tuple[AcquisitionRuleType[GaussianProcessRegression], int]
    ),
) -> None:
    _test_optimizer_finds_minimum(
        GaussianProcessRegression,
        num_steps,
        acquisition_rule,
        optimize_branin=True,
    )


@random_seed
@pytest.mark.parametrize(*GPR_OPTIMIZER_PARAMS())
def test_bayesian_optimizer_with_gpr_finds_minima_of_simple_quadratic(
    num_steps: int,
    acquisition_rule: (
        AcquisitionRuleType[GaussianProcessRegression]
        | Tuple[AcquisitionRuleType[GaussianProcessRegression], int]
    ),
) -> None:
    # for speed reasons we sometimes test with a simple quadratic defined on the same search space
    # branin; currently assume that every rule should be able to solve this in 6 steps
    _test_optimizer_finds_minimum(GaussianProcessRegression, min(num_steps, 6), acquisition_rule)


@random_seed
@pytest.mark.slow
def test_bayesian_optimizer_with_vgp_finds_minima_of_scaled_branin() -> None:
    _test_optimizer_finds_minimum(
        VariationalGaussianProcess,
        10,
        EfficientGlobalOptimization[SearchSpace, VariationalGaussianProcess](
            builder=ParallelContinuousThompsonSampling(), num_query_points=5
        ),
    )


@random_seed
@pytest.mark.parametrize("use_natgrads", [False, True])
def test_bayesian_optimizer_with_vgp_finds_minima_of_simple_quadratic(use_natgrads: bool) -> None:
    # regression test for [#406]; use natgrads doesn't work well as a model for the objective
    # so don't bother checking the results, just that it doesn't crash
    _test_optimizer_finds_minimum(
        VariationalGaussianProcess,
        None if use_natgrads else 5,
        EfficientGlobalOptimization[SearchSpace, GPflowPredictor](),
        model_args={"use_natgrads": use_natgrads},
    )


@random_seed
@pytest.mark.slow
def test_bayesian_optimizer_with_svgp_finds_minima_of_scaled_branin() -> None:
    _test_optimizer_finds_minimum(
        SparseVariational,
        40,
        EfficientGlobalOptimization[SearchSpace, SparseVariational](),
        optimize_branin=True,
        model_args={"optimizer": Optimizer(gpflow.optimizers.Scipy(), compile=True)},
    )
    _test_optimizer_finds_minimum(
        SparseVariational,
        25,
        EfficientGlobalOptimization[SearchSpace, SparseVariational](
            builder=ParallelContinuousThompsonSampling(), num_query_points=5
        ),
        optimize_branin=True,
        model_args={"optimizer": Optimizer(gpflow.optimizers.Scipy(), compile=True)},
    )


@random_seed
def test_bayesian_optimizer_with_svgp_finds_minima_of_simple_quadratic() -> None:
    _test_optimizer_finds_minimum(
        SparseVariational,
        5,
        EfficientGlobalOptimization[SearchSpace, SparseVariational](),
        model_args={"optimizer": Optimizer(gpflow.optimizers.Scipy(), compile=True)},
    )
    _test_optimizer_finds_minimum(
        SparseVariational,
        5,
        EfficientGlobalOptimization[SearchSpace, SparseVariational](
            builder=ParallelContinuousThompsonSampling(), num_query_points=5
        ),
        model_args={"optimizer": Optimizer(gpflow.optimizers.Scipy(), compile=True)},
    )


@random_seed
@pytest.mark.slow
def test_bayesian_optimizer_with_sgpr_finds_minima_of_scaled_branin() -> None:
    _test_optimizer_finds_minimum(
        SparseGaussianProcessRegression,
        9,
        EfficientGlobalOptimization[SearchSpace, SparseGaussianProcessRegression](),
        optimize_branin=True,
    )
    _test_optimizer_finds_minimum(
        SparseGaussianProcessRegression,
        20,
        EfficientGlobalOptimization[SearchSpace, SparseGaussianProcessRegression](
            builder=ParallelContinuousThompsonSampling(), num_query_points=5
        ),
        optimize_branin=True,
    )


@random_seed
def test_bayesian_optimizer_with_sgpr_finds_minima_of_simple_quadratic() -> None:
    _test_optimizer_finds_minimum(
        SparseGaussianProcessRegression,
        5,
        EfficientGlobalOptimization[SearchSpace, SparseGaussianProcessRegression](),
    )


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
                num_query_points=4,
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
            lambda: EfficientGlobalOptimization(),
            id="EfficientGlobalOptimization",
            marks=pytest.mark.skip(reason="too fragile"),
        ),
        pytest.param(
            60,
            lambda: EfficientGlobalOptimization(
                ParallelContinuousThompsonSampling(),
                num_query_points=4,
            ),
            id="ParallelContinuousThompsonSampling",
        ),
    ],
)
def test_bayesian_optimizer_with_deep_ensemble_finds_minima_of_scaled_branin(
    num_steps: int,
    acquisition_rule: Callable[[], AcquisitionRule[TensorType, SearchSpace, DeepEnsemble]],
) -> None:
    _test_optimizer_finds_minimum(
        DeepEnsemble,
        num_steps,
        acquisition_rule(),
        optimize_branin=True,
        model_args={"bootstrap": True, "diversify": False},
    )


@random_seed
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(tf.float64, id="float64"), pytest.param(tf.float32, id="float32")],
)
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        pytest.param(5, lambda: EfficientGlobalOptimization(), id="EfficientGlobalOptimization"),
        pytest.param(10, lambda: DiscreteThompsonSampling(1000, 1), id="DiscreteThompsonSampling"),
        pytest.param(
            5,
            lambda: DiscreteThompsonSampling(
                1000, 1, thompson_sampler=ThompsonSamplerFromTrajectory()
            ),
            id="DiscreteThompsonSampling/ThompsonSamplerFromTrajectory",
        ),
    ],
)
def test_bayesian_optimizer_with_deep_ensemble_finds_minima_of_simple_quadratic(
    dtype: tf.DType,
    num_steps: int,
    acquisition_rule: Callable[[], AcquisitionRule[TensorType, SearchSpace, DeepEnsemble]],
    request: Any,
) -> None:
    if request.node.callspec.id == "DiscreteThompsonSampling-float32":
        # TODO: DiscreteThompsonSampling with ExactThompsonSampler doesn't converge for some reason
        pytest.skip("skip DiscreteThompsonSampling(1000, 1) test with float32")

    _test_optimizer_finds_minimum(
        DeepEnsemble,
        num_steps,
        acquisition_rule(),
        single_precision=dtype == tf.float32,
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


def _test_optimizer_finds_minimum(
    model_type: Type[TrainableProbabilisticModelType],
    num_steps: Optional[int],
    acquisition_rule: (
        AcquisitionRuleType[TrainableProbabilisticModelType]
        | Tuple[AcquisitionRuleType[TrainableProbabilisticModelType], int]
    ),
    optimize_branin: bool = False,
    model_args: Optional[Mapping[str, Any]] = None,
    check_regret: bool = False,
    single_precision: bool = False,
) -> None:
    model_args = model_args or {}
    test_dtype = tf.float32 if single_precision else tf.float64

    if optimize_branin:
        search_space = ScaledBranin.search_space
        minimizers = ScaledBranin.minimizers
        minima = ScaledBranin.minimum
        rtol_level = 0.005
        num_initial_query_points = 5
    else:
        search_space = SimpleQuadratic.search_space
        minimizers = SimpleQuadratic.minimizers
        minima = SimpleQuadratic.minimum
        rtol_level = 0.05
        num_initial_query_points = 10

    if single_precision:
        minimizers = tf.cast(minimizers, dtype=tf.float32)
        minima = tf.cast(minima, dtype=tf.float32)
        search_space = Box(
            tf.cast(search_space.lower, dtype=tf.float32),
            tf.cast(search_space.upper, dtype=tf.float32),
        )

    original_tf_cast = tf.cast

    def patched_tf_cast(x: TensorType, dtype: tf.DType) -> TensorType:
        # ensure there are no unnecessary casts from float64 to float32 or vice versa
        # for now, only do this in the single_precision test as there are some false positives:
        # - VGP initialises q_mu to tf.zero and then immediately casts it to float64!
        # - write_summary_data_based_metrics casts observations to the model prediction dtype
        # - spo_improvement_on_initial_samples metric casts model initial_values to best_values
        if single_precision:
            if (
                isinstance(x, tf.Tensor)
                and x.dtype in (tf.float32, tf.float64)
                and x.dtype != dtype
            ):
                raise ValueError(f"unexpected cast: {x} to {dtype}")
        return original_tf_cast(x, dtype)

    with patch("tensorflow.cast", side_effect=patched_tf_cast):
        if model_type in [SparseVariational, DeepEnsemble]:
            num_initial_query_points = 20
        elif model_type in [DeepGaussianProcess]:
            num_initial_query_points = 25

        initial_query_points = search_space.sample(num_initial_query_points)
        assert initial_query_points.dtype is test_dtype
        observer = mk_observer(
            ScaledBranin.objective if optimize_branin else SimpleQuadratic.objective
        )
        initial_data = observer(initial_query_points)
        assert initial_data.observations.dtype is test_dtype

        if isinstance(acquisition_rule, tuple):
            acquisition_rule, num_models = acquisition_rule
        else:
            num_models = 1

        model: TrainableProbabilisticModel  # (really TPMType, but that's too complicated for mypy)

        if model_type is GaussianProcessRegression:
            if "LocalPenalization" in acquisition_rule.__repr__():
                likelihood_variance = 1e-3
            else:
                likelihood_variance = 1e-5
            gpr = build_gpr(initial_data, search_space, likelihood_variance=likelihood_variance)
            model = GaussianProcessRegression(gpr, **model_args)

        elif model_type is SparseGaussianProcessRegression:
            sgpr = build_sgpr(initial_data, search_space, num_inducing_points=50)
            model = SparseGaussianProcessRegression(
                sgpr,
                **model_args,
                inducing_point_selector=ConditionalImprovementReduction(),
            )

        elif model_type is VariationalGaussianProcess:
            empirical_variance = tf.math.reduce_variance(initial_data.observations)
            kernel = gpflow.kernels.Matern52(variance=empirical_variance, lengthscales=[0.2, 0.2])
            likelihood = gpflow.likelihoods.Gaussian(1e-3)
            vgp = gpflow.models.VGP(initial_data.astuple(), kernel, likelihood)
            gpflow.utilities.set_trainable(vgp.likelihood, False)
            model = VariationalGaussianProcess(vgp, **model_args)

        elif model_type is SparseVariational:
            svgp = build_svgp(initial_data, search_space, num_inducing_points=50)
            model = SparseVariational(
                svgp,
                **model_args,
                inducing_point_selector=ConditionalImprovementReduction(),
            )

        elif model_type is DeepGaussianProcess:
            model = DeepGaussianProcess(
                partial(build_vanilla_deep_gp, initial_data, search_space), **model_args
            )

        elif model_type is DeepEnsemble:
            keras_ensemble = build_keras_ensemble(initial_data, 5, 3, 25, "selu")
            fit_args = {
                "batch_size": 20,
                "epochs": 200,
                "callbacks": [
                    tf_keras.callbacks.EarlyStopping(
                        monitor="loss", patience=25, restore_best_weights=True
                    )
                ],
                "verbose": 0,
            }
            de_optimizer = KerasOptimizer(tf_keras.optimizers.Adam(0.01), fit_args)
            model = DeepEnsemble(keras_ensemble, de_optimizer, **model_args)

        else:
            raise ValueError(f"Unsupported model_type '{model_type}'")

        model = cast(TrainableProbabilisticModelType, model)
        models = copy_to_local_models(model, num_models) if num_models > 1 else {OBJECTIVE: model}
        dataset = {OBJECTIVE: initial_data}

        with tempfile.TemporaryDirectory() as tmpdirname:
            summary_writer = tf.summary.create_file_writer(tmpdirname)
            with tensorboard_writer(summary_writer):
                result = BayesianOptimizer(observer, search_space).optimize(
                    num_steps or 2,
                    dataset,
                    models,
                    acquisition_rule,
                    track_state=True,
                    track_path=Path(tmpdirname) / "history",
                    early_stop_callback=stop_at_minimum(
                        # stop as soon as we find the minimum (but always run at least one step)
                        minima,
                        minimizers,
                        minimum_rtol=rtol_level,
                        minimum_step_number=2,
                    ),
                    fit_initial_model=False,
                )

                # check history saved ok
                assert len(result.history) <= (num_steps or 2)
                assert len(result.loaded_history) == len(result.history)
                loaded_result: OptimizationResult[None, TrainableProbabilisticModel] = (
                    OptimizationResult.from_path(Path(tmpdirname) / "history")
                )
                assert loaded_result.final_result.is_ok
                assert len(loaded_result.history) == len(result.history)

                if num_steps is None:
                    # this test is just being run to check for crashes, not performance
                    pass
                elif check_regret:
                    # just check that the new observations are mostly better than the initial ones
                    assert isinstance(result.history[0], FrozenRecord)
                    initial_observations = result.history[0].load().dataset.observations
                    best_initial = tf.math.reduce_min(initial_observations)
                    better_than_initial = 0
                    num_points = len(initial_observations)
                    for i in range(1, len(result.history)):
                        step_history = result.history[i]
                        assert isinstance(step_history, FrozenRecord)
                        step_observations = step_history.load().dataset.observations
                        new_observations = step_observations[num_points:]
                        assert new_observations.dtype is test_dtype
                        if tf.math.reduce_min(new_observations) < best_initial:
                            better_than_initial += 1
                        num_points = len(step_observations)

                    assert better_than_initial / len(result.history) > 0.6
                else:
                    # this actually checks that we solved the problem
                    best_x, best_y, _ = result.try_get_optimal_point()
                    assert best_x.dtype is test_dtype
                    assert best_y.dtype is test_dtype

                    minimizer_err = tf.abs((best_x - minimizers) / minimizers)
                    assert tf.reduce_any(tf.reduce_all(minimizer_err < 0.05, axis=-1), axis=0)
                    npt.assert_allclose(best_y, minima, rtol=rtol_level)

                if isinstance(acquisition_rule, EfficientGlobalOptimization):
                    acq_function = acquisition_rule.acquisition_function
                    assert acq_function is not None

                    # check acquisition functions defined as classes aren't retraced unnecessarily
                    # they should be retraced for the optimizer's starting grid, L-BFGS, and logging
                    # (and possibly once more due to variable creation)
                    if isinstance(
                        acq_function, (AcquisitionFunctionClass, TrajectoryFunctionClass)
                    ):
                        assert acq_function.__call__._get_tracing_count() in {3, 4}  # type: ignore

                    # update trajectory function if necessary, so we can test it
                    if isinstance(acq_function, TrajectoryFunctionClass):
                        assert hasattr(acquisition_rule._builder, "single_builder")
                        sampler = acquisition_rule._builder.single_builder._trajectory_sampler
                        sampler.update_trajectory(acq_function)

                    # check that acquisition functions can be saved and reloaded
                    acq_function_copy = dill.loads(dill.dumps(acq_function))

                    # and that the copy gives the same values as the original
                    batch_size = (
                        1
                        if isinstance(acquisition_rule._builder, GreedyAcquisitionFunctionBuilder)
                        else acquisition_rule._num_query_points
                    )
                    random_batch = tf.expand_dims(search_space.sample(batch_size), 0)
                    npt.assert_allclose(
                        acq_function(random_batch), acq_function_copy(random_batch), rtol=5e-7
                    )

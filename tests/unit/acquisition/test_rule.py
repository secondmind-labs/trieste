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
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Callable, Optional

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import empty_dataset, quadratic, random_seed
from tests.util.models.gpflow.models import (
    GaussianProcess,
    QuadraticMeanAndRBFKernel,
    QuadraticMeanAndRBFKernelWithSamplers,
)
from trieste.acquisition import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    NegativeLowerConfidenceBound,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
    VectorizedAcquisitionFunctionBuilder,
)
from trieste.acquisition.optimizer import AcquisitionOptimizer
from trieste.acquisition.rule import (
    TURBO,
    AcquisitionRule,
    AsynchronousGreedy,
    AsynchronousOptimization,
    AsynchronousRuleState,
    BatchHypervolumeSharpeRatioIndicator,
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
    RandomSampling,
    TrustRegion,
)
from trieste.acquisition.sampler import (
    ExactThompsonSampler,
    GumbelSampler,
    ThompsonSampler,
    ThompsonSamplerFromTrajectory,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.models.interfaces import TrainableSupportsGetKernel
from trieste.observer import OBJECTIVE
from trieste.space import Box
from trieste.types import State, Tag, TensorType


def _line_search_maximize(
    search_space: Box, f: AcquisitionFunction, num_query_points: int = 1
) -> TensorType:
    if num_query_points != 1:
        raise ValueError("_line_search_maximizer only defined for batches of size 1")
    if len(search_space.lower) != 1:
        raise ValueError("_line_search_maximizer only defined for search spaces of dimension 1")
    xs = tf.linspace(search_space.lower, search_space.upper, 10**6)
    return xs[tf.squeeze(tf.argmax(f(tf.expand_dims(xs, 1)))), None]


@pytest.mark.parametrize(
    "num_search_space_samples, num_query_points",
    [
        (0, 50),
        (-2, 50),
        (10, 0),
        (10, -2),
    ],
)
def test_discrete_thompson_sampling_raises_for_invalid_init_params(
    num_search_space_samples: int, num_query_points: int
) -> None:
    with pytest.raises(ValueError):
        DiscreteThompsonSampling(num_search_space_samples, num_query_points)


@pytest.mark.parametrize(
    "models",
    [
        {},
        {"foo": QuadraticMeanAndRBFKernel()},
        {"foo": QuadraticMeanAndRBFKernel(), OBJECTIVE: QuadraticMeanAndRBFKernel()},
    ],
)
@pytest.mark.parametrize("datasets", [{}, {OBJECTIVE: empty_dataset([1], [1])}])
def test_discrete_thompson_sampling_raises_for_invalid_models_keys(
    datasets: dict[Tag, Dataset], models: dict[Tag, ProbabilisticModel]
) -> None:
    search_space = Box([-1], [1])
    rule = DiscreteThompsonSampling(100, 10)
    with pytest.raises(ValueError):
        rule.acquire(search_space, models, datasets=datasets)


@pytest.mark.parametrize("models", [{}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}])
@pytest.mark.parametrize(
    "datasets",
    [
        {},
        {"foo": empty_dataset([1], [1])},
        {"foo": empty_dataset([1], [1]), OBJECTIVE: empty_dataset([1], [1])},
    ],
)
def test_discrete_thompson_sampling_raises_for_invalid_dataset_keys(
    datasets: dict[Tag, Dataset], models: dict[Tag, ProbabilisticModel]
) -> None:
    search_space = Box([-1], [1])
    rule = DiscreteThompsonSampling(10, 100)
    with pytest.raises(ValueError):
        rule.acquire(search_space, models, datasets=datasets)


@pytest.mark.parametrize(
    "sampler",
    [
        ExactThompsonSampler(sample_min_value=True),
        ThompsonSamplerFromTrajectory(sample_min_value=True),
    ],
)
def test_discrete_thompson_sampling_raises_if_passed_sampler_with_sample_min_value_True(
    sampler: ThompsonSampler[GaussianProcess],
) -> None:
    with pytest.raises(ValueError):
        DiscreteThompsonSampling(100, 10, thompson_sampler=sampler)


@pytest.mark.parametrize(
    "thompson_sampler",
    [
        ExactThompsonSampler(sample_min_value=False),
        ThompsonSamplerFromTrajectory(sample_min_value=False),
    ],
)
def test_discrete_thompson_sampling_initialized_with_correct_sampler(
    thompson_sampler: ThompsonSampler[GaussianProcess],
) -> None:
    ts = DiscreteThompsonSampling(100, 10, thompson_sampler=thompson_sampler)
    assert ts._thompson_sampler == thompson_sampler


def test_discrete_thompson_sampling_raises_if_use_fourier_features_with_incorrect_model() -> None:
    search_space = Box([-2.2, -1.0], [1.3, 3.3])
    ts = DiscreteThompsonSampling(
        100, 10, thompson_sampler=ThompsonSamplerFromTrajectory(sample_min_value=False)
    )
    dataset = Dataset(tf.zeros([1, 2], dtype=tf.float64), tf.zeros([1, 1], dtype=tf.float64))
    model = QuadraticMeanAndRBFKernel(noise_variance=tf.constant(1.0, dtype=tf.float64))
    with pytest.raises(ValueError):
        ts.acquire_single(search_space, model, dataset=dataset)  # type: ignore


def test_discrete_thompson_sampling_raises_for_gumbel_sampler() -> None:
    with pytest.raises(ValueError):
        DiscreteThompsonSampling(100, 10, thompson_sampler=GumbelSampler(sample_min_value=False))


@pytest.mark.parametrize(
    "thompson_sampler",
    [
        ExactThompsonSampler(sample_min_value=False),
        ThompsonSamplerFromTrajectory(sample_min_value=False),
    ],
)
@pytest.mark.parametrize("num_query_points", [1, 10])
def test_discrete_thompson_sampling_acquire_returns_correct_shape(
    thompson_sampler: ThompsonSampler[GaussianProcess], num_query_points: int
) -> None:
    search_space = Box([-2.2, -1.0], [1.3, 3.3])
    ts = DiscreteThompsonSampling(100, num_query_points, thompson_sampler=thompson_sampler)
    dataset = Dataset(tf.zeros([1, 2], dtype=tf.float64), tf.zeros([1, 1], dtype=tf.float64))
    model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset=dataset, noise_variance=tf.constant(1.0, dtype=tf.float64)
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    query_points = ts.acquire_single(search_space, model, dataset=dataset)

    npt.assert_array_equal(query_points.shape, tf.constant([num_query_points, 2]))


@pytest.mark.parametrize("num_query_points", [-1, 0])
def test_random_sampling_raises_for_invalid_init_params(num_query_points: int) -> None:
    with pytest.raises(ValueError):
        RandomSampling(num_query_points)


@pytest.mark.parametrize("num_query_points", [1, 10, 50])
def test_random_sampling_acquire_returns_correct_shape(num_query_points: int) -> None:
    search_space = Box([-2.2, -1.0], [1.3, 3.3])
    rule = RandomSampling(num_query_points)
    dataset = Dataset(tf.zeros([1, 2], dtype=tf.float64), tf.zeros([1, 1], dtype=tf.float64))
    model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset=dataset, noise_variance=tf.constant(1.0, dtype=tf.float64)
    )
    query_points = rule.acquire_single(search_space, model)

    npt.assert_array_equal(query_points.shape, tf.constant([num_query_points, 2]))


def test_efficient_global_optimization_raises_for_no_query_points() -> None:
    with pytest.raises(ValueError):
        EfficientGlobalOptimization(num_query_points=0)


def test_efficient_global_optimization_raises_for_no_batch_fn_with_many_query_points() -> None:
    with pytest.raises(ValueError):
        EfficientGlobalOptimization(num_query_points=2)


@pytest.mark.parametrize("optimizer", [_line_search_maximize, None])
def test_efficient_global_optimization(optimizer: AcquisitionOptimizer[Box]) -> None:
    class NegQuadratic(SingleModelAcquisitionBuilder[ProbabilisticModel]):
        def __init__(self) -> None:
            self._updated = False

        def prepare_acquisition_function(
            self,
            model: ProbabilisticModel,
            dataset: Optional[Dataset] = None,
        ) -> AcquisitionFunction:
            return lambda x: -quadratic(tf.squeeze(x, -2) - 1)

        def update_acquisition_function(
            self,
            function: AcquisitionFunction,
            model: ProbabilisticModel,
            dataset: Optional[Dataset] = None,
        ) -> AcquisitionFunction:
            self._updated = True
            return function

    function = NegQuadratic()
    search_space = Box([-10], [10])
    ego = EfficientGlobalOptimization(function, optimizer)
    data, model = empty_dataset([1], [1]), QuadraticMeanAndRBFKernel(x_shift=1)
    query_point = ego.acquire_single(search_space, model, dataset=data)
    npt.assert_allclose(query_point, [[1]], rtol=1e-4)
    assert not function._updated
    query_point = ego.acquire(search_space, {OBJECTIVE: model})
    npt.assert_allclose(query_point, [[1]], rtol=1e-4)
    assert function._updated


def test_efficient_global_optimization_initial_acquisition_function() -> None:
    class NoisyNegQuadratic(SingleModelAcquisitionBuilder[ProbabilisticModel]):
        def prepare_acquisition_function(
            self,
            model: ProbabilisticModel,
            dataset: Optional[Dataset] = None,
        ) -> AcquisitionFunction:
            noise = tf.random.uniform([], -0.05, 0.05, dtype=tf.float64)
            return lambda x: -quadratic(tf.squeeze(x, -2) - 1) + noise

        def update_acquisition_function(
            self,
            function: AcquisitionFunction,
            model: ProbabilisticModel,
            dataset: Optional[Dataset] = None,
        ) -> AcquisitionFunction:
            return function

    builder = NoisyNegQuadratic()
    search_space = Box([-10], [10])
    ego = EfficientGlobalOptimization[Box, ProbabilisticModel](builder)
    data, model = empty_dataset([1], [1]), QuadraticMeanAndRBFKernel(x_shift=1)
    ego.acquire_single(search_space, model, dataset=data)
    assert ego.acquisition_function is not None

    # check that we can create a new EGO with the exact same AF state
    acq_func = copy.deepcopy(ego.acquisition_function)
    ego_copy = EfficientGlobalOptimization[Box, ProbabilisticModel](
        builder, initial_acquisition_function=acq_func
    )
    ego_copy.acquire_single(search_space, model, dataset=data)
    assert ego_copy.acquisition_function is not None
    x = search_space.sample(1)
    npt.assert_allclose(ego.acquisition_function(x), ego_copy.acquisition_function(x))

    # check that if we don't do this, the AF state might vary
    ego_non_copy = EfficientGlobalOptimization[Box, ProbabilisticModel](builder)
    ego_non_copy.acquire_single(search_space, model, dataset=data)
    assert ego_non_copy.acquisition_function is not None
    npt.assert_raises(
        AssertionError,
        npt.assert_allclose,
        ego.acquisition_function(x),
        ego_non_copy.acquisition_function(x),
    )


class _JointBatchModelMinusMeanMaximumSingleBuilder(AcquisitionFunctionBuilder[ProbabilisticModel]):
    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        return lambda at: -tf.reduce_max(models[OBJECTIVE].predict(at)[0], axis=-2)


@random_seed
@pytest.mark.parametrize(
    "rule_fn",
    [
        lambda acq, batch_size: EfficientGlobalOptimization(acq, num_query_points=batch_size),
        lambda acq, batch_size: AsynchronousOptimization(acq, num_query_points=batch_size),
    ],
)
# As a side effect, this test ensures and EGO and AsynchronousOptimization
# behave similarly in sync mode
def test_joint_batch_acquisition_rule_acquire(
    rule_fn: Callable[
        # callable input type(s)
        [_JointBatchModelMinusMeanMaximumSingleBuilder, int],
        # callable output type
        AcquisitionRule[TensorType, Box, ProbabilisticModel]
        | AcquisitionRule[State[TensorType, AsynchronousRuleState], Box, ProbabilisticModel],
    ]
) -> None:
    search_space = Box(tf.constant([-2.2, -1.0]), tf.constant([1.3, 3.3]))
    num_query_points = 4
    acq = _JointBatchModelMinusMeanMaximumSingleBuilder()
    acq_rule: AcquisitionRule[TensorType, Box, ProbabilisticModel] | AcquisitionRule[
        State[TensorType, AsynchronousRuleState], Box, ProbabilisticModel
    ] = rule_fn(acq, num_query_points)

    dataset = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))
    points_or_stateful = acq_rule.acquire_single(
        search_space, QuadraticMeanAndRBFKernel(), dataset=dataset
    )
    if callable(points_or_stateful):
        _, query_point = points_or_stateful(None)
    else:
        query_point = points_or_stateful
    npt.assert_allclose(query_point, [[0.0, 0.0]] * num_query_points, atol=1e-3)


class _GreedyBatchModelMinusMeanMaximumSingleBuilder(
    SingleModelGreedyAcquisitionBuilder[ProbabilisticModel]
):
    def __init__(self) -> None:
        self._update_count = 0

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: TensorType = None,
    ) -> AcquisitionFunction:
        if pending_points is None:
            return lambda at: -tf.reduce_max(model.predict(at)[0], axis=-2)
        else:
            best_pending_score = tf.reduce_max(model.predict(pending_points)[0])
            return lambda at: -tf.math.maximum(
                tf.reduce_max(model.predict(at)[0], axis=-2), best_pending_score
            )

    def update_acquisition_function(
        self,
        function: Optional[AcquisitionFunction],
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        self._update_count += 1
        return self.prepare_acquisition_function(
            model, dataset=dataset, pending_points=pending_points
        )


@random_seed
@pytest.mark.parametrize(
    "rule_fn",
    [
        lambda acq, batch_size: EfficientGlobalOptimization(acq, num_query_points=batch_size),
        lambda acq, batch_size: AsynchronousGreedy(acq, num_query_points=batch_size),
    ],
)
# As a side effect, this test ensures and EGO and AsynchronousGreedy
# behave similarly in sync mode
def test_greedy_batch_acquisition_rule_acquire(
    rule_fn: Callable[
        # callable input type(s)
        [_GreedyBatchModelMinusMeanMaximumSingleBuilder, int],
        # callable output type
        AcquisitionRule[TensorType, Box, ProbabilisticModel]
        | AcquisitionRule[State[TensorType, AsynchronousRuleState], Box, ProbabilisticModel],
    ]
) -> None:
    search_space = Box(tf.constant([-2.2, -1.0]), tf.constant([1.3, 3.3]))
    num_query_points = 4
    acq = _GreedyBatchModelMinusMeanMaximumSingleBuilder()
    assert acq._update_count == 0
    acq_rule: AcquisitionRule[TensorType, Box, ProbabilisticModel] | AcquisitionRule[
        State[TensorType, AsynchronousRuleState], Box, ProbabilisticModel
    ] = rule_fn(acq, num_query_points)
    dataset = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))
    points_or_stateful = acq_rule.acquire_single(
        search_space, QuadraticMeanAndRBFKernel(), dataset=dataset
    )
    if callable(points_or_stateful):
        _, query_points = points_or_stateful(None)
    else:
        query_points = points_or_stateful
    assert acq._update_count == num_query_points - 1
    npt.assert_allclose(query_points, [[0.0, 0.0]] * num_query_points, atol=1e-3)

    points_or_stateful = acq_rule.acquire_single(
        search_space, QuadraticMeanAndRBFKernel(), dataset=dataset
    )
    if callable(points_or_stateful):
        _, query_points = points_or_stateful(None)
    else:
        query_points = points_or_stateful
    npt.assert_allclose(query_points, [[0.0, 0.0]] * num_query_points, atol=1e-3)
    assert acq._update_count == 2 * num_query_points - 1


class _VectorizedBatchModelMinusMeanMaximumSingleBuilder(
    VectorizedAcquisitionFunctionBuilder[ProbabilisticModel]
):
    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        return lambda at: tf.squeeze(-models[OBJECTIVE].predict(at)[0], -1)


@random_seed
def test_vectorized_batch_acquisition_rule_acquire() -> None:
    search_space = Box(tf.constant([-2.2, -1.0]), tf.constant([1.3, 3.3]))
    num_query_points = 4
    acq = _VectorizedBatchModelMinusMeanMaximumSingleBuilder()
    acq_rule: AcquisitionRule[TensorType, Box, ProbabilisticModel] = EfficientGlobalOptimization(
        acq, num_query_points=num_query_points
    )
    dataset = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))
    points_or_stateful = acq_rule.acquire_single(
        search_space, QuadraticMeanAndRBFKernel(), dataset=dataset
    )
    if callable(points_or_stateful):
        _, query_point = points_or_stateful(None)
    else:
        query_point = points_or_stateful
    npt.assert_allclose(query_point, [[0.0, 0.0]] * num_query_points, atol=1e-3)


def test_async_greedy_raises_for_non_greedy_function() -> None:
    non_greedy_function_builder = NegativeLowerConfidenceBound()
    with pytest.raises(NotImplementedError):
        # we are deliberately passing in wrong object
        # hence type ignore
        AsynchronousGreedy(non_greedy_function_builder)  # type: ignore


def test_async_optimization_raises_for_incorrect_query_points() -> None:
    with pytest.raises(ValueError):
        AsynchronousOptimization(num_query_points=0)

    with pytest.raises(ValueError):
        AsynchronousOptimization(num_query_points=-5)


def test_async_greedy_raises_for_incorrect_query_points() -> None:
    with pytest.raises(ValueError):
        AsynchronousGreedy(
            builder=_GreedyBatchModelMinusMeanMaximumSingleBuilder(), num_query_points=0
        )

    with pytest.raises(ValueError):
        AsynchronousGreedy(
            builder=_GreedyBatchModelMinusMeanMaximumSingleBuilder(), num_query_points=-5
        )


@random_seed
@pytest.mark.parametrize(
    "async_rule",
    [
        AsynchronousOptimization(_JointBatchModelMinusMeanMaximumSingleBuilder()),
        AsynchronousGreedy(_GreedyBatchModelMinusMeanMaximumSingleBuilder()),
    ],
)
def test_async_keeps_track_of_pending_points(
    async_rule: AcquisitionRule[
        State[Optional[AsynchronousRuleState], TensorType], Box, ProbabilisticModel
    ]
) -> None:
    search_space = Box(tf.constant([-2.2, -1.0]), tf.constant([1.3, 3.3]))
    dataset = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))

    state_fn = async_rule.acquire_single(search_space, QuadraticMeanAndRBFKernel(), dataset=dataset)
    state, point1 = state_fn(None)
    state, point2 = state_fn(state)

    assert state is not None
    assert state.pending_points is not None
    assert len(state.pending_points) == 2

    # pretend we saw observation for the first point
    new_observations = Dataset(
        query_points=point1,
        observations=tf.constant([[1]], dtype=tf.float32),
    )
    state_fn = async_rule.acquire_single(
        search_space,
        QuadraticMeanAndRBFKernel(),
        dataset=dataset + new_observations,
    )
    state, point3 = state_fn(state)

    assert state is not None
    assert state.pending_points is not None
    assert len(state.pending_points) == 2

    # we saw first point, so pendings points are
    # second point and new third point
    npt.assert_allclose(state.pending_points, tf.concat([point2, point3], axis=0))


@pytest.mark.parametrize("datasets", [{}, {"foo": empty_dataset([1], [1])}])
@pytest.mark.parametrize(
    "models", [{}, {"foo": QuadraticMeanAndRBFKernel()}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}]
)
def test_trust_region_raises_for_missing_datasets_key(
    datasets: dict[Tag, Dataset], models: dict[Tag, ProbabilisticModel]
) -> None:
    search_space = Box([-1], [1])
    rule = TrustRegion()
    with pytest.raises(ValueError):
        rule.acquire(search_space, models, datasets=datasets)


class _Midpoint(AcquisitionRule[TensorType, Box, ProbabilisticModel]):
    def acquire(
        self,
        search_space: Box,
        models: Mapping[Tag, ProbabilisticModel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        return (search_space.upper[None] + search_space.lower[None]) / 2


@pytest.mark.parametrize(
    "rule, expected_query_point",
    [
        (EfficientGlobalOptimization(NegativeLowerConfidenceBound(0)), [[0.0, 0.0]]),
        (_Midpoint(), [[-0.45, 1.15]]),
    ],
)
def test_trust_region_for_default_state(
    rule: AcquisitionRule[TensorType, Box, ProbabilisticModel], expected_query_point: TensorType
) -> None:
    tr = TrustRegion(rule)
    dataset = Dataset(tf.constant([[0.1, 0.2]]), tf.constant([[0.012]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    state, query_point = tr.acquire_single(
        search_space, QuadraticMeanAndRBFKernel(), dataset=dataset
    )(None)

    assert state is not None
    npt.assert_array_almost_equal(query_point, expected_query_point, 5)
    npt.assert_array_almost_equal(state.acquisition_space.lower, lower_bound)
    npt.assert_array_almost_equal(state.acquisition_space.upper, upper_bound)
    npt.assert_array_almost_equal(state.y_min, [0.012])
    assert state.is_global


@pytest.mark.parametrize(
    "rule, expected_query_point",
    [
        (EfficientGlobalOptimization(NegativeLowerConfidenceBound(0)), [[0.0, 0.0]]),
        (_Midpoint(), [[-0.45, 1.15]]),
    ],
)
def test_trust_region_successful_global_to_global_trust_region_unchanged(
    rule: AcquisitionRule[TensorType, Box, ProbabilisticModel], expected_query_point: TensorType
) -> None:
    tr = TrustRegion(rule)
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.3]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = True
    previous_state = TrustRegion.State(search_space, eps, previous_y_min, is_global)

    current_state, query_point = tr.acquire(
        search_space,
        {OBJECTIVE: QuadraticMeanAndRBFKernel()},
        datasets={OBJECTIVE: dataset},
    )(previous_state)

    assert current_state is not None
    npt.assert_array_almost_equal(current_state.eps, previous_state.eps)
    assert current_state.is_global
    npt.assert_array_almost_equal(query_point, expected_query_point, 5)
    npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)
    npt.assert_array_almost_equal(current_state.acquisition_space.upper, upper_bound)


@pytest.mark.parametrize(
    "rule",
    [
        EfficientGlobalOptimization(NegativeLowerConfidenceBound(0)),
        _Midpoint(),
    ],
)
def test_trust_region_for_unsuccessful_global_to_local_trust_region_unchanged(
    rule: AcquisitionRule[TensorType, Box, ProbabilisticModel]
) -> None:
    tr = TrustRegion(rule)
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.5]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = True
    acquisition_space = search_space
    previous_state = TrustRegion.State(acquisition_space, eps, previous_y_min, is_global)

    current_state, query_point = tr.acquire(
        search_space,
        {OBJECTIVE: QuadraticMeanAndRBFKernel()},
        datasets={OBJECTIVE: dataset},
    )(previous_state)

    assert current_state is not None
    npt.assert_array_almost_equal(current_state.eps, previous_state.eps)
    assert not current_state.is_global
    npt.assert_array_less(lower_bound, current_state.acquisition_space.lower)
    npt.assert_array_less(current_state.acquisition_space.upper, upper_bound)
    assert query_point[0] in current_state.acquisition_space


@pytest.mark.parametrize(
    "rule",
    [
        EfficientGlobalOptimization(NegativeLowerConfidenceBound(0)),
        _Midpoint(),
    ],
)
def test_trust_region_for_successful_local_to_global_trust_region_increased(
    rule: AcquisitionRule[TensorType, Box, ProbabilisticModel]
) -> None:
    tr = TrustRegion(rule)
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.3]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = False
    acquisition_space = Box(dataset.query_points[0] - eps, dataset.query_points[0] + eps)
    previous_state = TrustRegion.State(acquisition_space, eps, previous_y_min, is_global)

    current_state, _ = tr.acquire(
        search_space,
        {OBJECTIVE: QuadraticMeanAndRBFKernel()},
        datasets={OBJECTIVE: dataset},
    )(previous_state)

    assert current_state is not None
    npt.assert_array_less(previous_state.eps, current_state.eps)  # current TR larger than previous
    assert current_state.is_global
    npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)
    npt.assert_array_almost_equal(current_state.acquisition_space.upper, upper_bound)


@pytest.mark.parametrize(
    "rule",
    [
        EfficientGlobalOptimization(NegativeLowerConfidenceBound(0)),
        _Midpoint(),
    ],
)
def test_trust_region_for_unsuccessful_local_to_global_trust_region_reduced(
    rule: AcquisitionRule[TensorType, Box, ProbabilisticModel]
) -> None:
    tr = TrustRegion(rule)
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.5]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = False
    acquisition_space = Box(dataset.query_points[0] - eps, dataset.query_points[0] + eps)
    previous_state = TrustRegion.State(acquisition_space, eps, previous_y_min, is_global)

    current_state, _ = tr.acquire(
        search_space,
        {OBJECTIVE: QuadraticMeanAndRBFKernel()},
        datasets={OBJECTIVE: dataset},
    )(previous_state)

    assert current_state is not None
    npt.assert_array_less(current_state.eps, previous_state.eps)  # current TR smaller than previous
    assert current_state.is_global
    npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)


def test_trust_region_state_deepcopy() -> None:
    tr_state = TrustRegion.State(
        Box(tf.constant([1.2]), tf.constant([3.4])), tf.constant(5.6), tf.constant(7.8), False
    )
    tr_state_copy = copy.deepcopy(tr_state)
    npt.assert_allclose(tr_state_copy.acquisition_space.lower, tr_state.acquisition_space.lower)
    npt.assert_allclose(tr_state_copy.acquisition_space.upper, tr_state.acquisition_space.upper)
    npt.assert_allclose(tr_state_copy.eps, tr_state.eps)
    npt.assert_allclose(tr_state_copy.y_min, tr_state.y_min)
    assert tr_state_copy.is_global == tr_state.is_global


@pytest.mark.parametrize("datasets", [{}, {"foo": empty_dataset([1], [1])}])
@pytest.mark.parametrize(
    "models",
    [
        {},
        {"foo": QuadraticMeanAndRBFKernelWithSamplers(empty_dataset([1], [1]))},
        {OBJECTIVE: QuadraticMeanAndRBFKernelWithSamplers(empty_dataset([1], [1]))},
    ],
)
def test_turbo_raises_for_missing_datasets_key(
    datasets: Mapping[Tag, Dataset], models: Mapping[Tag, TrainableSupportsGetKernel]
) -> None:
    search_space = Box([-1], [1])
    rule = TURBO(search_space)
    with pytest.raises(ValueError):
        rule.acquire(search_space, models, datasets=datasets)


@pytest.mark.parametrize("num_trust_regions", [-1, 0, 10])
def test_turbo_rasise_for_invalid_num_trust_regions(num_trust_regions: int) -> None:
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    if num_trust_regions == 10:  # to be removed once we enable multiple trust regions
        with pytest.raises(NotImplementedError):
            TURBO(search_space, num_trust_regions=num_trust_regions)
    else:
        with pytest.raises(ValueError):
            TURBO(search_space, num_trust_regions=num_trust_regions)


@pytest.mark.parametrize(
    "L_init, L_max, L_min, failure_tolerance, success_tolerance",
    [
        (-1.0, 0.1, 1.0, 1, 1),
        (10.0, -1.0, 1.0, 1, 1),
        (10.0, 1.0, -4.0, 1, 1),
        (10.0, 1.0, 4.0, -1, 2),
        (10.0, 1.0, 4.0, 1, -1),
    ],
)
def test_turbo_rasise_for_invalid_trust_region_params(
    L_init: float,
    L_max: float,
    L_min: float,
    failure_tolerance: int,
    success_tolerance: int,
) -> None:
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)
    with pytest.raises(ValueError):
        TURBO(
            search_space,
            L_init=L_init,
            L_max=L_max,
            L_min=L_min,
            failure_tolerance=failure_tolerance,
            success_tolerance=success_tolerance,
        )


def test_turbo_heuristics_for_param_init_work() -> None:
    lower_bound = tf.constant([-2.0] * 20)
    upper_bound = tf.constant([1.0] * 20)
    search_space = Box(lower_bound, upper_bound)
    rule = TURBO(search_space)

    assert rule._L_init == 0.8 * 3.0
    assert rule._L_min == (0.5**7) * 3.0
    assert rule._L_max == 1.6 * 3.0
    assert rule._failure_tolerance == 20
    assert isinstance(rule._rule, DiscreteThompsonSampling)
    assert rule._rule._num_search_space_samples == 2_000
    assert rule._local_models is None

    rule = TURBO(search_space, rule=EfficientGlobalOptimization())
    assert isinstance(rule._rule, EfficientGlobalOptimization)


def test_turbo_acquire_uses_and_updates_correct_local_model() -> None:
    dataset_1 = Dataset(
        tf.constant([[0.0, 0.0]], dtype=tf.float64), tf.constant([[0.012]], dtype=tf.float64)
    )
    dataset_2 = Dataset(
        tf.constant([[10.0, 10.0]], dtype=tf.float64), tf.constant([[1.012]], dtype=tf.float64)
    )
    lower_bound = tf.constant([0.0, 0.0], dtype=tf.float64)
    upper_bound = tf.constant([20.0, 20.0], dtype=tf.float64)
    search_space = Box(lower_bound, upper_bound)
    global_model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset_1, noise_variance=tf.constant(1e-5, dtype=tf.float64)
    )
    global_model.kernel = gpflow.kernels.RBF(
        lengthscales=tf.constant([4.0, 1.0], dtype=tf.float64), variance=1e-5
    )  # need a gpflow kernel for TURBO

    # if user doesnt give a local model, then we refit the global model
    tr = TURBO(search_space)
    assert tr._local_models is None
    _, _ = tr.acquire_single(search_space, global_model, dataset=dataset_2)(None)
    assert tr._local_models is not None
    assert isinstance(  # type: ignore[unreachable]
        tr._local_models[OBJECTIVE].kernel, gpflow.kernels.RBF
    )
    npt.assert_array_equal(tr._local_models[OBJECTIVE]._dataset[0], dataset_2.query_points)

    # if user gives a local model, then we use that one
    local_model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset_1, noise_variance=tf.constant(1e-5, dtype=tf.float64)
    )
    local_model.kernel = gpflow.kernels.Matern52(
        lengthscales=tf.constant([4.0, 1.0], dtype=tf.float64), variance=1e-5
    )  # need a gpflow kernel for TURBO
    tr = TURBO(search_space, local_models={OBJECTIVE: local_model})
    assert isinstance(tr._local_models[OBJECTIVE].get_kernel(), gpflow.kernels.Matern52)
    _, _ = tr.acquire_single(search_space, global_model, dataset=dataset_2)(None)
    # check updated correct model
    assert isinstance(tr._local_models[OBJECTIVE].get_kernel(), gpflow.kernels.Matern52)
    npt.assert_array_equal(tr._local_models[OBJECTIVE]._dataset[0], dataset_2.query_points)


@pytest.mark.parametrize("num_query_points", [1, 2])
def test_turbo_acquire_returns_correct_shape(num_query_points: int) -> None:
    dataset = Dataset(
        tf.constant([[0.0, 0.0]], dtype=tf.float64), tf.constant([[0.012]], dtype=tf.float64)
    )
    lower_bound = tf.constant([0.0, 0.0], dtype=tf.float64)
    upper_bound = tf.constant([1.0, 1.0], dtype=tf.float64)
    search_space = Box(lower_bound, upper_bound)
    rule = DiscreteThompsonSampling(1_000, num_query_points)
    tr = TURBO(search_space, rule=rule)
    model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset, noise_variance=tf.constant(1e-5, dtype=tf.float64)
    )
    model.kernel = gpflow.kernels.RBF(
        lengthscales=tf.constant([4.0, 1.0], dtype=tf.float64), variance=1e-5
    )  # need a gpflow kernel for TURBO
    _, query_points = tr.acquire_single(search_space, model, dataset=dataset)(None)
    npt.assert_array_equal(tf.shape(query_points), [num_query_points, 2])


@random_seed
def test_turbo_for_default_state() -> None:
    dataset = Dataset(
        tf.constant([[0.0, 0.0]], dtype=tf.float64), tf.constant([[0.012]], dtype=tf.float64)
    )
    lower_bound = tf.constant([0.0, 0.0], dtype=tf.float64)
    upper_bound = tf.constant([1.0, 1.0], dtype=tf.float64)
    search_space = Box(lower_bound, upper_bound)
    tr = TURBO(search_space, rule=DiscreteThompsonSampling(100, 1))
    model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset, noise_variance=tf.constant(1e-5, dtype=tf.float64)
    )
    model.kernel = gpflow.kernels.RBF(
        lengthscales=tf.constant([4.0, 1.0], dtype=tf.float64), variance=1e-5
    )  # need a gpflow kernel for TURBO
    state, query_point = tr.acquire_single(search_space, model, dataset=dataset)(None)

    assert state is not None
    npt.assert_array_almost_equal(state.acquisition_space.lower, lower_bound)
    npt.assert_array_almost_equal(
        state.acquisition_space.upper, tf.constant([0.8, 0.2], dtype=tf.float64)
    )
    npt.assert_array_almost_equal(state.y_min, [0.012])
    npt.assert_array_almost_equal(state.L, tf.cast(0.8, dtype=tf.float64))
    assert state.success_counter == 0
    assert state.failure_counter == 0


def test_turbo_doesnt_change_size_unless_needed() -> None:
    dataset = Dataset(
        tf.constant([[0.0, 0.0]], dtype=tf.float64), tf.constant([[0.012]], dtype=tf.float64)
    )
    models = {
        OBJECTIVE: QuadraticMeanAndRBFKernelWithSamplers(
            dataset, noise_variance=tf.constant(1e-5, dtype=tf.float64)
        )
    }
    models[OBJECTIVE].kernel = gpflow.kernels.RBF(
        lengthscales=tf.constant([4.0, 1.0], dtype=tf.float64), variance=1e-5
    )  # need a gpflow kernel for TURBO
    lower_bound = tf.constant([0.0, 0.0], dtype=tf.float64)
    upper_bound = tf.constant([1.0, 1.0], dtype=tf.float64)
    search_space = Box(lower_bound, upper_bound)
    tr = TURBO(search_space)

    # success but not enough to trigger size change
    previous_y_min = dataset.observations[0] + 2.0  # force success
    for failure_counter in [0, 1]:
        for success_counter in [0, 1]:
            previous_state = TURBO.State(
                search_space,
                tf.constant(0.8, dtype=tf.float64),
                failure_counter,
                success_counter,
                previous_y_min,
            )
            current_state, _ = tr.acquire(
                search_space,
                models,
                datasets={OBJECTIVE: dataset},
            )(previous_state)

            assert current_state is not None
            npt.assert_array_almost_equal(current_state.L, tf.cast(0.8, dtype=tf.float64))
            npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)
            npt.assert_array_almost_equal(
                current_state.acquisition_space.upper, tf.constant([0.8, 0.2], dtype=tf.float64)
            )
            assert current_state.success_counter == success_counter + 1
            assert current_state.failure_counter == 0

    # failure but not enough to trigger size change
    previous_y_min = dataset.observations[0]  # force failure
    for success_counter in [0, 1, 2]:
        previous_state = TURBO.State(
            search_space, tf.constant(0.8, dtype=tf.float64), 0, success_counter, previous_y_min
        )
        current_state, _ = tr.acquire(
            search_space,
            models,
            datasets={OBJECTIVE: dataset},
        )(previous_state)

        assert current_state is not None
        npt.assert_array_almost_equal(current_state.L, tf.cast(0.8, dtype=tf.float64))
        npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)
        npt.assert_array_almost_equal(
            current_state.acquisition_space.upper, tf.constant([0.8, 0.2], dtype=tf.float64)
        )
        assert current_state.success_counter == 0
        assert current_state.failure_counter == 1


def test_turbo_does_change_size_correctly_when_needed() -> None:
    dataset = Dataset(
        tf.constant([[0.0, 0.0]], dtype=tf.float64), tf.constant([[0.012]], dtype=tf.float64)
    )
    models = {
        OBJECTIVE: QuadraticMeanAndRBFKernelWithSamplers(
            dataset, noise_variance=tf.constant(1e-5, dtype=tf.float64)
        )
    }
    models[OBJECTIVE].kernel = gpflow.kernels.RBF(
        lengthscales=tf.constant([4.0, 1.0], dtype=tf.float64), variance=1e-5
    )  # need a gpflow kernel for TURBO
    lower_bound = tf.constant([0.0, 0.0], dtype=tf.float64)
    upper_bound = tf.constant([1.0, 1.0], dtype=tf.float64)
    search_space = Box(lower_bound, upper_bound)
    tr = TURBO(search_space, failure_tolerance=2)

    # hits success limit
    previous_y_min = dataset.observations[0] + 2.0  # force success
    for failure_counter in [0, 1]:
        previous_state = TURBO.State(
            search_space, tf.constant(0.8, dtype=tf.float64), failure_counter, 2, previous_y_min
        )
        current_state, _ = tr.acquire(
            search_space,
            models,
            datasets={OBJECTIVE: dataset},
        )(previous_state)

        assert current_state is not None
        npt.assert_array_almost_equal(current_state.L, tf.cast(1.6, dtype=tf.float64))
        npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)
        npt.assert_array_almost_equal(
            current_state.acquisition_space.upper, tf.constant([1.0, 0.4], dtype=tf.float64)
        )
        assert current_state.success_counter == 0
        assert current_state.failure_counter == 0
    # hits failure limit
    previous_y_min = dataset.observations[0]  # force failure
    for success_counter in [0, 1, 2]:
        previous_state = TURBO.State(
            search_space, tf.constant(0.8, dtype=tf.float64), 1, success_counter, previous_y_min
        )
        current_state, _ = tr.acquire(
            search_space,
            models,
            datasets={OBJECTIVE: dataset},
        )(previous_state)

        assert current_state is not None
        npt.assert_array_almost_equal(current_state.L, tf.cast(0.4, dtype=tf.float64))
        npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)
        npt.assert_array_almost_equal(
            current_state.acquisition_space.upper, tf.constant([0.4, 0.1], dtype=tf.float64)
        )
        assert current_state.success_counter == 0
        assert current_state.failure_counter == 0


def test_turbo_restarts_tr_when_too_small() -> None:
    dataset = Dataset(
        tf.constant([[0.0, 0.0]], dtype=tf.float64), tf.constant([[0.012]], dtype=tf.float64)
    )
    models = {
        OBJECTIVE: QuadraticMeanAndRBFKernelWithSamplers(
            dataset, noise_variance=tf.constant(1e-5, dtype=tf.float64)
        )
    }
    models[OBJECTIVE].kernel = gpflow.kernels.RBF(
        variance=1e-5, lengthscales=tf.constant([4.0, 1.0], dtype=tf.float64)
    )  # need a gpflow kernel for TURBO
    lower_bound = tf.constant([0.0, 0.0], dtype=tf.float64)
    upper_bound = tf.constant([1.0, 1.0], dtype=tf.float64)
    search_space = Box(lower_bound, upper_bound)
    tr = TURBO(search_space)

    # first check what happens if L is too small from the start
    previous_y_min = dataset.observations[0]
    failure_counter = 1
    success_counter = 1
    L = tf.constant(1e-10, dtype=tf.float64)
    previous_search_space = Box(lower_bound / 2.0, upper_bound / 5.0)
    previous_state = TURBO.State(
        previous_search_space, L, failure_counter, success_counter, previous_y_min
    )
    current_state, _ = tr.acquire(
        search_space,
        models,
        datasets={OBJECTIVE: dataset},
    )(previous_state)

    assert current_state is not None
    npt.assert_array_almost_equal(current_state.L, tf.cast(0.8, dtype=tf.float64))
    npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)
    npt.assert_array_almost_equal(
        current_state.acquisition_space.upper, tf.constant([0.8, 0.2], dtype=tf.float64)
    )
    assert current_state.success_counter == 0
    assert current_state.failure_counter == 0

    # secondly check what happens if L is too small after triggering decreasing the region
    previous_state = TURBO.State(
        previous_search_space, 0.5**6 - 0.1, 1, success_counter, previous_y_min
    )
    current_state, _ = tr.acquire(
        search_space,
        models,
        datasets={OBJECTIVE: dataset},
    )(previous_state)

    assert current_state is not None
    npt.assert_array_almost_equal(current_state.L, tf.cast(0.8, dtype=tf.float64))
    npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)
    npt.assert_array_almost_equal(
        current_state.acquisition_space.upper, tf.constant([0.8, 0.2], dtype=tf.float64)
    )
    assert current_state.success_counter == 0
    assert current_state.failure_counter == 0


def test_turbo_state_deepcopy() -> None:
    tr_state = TURBO.State(
        acquisition_space=Box(tf.constant([1.2]), tf.constant([3.4])),
        L=0.8,
        failure_counter=0,
        success_counter=0,
        y_min=tf.constant(7.8),
    )
    tr_state_copy = copy.deepcopy(tr_state)
    npt.assert_allclose(tr_state_copy.acquisition_space.lower, tr_state.acquisition_space.lower)
    npt.assert_allclose(tr_state_copy.acquisition_space.upper, tr_state.acquisition_space.upper)
    npt.assert_allclose(tr_state_copy.L, tr_state.L)
    npt.assert_allclose(tr_state_copy.failure_counter, tr_state.failure_counter)
    npt.assert_allclose(tr_state_copy.success_counter, tr_state.success_counter)
    npt.assert_allclose(tr_state_copy.y_min, tr_state.y_min)


def test_asynchronous_rule_state_pending_points() -> None:
    pending_points = tf.constant([[1], [2], [3]])

    state = AsynchronousRuleState(pending_points)
    npt.assert_array_equal(pending_points, state.pending_points)


def test_asynchronous_rule_state_raises_incorrect_shape() -> None:
    with pytest.raises(ValueError):
        AsynchronousRuleState(tf.constant([1, 2]))

    with pytest.raises(ValueError):
        AsynchronousRuleState(tf.constant([[[1], [2]]]))


def test_asynchronous_rule_state_has_pending_points() -> None:
    state = AsynchronousRuleState(None)
    assert not state.has_pending_points

    state = AsynchronousRuleState(tf.zeros([0, 2]))
    assert not state.has_pending_points

    pending_points = tf.constant([[1], [2], [3]])
    state = AsynchronousRuleState(pending_points)
    assert state.has_pending_points


def test_asynchronous_rule_remove_points_raises_shape_mismatch() -> None:
    state = AsynchronousRuleState(tf.constant([[1], [2], [3]]))
    with pytest.raises(ValueError):
        state.remove_points(tf.constant([[1, 1]]))

    state = AsynchronousRuleState(tf.constant([[1, 1], [2, 2]]))
    with pytest.raises(ValueError):
        state.remove_points(tf.constant([[1]]))

    state = AsynchronousRuleState(tf.constant([[1, 1], [2, 2]]))
    with pytest.raises(ValueError):
        state.remove_points(tf.constant([[[1, 1], [2, 2]]]))


def test_asynchronous_rule_state_remove_points() -> None:
    # brace yourself, there are many test cases here

    pending_points = tf.constant([[1], [2], [3]])

    # first
    state = AsynchronousRuleState(pending_points)
    state = state.remove_points(tf.constant([[1]]))
    npt.assert_array_equal(state.pending_points, [[2], [3]])

    # neither first nor last
    state = AsynchronousRuleState(pending_points)
    state = state.remove_points(tf.constant([[2]]))
    npt.assert_array_equal(state.pending_points, [[1], [3]])

    # last
    state = AsynchronousRuleState(pending_points)
    state = state.remove_points(tf.constant([[3]]))
    npt.assert_array_equal(state.pending_points, [[1], [2]])

    # unknown point, nothing to remove
    state = AsynchronousRuleState(pending_points)
    state = state.remove_points(tf.constant([[4]]))
    npt.assert_array_equal(state.pending_points, [[1], [2], [3]])

    # duplicated pending points - only remove one occurence
    state = AsynchronousRuleState(tf.constant([[1], [2], [3], [2]]))
    state = state.remove_points(tf.constant([[2]]))
    npt.assert_array_equal(state.pending_points, [[1], [3], [2]])

    # duplicated pending points - remove a dupe and not a dupe
    state = AsynchronousRuleState(tf.constant([[1], [2], [3], [2]]))
    state = state.remove_points(tf.constant([[2], [3]]))
    npt.assert_array_equal(state.pending_points, [[1], [2]])

    # duplicated pending points - remove both dupes
    state = AsynchronousRuleState(tf.constant([[1], [2], [3], [2]]))
    state = state.remove_points(tf.constant([[2], [2]]))
    npt.assert_array_equal(state.pending_points, [[1], [3]])

    # duplicated pending points - dupe, not a dupe, unknown point
    state = AsynchronousRuleState(tf.constant([[1], [2], [3], [2]]))
    state = state.remove_points(tf.constant([[2], [3], [4]]))
    npt.assert_array_equal(state.pending_points, [[1], [2]])

    # remove from empty
    state = AsynchronousRuleState(None)
    state = state.remove_points(tf.constant([[2]]))
    assert not state.has_pending_points

    # remove all
    state = AsynchronousRuleState(pending_points)
    state = state.remove_points(pending_points)
    assert not state.has_pending_points

    # bigger last dimension
    state = AsynchronousRuleState(tf.constant([[1, 1], [2, 3]]))
    state = state.remove_points(tf.constant([[1, 1], [2, 2], [3, 3], [1, 2]]))
    npt.assert_array_equal(state.pending_points, [[2, 3]])


def test_asynchronous_rule_add_pending_points_raises_shape_mismatch() -> None:
    state = AsynchronousRuleState(tf.constant([[1], [2], [3]]))
    with pytest.raises(ValueError):
        state.add_pending_points(tf.constant([[1, 1]]))

    state = AsynchronousRuleState(tf.constant([[1, 1], [2, 2]]))
    with pytest.raises(ValueError):
        state.add_pending_points(tf.constant([[1]]))

    state = AsynchronousRuleState(tf.constant([[1, 1], [2, 2]]))
    with pytest.raises(ValueError):
        state.add_pending_points(tf.constant([[[1, 1], [2, 2]]]))


def test_asynchronous_rule_add_pending_points() -> None:
    state = AsynchronousRuleState(None)
    state = state.add_pending_points(tf.constant([[1]]))
    npt.assert_array_equal(state.pending_points, [[1]])

    state = AsynchronousRuleState(tf.constant([[1], [2]]))
    state = state.add_pending_points(tf.constant([[1]]))
    npt.assert_array_equal(state.pending_points, [[1], [2], [1]])

    state = AsynchronousRuleState(tf.constant([[1, 1], [2, 2]]))
    state = state.add_pending_points(tf.constant([[3, 3], [4, 4]]))
    npt.assert_array_equal(state.pending_points, [[1, 1], [2, 2], [3, 3], [4, 4]])


@pytest.mark.parametrize(
    "batch_size,ga_population_size,ga_n_generations,filter_threshold",
    [
        (-2, 500, 200, 0.1),
        (0, 500, 200, 0.1),
        (10, -2, 200, 0.1),
        (10, 0, 200, 0.1),
        (10, 500, -2, 0.1),
        (10, 500, 0, 0.1),
        (10, 500, 200, -0.1),
        (10, 500, 200, 1.1),
    ],
)
@pytest.mark.qhsri
def test_qhsri_raises_invalid_parameters(
    batch_size: int, ga_population_size: int, ga_n_generations: int, filter_threshold: float
) -> None:
    with pytest.raises(ValueError):
        BatchHypervolumeSharpeRatioIndicator(
            batch_size, ga_population_size, ga_n_generations, filter_threshold
        )


@pytest.mark.parametrize(
    "models",
    [
        {},
        {"foo": QuadraticMeanAndRBFKernel()},
        {"foo": QuadraticMeanAndRBFKernel(), OBJECTIVE: QuadraticMeanAndRBFKernel()},
    ],
)
@pytest.mark.parametrize("datasets", [{}, {OBJECTIVE: empty_dataset([1], [1])}])
@pytest.mark.qhsri
def test_qhsri_raises_for_invalid_models_keys(
    datasets: dict[Tag, Dataset], models: dict[Tag, ProbabilisticModel]
) -> None:
    search_space = Box([-1], [1])
    rule = BatchHypervolumeSharpeRatioIndicator()
    with pytest.raises(ValueError):
        rule.acquire(search_space, models, datasets=datasets)


@pytest.mark.parametrize("models", [{}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}])
@pytest.mark.parametrize(
    "datasets",
    [
        {},
        {"foo": empty_dataset([1], [1])},
        {"foo": empty_dataset([1], [1]), OBJECTIVE: empty_dataset([1], [1])},
    ],
)
@pytest.mark.qhsri
def test_qhsri_raises_for_invalid_dataset_keys(
    datasets: dict[Tag, Dataset], models: dict[Tag, ProbabilisticModel]
) -> None:
    search_space = Box([-1], [1])
    rule = BatchHypervolumeSharpeRatioIndicator()
    with pytest.raises(ValueError):
        rule.acquire(search_space, models, datasets=datasets)

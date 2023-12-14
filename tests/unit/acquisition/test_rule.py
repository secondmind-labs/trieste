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
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability as tfp

from tests.util.misc import empty_dataset, mk_dataset, quadratic, random_seed
from tests.util.models.gpflow.models import (
    GaussianProcess,
    QuadraticMeanAndRBFKernel,
    QuadraticMeanAndRBFKernelWithSamplers,
)
from trieste.acquisition import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    MultipleOptimismNegativeLowerConfidenceBound,
    NegativeLowerConfidenceBound,
    ParallelContinuousThompsonSampling,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
    VectorizedAcquisitionFunctionBuilder,
)
from trieste.acquisition.optimizer import AcquisitionOptimizer
from trieste.acquisition.rule import (
    AcquisitionRule,
    AsynchronousGreedy,
    AsynchronousOptimization,
    AsynchronousRuleState,
    BatchHypervolumeSharpeRatioIndicator,
    BatchTrustRegionBox,
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
    RandomSampling,
    SingleObjectiveTrustRegionBox,
    TREGOBox,
    TURBOBox,
)
from trieste.acquisition.sampler import (
    ExactThompsonSampler,
    GumbelSampler,
    ThompsonSampler,
    ThompsonSamplerFromTrajectory,
)
from trieste.acquisition.utils import copy_to_local_models
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.models.interfaces import TrainableSupportsGetKernel
from trieste.objectives.utils import mk_batch_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box, SearchSpace, TaggedMultiSearchSpace
from trieste.types import State, Tag, TensorType
from trieste.utils.misc import LocalizedTag, get_value_for_tag


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
    npt.assert_allclose(tf.constant(query_point), [[0.0, 0.0]] * num_query_points, atol=1e-3)


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
    npt.assert_allclose(tf.constant(query_points), [[0.0, 0.0]] * num_query_points, atol=1e-3)

    points_or_stateful = acq_rule.acquire_single(
        search_space, QuadraticMeanAndRBFKernel(), dataset=dataset
    )
    if callable(points_or_stateful):
        _, query_points = points_or_stateful(None)
    else:
        query_points = points_or_stateful
    npt.assert_allclose(tf.constant(query_points), [[0.0, 0.0]] * num_query_points, atol=1e-3)
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


@pytest.mark.parametrize(
    "datasets",
    [
        {},
        {"foo": empty_dataset([1], [1])},
        {OBJECTIVE: empty_dataset([1], [1]), "foo": empty_dataset([1], [1])},
    ],
)
@pytest.mark.parametrize(
    "models", [{}, {"foo": QuadraticMeanAndRBFKernel()}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}]
)
def test_trego_raises_for_missing_datasets_key(
    datasets: Mapping[Tag, Dataset], models: dict[Tag, ProbabilisticModel]
) -> None:
    search_space = Box([-1], [1])
    subspace = TREGOBox(search_space)
    with pytest.raises(ValueError, match="a single OBJECTIVE dataset must be provided"):
        subspace.update(models, datasets)


class _Midpoint(AcquisitionRule[TensorType, Box, ProbabilisticModel]):
    def acquire(
        self,
        search_space: Box,
        models: Mapping[Tag, ProbabilisticModel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        return tf.reshape(
            (search_space.upper[None] + search_space.lower[None]) / 2, (-1, search_space.dimension)
        )


@pytest.mark.parametrize(
    "rule, expected_query_point",
    [
        (EfficientGlobalOptimization(NegativeLowerConfidenceBound(0)), [[0.0, 0.0]]),
        (_Midpoint(), [[-0.45, 1.15]]),
    ],
)
def test_trego_for_default_state(
    rule: AcquisitionRule[TensorType, SearchSpace, ProbabilisticModel],
    expected_query_point: TensorType,
) -> None:
    dataset = Dataset(tf.constant([[0.1, 0.2]]), tf.constant([[0.012]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)
    subspace = TREGOBox(search_space)
    tr = BatchTrustRegionBox(subspace, rule)

    model = QuadraticMeanAndRBFKernel()
    state, query_point = tr.acquire_single(search_space, model, dataset=dataset)(None)
    tr.filter_datasets({OBJECTIVE: model}, {OBJECTIVE: dataset})

    assert state is not None
    ret_subspace = state.acquisition_space.get_subspace("0")
    assert isinstance(ret_subspace, TREGOBox)
    npt.assert_array_almost_equal(ret_subspace.lower, lower_bound)
    npt.assert_array_almost_equal(ret_subspace.upper, upper_bound)

    npt.assert_array_almost_equal(query_point, [expected_query_point], 5)
    npt.assert_array_almost_equal(subspace.lower, lower_bound)
    npt.assert_array_almost_equal(subspace.upper, upper_bound)
    npt.assert_array_almost_equal(subspace._y_min, [0.012])
    assert subspace._is_global


def trego_create_subspace(
    search_space: Box,
    acquisition_space: Box,
    dataset: Dataset,
    eps: TensorType,
    previous_y_min: TensorType,
    is_global: bool,
) -> TREGOBox:
    subspace = TREGOBox(search_space, region_index=0)
    subspace.initialize(datasets={OBJECTIVE: dataset})
    subspace._eps = eps
    subspace._y_min = previous_y_min
    subspace._is_global = is_global
    subspace._lower = acquisition_space.lower
    subspace._upper = acquisition_space.upper
    subspace.location = (acquisition_space.lower + acquisition_space.upper) / 2
    return subspace


@pytest.mark.parametrize(
    "rule, expected_query_point",
    [
        (EfficientGlobalOptimization(NegativeLowerConfidenceBound(0)), [[0.0, 0.0]]),
        (_Midpoint(), [[-0.45, 1.15]]),
    ],
)
def test_trego_successful_global_to_global_trust_region_unchanged(
    rule: AcquisitionRule[TensorType, SearchSpace, ProbabilisticModel],
    expected_query_point: TensorType,
) -> None:
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.3]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)
    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = True
    subspace = trego_create_subspace(
        search_space, search_space, dataset, eps, previous_y_min, is_global
    )
    tr = BatchTrustRegionBox(subspace, rule)

    previous_state = BatchTrustRegionBox.State(TaggedMultiSearchSpace([subspace]))
    model = {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    current_state, query_point = tr.acquire(
        search_space,
        model,
        datasets={OBJECTIVE: dataset},
    )(previous_state)
    tr.filter_datasets(model, {OBJECTIVE: dataset})

    assert current_state is not None
    current_subspace = current_state.acquisition_space.get_subspace("0")
    assert isinstance(current_subspace, TREGOBox)
    npt.assert_array_almost_equal(current_subspace.lower, lower_bound)
    npt.assert_array_almost_equal(current_subspace.upper, upper_bound)

    npt.assert_array_almost_equal(subspace._eps, eps)
    assert subspace._is_global
    npt.assert_array_almost_equal(query_point, [expected_query_point], 5)
    npt.assert_array_almost_equal(subspace.lower, lower_bound)
    npt.assert_array_almost_equal(subspace.upper, upper_bound)


@pytest.mark.parametrize(
    "rule",
    [
        EfficientGlobalOptimization(NegativeLowerConfidenceBound(0)),
        _Midpoint(),
    ],
)
def test_trego_for_unsuccessful_global_to_local_trust_region_unchanged(
    rule: AcquisitionRule[TensorType, SearchSpace, ProbabilisticModel]
) -> None:
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.5]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)
    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = True
    acquisition_space = search_space
    subspace = trego_create_subspace(
        search_space, acquisition_space, dataset, eps, previous_y_min, is_global
    )
    tr = BatchTrustRegionBox(subspace, rule)

    previous_subspace_copy = copy.deepcopy(subspace)

    previous_state = BatchTrustRegionBox.State(TaggedMultiSearchSpace([subspace]))
    model = {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    current_state, query_point = tr.acquire(
        search_space,
        model,
        datasets={OBJECTIVE: dataset},
    )(previous_state)
    tr.filter_datasets(model, {OBJECTIVE: dataset})

    assert current_state is not None
    current_subspace = current_state.acquisition_space.get_subspace("0")
    assert isinstance(current_subspace, TREGOBox)
    npt.assert_array_almost_equal(current_subspace.lower, acquisition_space.lower)
    npt.assert_array_almost_equal(current_subspace.upper, acquisition_space.upper)

    npt.assert_array_almost_equal(subspace._eps, eps)
    assert not subspace._is_global
    npt.assert_array_less(lower_bound, subspace.lower)
    npt.assert_array_less(subspace.upper, upper_bound)
    assert query_point[0][0] in previous_subspace_copy


@pytest.mark.parametrize(
    "rule",
    [
        EfficientGlobalOptimization(NegativeLowerConfidenceBound(0)),
        _Midpoint(),
    ],
)
def test_trego_for_successful_local_to_global_trust_region_increased(
    rule: AcquisitionRule[TensorType, SearchSpace, ProbabilisticModel]
) -> None:
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.3]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)
    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = False
    acquisition_space = Box(dataset.query_points[0] - eps, dataset.query_points[0] + eps)
    subspace = trego_create_subspace(
        search_space, acquisition_space, dataset, eps, previous_y_min, is_global
    )
    tr = BatchTrustRegionBox(subspace, rule)

    previous_state = BatchTrustRegionBox.State(TaggedMultiSearchSpace([subspace]))
    model = {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    current_state, _ = tr.acquire(
        search_space,
        model,
        datasets={OBJECTIVE: dataset},
    )(previous_state)
    tr.filter_datasets(model, {OBJECTIVE: dataset})

    assert current_state is not None
    current_subspace = current_state.acquisition_space.get_subspace("0")
    assert isinstance(current_subspace, TREGOBox)
    npt.assert_array_almost_equal(current_subspace.lower, acquisition_space.lower)
    npt.assert_array_almost_equal(current_subspace.upper, acquisition_space.upper)

    npt.assert_array_less(eps, subspace._eps)  # current TR larger than previous
    assert subspace._is_global
    npt.assert_array_almost_equal(subspace.lower, lower_bound)
    npt.assert_array_almost_equal(subspace.upper, upper_bound)


@pytest.mark.parametrize(
    "rule",
    [
        EfficientGlobalOptimization(NegativeLowerConfidenceBound(0)),
        _Midpoint(),
    ],
)
def test_trego_for_unsuccessful_local_to_global_trust_region_reduced(
    rule: AcquisitionRule[TensorType, SearchSpace, ProbabilisticModel]
) -> None:
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.5]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)
    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = False
    acquisition_space = Box(dataset.query_points[0] - eps, dataset.query_points[0] + eps)
    subspace = trego_create_subspace(
        search_space, acquisition_space, dataset, eps, previous_y_min, is_global
    )
    tr = BatchTrustRegionBox(subspace, rule)

    previous_state = BatchTrustRegionBox.State(TaggedMultiSearchSpace([subspace]))
    model = {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    current_state, _ = tr.acquire(
        search_space,
        model,
        datasets={OBJECTIVE: dataset},
    )(previous_state)
    tr.filter_datasets(model, {OBJECTIVE: dataset})

    assert current_state is not None
    current_subspace = current_state.acquisition_space.get_subspace("0")
    assert isinstance(current_subspace, TREGOBox)
    npt.assert_array_almost_equal(current_subspace.lower, acquisition_space.lower)
    npt.assert_array_almost_equal(current_subspace.upper, acquisition_space.upper)

    npt.assert_array_less(subspace._eps, eps)  # current TR smaller than previous
    assert subspace._is_global
    npt.assert_array_almost_equal(subspace.lower, lower_bound)
    npt.assert_array_almost_equal(subspace.upper, upper_bound)


def test_trego_always_uses_global_dataset() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    dataset = Dataset(
        tf.constant([[0.1, 0.2], [-0.1, -0.2], [1.1, 2.3]]), tf.constant([[0.4], [0.5], [0.6]])
    )
    tr = BatchTrustRegionBox(TREGOBox(search_space))  # type: ignore[var-annotated]
    new_data = Dataset(
        tf.constant([[0.5, -0.2], [0.7, 0.2], [1.1, 0.3], [0.5, 0.5]]),
        tf.constant([[0.7], [0.8], [0.9], [1.0]]),
    )
    updated_datasets = tr.filter_datasets(
        {LocalizedTag(OBJECTIVE, 0): QuadraticMeanAndRBFKernel()},
        {OBJECTIVE: dataset + new_data, LocalizedTag(OBJECTIVE, 0): dataset + new_data},
    )

    # Both the local and global datasets should match.
    assert updated_datasets.keys() == {OBJECTIVE, LocalizedTag(OBJECTIVE, 0)}
    # Updated dataset should contain all the points, including ones outside the search space.
    exp_dataset = dataset + new_data
    for key in updated_datasets.keys():
        npt.assert_array_equal(exp_dataset.query_points, updated_datasets[key].query_points)
        npt.assert_array_equal(exp_dataset.observations, updated_datasets[key].observations)


def test_trego_state_deepcopy() -> None:
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.5]]))
    search_space = Box(tf.constant([1.2]), tf.constant([3.4]))
    subspace = trego_create_subspace(
        search_space,
        search_space,
        dataset,
        tf.constant(5.6),
        tf.constant(7.8),
        False,
    )
    tr_state = BatchTrustRegionBox.State(TaggedMultiSearchSpace([subspace]))
    tr_state_copy = copy.deepcopy(tr_state)
    tr_subspace = tr_state.acquisition_space.get_subspace("0")
    tr_subspace_copy = tr_state_copy.acquisition_space.get_subspace("0")
    assert isinstance(tr_subspace, TREGOBox)
    assert isinstance(tr_subspace_copy, TREGOBox)
    npt.assert_allclose(tr_subspace_copy.lower, tr_subspace.lower)
    npt.assert_allclose(tr_subspace_copy.upper, tr_subspace.upper)
    npt.assert_allclose(tr_subspace_copy._eps, tr_subspace._eps)
    npt.assert_allclose(tr_subspace_copy._y_min, tr_subspace._y_min)
    assert tr_subspace_copy._is_global == tr_subspace._is_global


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
    region = TURBOBox(search_space)
    with pytest.raises(ValueError, match="a single OBJECTIVE dataset must be provided"):
        region.update(models, datasets)


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
        TURBOBox(
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
    rule = BatchTrustRegionBox(TURBOBox(search_space))  # type: ignore[var-annotated]
    rule.acquire(search_space, {OBJECTIVE: QuadraticMeanAndRBFKernel()})
    assert rule._subspaces is not None
    region = rule._subspaces[0]
    assert isinstance(region, TURBOBox)

    assert region.L_init == 0.8 * 3.0
    assert region.L_min == (0.5**7) * 3.0
    assert region.L_max == 1.6 * 3.0
    assert region.failure_tolerance == 20
    assert isinstance(rule._rule, DiscreteThompsonSampling)
    assert rule._rule._num_search_space_samples == 2_000

    rule = BatchTrustRegionBox(TURBOBox(search_space), rule=EfficientGlobalOptimization())
    rule.acquire(search_space, {OBJECTIVE: QuadraticMeanAndRBFKernel()})
    assert isinstance(rule._rule, EfficientGlobalOptimization)


@pytest.mark.parametrize("num_query_points", [1, 2])
def test_turbo_acquire_returns_correct_shape(num_query_points: int) -> None:
    dataset = Dataset(
        tf.constant([[0.0, 0.0]], dtype=tf.float64), tf.constant([[0.012]], dtype=tf.float64)
    )
    lower_bound = tf.constant([0.0, 0.0], dtype=tf.float64)
    upper_bound = tf.constant([1.0, 1.0], dtype=tf.float64)
    search_space = Box(lower_bound, upper_bound)
    rule = DiscreteThompsonSampling(1_000, num_query_points)
    tr = BatchTrustRegionBox(TURBOBox(search_space), rule=rule)
    model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset, noise_variance=tf.constant(1e-5, dtype=tf.float64)
    )
    model.kernel = gpflow.kernels.RBF(
        lengthscales=tf.constant([4.0, 1.0], dtype=tf.float64), variance=1e-5
    )  # need a gpflow kernel for TURBOBox
    _, query_points = tr.acquire_single(search_space, model, dataset=dataset)(None)
    npt.assert_array_equal(tf.shape(query_points), [num_query_points, 1, 2])


@random_seed
def test_turbo_for_default_state() -> None:
    dataset = Dataset(
        tf.constant([[0.0, 0.0]], dtype=tf.float64), tf.constant([[0.012]], dtype=tf.float64)
    )
    lower_bound = tf.constant([0.0, 0.0], dtype=tf.float64)
    upper_bound = tf.constant([1.0, 1.0], dtype=tf.float64)
    search_space = Box(lower_bound, upper_bound)
    orig_region = TURBOBox(search_space)
    region = copy.deepcopy(orig_region)
    tr = BatchTrustRegionBox(region, rule=DiscreteThompsonSampling(100, 1))
    model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset, noise_variance=tf.constant(1e-5, dtype=tf.float64)
    )
    model.kernel = gpflow.kernels.RBF(
        lengthscales=tf.constant([4.0, 1.0], dtype=tf.float64), variance=1e-5
    )  # need a gpflow kernel for TURBOBox
    state, query_point = tr.acquire_single(search_space, model, dataset=dataset)(None)
    tr.filter_datasets({OBJECTIVE: model}, {OBJECTIVE: dataset})

    assert state is not None
    state_region = state.acquisition_space.get_subspace("0")
    assert isinstance(state_region, TURBOBox)
    npt.assert_array_almost_equal(state_region.lower, orig_region.lower)
    npt.assert_array_almost_equal(state_region.upper, orig_region.upper)
    npt.assert_array_almost_equal(region.lower, lower_bound)
    npt.assert_array_almost_equal(region.upper, tf.constant([0.8, 0.2], dtype=tf.float64))
    npt.assert_array_almost_equal(region.y_min, [0.012])
    npt.assert_array_almost_equal(region.L, tf.cast(0.8, dtype=tf.float64))
    assert region.success_counter == 0
    assert region.failure_counter == 0


def turbo_create_region(
    search_space: Box,
    acquisition_space: Box,
    L: float,
    failure_counter: int,
    success_counter: int,
    previous_y_min: TensorType,
) -> TURBOBox:
    subspace = TURBOBox(search_space, region_index=0)
    subspace._lower = acquisition_space.lower
    subspace._upper = acquisition_space.upper
    subspace.L = L
    subspace.failure_counter = failure_counter
    subspace.success_counter = success_counter
    subspace.y_min = previous_y_min
    subspace._initialized = True
    return subspace


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
    )  # need a gpflow kernel for TURBOBox
    lower_bound = tf.constant([0.0, 0.0], dtype=tf.float64)
    upper_bound = tf.constant([1.0, 1.0], dtype=tf.float64)
    search_space = Box(lower_bound, upper_bound)
    orig_region = TURBOBox(search_space)
    tr = BatchTrustRegionBox(orig_region)  # type: ignore[var-annotated]

    # success but not enough to trigger size change
    previous_y_min = dataset.observations[0] + 2.0  # force success
    for failure_counter in [0, 1]:
        for success_counter in [0, 1]:
            region = turbo_create_region(
                search_space,
                search_space,
                tf.constant(0.8, dtype=tf.float64),
                failure_counter,
                success_counter,
                previous_y_min,
            )
            previous_state = BatchTrustRegionBox.State(TaggedMultiSearchSpace([region]))
            tr._subspaces = (region,)
            current_state, _ = tr.acquire(
                search_space,
                models,
                datasets={OBJECTIVE: dataset},
            )(previous_state)
            tr.filter_datasets(models, {OBJECTIVE: dataset})

            assert current_state is not None
            state_region = current_state.acquisition_space.get_subspace("0")
            assert isinstance(state_region, TURBOBox)
            npt.assert_array_almost_equal(region.L, tf.cast(0.8, dtype=tf.float64))
            npt.assert_array_almost_equal(region.lower, lower_bound)
            npt.assert_array_almost_equal(region.upper, tf.constant([0.8, 0.2], dtype=tf.float64))
            assert region.success_counter == success_counter + 1
            assert region.failure_counter == 0

    # failure but not enough to trigger size change
    previous_y_min = dataset.observations[0]  # force failure
    for success_counter in [0, 1, 2]:
        region = turbo_create_region(
            search_space,
            search_space,
            tf.constant(0.8, dtype=tf.float64),
            0,
            success_counter,
            previous_y_min,
        )
        previous_state = BatchTrustRegionBox.State(TaggedMultiSearchSpace([region]))
        tr._subspaces = (region,)
        current_state, _ = tr.acquire(
            search_space,
            models,
            datasets={OBJECTIVE: dataset},
        )(previous_state)
        tr.filter_datasets(models, {OBJECTIVE: dataset})

        assert current_state is not None
        state_region = current_state.acquisition_space.get_subspace("0")
        assert isinstance(state_region, TURBOBox)
        npt.assert_array_almost_equal(region.L, tf.cast(0.8, dtype=tf.float64))
        npt.assert_array_almost_equal(region.lower, lower_bound)
        npt.assert_array_almost_equal(region.upper, tf.constant([0.8, 0.2], dtype=tf.float64))
        assert region.success_counter == 0
        assert region.failure_counter == 1


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
    )  # need a gpflow kernel for TURBOBox
    lower_bound = tf.constant([0.0, 0.0], dtype=tf.float64)
    upper_bound = tf.constant([1.0, 1.0], dtype=tf.float64)
    search_space = Box(lower_bound, upper_bound)
    orig_region = TURBOBox(search_space, failure_tolerance=2, region_index=0)
    tr = BatchTrustRegionBox(orig_region)  # type: ignore[var-annotated]

    # hits success limit
    previous_y_min = dataset.observations[0] + 2.0  # force success
    for failure_counter in [0, 1]:
        region = turbo_create_region(
            search_space,
            search_space,
            tf.constant(0.8, dtype=tf.float64),
            failure_counter,
            2,
            previous_y_min,
        )
        previous_state = BatchTrustRegionBox.State(TaggedMultiSearchSpace([region]))
        tr._subspaces = (region,)
        current_state, _ = tr.acquire(
            search_space,
            models,
            datasets={OBJECTIVE: dataset},
        )(previous_state)
        tr.filter_datasets(models, {OBJECTIVE: dataset})

        assert current_state is not None
        state_region = current_state.acquisition_space.get_subspace("0")
        assert isinstance(state_region, TURBOBox)
        npt.assert_array_almost_equal(region.L, tf.cast(1.6, dtype=tf.float64))
        npt.assert_array_almost_equal(region.lower, lower_bound)
        npt.assert_array_almost_equal(region.upper, tf.constant([1.0, 0.4], dtype=tf.float64))
        assert region.success_counter == 0
        assert region.failure_counter == 0
    # hits failure limit
    previous_y_min = dataset.observations[0]  # force failure
    for success_counter in [0, 1, 2]:
        region = turbo_create_region(
            search_space,
            search_space,
            tf.constant(0.8, dtype=tf.float64),
            1,
            success_counter,
            previous_y_min,
        )
        previous_state = BatchTrustRegionBox.State(TaggedMultiSearchSpace([region]))
        tr._subspaces = (region,)
        current_state, _ = tr.acquire(
            search_space,
            models,
            datasets={OBJECTIVE: dataset},
        )(previous_state)
        tr.filter_datasets(models, {OBJECTIVE: dataset})

        assert current_state is not None
        state_region = current_state.acquisition_space.get_subspace("0")
        assert isinstance(state_region, TURBOBox)
        npt.assert_array_almost_equal(region.L, tf.cast(0.4, dtype=tf.float64))
        npt.assert_array_almost_equal(region.lower, lower_bound)
        npt.assert_array_almost_equal(region.upper, tf.constant([0.4, 0.1], dtype=tf.float64))
        assert region.success_counter == 0
        assert region.failure_counter == 0


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
    )  # need a gpflow kernel for TURBOBox
    lower_bound = tf.constant([0.0, 0.0], dtype=tf.float64)
    upper_bound = tf.constant([1.0, 1.0], dtype=tf.float64)
    search_space = Box(lower_bound, upper_bound)
    orig_region = TURBOBox(search_space, region_index=0)
    tr = BatchTrustRegionBox(orig_region)  # type: ignore[var-annotated]

    # first check what happens if L is too small from the start
    previous_y_min = dataset.observations[0]
    failure_counter = 1
    success_counter = 1
    L = tf.constant(1e-10, dtype=tf.float64)
    previous_search_space = Box(lower_bound / 2.0, upper_bound / 5.0)
    region = turbo_create_region(
        search_space, previous_search_space, L, failure_counter, success_counter, previous_y_min
    )
    previous_state = BatchTrustRegionBox.State(TaggedMultiSearchSpace([region]))
    tr._subspaces = (region,)
    current_state, _ = tr.acquire(
        search_space,
        models,
        datasets={OBJECTIVE: dataset},
    )(previous_state)
    tr.filter_datasets(models, {OBJECTIVE: dataset})

    assert current_state is not None
    state_region = current_state.acquisition_space.get_subspace("0")
    assert isinstance(state_region, TURBOBox)
    npt.assert_array_almost_equal(region.L, tf.cast(0.8, dtype=tf.float64))
    npt.assert_array_almost_equal(region.lower, lower_bound)
    npt.assert_array_almost_equal(region.upper, tf.constant([0.8, 0.2], dtype=tf.float64))
    assert region.success_counter == 0
    assert region.failure_counter == 0

    # secondly check what happens if L is too small after triggering decreasing the region
    region = turbo_create_region(
        search_space, previous_search_space, 0.5**6 - 0.1, 1, success_counter, previous_y_min
    )
    previous_state = BatchTrustRegionBox.State(TaggedMultiSearchSpace([region]))
    tr._subspaces = (region,)
    current_state, _ = tr.acquire(
        search_space,
        models,
        datasets={OBJECTIVE: dataset},
    )(previous_state)
    tr.filter_datasets(models, {OBJECTIVE: dataset})

    assert current_state is not None
    state_region = current_state.acquisition_space.get_subspace("0")
    assert isinstance(state_region, TURBOBox)
    npt.assert_array_almost_equal(region.L, tf.cast(0.8, dtype=tf.float64))
    npt.assert_array_almost_equal(region.lower, lower_bound)
    npt.assert_array_almost_equal(region.upper, tf.constant([0.8, 0.2], dtype=tf.float64))
    assert region.success_counter == 0
    assert region.failure_counter == 0


def test_turbo_state_deepcopy() -> None:
    search_space = Box(tf.constant([1.2]), tf.constant([3.4]))
    subspace = turbo_create_region(search_space, search_space, 0.8, 0, 0, tf.constant(7.8))
    tr_state = BatchTrustRegionBox.State(TaggedMultiSearchSpace([subspace]))
    tr_state_copy = copy.deepcopy(tr_state)
    tr_subspace = tr_state.acquisition_space.get_subspace("0")
    tr_subspace_copy = tr_state_copy.acquisition_space.get_subspace("0")
    assert isinstance(tr_subspace, TURBOBox)
    assert isinstance(tr_subspace_copy, TURBOBox)
    npt.assert_allclose(tr_subspace_copy.lower, tr_subspace.lower)
    npt.assert_allclose(tr_subspace_copy.upper, tr_subspace.upper)
    npt.assert_allclose(tr_subspace_copy.L, tr_subspace.L)
    npt.assert_allclose(tr_subspace_copy.failure_counter, tr_subspace.failure_counter)
    npt.assert_allclose(tr_subspace_copy.success_counter, tr_subspace.success_counter)
    npt.assert_allclose(tr_subspace_copy.y_min, tr_subspace.y_min)


@pytest.mark.parametrize(
    "datasets",
    [
        {},
        {"foo": empty_dataset([1], [1])},
        {OBJECTIVE: empty_dataset([1], [1]), "foo": empty_dataset([1], [1])},
    ],
)
def test_trust_region_box_get_dataset_min_raises_if_dataset_is_faulty(
    datasets: Mapping[Tag, Dataset]
) -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    trb = SingleObjectiveTrustRegionBox(search_space)
    with pytest.raises(ValueError, match="a single OBJECTIVE dataset must be provided"):
        trb.get_dataset_min(datasets)


# get_dataset_min picks the minimum x and y values from the dataset.
def test_trust_region_box_get_dataset_min() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    dataset = Dataset(
        tf.constant([[0.1, 0.1], [0.5, 0.5], [0.3, 0.4], [0.8, 0.8], [0.4, 0.4]], dtype=tf.float64),
        tf.constant([[0.0], [0.5], [0.2], [0.1], [1.0]], dtype=tf.float64),
    )
    trb = SingleObjectiveTrustRegionBox(search_space)
    trb._lower = tf.constant([0.2, 0.2], dtype=tf.float64)
    trb._upper = tf.constant([0.7, 0.7], dtype=tf.float64)
    x_min, y_min = trb.get_dataset_min({OBJECTIVE: dataset})
    npt.assert_array_equal(x_min, tf.constant([0.3, 0.4], dtype=tf.float64))
    npt.assert_array_equal(y_min, tf.constant([0.2], dtype=tf.float64))


# get_dataset_min returns first x value and inf y value when points in dataset are outside the
# search space.
def test_trust_region_box_get_dataset_min_outside_search_space() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    dataset = Dataset(
        tf.constant([[1.2, 1.3], [-0.4, -0.5]], dtype=tf.float64),
        tf.constant([[0.7], [0.9]], dtype=tf.float64),
    )
    trb = SingleObjectiveTrustRegionBox(search_space)
    x_min, y_min = trb.get_dataset_min({OBJECTIVE: dataset})
    npt.assert_array_equal(x_min, tf.constant([1.2, 1.3], dtype=tf.float64))
    npt.assert_array_equal(y_min, tf.constant([np.inf], dtype=tf.float64))


# Initialize sets the box to a random location, and sets the eps and y_min values.
def test_trust_region_box_initialize() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    datasets = {
        OBJECTIVE: Dataset(  # Points outside the search space should be ignored.
            tf.constant([[1.2, 1.3], [-0.4, -0.5]], dtype=tf.float64),
            tf.constant([[0.7], [0.9]], dtype=tf.float64),
        )
    }
    trb = SingleObjectiveTrustRegionBox(search_space)
    trb.initialize(datasets=datasets)

    exp_eps = 0.5 * (search_space.upper - search_space.lower) / 5.0 ** (1.0 / 2.0)
    npt.assert_array_equal(trb.eps, exp_eps)
    npt.assert_array_compare(np.less_equal, search_space.lower, trb.location)
    npt.assert_array_compare(np.less_equal, trb.location, search_space.upper)
    npt.assert_array_compare(np.less_equal, search_space.lower, trb.lower)
    npt.assert_array_compare(np.less_equal, trb.upper, search_space.upper)
    npt.assert_array_compare(np.less_equal, trb.upper - trb.lower, 2 * exp_eps)
    npt.assert_array_equal(trb._y_min, tf.constant([np.inf], dtype=tf.float64))


# Update call initializes the box if eps is smaller than min_eps.
def test_trust_region_box_update_initialize() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    datasets = {
        OBJECTIVE: Dataset(  # Points outside the search space should be ignored.
            tf.constant([[1.2, 1.3], [-0.4, -0.5]], dtype=tf.float64),
            tf.constant([[0.7], [0.9]], dtype=tf.float64),
        )
    }
    trb = SingleObjectiveTrustRegionBox(search_space, min_eps=0.5)
    trb.initialize(datasets=datasets)
    location = trb.location

    trb.update(datasets=datasets)
    npt.assert_array_compare(np.not_equal, location, trb.location)
    location = trb.location

    trb.update(datasets=datasets)
    npt.assert_array_compare(np.not_equal, location, trb.location)


# Update call does not initialize the box if eps is larger than min_eps.
def test_trust_region_box_update_no_initialize() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    datasets = {
        OBJECTIVE: Dataset(
            tf.constant([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]], dtype=tf.float64),
            tf.constant([[0.5], [0.0], [1.0]], dtype=tf.float64),
        )
    }
    trb = SingleObjectiveTrustRegionBox(search_space, min_eps=0.1)
    trb.initialize(datasets=datasets)
    trb.location = tf.constant([0.5, 0.5], dtype=tf.float64)
    location = trb.location

    trb.update(datasets=datasets)
    npt.assert_array_equal(location, trb.location)


# Update shrinks/expands box on successful/unsuccessful step.
@pytest.mark.parametrize("success", [True, False])
def test_trust_region_box_update_size(success: bool) -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    datasets = {
        OBJECTIVE: Dataset(
            tf.constant([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]], dtype=tf.float64),
            tf.constant([[0.5], [0.3], [1.0]], dtype=tf.float64),
        )
    }
    trb = SingleObjectiveTrustRegionBox(search_space, min_eps=0.1)
    trb.initialize(datasets=datasets)

    # Ensure there is at least one point captured in the box.
    orig_point = trb.sample(1)
    orig_min = tf.constant([[0.1]], dtype=tf.float64)
    datasets[OBJECTIVE] = Dataset(
        np.concatenate([datasets[OBJECTIVE].query_points, orig_point], axis=0),
        np.concatenate([datasets[OBJECTIVE].observations, orig_min], axis=0),
    )
    trb.update(datasets=datasets)

    eps = trb.eps

    if success:
        # Sample a point from the box.
        new_point = trb.sample(1)
    else:
        # Pick point outside the box.
        new_point = tf.constant([[1.2, 1.3]], dtype=tf.float64)

    # Add a new min point to the dataset.
    new_min = tf.constant([[-0.1]], dtype=tf.float64)
    datasets[OBJECTIVE] = Dataset(
        np.concatenate([datasets[OBJECTIVE].query_points, new_point], axis=0),
        np.concatenate([datasets[OBJECTIVE].observations, new_min], axis=0),
    )
    # Update the box.
    trb.update(datasets=datasets)

    if success:
        # Check that the location is the new min point.
        new_point = np.squeeze(new_point)
        npt.assert_allclose(new_point, trb.location)
        npt.assert_allclose(new_min, trb._y_min)
        # Check that the box is larger by beta.
        npt.assert_allclose(eps / trb._beta, trb.eps)
    else:
        # Check that the location is the old min point.
        orig_point = np.squeeze(orig_point)
        npt.assert_allclose(orig_point, trb.location)
        npt.assert_allclose(orig_min, trb._y_min)
        # Check that the box is smaller by beta.
        npt.assert_allclose(eps * trb._beta, trb.eps)

    # Check the new box bounds.
    npt.assert_allclose(trb.lower, np.maximum(trb.location - trb.eps, search_space.lower))
    npt.assert_allclose(trb.upper, np.minimum(trb.location + trb.eps, search_space.upper))


# Check multi trust region works when no subspace is provided.
@pytest.mark.parametrize(
    "rule, exp_num_subspaces",
    [
        (EfficientGlobalOptimization(), 1),
        (EfficientGlobalOptimization(ParallelContinuousThompsonSampling(), num_query_points=2), 2),
        (RandomSampling(num_query_points=2), 1),
    ],
)
def test_multi_trust_region_box_no_subspace(
    rule: AcquisitionRule[TensorType, SearchSpace, ProbabilisticModel],
    exp_num_subspaces: int,
) -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    mtb = BatchTrustRegionBox(rule=rule)
    mtb.acquire(search_space, {})

    assert mtb._tags is not None
    assert mtb._subspaces is not None
    assert len(mtb._subspaces) == exp_num_subspaces
    for i, (subspace, tag) in enumerate(zip(mtb._subspaces, mtb._tags)):
        assert isinstance(subspace, SingleObjectiveTrustRegionBox)
        assert subspace.global_search_space == search_space
        assert tag == f"{i}"


# Check multi trust region works when a single subspace is provided.
def test_multi_trust_region_box_single_subspace() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    subspace = SingleObjectiveTrustRegionBox(search_space)
    mtb = BatchTrustRegionBox(subspace)  # type: ignore[var-annotated]
    assert mtb._subspaces == (subspace,)
    assert mtb._tags == ("0",)


# When state is None, acquire returns a multi search space of the correct type.
def test_multi_trust_region_box_acquire_no_state() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    dataset = Dataset(
        tf.constant([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]], dtype=tf.float64),
        tf.constant([[0.5], [0.0], [1.0]], dtype=tf.float64),
    )
    datasets = {OBJECTIVE: dataset}
    model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset=dataset, noise_variance=tf.constant(1.0, dtype=tf.float64)
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    models = {OBJECTIVE: model}
    base_rule = EfficientGlobalOptimization(  # type: ignore[var-annotated]
        builder=ParallelContinuousThompsonSampling(), num_query_points=2
    )
    subspaces = [
        SingleObjectiveTrustRegionBox(search_space, beta=0.1, kappa=1e-3, min_eps=1e-1)
        for _ in range(2)
    ]
    prev_subspaces = [copy.deepcopy(subspace) for subspace in subspaces]
    mtb = BatchTrustRegionBox(subspaces, base_rule)
    state, points = mtb.acquire(search_space, models, datasets)(None)
    mtb.filter_datasets(models, datasets)

    assert state is not None
    assert isinstance(state.acquisition_space, TaggedMultiSearchSpace)
    assert len(state.acquisition_space.subspace_tags) == 2

    for index, (tag, point) in enumerate(zip(state.acquisition_space.subspace_tags, points[0])):
        subspace = state.acquisition_space.get_subspace(tag)
        assert isinstance(subspace, SingleObjectiveTrustRegionBox)
        assert subspace.global_search_space == search_space
        assert subspace.region_index == index
        assert subspace._beta == 0.1
        assert subspace._kappa == 1e-3
        assert subspace._min_eps == 1e-1
        assert point in prev_subspaces[index]


def test_multi_trust_region_box_raises_on_mismatched_global_search_space() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    base_rule = EfficientGlobalOptimization(  # type: ignore[var-annotated]
        builder=ParallelContinuousThompsonSampling(), num_query_points=2
    )
    subspaces = [SingleObjectiveTrustRegionBox(search_space) for _ in range(2)]
    mtb = BatchTrustRegionBox(subspaces, base_rule)

    with pytest.raises(AssertionError, match="The global search space of the subspaces should "):
        mtb.acquire(Box([0.0, 0.0], [2.0, 2.0]), {})


def test_multi_trust_region_box_raises_on_mismatched_tags() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    dataset = Dataset(
        tf.constant([[0.0, 0.0], [1.0, 1.0]], dtype=tf.float64),
        tf.constant([[0.0], [1.0]], dtype=tf.float64),
    )
    base_rule = EfficientGlobalOptimization(  # type: ignore[var-annotated]
        builder=ParallelContinuousThompsonSampling(), num_query_points=2
    )
    subspaces = [SingleObjectiveTrustRegionBox(search_space) for _ in range(2)]
    mtb = BatchTrustRegionBox(subspaces, base_rule)

    state = BatchTrustRegionBox.State(
        acquisition_space=TaggedMultiSearchSpace(subspaces, tags=["a", "b"])
    )
    models = {OBJECTIVE: QuadraticMeanAndRBFKernelWithSamplers(dataset)}
    state_func = mtb.acquire(
        search_space,
        models,
        {OBJECTIVE: dataset},
    )
    mtb.filter_datasets(models, {OBJECTIVE: dataset})

    with pytest.raises(AssertionError, match="The tags of the state acquisition space"):
        _, _ = state_func(state)


class TestTrustRegionBox(SingleObjectiveTrustRegionBox):
    def __init__(
        self,
        fixed_location: TensorType,
        global_search_space: SearchSpace,
        beta: float = 0.7,
        kappa: float = 1e-4,
        min_eps: float = 1e-2,
        init_eps: float = 0.07,
    ):
        self._location = fixed_location
        self._init_eps_val = init_eps
        super().__init__(global_search_space, beta, kappa, min_eps)

    @property
    def location(self) -> TensorType:
        return self._location

    @location.setter
    def location(self, location: TensorType) -> None:
        ...

    def _init_eps(self) -> None:
        self.eps = tf.constant(self._init_eps_val, dtype=tf.float64)


# Start with a defined state and dataset. Acquire should return an updated state.
def test_multi_trust_region_box_acquire_with_state() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    init_dataset = Dataset(
        tf.constant([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]], dtype=tf.float64),
        tf.constant([[1.0], [1.0], [1.0]], dtype=tf.float64),
    )
    model = QuadraticMeanAndRBFKernelWithSamplers(
        dataset=init_dataset, noise_variance=tf.constant(1e-6, dtype=tf.float64)
    )
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    models = {OBJECTIVE: model}
    base_rule = EfficientGlobalOptimization(  # type: ignore[var-annotated]
        builder=ParallelContinuousThompsonSampling(), num_query_points=3
    )

    # Third region is close to the first.
    subspaces = [
        TestTrustRegionBox(tf.constant([0.3, 0.3], dtype=tf.float64), search_space),
        TestTrustRegionBox(tf.constant([0.7, 0.7], dtype=tf.float64), search_space),
        TestTrustRegionBox(tf.constant([0.3, 0.3], dtype=tf.float64) + 1e-7, search_space),
    ]
    mtb = BatchTrustRegionBox(subspaces, base_rule)
    state = BatchTrustRegionBox.State(acquisition_space=TaggedMultiSearchSpace(subspaces))
    for subspace in subspaces:
        subspace.initialize(datasets={OBJECTIVE: init_dataset})

    dataset = Dataset(
        init_dataset.query_points,
        tf.constant([[0.1], [0.2], [0.3]], dtype=tf.float64),
    )
    state_func = mtb.acquire(search_space, models, {OBJECTIVE: dataset})
    next_state, points = state_func(state)
    mtb.filter_datasets(models, {OBJECTIVE: dataset})

    assert next_state is not None
    assert points.shape == [1, 3, 2]
    # The regions correspond to first, third and first points in the dataset.
    # First two regions should be updated.
    # The third region should be initialized and not updated, as it is too close to the first
    # subspace.
    for point, subspace, exp_obs, exp_eps in zip(
        points[0],
        subspaces,
        [dataset.observations[0], dataset.observations[2], dataset.observations[0]],
        [0.1, 0.1, 0.07],  # First two regions updated, third region initialized.
    ):
        assert point in subspace
        npt.assert_array_equal(subspace._y_min, exp_obs)
        # Check the box was updated/initialized correctly.
        npt.assert_allclose(subspace.eps, exp_eps)


# Test case with multiple local models and multiple regions for batch trust regions.
# It checks that the correct model is passed to each region, and that the correct dataset is
# passed to each instance of the base rule (note: the base rule is deep-copied for each region).
# This is done by mapping each region to a model. For each region the model has a local quadratic
# shape with the minimum at the center of the region. The overal model is creating by creating
# a product of all regions using that model. The end expected result is that each region should find
# its center after optimization. If the wrong model is being used by a region, then instead it would
# find one of its boundaries.
# Note that the implementation of this test is more general than strictly required. It can support
# fewer models than regions (as long as the number of regions is a multiple of the number of
# models). However, currently trieste only supports either a global model or a one to one mapping
# between models and regions.
@pytest.mark.parametrize("use_global_model", [True, False])
@pytest.mark.parametrize("use_global_dataset", [True, False])
@pytest.mark.parametrize("num_regions", [2, 4])
@pytest.mark.parametrize("num_query_points_per_region", [1, 2])
def test_multi_trust_region_box_with_multiple_models_and_regions(
    use_global_model: bool,
    use_global_dataset: bool,
    num_regions: int,
    num_query_points_per_region: int,
) -> None:
    search_space = Box([0.0, 0.0], [6.0, 6.0])
    base_shift = tf.constant([2.0, 2.0], dtype=tf.float64)  # Common base shift for all regions.
    eps = 0.9
    subspaces = [
        TestTrustRegionBox(base_shift + i, search_space, init_eps=eps) for i in range(num_regions)
    ]

    # Define the models and acquisition functions for each region
    noise_variance = tf.constant(1e-6, dtype=tf.float64)
    kernel_variance = tf.constant(1e-3, dtype=tf.float64)

    global_dataset = Dataset(
        tf.constant([[0.0, 0.0]], dtype=tf.float64),
        tf.constant([[1.0]], dtype=tf.float64),
    )
    init_datasets = {OBJECTIVE: global_dataset}
    models = {}
    r = range(1) if use_global_model else range(num_regions)
    for i in r:
        if use_global_model:
            tag = OBJECTIVE
            num_models = 1
        else:
            tag = LocalizedTag(OBJECTIVE, i)
            num_models = num_regions

        num_regions_per_model = num_regions // num_models
        query_points = tf.stack([base_shift + j for j in range(i, num_regions, num_models)])
        observations = tf.constant([0.0] * num_regions_per_model, dtype=tf.float64)[:, None]

        if not use_global_dataset:
            init_datasets[tag] = Dataset(query_points, observations)

        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(kernel_variance)

        # Overall mean function is a product of local mean functions.
        def mean_function(x: TensorType, i: int = i) -> TensorType:
            return tf.reduce_prod(
                tf.stack(
                    [
                        quadratic(x - tf.cast(base_shift + j, dtype=x.dtype))
                        for j in range(i, num_regions, num_models)
                    ]
                ),
                axis=0,
            )

        models[tag] = GaussianProcess([mean_function], [kernel], noise_variance)
        models[tag]._exp_dataset = (  # type: ignore[attr-defined]
            global_dataset if use_global_dataset else init_datasets[tag]
        )

    if use_global_model:
        # Global model; acquire in parallel.
        num_query_points = num_regions * num_query_points_per_region
    else:
        # Local models; acquire sequentially.
        num_query_points = num_query_points_per_region

    class TestMultipleOptimismNegativeLowerConfidenceBound(
        MultipleOptimismNegativeLowerConfidenceBound
    ):
        # Override the prepare_acquisition_function method to check that the dataset is correct.
        def prepare_acquisition_function(
            self,
            model: ProbabilisticModel,
            dataset: Optional[Dataset] = None,
        ) -> AcquisitionFunction:
            assert dataset is model._exp_dataset  # type: ignore[attr-defined]
            return super().prepare_acquisition_function(model, dataset)

    base_rule = EfficientGlobalOptimization(  # type: ignore[var-annotated]
        builder=TestMultipleOptimismNegativeLowerConfidenceBound(search_space),
        num_query_points=num_query_points,
    )

    mtb = BatchTrustRegionBox(subspaces, base_rule)
    _, points = mtb.acquire(search_space, models, init_datasets)(None)
    mtb.filter_datasets(models, init_datasets)

    npt.assert_array_equal(points.shape, [num_query_points_per_region, num_regions, 2])

    # Each region should find the minimum of its local model, which will be the center of
    # the region.
    exp_points = tf.stack([base_shift + i for i in range(num_regions)])
    exp_points = tf.tile(exp_points[None, :, :], [num_query_points_per_region, 1, 1])
    npt.assert_allclose(points, exp_points)


# This test ensures that the datasets for each region are updated correctly. The datasets should
# contain filtered data, i.e. only points in the respective regions.
@pytest.mark.parametrize(
    "datasets, exp_num_init_points",
    [
        ({OBJECTIVE: mk_dataset([[0.0], [1.0], [2.0]], [[1.0], [1.0], [1.0]])}, 1),
        (
            {
                OBJECTIVE: mk_dataset(
                    [[0.0], [1.0], [0.3], [2.0], [0.7], [1.7]],
                    [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
                )
            },
            2,
        ),
        (
            {
                OBJECTIVE: mk_dataset([[-1.0]], [[-1.0]]),  # Should be ignored.
                LocalizedTag(OBJECTIVE, 0): mk_dataset([[0.0]], [[1.0]]),
                LocalizedTag(OBJECTIVE, 1): mk_dataset([[1.0]], [[1.0]]),
                LocalizedTag(OBJECTIVE, 2): mk_dataset([[2.0]], [[1.0]]),
            },
            1,
        ),
        (
            {
                OBJECTIVE: mk_dataset([[-1.0]], [[-1.0]]),  # Should be ignored.
                LocalizedTag(OBJECTIVE, 0): mk_dataset([[0.0], [1.0]], [[1.0], [1.0]]),
                LocalizedTag(OBJECTIVE, 1): mk_dataset([[2.0], [1.0]], [[1.0], [1.0]]),
                LocalizedTag(OBJECTIVE, 2): mk_dataset([[2.0], [3.0]], [[1.0], [1.0]]),
            },
            1,
        ),
    ],
)
@pytest.mark.parametrize("num_query_points_per_region", [1, 2])
def test_multi_trust_region_box_updated_datasets_are_in_regions(
    datasets: Mapping[Tag, Dataset], exp_num_init_points: int, num_query_points_per_region: int
) -> None:
    num_local_models = 3
    search_space = Box([0.0], [3.0])
    # Non-overlapping regions.
    subspaces = [
        TestTrustRegionBox(tf.constant([i], dtype=tf.float64), search_space, init_eps=0.4)
        for i in range(num_local_models)
    ]
    models = copy_to_local_models(QuadraticMeanAndRBFKernel(), num_local_models)
    base_rule = EfficientGlobalOptimization(  # type: ignore[var-annotated]
        builder=MultipleOptimismNegativeLowerConfidenceBound(search_space),
        num_query_points=num_query_points_per_region,
    )
    rule = BatchTrustRegionBox(subspaces, base_rule)
    _, points = rule.acquire(search_space, models, datasets)(None)
    observer = mk_batch_observer(quadratic)
    new_data = observer(points)
    assert not isinstance(new_data, Dataset)

    updated_datasets = {}
    for tag in new_data:
        _, dataset = get_value_for_tag(datasets, *[tag, LocalizedTag.from_tag(tag).global_tag])
        assert dataset is not None
        updated_datasets[tag] = dataset + new_data[tag]
    filtered_datasets = rule.filter_datasets(models, updated_datasets)

    # Check local datasets.
    for i, subspace in enumerate(subspaces):
        assert (
            filtered_datasets[LocalizedTag(OBJECTIVE, i)].query_points.shape[0]
            == exp_num_init_points + num_query_points_per_region
        )
        assert np.all(subspace.contains(filtered_datasets[LocalizedTag(OBJECTIVE, i)].query_points))

    # Check global dataset.
    assert filtered_datasets[OBJECTIVE].query_points.shape[0] == (
        datasets[OBJECTIVE].query_points.shape[0] + num_local_models * num_query_points_per_region
    )
    # Global dataset should be the unfiltered full dataset.
    npt.assert_array_almost_equal(
        filtered_datasets[OBJECTIVE].query_points, updated_datasets[OBJECTIVE].query_points
    )


def test_multi_trust_region_box_state_deepcopy() -> None:
    search_space = Box([0.0, 0.0], [1.0, 1.0])
    dataset = Dataset(
        tf.constant([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]], dtype=tf.float64),
        tf.constant([[1.0], [1.0], [1.0]], dtype=tf.float64),
    )
    subspaces = [
        SingleObjectiveTrustRegionBox(search_space, beta=0.07, kappa=1e-5, min_eps=1e-3)
        for _ in range(3)
    ]
    for _subspace in subspaces:
        _subspace.initialize(datasets={OBJECTIVE: dataset})
    state = BatchTrustRegionBox.State(acquisition_space=TaggedMultiSearchSpace(subspaces))

    state_copy = copy.deepcopy(state)
    assert state_copy is not state
    assert state_copy.acquisition_space is not state.acquisition_space
    assert state_copy.acquisition_space._spaces is not state.acquisition_space._spaces
    assert state_copy.acquisition_space.subspace_tags == state.acquisition_space.subspace_tags

    for subspace, subspace_copy in zip(
        state.acquisition_space._spaces.values(), state_copy.acquisition_space._spaces.values()
    ):
        assert subspace is not subspace_copy
        assert isinstance(subspace, SingleObjectiveTrustRegionBox)
        assert isinstance(subspace_copy, SingleObjectiveTrustRegionBox)
        assert subspace._beta == subspace_copy._beta
        assert subspace._kappa == subspace_copy._kappa
        assert subspace._min_eps == subspace_copy._min_eps
        npt.assert_array_equal(subspace.eps, subspace_copy.eps)
        npt.assert_array_equal(subspace.location, subspace_copy.location)
        npt.assert_array_equal(subspace._y_min, subspace_copy._y_min)


def test_asynchronous_rule_state_pending_points() -> None:
    pending_points = tf.constant([[1], [2], [3]])

    state = AsynchronousRuleState(pending_points)
    assert state.pending_points is not None
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
    assert state.pending_points is not None
    npt.assert_array_equal(state.pending_points, [[2], [3]])

    # neither first nor last
    state = AsynchronousRuleState(pending_points)
    state = state.remove_points(tf.constant([[2]]))
    assert state.pending_points is not None
    npt.assert_array_equal(state.pending_points, [[1], [3]])

    # last
    state = AsynchronousRuleState(pending_points)
    state = state.remove_points(tf.constant([[3]]))
    assert state.pending_points is not None
    npt.assert_array_equal(state.pending_points, [[1], [2]])

    # unknown point, nothing to remove
    state = AsynchronousRuleState(pending_points)
    state = state.remove_points(tf.constant([[4]]))
    assert state.pending_points is not None
    npt.assert_array_equal(state.pending_points, [[1], [2], [3]])

    # duplicated pending points - only remove one occurence
    state = AsynchronousRuleState(tf.constant([[1], [2], [3], [2]]))
    state = state.remove_points(tf.constant([[2]]))
    assert state.pending_points is not None
    npt.assert_array_equal(state.pending_points, [[1], [3], [2]])

    # duplicated pending points - remove a dupe and not a dupe
    state = AsynchronousRuleState(tf.constant([[1], [2], [3], [2]]))
    state = state.remove_points(tf.constant([[2], [3]]))
    assert state.pending_points is not None
    npt.assert_array_equal(state.pending_points, [[1], [2]])

    # duplicated pending points - remove both dupes
    state = AsynchronousRuleState(tf.constant([[1], [2], [3], [2]]))
    state = state.remove_points(tf.constant([[2], [2]]))
    assert state.pending_points is not None
    npt.assert_array_equal(state.pending_points, [[1], [3]])

    # duplicated pending points - dupe, not a dupe, unknown point
    state = AsynchronousRuleState(tf.constant([[1], [2], [3], [2]]))
    state = state.remove_points(tf.constant([[2], [3], [4]]))
    assert state.pending_points is not None
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
    assert state.pending_points is not None
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
    assert state.pending_points is not None
    npt.assert_array_equal(state.pending_points, [[1]])

    state = AsynchronousRuleState(tf.constant([[1], [2]]))
    state = state.add_pending_points(tf.constant([[1]]))
    assert state.pending_points is not None
    npt.assert_array_equal(state.pending_points, [[1], [2], [1]])

    state = AsynchronousRuleState(tf.constant([[1, 1], [2, 2]]))
    state = state.add_pending_points(tf.constant([[3, 3], [4, 4]]))
    assert state.pending_points is not None
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

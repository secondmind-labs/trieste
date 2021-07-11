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

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import empty_dataset, quadratic, random_seed
from tests.util.model import QuadraticMeanAndRBFKernel
from trieste.acquisition import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    NegativeLowerConfidenceBound,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
)
from trieste.acquisition.optimizer import AcquisitionOptimizer
from trieste.acquisition.rule import (
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
    TrustRegion,
    TrustRegionState,
    continuous_trust_region,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.observer import OBJECTIVE
from trieste.space import Box
from trieste.types import TensorType


def _line_search_maximize(
    search_space: Box, f: AcquisitionFunction, num_query_points: int = 1
) -> TensorType:
    if num_query_points != 1:
        raise ValueError("_line_search_maximizer only defined for batches of size 1")
    if len(search_space.lower) != 1:
        raise ValueError("_line_search_maximizer only defined for search spaces of dimension 1")
    xs = tf.linspace(search_space.lower, search_space.upper, 10 ** 6)
    return xs[tf.squeeze(tf.argmax(f(tf.expand_dims(xs, 1)))), None]


@pytest.mark.parametrize(
    "num_search_space_samples, num_query_points, num_fourier_features",
    [
        (0, 50, 100),
        (-2, 50, 100),
        (10, 0, 100),
        (10, -2, 100),
        (10, 50, 0),
        (10, 50, -2),
    ],
)
def test_discrete_thompson_sampling_raises_for_invalid_init_params(
    num_search_space_samples, num_query_points, num_fourier_features
) -> None:
    with pytest.raises(ValueError):
        DiscreteThompsonSampling(num_search_space_samples, num_query_points, num_fourier_features)


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
    datasets: dict[str, Dataset], models: dict[str, ProbabilisticModel]
) -> None:
    search_space = Box([-1], [1])
    rule = DiscreteThompsonSampling(100, 10)
    with pytest.raises(ValueError):
        rule.acquire(search_space, datasets, models)


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
    datasets: dict[str, Dataset], models: dict[str, ProbabilisticModel]
) -> None:
    search_space = Box([-1], [1])
    rule = DiscreteThompsonSampling(10, 100)
    with pytest.raises(ValueError):
        rule.acquire(search_space, datasets, models)


@pytest.mark.parametrize("num_fourier_features", [None, 100])
@pytest.mark.parametrize("num_query_points", [1, 10])
def test_discrete_thompson_sampling_acquire_returns_correct_shape(
    num_fourier_features: bool, num_query_points: int
) -> None:
    search_space = Box(tf.constant([-2.2, -1.0]), tf.constant([1.3, 3.3]))
    ts = DiscreteThompsonSampling(100, num_query_points, num_fourier_features=num_fourier_features)
    dataset = Dataset(tf.zeros([1, 2], dtype=tf.float64), tf.zeros([1, 1], dtype=tf.float64))
    model = QuadraticMeanAndRBFKernel(noise_variance=tf.constant(1.0, dtype=tf.float64))
    model.kernel = (
        gpflow.kernels.RBF()
    )  # need a gpflow kernel object for random feature decompositions
    query_points = ts.acquire_single(search_space, dataset, model)

    npt.assert_array_equal(query_points.shape, tf.constant([num_query_points, 2]))


def test_efficient_global_optimization_raises_for_no_query_points() -> None:
    with pytest.raises(ValueError):
        EfficientGlobalOptimization(num_query_points=0)


def test_efficient_global_optimization_raises_for_no_batch_fn_with_many_query_points() -> None:
    with pytest.raises(ValueError):
        EfficientGlobalOptimization(num_query_points=2)


@pytest.mark.parametrize("optimizer", [_line_search_maximize, None])
def test_efficient_global_optimization(optimizer: AcquisitionOptimizer[Box]) -> None:
    class NegQuadratic(SingleModelAcquisitionBuilder):
        def prepare_acquisition_function(
            self, dataset: Dataset, model: ProbabilisticModel
        ) -> AcquisitionFunction:
            return lambda x: -quadratic(tf.squeeze(x, -2) - 1)

    search_space = Box([-10], [10])
    ego = EfficientGlobalOptimization(NegQuadratic(), optimizer)
    data, model = empty_dataset([1], [1]), QuadraticMeanAndRBFKernel(x_shift=1)
    query_point = ego.acquire_single(search_space, data, model)
    npt.assert_allclose(query_point, [[1]], rtol=1e-4)
    query_point = ego.acquire(search_space, {OBJECTIVE: data}, {OBJECTIVE: model})
    npt.assert_allclose(query_point, [[1]], rtol=1e-4)


class _JointBatchModelMinusMeanMaximumSingleBuilder(AcquisitionFunctionBuilder):
    def prepare_acquisition_function(
        self, dataset: Mapping[str, Dataset], model: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        return lambda at: -tf.reduce_max(model[OBJECTIVE].predict(at)[0], axis=-2)


@random_seed
def test_joint_batch_acquisition_rule_acquire() -> None:
    search_space = Box(tf.constant([-2.2, -1.0]), tf.constant([1.3, 3.3]))
    num_query_points = 4
    acq = _JointBatchModelMinusMeanMaximumSingleBuilder()
    ego: EfficientGlobalOptimization[Box] = EfficientGlobalOptimization(
        acq, num_query_points=num_query_points
    )
    dataset = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))
    query_point = ego.acquire_single(search_space, dataset, QuadraticMeanAndRBFKernel())

    npt.assert_allclose(query_point, [[0.0, 0.0]] * num_query_points, atol=1e-3)


class _GreedyBatchModelMinusMeanMaximumSingleBuilder(SingleModelGreedyAcquisitionBuilder):
    def prepare_acquisition_function(
        self,
        dataset: Dataset,
        model: ProbabilisticModel,
        pending_points: TensorType = None,
    ) -> AcquisitionFunction:
        if pending_points is None:
            return lambda at: -tf.reduce_max(model.predict(at)[0], axis=-2)
        else:
            best_pending_score = tf.reduce_max(model.predict(pending_points)[0])
            return lambda at: -tf.math.maximum(
                tf.reduce_max(model.predict(at)[0], axis=-2), best_pending_score
            )


@random_seed
def test_greedy_batch_acquisition_rule_acquire() -> None:
    search_space = Box(tf.constant([-2.2, -1.0]), tf.constant([1.3, 3.3]))
    num_query_points = 4
    acq = _GreedyBatchModelMinusMeanMaximumSingleBuilder()
    ego: EfficientGlobalOptimization[Box] = EfficientGlobalOptimization(
        acq, num_query_points=num_query_points
    )
    dataset = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))
    query_point = ego.acquire_single(search_space, dataset, QuadraticMeanAndRBFKernel())

    npt.assert_allclose(query_point, [[0.0, 0.0]] * num_query_points, atol=1e-3)


@pytest.mark.parametrize("datasets", [{}, {"foo": empty_dataset([1], [1])}])
@pytest.mark.parametrize(
    "models", [{}, {"foo": QuadraticMeanAndRBFKernel()}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}]
)
def test_trust_region_raises_for_missing_datasets_key(
    datasets: dict[str, Dataset], models: dict[str, ProbabilisticModel]
) -> None:
    rule = continuous_trust_region()(Box([-1], [1]))
    with pytest.raises(KeyError):
        rule.acquire(datasets, models)


def test_trust_region_for_default_state() -> None:
    dataset = Dataset(tf.constant([[0.1, 0.2]]), tf.constant([[0.012]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    state, new_acquisition_space = continuous_trust_region()(search_space).acquire(
        {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    )(None)

    npt.assert_array_equal(state.acquisition_space.lower, new_acquisition_space.lower)
    npt.assert_array_equal(state.acquisition_space.upper, new_acquisition_space.upper)

    npt.assert_array_almost_equal(state.acquisition_space.lower, lower_bound)
    npt.assert_array_almost_equal(state.acquisition_space.upper, upper_bound)
    npt.assert_array_almost_equal(state.y_min, [0.012])
    assert state.is_global


def test_trust_region_successful_global_to_global_trust_region_unchanged() -> None:
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.3]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = True
    previous_state = TrustRegionState(search_space, eps, previous_y_min, is_global)

    current_state, new_acquisition_space = continuous_trust_region()(search_space).acquire(
        {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    )(previous_state)

    npt.assert_array_equal(current_state.acquisition_space.lower, new_acquisition_space.lower)
    npt.assert_array_equal(current_state.acquisition_space.upper, new_acquisition_space.upper)

    npt.assert_array_almost_equal(current_state.eps, previous_state.eps)
    assert current_state.is_global
    npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)
    npt.assert_array_almost_equal(current_state.acquisition_space.upper, upper_bound)


def test_trust_region_for_unsuccessful_global_to_local_trust_region_unchanged() -> None:
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.5]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = True
    acquisition_space = search_space
    previous_state = TrustRegionState(acquisition_space, eps, previous_y_min, is_global)

    current_state, new_acquisition_space = continuous_trust_region()(search_space).acquire(
        {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    )(previous_state)

    npt.assert_array_equal(current_state.acquisition_space.lower, new_acquisition_space.lower)
    npt.assert_array_equal(current_state.acquisition_space.upper, new_acquisition_space.upper)

    npt.assert_array_almost_equal(current_state.eps, previous_state.eps)
    assert not current_state.is_global
    npt.assert_array_less(lower_bound, current_state.acquisition_space.lower)
    npt.assert_array_less(current_state.acquisition_space.upper, upper_bound)


def test_trust_region_for_successful_local_to_global_trust_region_increased() -> None:
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.3]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = False
    acquisition_space = Box(dataset.query_points[0] - eps, dataset.query_points[0] + eps)
    previous_state = TrustRegionState(acquisition_space, eps, previous_y_min, is_global)

    current_state, new_acquisition_space = continuous_trust_region()(search_space).acquire(
        {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    )(previous_state)

    npt.assert_array_equal(current_state.acquisition_space.lower, new_acquisition_space.lower)
    npt.assert_array_equal(current_state.acquisition_space.upper, new_acquisition_space.upper)

    npt.assert_array_less(previous_state.eps, current_state.eps)  # current TR larger than previous
    assert current_state.is_global
    npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)
    npt.assert_array_almost_equal(current_state.acquisition_space.upper, upper_bound)


def test_trust_region_for_unsuccessful_local_to_global_trust_region_reduced() -> None:
    dataset = Dataset(tf.constant([[0.1, 0.2], [-0.1, -0.2]]), tf.constant([[0.4], [0.5]]))
    lower_bound = tf.constant([-2.2, -1.0])
    upper_bound = tf.constant([1.3, 3.3])
    search_space = Box(lower_bound, upper_bound)

    eps = 0.5 * (search_space.upper - search_space.lower) / 10
    previous_y_min = dataset.observations[0]
    is_global = False
    acquisition_space = Box(dataset.query_points[0] - eps, dataset.query_points[0] + eps)
    previous_state = TrustRegionState(acquisition_space, eps, previous_y_min, is_global)

    current_state, new_acquisition_space = continuous_trust_region()(search_space).acquire(
        {OBJECTIVE: dataset}, {OBJECTIVE: QuadraticMeanAndRBFKernel()}
    )(previous_state)

    npt.assert_array_equal(current_state.acquisition_space.lower, new_acquisition_space.lower)
    npt.assert_array_equal(current_state.acquisition_space.upper, new_acquisition_space.upper)

    npt.assert_array_less(current_state.eps, previous_state.eps)  # current TR smaller than previous
    assert current_state.is_global
    npt.assert_array_almost_equal(current_state.acquisition_space.lower, lower_bound)


def test_trust_region_state_deepcopy() -> None:
    tr_state = TrustRegionState(
        Box(tf.constant([1.2]), tf.constant([3.4])), tf.constant(5.6), tf.constant(7.8), False
    )
    tr_state_copy = copy.deepcopy(tr_state)
    npt.assert_allclose(tr_state_copy.acquisition_space.lower, tr_state.acquisition_space.lower)
    npt.assert_allclose(tr_state_copy.acquisition_space.upper, tr_state.acquisition_space.upper)
    npt.assert_allclose(tr_state_copy.eps, tr_state.eps)
    npt.assert_allclose(tr_state_copy.y_min, tr_state.y_min)
    assert tr_state_copy.is_global == tr_state.is_global

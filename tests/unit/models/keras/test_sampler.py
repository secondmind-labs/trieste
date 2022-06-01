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

from typing import Any, Callable, cast

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import empty_dataset, quadratic, random_seed
from tests.util.models.keras.models import trieste_deep_ensemble_model
from trieste.data import Dataset
from trieste.models.keras import DeepEnsemble, DeepEnsembleTrajectorySampler
from trieste.types import TensorType

_ENSEMBLE_SIZE = 3


@pytest.fixture(name="diversify", params=[True, False])
def _diversify_fixture(request: Any) -> bool:
    return request.param


@pytest.fixture(name="num_evals", params=[9, 19])
def _num_evals_fixture(request: Any) -> int:
    return request.param


@pytest.fixture(name="batch_size", params=[1, 2])
def _batch_size_fixture(request: Any) -> int:
    return request.param


def test_ensemble_trajectory_sampler_returns_trajectory_function_with_correctly_shaped_output(
    num_evals: int,
    batch_size: int,
    dim: int,
    diversify: bool,
) -> None:
    """
    Inputs should be [N,B,d] while output should be [N,B]
    """
    example_data = empty_dataset([dim], [1])
    test_data = tf.random.uniform([num_evals, batch_size, dim])  # [N, B, d]

    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE)

    sampler = DeepEnsembleTrajectorySampler(model, diversify=diversify)
    trajectory = sampler.get_trajectory()

    assert trajectory(test_data).shape == (num_evals, batch_size)


def test_ensemble_trajectory_sampler_returns_deterministic_trajectory(
    num_evals: int, batch_size: int, dim: int, diversify: bool
) -> None:
    """
    Evaluating the same data with the same trajectory multiple times should yield
    exactly the same output.
    """
    example_data = empty_dataset([dim], [1])
    test_data = tf.random.uniform([num_evals, batch_size, dim])  # [N, B, d]

    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE)

    sampler = DeepEnsembleTrajectorySampler(model, diversify=diversify)
    trajectory = sampler.get_trajectory()

    eval_1 = trajectory(test_data)
    eval_2 = trajectory(test_data)

    npt.assert_allclose(eval_1, eval_2)


def test_ensemble_trajectory_sampler_samples_are_distinct_for_new_instances(
    diversify: bool,
) -> None:
    """
    If seeds are not fixed instantiating a new sampler should give us different trajectories.
    """
    example_data = empty_dataset([1], [1])
    test_data = tf.linspace([-10.0], [10.0], 100)
    test_data = tf.expand_dims(test_data, -2)  # [N, 1, d]
    test_data = tf.tile(test_data, [1, 2, 1])  # [N, 2, D]

    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE * 10)

    def _get_trajectory_evaluation(
        model: DeepEnsemble, diversify: bool, seed: int
    ) -> Callable[[TensorType], TensorType]:
        """This allows us to set a different seed for each instance"""

        @random_seed(seed=seed)
        def foo(query_points: TensorType) -> TensorType:
            sampler = DeepEnsembleTrajectorySampler(model, diversify=diversify)
            trajectory = sampler.get_trajectory()
            return trajectory(query_points)

        return foo

    eval_1 = _get_trajectory_evaluation(model, diversify, 0)(test_data)
    eval_2 = _get_trajectory_evaluation(model, diversify, 1)(test_data)

    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(eval_1 - eval_2))
    )  # distinct between seperate draws
    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(eval_1[:, 0] - eval_1[:, 1]))
    )  # distinct for two samples within same draw
    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(eval_2[:, 0] - eval_2[:, 1]))
    )  # distinct for two samples within same draw


@random_seed
def test_ensemble_trajectory_sampler_samples_are_distinct_within_batch(diversify: bool) -> None:
    """
    Samples for elements of the batch should be different. Note that when diversify is not used,
    for small ensembles we could randomnly choose the same network and then we would get the same
    result.
    """
    example_data = empty_dataset([1], [1])
    test_data = tf.linspace([-10.0], [10.0], 100)
    test_data = tf.expand_dims(test_data, -2)  # [N, 1, d]
    test_data = tf.tile(test_data, [1, 2, 1])  # [N, 2, D]

    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE * 3)

    sampler1 = DeepEnsembleTrajectorySampler(model, diversify=diversify)
    trajectory1 = sampler1.get_trajectory()

    sampler2 = DeepEnsembleTrajectorySampler(model, diversify=diversify)
    trajectory2 = sampler2.get_trajectory()

    eval_1 = trajectory1(test_data)
    eval_2 = trajectory2(test_data)

    npt.assert_allclose(eval_1, eval_2)  # same for seperate draws
    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(eval_1[:, 0] - eval_1[:, 1]))
    )  # distinct for two samples within same draw
    npt.assert_array_less(
        1e-1, tf.reduce_max(tf.abs(eval_2[:, 0] - eval_2[:, 1]))
    )  # distinct for two samples within same draw


@random_seed
def test_ensemble_trajectory_sampler_resample_with_new_sampler_does_not_change_old_sampler(
    diversify: bool,
) -> None:
    """
    Generating a new trajectory and resampling it will not affect a previous
    trajectory instance. Before resampling evaluations from both trajectories
    are the same.
    """
    example_data = empty_dataset([1], [1])
    test_data = tf.linspace([-10.0], [10.0], 100)
    test_data = tf.expand_dims(test_data, -2)  # [N, 1, d]
    test_data = tf.tile(test_data, [1, 2, 1])  # [N, 2, D]

    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE * 3)

    sampler = DeepEnsembleTrajectorySampler(model, diversify)
    trajectory1 = sampler.get_trajectory()
    evals_11 = trajectory1(test_data)

    trajectory2 = sampler.get_trajectory()
    evals_21 = trajectory2(test_data)

    trajectory2 = sampler.resample_trajectory(trajectory2)
    evals_22 = trajectory2(test_data)
    evals_12 = trajectory1(test_data)

    npt.assert_array_less(1e-1, tf.reduce_max(tf.abs(evals_22 - evals_21)))
    npt.assert_allclose(evals_11, evals_21)
    npt.assert_allclose(evals_11, evals_12)


@random_seed
def test_ensemble_trajectory_sampler_new_trajectories_diverge(diversify: bool) -> None:
    """
    Generating two trajectories from the same sampler and resampling them will lead to different
    trajectories, even though they were initially the same.
    """
    example_data = empty_dataset([1], [1])
    test_data = tf.linspace([-10.0], [10.0], 100)
    test_data = tf.expand_dims(test_data, -2)  # [N, 1, d]
    test_data = tf.tile(test_data, [1, 2, 1])  # [N, 2, D]

    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE * 3)

    sampler = DeepEnsembleTrajectorySampler(model, diversify=diversify)

    trajectory11 = sampler.get_trajectory()
    evals_11 = trajectory11(test_data)
    trajectory12 = sampler.resample_trajectory(trajectory11)
    evals_12 = trajectory12(test_data)

    trajectory21 = sampler.get_trajectory()
    evals_21 = trajectory21(test_data)
    trajectory22 = sampler.resample_trajectory(trajectory21)
    evals_22 = trajectory22(test_data)

    npt.assert_allclose(evals_11, evals_21)
    npt.assert_array_less(1e-1, tf.reduce_max(tf.abs(evals_22 - evals_12)))
    npt.assert_array_less(1e-1, tf.reduce_max(tf.abs(evals_11 - evals_12)))
    npt.assert_array_less(1e-1, tf.reduce_max(tf.abs(evals_21 - evals_22)))


@random_seed
def test_ensemble_trajectory_sampler_resample_provides_new_samples_without_retracing(
    diversify: bool,
) -> None:
    """
    Resampling a trajectory should be done without retracing, we also check whether we
    get different samples.
    """
    example_data = empty_dataset([1], [1])
    test_data = tf.linspace([-10.0], [10.0], 100)
    test_data = tf.expand_dims(test_data, -2)  # [N, 1, d]

    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE * 2)

    sampler = DeepEnsembleTrajectorySampler(model, diversify=diversify)

    trajectory = sampler.get_trajectory()
    evals_1 = trajectory(test_data)

    trajectory = sampler.resample_trajectory(trajectory)
    evals_2 = trajectory(test_data)

    trajectory = sampler.resample_trajectory(trajectory)
    evals_3 = trajectory(test_data)

    # no retracing
    assert trajectory.__call__._get_tracing_count() == 1  # type: ignore

    # check all samples are different
    npt.assert_array_less(1e-4, tf.abs(evals_1 - evals_2))
    npt.assert_array_less(1e-4, tf.abs(evals_2 - evals_3))
    npt.assert_array_less(1e-4, tf.abs(evals_1 - evals_3))


@random_seed
def test_ensemble_trajectory_sampler_update_trajectory_updates_and_doesnt_retrace(
    diversify: bool,
) -> None:
    """
    We do updates after updating the model, check if model is indeed changed and verify
    that samples are new.
    """
    dim = 3
    batch_size = 2
    num_data = 100

    example_data = empty_dataset([dim], [1])
    test_data = tf.random.uniform([num_data, batch_size, dim])  # [N, B, d]

    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE)

    trajectory_sampler = DeepEnsembleTrajectorySampler(model, diversify=diversify)
    trajectory = trajectory_sampler.get_trajectory()

    eval_before = trajectory(test_data)

    for _ in range(3):
        x_train = tf.random.uniform([num_data, dim])  # [N, d]
        new_dataset = Dataset(x_train, quadratic(x_train))
        model = cast(DeepEnsemble, trajectory_sampler._model)
        old_weights = model.model.get_weights()
        model.optimize(new_dataset)

        trajectory_updated = trajectory_sampler.update_trajectory(trajectory)
        eval_after = trajectory(test_data)

        assert trajectory_updated is trajectory  # check update was in place

        npt.assert_array_less(1e-4, tf.abs(model.model.get_weights()[0], old_weights[0]))
        npt.assert_array_less(
            0.01, tf.reduce_max(tf.abs(eval_before - eval_after))
        )  # two samples should be different

    assert trajectory.__call__._get_tracing_count() == 1  # type: ignore


@random_seed
def test_ensemble_trajectory_sampler_trajectory_on_subsets_same_as_set(diversify: bool) -> None:
    """
    We check if the trajectory called on a set of data is the same as calling it on subsets.
    """
    x_train = 10 * tf.random.uniform([10000, 1])  # [N, d]
    train_data = Dataset(x_train, quadratic(x_train))

    test_data = tf.linspace([-10.0], [10.0], 300)
    test_data = tf.expand_dims(test_data, -2)  # [N, 1, d]
    test_data = tf.tile(test_data, [1, 2, 1])  # [N, 2, d]

    model, _, _ = trieste_deep_ensemble_model(train_data, _ENSEMBLE_SIZE)
    model.optimize(train_data)

    trajectory_sampler = DeepEnsembleTrajectorySampler(model, diversify)
    trajectory = trajectory_sampler.get_trajectory()

    eval_all = trajectory(test_data)
    eval_1 = trajectory(test_data[0:100, :])
    eval_2 = trajectory(test_data[100:200, :])
    eval_3 = trajectory(test_data[200:300, :])

    npt.assert_allclose(eval_all, tf.concat([eval_1, eval_2, eval_3], axis=0), rtol=2e-6)


@random_seed
def test_ensemble_trajectory_sampler_trajectory_is_continuous(diversify: bool) -> None:
    """
    We check if the trajectory seems to give continuous output, for delta x we get delta y.
    """
    x_train = 10 * tf.random.uniform([10000, 1])  # [N, d]
    train_data = Dataset(x_train, quadratic(x_train))

    test_data = tf.linspace([-10.0], [10.0], 300)
    test_data = tf.expand_dims(test_data, -2)  # [N, 1, d]
    test_data = tf.tile(test_data, [1, 2, 1])  # [N, 2, d]

    model, _, _ = trieste_deep_ensemble_model(train_data, _ENSEMBLE_SIZE)

    trajectory_sampler = DeepEnsembleTrajectorySampler(model, diversify=diversify)
    trajectory = trajectory_sampler.get_trajectory()

    npt.assert_array_less(tf.abs(trajectory(test_data + 1e-20) - trajectory(test_data)), 1e-20)

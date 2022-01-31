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

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import empty_dataset, random_seed
from tests.util.models.keras.models import trieste_deep_ensemble_model
from trieste.models.keras import EnsembleTrajectorySampler

_ENSEMBLE_SIZE = 3


@pytest.mark.parametrize("num_evals", [10, 20])
def test_ensemble_trajectory_sampler_returns_trajectory_function_with_correctly_shaped_output(
    num_evals: int,
) -> None:
    example_data = empty_dataset([1], [1])
    test_data = tf.linspace([-10.0], [10.0], num_evals)
    test_data = tf.expand_dims(test_data, -2)  # [N, 1, d]

    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE)

    sampler = EnsembleTrajectorySampler(model)
    trajectory = sampler.get_trajectory()

    assert trajectory(test_data).shape == (num_evals, 1)


def test_ensemble_trajectory_sampler_returns_deterministic_trajectory() -> None:
    example_data = empty_dataset([1], [1])
    test_data = tf.linspace([-10.0], [10.0], 100)
    test_data = tf.expand_dims(test_data, -2)  # [N, 1, d]

    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE)

    sampler = EnsembleTrajectorySampler(model)
    trajectory = sampler.get_trajectory()

    trajectory_eval_1 = trajectory(test_data)
    trajectory_eval_2 = trajectory(test_data)

    npt.assert_allclose(trajectory_eval_1, trajectory_eval_2)


@random_seed
def test_ensemble_trajectory_sampler_samples_are_distinct_for_new_instances() -> None:
    example_data = empty_dataset([1], [1])
    test_data = tf.linspace([-10.0], [10.0], 100)
    test_data = tf.expand_dims(test_data, -2)  # [N, 1, d]

    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE * 2)

    sampler1 = EnsembleTrajectorySampler(model)
    trajectory1 = sampler1.get_trajectory()

    sampler2 = EnsembleTrajectorySampler(model)
    trajectory2 = sampler2.get_trajectory()

    assert tf.reduce_any(trajectory1(test_data) != trajectory2(test_data))


@random_seed
def test_ensemble_trajectory_sampler_resample_provides_new_samples_without_retracing() -> None:
    example_data = empty_dataset([1], [1])
    test_data = tf.linspace([-10.0], [10.0], 100)
    test_data = tf.expand_dims(test_data, -2)  # [N, 1, d]

    model, _, _ = trieste_deep_ensemble_model(example_data, _ENSEMBLE_SIZE * 2)

    sampler = EnsembleTrajectorySampler(model)

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

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
import itertools
import numpy as np

from tests.util.misc import (
    TF_DEBUGGING_ERROR_TYPES,
    mk_dataset,
    random_seed,
)
from tests.util.model import QuadraticMeanAndRBFKernel, LinearMeanAndRBFKernel
from trieste.acquisition.multiobjective.qehvi import BatchMonteCarloHypervolumeExpectedImprovement
from trieste.acquisition.multiobjective.analytic import expected_hv_improvement

from trieste.data import Dataset

from trieste.utils.pareto import Pareto
from trieste.acquisition.multiobjective.function import get_nadir_point
from tests.unit.acquisition.test_function import _dim_two_gp

test_models = (QuadraticMeanAndRBFKernel, LinearMeanAndRBFKernel, QuadraticMeanAndRBFKernel)


@pytest.mark.parametrize("sample_size", [-2, 0])
def test_batch_monte_carlo_hypervolume_expected_improvement_raises_for_invalid_sample_size(
        sample_size: int,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMonteCarloHypervolumeExpectedImprovement(sample_size)


def test_batch_monte_carlo_expected_hypervolume_improvement_raises_for_invalid_jitter() -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        BatchMonteCarloHypervolumeExpectedImprovement(100, jitter=-1.0)


def test_batch_monte_carlo_expected_hypervolume_improvement_raises_for_empty_data() -> None:
    builder = BatchMonteCarloHypervolumeExpectedImprovement(100)
    datastes = {f'OBJECTIVE_{i + 1}': Dataset(tf.zeros([0, 2]), tf.zeros([0, 1])) for i in range(2)}
    model = {f'OBJECTIVE_{i + 1}': QuadraticMeanAndRBFKernel() for i in range(2)}
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(datastes, model)


def test_batch_monte_carlo_expected_hypervolume_improvement_raises_for_model_with_wrong_event_shape() -> None:
    builder = BatchMonteCarloHypervolumeExpectedImprovement(100)
    datastes = {f'OBJECTIVE_{i + 1}': mk_dataset([[0.0, 0.0]], [[0.0, 0.0]]) for i in range(2)}
    models = {f'OBJECTIVE_{i + 1}': _dim_two_gp() for i in range(2)}
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(datastes, models)


@random_seed
@pytest.mark.parametrize(
    "input_dim, num_samples_per_point, training_input, obj_num, variance_scale ,rtol, atol",
    [
        (1, 50_000, tf.constant([[0.3], [0.22], [0.1], [0.35]]), 2, 1.0, 0.01, 1e-2),
        (1, 50_000, tf.constant([[0.3], [0.22], [0.1], [0.35]]), 2, 2.0, 0.01, 1e-2),
        (2, 50_000, tf.constant([[0.0, 0.0], [0.2, 0.5]]), 2, 1.0, 0.01, 1e-2),
        (1, 100_000, tf.constant([[0.3], [0.22], [0.1], [0.35]]), 3, 1.0, 0.01, 1e-2),
    ],
)
def test_batch_monte_carlo_expected_hv_improvement_can_approx_analutical_ehvi(
        input_dim: int, num_samples_per_point: int, training_input: tf.Tensor, obj_num: int,
        variance_scale: float, rtol: float, atol: float) -> None:
    # Note: the test data number grows exponentially with num of obj
    data_num_seg_per_dim = 10  # test data number per input dim
    xs = tf.constant(list(itertools.product(*[list(np.linspace(-1, 1, data_num_seg_per_dim))] * input_dim)))
    xs = tf.cast(xs, dtype=training_input.dtype)

    models = {f'OBJECTIVE{i}': test_models[i](variance_scale) for i in range(obj_num)}
    # gen prepare Pareto
    predicts = [models[model_tag].predict(training_input) for model_tag in models]
    mean, _ = (tf.concat(moment, 1) for moment in zip(*predicts))
    _model_based_tr_dataset = {f'OBJECTIVE{i}': Dataset(training_input, mean[:, i, tf.newaxis])
                               for i in range(obj_num)}

    _model_based_pareto = Pareto(Dataset(tf.zeros_like(mean), mean))
    nadir = get_nadir_point(_model_based_pareto.front)

    qehvi_builder = BatchMonteCarloHypervolumeExpectedImprovement(sample_size=num_samples_per_point)
    qehvi = qehvi_builder.prepare_acquisition_function(_model_based_tr_dataset, models)(xs[:, tf.newaxis, :])
    ehvi = expected_hv_improvement(models, xs, _model_based_pareto, nadir)

    npt.assert_allclose(ehvi, qehvi, rtol=rtol, atol=atol)

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

import itertools

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.unit.acquisition.test_ehvi import _linear_mean_gaussian_process
from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, random_seed
from tests.util.model import QuadraticMeanAndRBFKernel
from trieste.acquisition.function import (
    BatchMonteCarloHypervolumeExpectedImprovement,
    expected_hv_improvement,
    get_reference_point,
)
from trieste.data import Dataset
from trieste.models.model_interfaces import ModelStack
from trieste.utils.pareto import Pareto

test_models = (QuadraticMeanAndRBFKernel, _linear_mean_gaussian_process, QuadraticMeanAndRBFKernel)


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
    dataste = Dataset(tf.zeros([0, 2]), tf.zeros([0, 1]))
    model = ModelStack(*[(QuadraticMeanAndRBFKernel(), 1) for _ in range(2)])
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        builder.prepare_acquisition_function(dataste, model)


@random_seed
@pytest.mark.parametrize(
    "input_dim, num_samples_per_point, training_input, obj_num, variance_scale ,rtol, atol",
    [
        (1, 50_000, tf.constant([[0.3], [0.22], [0.1], [0.35]]), 2, 1.0, 0.01, 1e-2),
        (1, 50_000, tf.constant([[0.3], [0.22], [0.1], [0.35]]), 2, 2.0, 0.01, 1e-2),
        (2, 50_000, tf.constant([[0.0, 0.0], [0.2, 0.5]]), 2, 1.0, 0.01, 1e-2),
        # (1, 100_000, tf.constant([[0.3], [0.22], [0.1], [0.35]]), 3, 1.0, 0.01, 1e-2),
    ],
)
def test_batch_monte_carlo_expected_hv_improvement_can_approx_analytical_ehvi(
    input_dim: int,
    num_samples_per_point: int,
    training_input: tf.Tensor,
    obj_num: int,
    variance_scale: float,
    rtol: float,
    atol: float,
) -> None:
    # Note: the test data number grows exponentially with num of obj
    data_num_seg_per_dim = 10  # test data number per input dim
    xs = tf.constant(
        list(itertools.product(*[list(np.linspace(-1, 1, data_num_seg_per_dim))] * input_dim))
    )
    xs = tf.cast(xs, dtype=training_input.dtype)

    model = ModelStack(*[(test_models[_](), 1) for _ in range(obj_num)])
    # gen prepare Pareto
    mean, _ = model.predict(training_input)
    _model_based_tr_dataset = Dataset(training_input, mean)

    _model_based_pareto = Pareto(mean)
    _reference_pt = get_reference_point(_model_based_pareto.front)

    qehvi_builder = BatchMonteCarloHypervolumeExpectedImprovement(sample_size=num_samples_per_point)
    qehvi = qehvi_builder.prepare_acquisition_function(_model_based_tr_dataset, model)(
        tf.expand_dims(xs, -2)
    )
    ehvi = expected_hv_improvement(model, _model_based_pareto, _reference_pt)(
        tf.expand_dims(xs, -2)
    )

    npt.assert_allclose(ehvi, qehvi, rtol=rtol, atol=atol)


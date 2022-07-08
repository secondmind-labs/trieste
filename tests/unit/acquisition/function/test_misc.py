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

from unittest.mock import MagicMock

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, ShapeLike, various_shapes
from tests.util.models.gpflow.models import QuadraticMeanAndRBFKernel
from trieste.acquisition import MakePositive, ProbabilityOfFeasibility, probability_of_feasibility
from trieste.models import ProbabilisticModel


@pytest.mark.parametrize(
    "threshold, at, expected",
    [
        (0.0, tf.constant([[0.0]]), 0.5),
        # values looked up on a standard normal table
        (2.0, tf.constant([[1.0]]), 0.5 + 0.34134),
        (-0.25, tf.constant([[-0.5]]), 0.5 - 0.19146),
    ],
)
def test_probability_of_feasibility(threshold: float, at: tf.Tensor, expected: float) -> None:
    actual = probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold)(at)
    npt.assert_allclose(actual, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "at",
    [
        tf.constant([[0.0]], tf.float64),
        tf.constant([[-3.4]], tf.float64),
        tf.constant([[0.2]], tf.float64),
    ],
)
@pytest.mark.parametrize("threshold", [-2.3, 0.2])
def test_probability_of_feasibility_builder_builds_pof(threshold: float, at: tf.Tensor) -> None:
    builder = ProbabilityOfFeasibility(threshold)
    acq = builder.prepare_acquisition_function(QuadraticMeanAndRBFKernel())
    expected = probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold)(at)

    npt.assert_allclose(acq(at), expected)


@pytest.mark.parametrize("shape", various_shapes() - {()})
def test_probability_of_feasibility_raises_on_non_scalar_threshold(shape: ShapeLike) -> None:
    threshold = tf.ones(shape)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold)


@pytest.mark.parametrize("shape", [[], [0], [2], [2, 1], [1, 2, 1]])
def test_probability_of_feasibility_raises_on_invalid_at_shape(shape: ShapeLike) -> None:
    at = tf.ones(shape)
    pof = probability_of_feasibility(QuadraticMeanAndRBFKernel(), 0.0)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        pof(at)


@pytest.mark.parametrize("shape", various_shapes() - {()})
def test_probability_of_feasibility_builder_raises_on_non_scalar_threshold(
    shape: ShapeLike,
) -> None:
    threshold = tf.ones(shape)
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        ProbabilityOfFeasibility(threshold)


@pytest.mark.parametrize("at", [tf.constant([[0.0]], tf.float64)])
@pytest.mark.parametrize("threshold", [-2.3, 0.2])
def test_probability_of_feasibility_builder_updates_without_retracing(
    threshold: float, at: tf.Tensor
) -> None:
    builder = ProbabilityOfFeasibility(threshold)
    model = QuadraticMeanAndRBFKernel()
    expected = probability_of_feasibility(QuadraticMeanAndRBFKernel(), threshold)(at)
    acq = builder.prepare_acquisition_function(model)
    assert acq._get_tracing_count() == 0  # type: ignore
    npt.assert_allclose(acq(at), expected)
    assert acq._get_tracing_count() == 1  # type: ignore
    up_acq = builder.update_acquisition_function(acq, model)
    assert up_acq == acq
    npt.assert_allclose(acq(at), expected)
    assert acq._get_tracing_count() == 1  # type: ignore


@pytest.mark.parametrize("in_place_update", [False, True])
def test_make_positive(in_place_update: bool) -> None:
    base = MagicMock()
    base.prepare_acquisition_function.side_effect = lambda *args: lambda x: x
    if in_place_update:
        base.update_acquisition_function.side_effect = lambda f, *args: f
    else:
        base.update_acquisition_function.side_effect = lambda *args: lambda x: 3.0
    builder: MakePositive[ProbabilisticModel] = MakePositive(base)

    model = QuadraticMeanAndRBFKernel()
    acq_fn = builder.prepare_acquisition_function(model)
    xs = tf.linspace([-1], [1], 10)
    npt.assert_allclose(acq_fn(xs), tf.math.log(1 + tf.math.exp(xs)))
    assert base.prepare_acquisition_function.call_count == 1
    assert base.update_acquisition_function.call_count == 0

    up_acq_fn = builder.update_acquisition_function(acq_fn, model)
    assert base.prepare_acquisition_function.call_count == 1
    assert base.update_acquisition_function.call_count == 1
    if in_place_update:
        assert up_acq_fn is acq_fn
    else:
        npt.assert_allclose(up_acq_fn(xs), tf.math.log(1 + tf.math.exp(3.0)))

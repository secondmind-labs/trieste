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

from typing import Mapping, Optional, cast

import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, empty_dataset, quadratic, random_seed
from tests.util.models.gpflow.models import (
    GaussianProcess,
    GaussianProcessWithBatchSamplers,
    QuadraticMeanAndRBFKernel,
)
from trieste.acquisition import AcquisitionFunction, AcquisitionFunctionBuilder
from trieste.acquisition.interface import GreedyAcquisitionFunctionBuilder
from trieste.acquisition.multi_objective import (
    HIPPO,
    ExpectedConstrainedHypervolumeImprovement,
    ExpectedHypervolumeImprovement,
    hippo_penalizer,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.types import TensorType


def _mo_test_model(
    num_obj: int, *kernel_amplitudes: float | TensorType | None, with_reparam_sampler: bool = True
) -> GaussianProcess:
    means = [quadratic, lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), quadratic]
    kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(k_amp) for k_amp in kernel_amplitudes]
    if with_reparam_sampler:
        return GaussianProcessWithBatchSamplers(means[:num_obj], kernels[:num_obj])
    else:
        return GaussianProcess(means[:num_obj], kernels[:num_obj])


class _Certainty(AcquisitionFunctionBuilder[ProbabilisticModel]):
    def prepare_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModel],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> AcquisitionFunction:
        return lambda x: tf.ones((tf.shape(x)[0], 1), dtype=tf.float64)


def test_hippo_builder_raises_for_empty_data() -> None:
    num_obj = 3
    dataset = {"": empty_dataset([2], [num_obj])}
    model = {"": QuadraticMeanAndRBFKernel()}
    hippo = cast(GreedyAcquisitionFunctionBuilder[QuadraticMeanAndRBFKernel], HIPPO(""))

    with pytest.raises(tf.errors.InvalidArgumentError):
        hippo.prepare_acquisition_function(model, dataset)
    with pytest.raises(tf.errors.InvalidArgumentError):
        hippo.prepare_acquisition_function(model, dataset)


@pytest.mark.parametrize("at", [tf.constant([[0.0], [1.0]]), tf.constant([[[0.0], [1.0]]])])
def test_hippo_penalizer_raises_for_invalid_batch_size(at: TensorType) -> None:
    pending_points = tf.zeros([1, 2], dtype=tf.float64)
    hp = hippo_penalizer(QuadraticMeanAndRBFKernel(), pending_points)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        hp(at)


def test_hippo_penalizer_raises_for_empty_pending_points() -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        hippo_penalizer(QuadraticMeanAndRBFKernel(), None)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        hippo_penalizer(QuadraticMeanAndRBFKernel(), tf.zeros([0, 2]))


def test_hippo_penalizer_update_raises_for_empty_pending_points() -> None:
    pending_points = tf.zeros([1, 2], dtype=tf.float64)
    hp = hippo_penalizer(QuadraticMeanAndRBFKernel(), pending_points)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        hp.update(None)

    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        hp.update(tf.zeros([0, 2]))


@pytest.mark.parametrize(
    "point_to_penalize", [tf.constant([[[0.0, 1.0]]]), tf.constant([[[3.0, 4.0]]])]
)
def test_hippo_penalizer_penalizes_pending_point(point_to_penalize: TensorType) -> None:
    pending_points = tf.constant([[0.0, 1.0], [2.0, 3.0], [3.0, 4.0]])
    hp = hippo_penalizer(QuadraticMeanAndRBFKernel(), pending_points)

    penalty = hp(point_to_penalize)

    # if the point is already collected, it shall be penalized to 0
    npt.assert_allclose(penalty, tf.zeros((1, 1)))


@random_seed
@pytest.mark.parametrize(
    "base_builder",
    [
        ExpectedHypervolumeImprovement().using(""),
        ExpectedConstrainedHypervolumeImprovement("", _Certainty(), 0.0),
    ],
)
def test_hippo_penalized_acquisitions_match_base_acquisition(
    base_builder: AcquisitionFunctionBuilder[ProbabilisticModel],
) -> None:
    data = {"": Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))}
    model = {"": _mo_test_model(2, *[None] * 2)}

    hippo_acq_builder: HIPPO[ProbabilisticModel] = HIPPO(
        "", base_acquisition_function_builder=base_builder
    )
    hippo_acq = hippo_acq_builder.prepare_acquisition_function(model, data, None)

    base_acq = base_builder.prepare_acquisition_function(model, data)

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))
    hippo_acq_values = hippo_acq(xs[..., None, :])
    base_acq_values = base_acq(xs[..., None, :])

    npt.assert_array_equal(hippo_acq_values, base_acq_values)


@random_seed
@pytest.mark.parametrize(
    "base_builder",
    [
        ExpectedHypervolumeImprovement().using(""),
        ExpectedConstrainedHypervolumeImprovement("", _Certainty(), 0.0),
    ],
)
def test_hippo_penalized_acquisitions_combine_base_and_penalization_correctly(
    base_builder: AcquisitionFunctionBuilder[ProbabilisticModel],
) -> None:
    data = {"": Dataset(tf.zeros([3, 2], dtype=tf.float64), tf.ones([3, 2], dtype=tf.float64))}
    model = {"": _mo_test_model(2, *[None] * 2)}
    pending_points = tf.zeros([2, 2], dtype=tf.float64)

    hippo_acq_builder: HIPPO[ProbabilisticModel] = HIPPO(
        "", base_acquisition_function_builder=base_builder
    )
    hippo_acq = hippo_acq_builder.prepare_acquisition_function(model, data, pending_points)
    base_acq = base_builder.prepare_acquisition_function(model, data)
    penalizer = hippo_penalizer(model[""], pending_points)
    assert hippo_acq._get_tracing_count() == 0  # type: ignore

    x_range = tf.linspace(0.0, 1.0, 11)
    x_range = tf.cast(x_range, dtype=tf.float64)
    xs = tf.reshape(tf.stack(tf.meshgrid(x_range, x_range, indexing="ij"), axis=-1), (-1, 2))

    hippo_acq_values = hippo_acq(xs[..., None, :])
    base_acq_values = base_acq(xs[..., None, :])
    penalty_values = penalizer(xs[..., None, :])
    penalized_base_acq = tf.math.exp(tf.math.log(base_acq_values) + tf.math.log(penalty_values))

    npt.assert_array_equal(hippo_acq_values, penalized_base_acq)
    assert hippo_acq._get_tracing_count() == 1  # type: ignore

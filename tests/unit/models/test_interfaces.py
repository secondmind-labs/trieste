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

from collections.abc import Callable, Sequence
from typing import Optional

import gpflow
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.keras import tf_keras

from tests.util.misc import assert_datasets_allclose, quadratic, random_seed
from tests.util.models.gpflow.models import (
    GaussianProcessWithBatchSamplers,
    PseudoTrainableProbModel,
    QuadraticMeanAndRBFKernel,
    gpr_model,
    sgpr_model,
)
from tests.util.models.models import fnc_2sin_x_over_3, fnc_3x_plus_10
from trieste.data import Dataset
from trieste.models import TrainableModelStack, TrainableProbabilisticModel
from trieste.models.interfaces import (
    EncodedProbabilisticModel,
    EncodedSupportsPredictJoint,
    EncodedSupportsPredictY,
    EncodedTrainableProbabilisticModel,
    TrainablePredictJointReparamModelStack,
    TrainablePredictYModelStack,
    TrainableSupportsPredictJoint,
    TrainableSupportsPredictJointHasReparamSampler,
)
from trieste.models.utils import get_last_optimization_result, optimize_model_and_save_result
from trieste.space import EncoderFunction
from trieste.types import TensorType


class _QuadraticModel(
    GaussianProcessWithBatchSamplers, PseudoTrainableProbModel, TrainableSupportsPredictJoint
):
    def __init__(
        self,
        mean_shifts: list[float],
        kernel_amplitudes: list[float],
        observations_noise: float = 1.0,
    ):
        super().__init__(
            [(lambda y: lambda x: quadratic(x) + y)(shift) for shift in mean_shifts],
            [tfp.math.psd_kernels.ExponentiatedQuadratic(x) for x in kernel_amplitudes],
            observations_noise,
        )


def _model_stack() -> tuple[
    TrainablePredictJointReparamModelStack,
    tuple[TrainableSupportsPredictJointHasReparamSampler, ...],
]:
    model01 = _QuadraticModel([0.0, 0.5], [1.0, 0.3])
    model2 = _QuadraticModel([2.0], [2.0])
    model3 = _QuadraticModel([-1.0], [0.1])
    return TrainablePredictJointReparamModelStack((model01, 2), (model2, 1), (model3, 1)), (
        model01,
        model2,
        model3,
    )


def test_model_stack_predict() -> None:
    stack, (model01, model2, model3) = _model_stack()
    assert all(
        isinstance(model, TrainableProbabilisticModel) for model in (stack, model01, model2, model3)
    )
    query_points = tf.random.uniform([5, 7, 3])
    mean, var = stack.predict(query_points)

    assert mean.shape == [5, 7, 4]
    assert var.shape == [5, 7, 4]

    mean01, var01 = model01.predict(query_points)
    mean2, var2 = model2.predict(query_points)
    mean3, var3 = model3.predict(query_points)

    npt.assert_allclose(mean[..., :2], mean01)
    npt.assert_allclose(mean[..., 2:3], mean2)
    npt.assert_allclose(mean[..., 3:], mean3)
    npt.assert_allclose(var[..., :2], var01)
    npt.assert_allclose(var[..., 2:3], var2)
    npt.assert_allclose(var[..., 3:], var3)


def test_model_stack_predict_joint() -> None:
    stack, (model01, model2, model3) = _model_stack()
    query_points = tf.random.uniform([5, 7, 3])
    mean, cov = stack.predict_joint(query_points)

    assert mean.shape == [5, 7, 4]
    assert cov.shape == [5, 4, 7, 7]

    mean01, cov01 = model01.predict_joint(query_points)
    mean2, cov2 = model2.predict_joint(query_points)
    mean3, cov3 = model3.predict_joint(query_points)

    npt.assert_allclose(mean[..., :2], mean01)
    npt.assert_allclose(mean[..., 2:3], mean2)
    npt.assert_allclose(mean[..., 3:], mean3)
    npt.assert_allclose(cov[..., :2, :, :], cov01)
    npt.assert_allclose(cov[..., 2:3, :, :], cov2)
    npt.assert_allclose(cov[..., 3:, :, :], cov3)


def test_model_stack_predict_y() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model1 = gpr_model(x, fnc_3x_plus_10(x))
    model2 = sgpr_model(x, fnc_2sin_x_over_3(x))
    stack = TrainablePredictYModelStack((model1, 1), (model2, 1))
    mean, variance = stack.predict_y(x)
    npt.assert_allclose(mean[:, 0:1], model1.predict_y(x)[0])
    npt.assert_allclose(mean[:, 1:2], model2.predict_y(x)[0])
    npt.assert_allclose(variance[:, 0:1], model1.predict_y(x)[1])
    npt.assert_allclose(variance[:, 1:2], model2.predict_y(x)[1])


@random_seed
def test_model_stack_sample() -> None:
    query_points = tf.random.uniform([5, 7, 3], maxval=10.0)
    stack, (model01, model2, model3) = _model_stack()
    samples = stack.sample(query_points, 10_000)

    assert samples.shape == [5, 10_000, 7, 4]

    mean = tf.reduce_mean(samples, axis=1)
    var = tf.math.reduce_variance(samples, axis=1)

    mean01, var01 = model01.predict(query_points)
    mean2, var2 = model2.predict(query_points)
    mean3, var3 = model3.predict(query_points)

    npt.assert_allclose(mean[..., :2], mean01, rtol=0.01)
    npt.assert_allclose(mean[..., 2:3], mean2, rtol=0.01)
    npt.assert_allclose(mean[..., 3:], mean3, rtol=0.01)
    npt.assert_allclose(var[..., :2], var01, rtol=0.04)
    npt.assert_allclose(var[..., 2:3], var2, rtol=0.04)
    npt.assert_allclose(var[..., 3:], var3, rtol=0.04)


def test_model_stack_training() -> None:
    class Model(GaussianProcessWithBatchSamplers, TrainableProbabilisticModel):
        def __init__(
            self,
            mean_functions: Sequence[Callable[[TensorType], TensorType]],
            kernels: Sequence[tfp.math.psd_kernels.PositiveSemidefiniteKernel],
            output_dims: slice,
        ):
            super().__init__(mean_functions, kernels)
            self._output_dims = output_dims

        def _assert_data(self, dataset: Dataset) -> None:
            qp, obs = dataset.astuple()
            expected_obs = data.observations[..., self._output_dims]
            assert_datasets_allclose(dataset, Dataset(qp, expected_obs))

        optimize = _assert_data
        update = _assert_data

    rbf = tfp.math.psd_kernels.ExponentiatedQuadratic()
    model01 = Model([quadratic, quadratic], [rbf, rbf], slice(0, 2))
    model2 = Model([quadratic], [rbf], slice(2, 3))
    model3 = Model([quadratic], [rbf], slice(3, 4))

    stack = TrainableModelStack((model01, 2), (model2, 1), (model3, 1))
    data = Dataset(tf.random.uniform([5, 7, 3]), tf.random.uniform([5, 7, 4]))
    stack.update(data)
    optimize_model_and_save_result(stack, data)
    assert get_last_optimization_result(stack) == [None] * 3


def test_model_stack_reparam_sampler_raises_for_submodels_without_reparam_sampler() -> None:
    model01 = _QuadraticModel([0.0, 0.5], [1.0, 0.3])
    model2 = QuadraticMeanAndRBFKernel()
    model_stack = TrainableModelStack((model01, 2), (model2, 1))  # type: ignore

    with pytest.raises(AttributeError):
        model_stack.reparam_sampler(1)  # type: ignore


def test_model_stack_reparam_sampler() -> None:
    query_points = tf.random.uniform([5, 7, 3], maxval=10.0)
    stack, (model01, model2, model3) = _model_stack()
    sampler = stack.reparam_sampler(10_000)

    samples = sampler.sample(query_points)

    assert samples.shape == [5, 10_000, 7, 4]

    mean = tf.reduce_mean(samples, axis=1)
    var = tf.math.reduce_variance(samples, axis=1)

    mean01, var01 = model01.predict(query_points)
    mean2, var2 = model2.predict(query_points)
    mean3, var3 = model3.predict(query_points)

    npt.assert_allclose(mean[..., :2], mean01, rtol=0.01)
    npt.assert_allclose(mean[..., 2:3], mean2, rtol=0.01)
    npt.assert_allclose(mean[..., 3:], mean3, rtol=0.01)
    npt.assert_allclose(var[..., :2], var01, rtol=0.04)
    npt.assert_allclose(var[..., 2:3], var2, rtol=0.04)
    npt.assert_allclose(var[..., 3:], var3, rtol=0.04)


class _EncodedModel(
    EncodedTrainableProbabilisticModel,
    EncodedSupportsPredictJoint,
    EncodedSupportsPredictY,
    EncodedProbabilisticModel,
):
    def __init__(self, encoder: EncoderFunction | None = None) -> None:
        self.dataset: Dataset | None = None
        self._encoder = (lambda x: x + 1) if encoder is None else encoder

    @property
    def encoder(self) -> EncoderFunction | None:
        return self._encoder

    def predict_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return query_points, query_points

    def sample_encoded(self, query_points: TensorType, num_samples: int) -> TensorType:
        return tf.tile(tf.expand_dims(query_points, 0), [num_samples, 1, 1])

    def log(self, dataset: Optional[Dataset] = None) -> None:
        pass

    def update_encoded(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def optimize_encoded(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def predict_joint_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        b, d = query_points.shape
        return query_points, tf.zeros([d, b, b])

    def predict_y_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.predict_encoded(query_points)


def test_encoded_probabilistic_model() -> None:
    model = _EncodedModel()
    query_points = tf.random.uniform([3, 5])
    mean, var = model.predict(query_points)
    npt.assert_allclose(mean, query_points + 1)
    npt.assert_allclose(var, query_points + 1)
    samples = model.sample(query_points, 10)
    assert len(samples) == 10
    for i in range(10):
        npt.assert_allclose(samples[i], query_points + 1)


def test_encoded_trainable_probabilistic_model() -> None:
    model = _EncodedModel()
    assert model.dataset is None
    for method in model.update, model.optimize:
        query_points = tf.random.uniform([3, 5])
        observations = tf.random.uniform([3, 1])
        dataset = Dataset(query_points, observations)
        method(dataset)
        assert model.dataset is not None
        # no idea why mypy thinks model.dataset couldn't have changed here
        npt.assert_allclose(  # type: ignore[unreachable]
            model.dataset.query_points, query_points + 1
        )
        npt.assert_allclose(model.dataset.observations, observations)


def test_encoded_supports_predict_joint() -> None:
    model = _EncodedModel()
    query_points = tf.random.uniform([3, 5])
    mean, var = model.predict_joint(query_points)
    npt.assert_allclose(mean, query_points + 1)
    npt.assert_allclose(var, tf.zeros([5, 3, 3]))


def test_encoded_supports_predict_y() -> None:
    model = _EncodedModel()
    query_points = tf.random.uniform([3, 5])
    mean, var = model.predict_y(query_points)
    npt.assert_allclose(mean, query_points + 1)
    npt.assert_allclose(var, query_points + 1)


def test_encoded_probabilistic_model_keras_embedding() -> None:
    encoder = tf_keras.layers.Embedding(3, 2)
    model = _EncodedModel(encoder=encoder)
    query_points = tf.random.uniform([3, 5], minval=0, maxval=3, dtype=tf.int32)
    mean, var = model.predict(query_points)
    assert mean.shape == (3, 5, 2)
    npt.assert_allclose(mean, encoder(query_points))

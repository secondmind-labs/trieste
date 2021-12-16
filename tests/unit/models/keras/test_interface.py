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

import copy

import gpflow
import pytest
import tensorflow as tf

from tests.util.misc import empty_dataset, raise_exc
from trieste.models.keras import NeuralNetworkPredictor
from trieste.models.optimizer import BatchOptimizer


class _DummyNeuralNetworkPredictor(NeuralNetworkPredictor):
    @property
    def model(self) -> tf.keras.Model:
        return raise_exc


def test_keras_predictor_repr_includes_class_name() -> None:
    model = _DummyNeuralNetworkPredictor()

    assert type(model).__name__ in repr(model)


def test_keras_predictor_default_optimizer_is_correct() -> None:
    model = _DummyNeuralNetworkPredictor()

    assert isinstance(model._optimizer, BatchOptimizer)
    assert isinstance(model._optimizer.optimizer, tf.optimizers.Adam)
    assert isinstance(model.optimizer, BatchOptimizer)
    assert isinstance(model.optimizer.optimizer, tf.optimizers.Adam)


def test_keras_predictor_check_optimizer_property() -> None:
    optimizer = BatchOptimizer(tf.optimizers.RMSprop())
    model = _DummyNeuralNetworkPredictor(optimizer)

    assert model.optimizer == optimizer


def test_keras_predictor_raises_on_predict_joint_call() -> None:
    model = _DummyNeuralNetworkPredictor()

    with pytest.raises(NotImplementedError):
        model.predict_joint(empty_dataset([1], [1]).query_points)


def test_keras_predictor_raises_on_sample_call() -> None:
    model = _DummyNeuralNetworkPredictor()

    with pytest.raises(NotImplementedError):
        model.sample(empty_dataset([1], [1]).query_points, 1)


def test_keras_predictor_raises_for_non_tf_optimizer() -> None:

    with pytest.raises(ValueError):
        _DummyNeuralNetworkPredictor(optimizer=BatchOptimizer(gpflow.optimizers.Scipy()))


def test_keras_predictor_deepcopy_raises_not_implemented() -> None:
    model = _DummyNeuralNetworkPredictor()

    with pytest.raises(NotImplementedError):
        copy.deepcopy(model)

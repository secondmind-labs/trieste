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

import gpflow
import pytest
from gpflow.keras import tf_keras

from tests.util.misc import empty_dataset, raise_exc
from trieste.models.keras import KerasPredictor
from trieste.models.optimizer import KerasOptimizer


class _DummyKerasPredictor(KerasPredictor):
    @property
    def model(self) -> tf_keras.Model:
        return raise_exc


def test_keras_predictor_repr_includes_class_name() -> None:
    model = _DummyKerasPredictor()

    assert type(model).__name__ in repr(model)


def test_keras_predictor_default_optimizer_is_correct() -> None:
    model = _DummyKerasPredictor()

    assert isinstance(model._optimizer, KerasOptimizer)
    assert isinstance(model._optimizer.optimizer, tf_keras.optimizers.Adam)
    assert isinstance(model.optimizer, KerasOptimizer)
    assert isinstance(model.optimizer.optimizer, tf_keras.optimizers.Adam)


def test_keras_predictor_check_optimizer_property() -> None:
    optimizer = KerasOptimizer(tf_keras.optimizers.RMSprop())
    model = _DummyKerasPredictor(optimizer)

    assert model.optimizer == optimizer


def test_keras_predictor_raises_on_sample_call() -> None:
    model = _DummyKerasPredictor()

    with pytest.raises(NotImplementedError):
        model.sample(empty_dataset([1], [1]).query_points, 1)


def test_keras_predictor_raises_for_non_tf_optimizer() -> None:
    with pytest.raises(ValueError):
        _DummyKerasPredictor(optimizer=KerasOptimizer(gpflow.optimizers.Scipy()))

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

from typing import Any, Callable

import pytest
import tensorflow as tf
from gpflux.models import DeepGP

from tests.util.models.gpflux.models import simple_two_layer_dgp_model, two_layer_dgp_model
from trieste.types import TensorType


@pytest.fixture(name="two_layer_model", params=[two_layer_dgp_model, simple_two_layer_dgp_model])
def _two_layer_model_fixture(request: Any) -> Callable[[TensorType], DeepGP]:
    return request.param


@pytest.fixture(name="keras_float")
def _keras_float() -> None:
    curr_float = tf.keras.backend.floatx()
    tf.keras.backend.set_floatx("float64")
    yield
    tf.keras.backend.set_floatx(curr_float)

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
import numpy as np
import pytest
import tensorflow as tf

from tests.util.models.gpflow.models import gpr_model
from tests.util.models.models import fnc_3x_plus_10
from trieste.models import ModelConfig


def test_model_missing_abstract_method() -> None:
    x = tf.constant(np.arange(5).reshape(-1, 1), dtype=gpflow.default_float())
    model_specs = {"model": gpr_model(x, fnc_3x_plus_10(x))}
    with pytest.raises(NotImplementedError):
        ModelConfig(**model_specs)

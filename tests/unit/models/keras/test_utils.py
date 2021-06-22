# Copyright 2021 The Bellman Contributors
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

import pytest

import numpy as np
import tensorflow as tf

from trieste.models.keras.utils import size


@pytest.mark.parametrize("n_dims", list(range(10)))
def test_size(n_dims):

    shape = np.random.randint(1, 10, (n_dims,))
    tensor = np.random.randint(0, 1, shape)
    tensor_spec = tf.TensorSpec(shape)

    assert size(tensor_spec) == np.size(tensor)

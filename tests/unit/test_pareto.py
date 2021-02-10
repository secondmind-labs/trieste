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

import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.data import Dataset
from trieste.utils.pareto import non_dominated_sort


@pytest.mark.parametrize(
    "scores, pareto_set, dominance",
    [
        (
            Dataset(
                tf.ones([8, 2]),
                tf.constant(
                    [
                        [0.9575, 0.4218],
                        [0.9649, 0.9157],
                        [0.1576, 0.7922],
                        [0.9706, 0.9595],
                        [0.9572, 0.6557],
                        [0.4854, 0.0357],
                        [0.8003, 0.8491],
                        [0.1419, 0.9340],
                    ]
                ),
            ),
            tf.constant([[0.1576, 0.7922], [0.4854, 0.0357], [0.1419, 0.934]]),
            tf.constant([1, 5, 0, 7, 1, 0, 2, 0]),
        )
    ],
)
def test_dominated_sort(scores: tf.Tensor, pareto_set: tf.Tensor, dominance: tf.Tensor) -> None:
    d1, d2 = non_dominated_sort(scores.observations)
    npt.assert_allclose(d1, pareto_set)
    npt.assert_allclose(d2, dominance)

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
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.acquisition.multi_objective.dominance import non_dominated


@pytest.mark.parametrize(
    "scores, pareto_set, dominance",
    [
        (
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
            tf.constant([[0.1576, 0.7922], [0.4854, 0.0357], [0.1419, 0.934]]),
            tf.constant([1, 5, 0, 7, 1, 0, 2, 0]),
        ),
        (
            tf.constant(
                [
                    [0.9575, 0.4218],
                    [0.9649, 0.9157],
                    [0.1576, 0.7922],
                    [0.9706, 0.9595],
                    [0.9572, 0.6557],
                    [0.4854, 0.0357],
                    [0.4954, 0.0357],
                    [0.8003, 0.8491],
                    [0.1419, 0.9340],
                    [0.1419, 0.9440],
                ]
            ),
            tf.constant([[0.1576, 0.7922], [0.4854, 0.0357], [0.1419, 0.934]]),
            tf.constant([2, 6, 0, 9, 2, 0, 1, 3, 0, 1]),
        ),
        (
            tf.constant(
                [
                    [0.9575, 0.4218],
                    [0.9649, 0.9157],
                    [0.1576, 0.7922],
                    [0.9706, 0.9595],
                    [0.9572, 0.6557],
                    [0.4854, 0.0357],
                    [0.4854, 0.0357],
                    [0.8003, 0.8491],
                    [0.1419, 0.9340],
                    [0.1419, 0.9340],
                ]
            ),
            tf.constant(
                [
                    [0.1576, 0.7922],
                    [0.4854, 0.0357],
                    [0.4854, 0.0357],
                    [0.1419, 0.934],
                    [0.1419, 0.934],
                ]
            ),
            tf.constant([2, 6, 0, 9, 2, 0, 0, 3, 0, 0]),
        ),
        (
            tf.constant(
                [
                    [0.90234935, 0.02297473, 0.05389869],
                    [0.98328614, 0.44182944, 0.6975261],
                    [0.39555323, 0.3040712, 0.3433497],
                    [0.72582424, 0.55389977, 0.00330079],
                    [0.9590585, 0.03233206, 0.2403127],
                    [0.04540098, 0.22407162, 0.11227596],
                ]
            ),
            tf.constant(
                [
                    [0.90234935, 0.02297473, 0.05389869],
                    [0.72582424, 0.55389977, 0.00330079],
                    [0.04540098, 0.22407162, 0.11227596],
                ]
            ),
            tf.constant([0, 4, 1, 0, 1, 0]),
        ),
        (
            tf.constant([]),
            tf.constant([]),
            tf.constant([]),
        ),
    ],
)
def test_dominated_sort(scores: tf.Tensor, pareto_set: tf.Tensor, dominance: tf.Tensor) -> None:
    ret_pareto_set, ret_nondominated = tf.function(non_dominated)(scores)
    npt.assert_allclose(tf.sort(ret_pareto_set, 0), tf.sort(pareto_set, 0))
    npt.assert_array_equal(ret_nondominated, dominance == 0)


@pytest.mark.parametrize("num_objectives", [2, 4, 6])
def test_dominated_sort_scales_ok(num_objectives: int) -> None:

    rng = np.random.RandomState(1234)
    dataset = rng.rand(10000, num_objectives)
    front, idx = non_dominated(dataset)

    for f in front:
        assert np.all(np.any(f <= dataset, axis=1))
    assert np.all(np.sort(front, axis=0) == np.sort(dataset[idx], axis=0))

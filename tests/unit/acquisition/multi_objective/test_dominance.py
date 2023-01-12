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
from datetime import timedelta
from time import perf_counter

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.acquisition.multi_objective.dominance import non_dominated

_COMPILERS = {
    "no_compiler": lambda f: f,
    "tf_function": tf.function,
}


@pytest.mark.parametrize(
    "scores, pareto_set, nondominated",
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
            tf.constant([False, False, True, False, False, True, False, True]),
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
            tf.constant([False, False, True, False, False, True, False, False, True, False]),
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
            tf.constant([False, False, True, False, False, True, True, False, True, True]),
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
            tf.constant([True, False, False, True, False, True]),
        ),
        (
            tf.zeros((0, 3)),
            tf.zeros((0, 3)),
            tf.ones((0,), dtype=tf.bool),
        ),
    ],
)
@pytest.mark.parametrize("compiler_name", _COMPILERS)
def test_dominated_sort(
    compiler_name: str, scores: tf.Tensor, pareto_set: tf.Tensor, nondominated: tf.Tensor
) -> None:
    compiled_non_dominated = _COMPILERS[compiler_name](non_dominated)
    ret_pareto_set, ret_nondominated = compiled_non_dominated(scores)
    npt.assert_allclose(tf.sort(ret_pareto_set, 0), tf.sort(pareto_set, 0))
    npt.assert_array_equal(ret_nondominated, nondominated)


@pytest.mark.parametrize("num_objectives", [2, 4, 6])
@pytest.mark.parametrize("compiler_name", _COMPILERS)
def test_dominated_scales_ok(compiler_name: str, num_objectives: int) -> None:
    num_points = 10_000
    compiled_non_dominated = _COMPILERS[compiler_name](non_dominated)

    rng = np.random.RandomState(1234)
    dataset = tf.Variable(rng.rand(num_points, num_objectives), shape=[None, num_objectives])

    before = perf_counter()
    front, idx = compiled_non_dominated(dataset)
    after = perf_counter()

    print()
    print(
        f"{num_points} x {num_objectives} ({compiler_name})"
        f" -> {timedelta(seconds=after - before)}"
    )

    for f in front:
        assert np.all(np.any(f <= dataset, axis=1))
    assert np.all(np.sort(front, axis=0) == np.sort(dataset[idx], axis=0))

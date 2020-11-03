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
from typing import Any, Callable, Dict, Tuple

import pytest
import numpy as np
import numpy.testing as npt
import tensorflow as tf

from trieste.type import TensorType
from trieste.utils.misc import jit, shapes_equal, to_numpy, zip_with
from tests.util.misc import ShapeLike, various_shapes


@pytest.mark.parametrize('apply', [True, False])
@pytest.mark.parametrize('kwargs', [
    {},
    {'autograph': False},
    {'input_signature': [tf.TensorSpec(()), tf.TensorSpec(())]},
])
def test_jit_function_behaviour_unchanged(apply: bool, kwargs: Any) -> None:
    @jit(apply, **kwargs)
    def add(t: tf.Tensor, u: tf.Tensor) -> tf.Tensor:
        return t + u

    assert add(tf.constant(1.), tf.constant(2.)) == tf.constant(3.)


@pytest.mark.parametrize('apply', [True, False])
@pytest.mark.parametrize('kwargs', [{}, {'autograph': False}])
def test_jit_compiles_function(apply: bool, kwargs: Any) -> None:
    @jit(apply, **kwargs)
    def one() -> tf.Tensor:
        return tf.constant(0)

    tf_function_type = type(tf.function(lambda x: x))
    assert isinstance(one, tf_function_type) == apply


@pytest.mark.parametrize('this_shape', various_shapes())
@pytest.mark.parametrize('that_shape', various_shapes())
def test_shapes_equal(this_shape: ShapeLike, that_shape: ShapeLike) -> None:
    assert shapes_equal(tf.ones(this_shape), tf.ones(that_shape)) == (this_shape == that_shape)


@pytest.mark.parametrize('t, expected', [
    (tf.constant(0), np.array(0)),
    (np.arange(12).reshape(3, -1), np.arange(12).reshape(3, -1)),
    (tf.reshape(tf.range(12), [3, -1]), np.arange(12).reshape(3, -1))
])
def test_to_numpy(t: TensorType, expected: np.ndarray) -> None:
    npt.assert_array_equal(to_numpy(t), expected)


@pytest.mark.parametrize('u, v, expected', [
    ({}, {}, {}),
    ({'a': 1}, {'a': 2}, {'a': (1, 2)}),
    ({'a': 1, 'b': 3}, {'a': 2, 'b': 4}, {'a': (1, 2), 'b': (3, 4)})
])
def test_zip_with_default(
    u: Dict[str, int], v: Dict[str, int], expected: Dict[str, Tuple[int, int]]
) -> None:
    assert zip_with(u, v) == expected


@pytest.mark.parametrize('u, v, expected', [
    ({}, {}, {}),
    ({'a': 1}, {'a': 2}, {'a': 3}),
    ({'a': 1, 'b': 3}, {'a': 2, 'b': 4}, {'a': 3, 'b': 7})
])
def test_zip_with_add(u: Dict[str, int], v: Dict[str, int], expected: Dict[str, int]) -> None:
    assert zip_with(u, v, lambda x, y: x + y) == expected


_zip_with_raises_cases = [({}, {'a': 1}), ({'a': 1}, {'a': 2, 'b': 3})]


@pytest.mark.parametrize('f', [lambda x, y: (x, y), lambda x, y: x + y])
@pytest.mark.parametrize(
    'u, v', _zip_with_raises_cases + [(v, u) for u, v in _zip_with_raises_cases]
)
def test_zip_with_raises_for_different_keys(
    u: Dict[str, int], v: Dict[str, int], f: Callable[[int, int], object]
) -> None:
    with pytest.raises(ValueError):
        zip_with(u, v, f)

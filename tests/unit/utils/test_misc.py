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
from __future__ import annotations

from time import sleep
from typing import Any, Optional, Union

import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES, ShapeLike, various_shapes
from trieste.observer import OBJECTIVE
from trieste.types import Tag, TensorType
from trieste.utils.misc import (
    Err,
    LocalizedTag,
    Ok,
    Timer,
    ensure_positive,
    flatten_leading_dims,
    get_value_for_tag,
    jit,
    shapes_equal,
    to_numpy,
)


@pytest.mark.parametrize("apply", [True, False])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"autograph": False},
        {"input_signature": [tf.TensorSpec(()), tf.TensorSpec(())]},
    ],
)
def test_jit_function_behaviour_unchanged(apply: bool, kwargs: Any) -> None:
    @jit(apply, **kwargs)
    def add(t: tf.Tensor, u: tf.Tensor) -> tf.Tensor:
        return t + u

    assert add(tf.constant(1.0), tf.constant(2.0)) == tf.constant(3.0)


@pytest.mark.parametrize("apply", [True, False])
@pytest.mark.parametrize("kwargs", [{}, {"autograph": False}])
def test_jit_compiles_function(apply: bool, kwargs: Any) -> None:
    @jit(apply, **kwargs)
    def one() -> tf.Tensor:
        return tf.constant(0)

    tf_function_type = type(tf.function(lambda x: x))
    assert isinstance(one, tf_function_type) == apply


@pytest.mark.parametrize("this_shape", various_shapes())
@pytest.mark.parametrize("that_shape", various_shapes())
def test_shapes_equal(this_shape: ShapeLike, that_shape: ShapeLike) -> None:
    assert shapes_equal(tf.ones(this_shape), tf.ones(that_shape)) == (this_shape == that_shape)


@pytest.mark.parametrize(
    "t, expected",
    [
        (tf.constant(0), np.array(0)),
        (np.arange(12).reshape(3, -1), np.arange(12).reshape(3, -1)),
        (tf.reshape(tf.range(12), [3, -1]), np.arange(12).reshape(3, -1)),
    ],
)
def test_to_numpy(t: TensorType, expected: "np.ndarray[Any, Any]") -> None:
    npt.assert_array_equal(to_numpy(t), expected)


def test_ok() -> None:
    assert Ok(1).unwrap() == 1
    assert Ok(1).is_ok is True
    assert Ok(1).is_err is False


def test_err() -> None:
    with pytest.raises(ValueError):
        Err(ValueError()).unwrap()

    assert Err(ValueError()).is_ok is False
    assert Err(ValueError()).is_err is True


def test_get_value_for_tag_returns_none_if_mapping_is_none() -> None:
    assert get_value_for_tag(None) == (None, None)


def test_get_value_for_tag_raises_if_tag_not_in_mapping() -> None:
    with pytest.raises(ValueError, match="none of the tags '.'baz',.' found in mapping"):
        get_value_for_tag({"foo": "bar"}, "baz")


def test_get_value_for_tag_returns_value_for_default_tag() -> None:
    assert get_value_for_tag({"foo": "bar", OBJECTIVE: "baz"}) == (OBJECTIVE, "baz")


def test_get_value_for_tag_returns_value_for_specified_tag() -> None:
    assert get_value_for_tag({"foo": "bar", OBJECTIVE: "baz"}, "foo") == ("foo", "bar")


def test_get_value_for_tag_returns_first_matching_tag() -> None:
    assert get_value_for_tag(
        {"foo": "bar", OBJECTIVE: "baz", "qux": "quux", "bar": "baz"}, *["far", "qux", "foo"]
    ) == ("qux", "quux")


@pytest.mark.parametrize("tag_name", ["test_tag_1", "test_tag_2"])
@pytest.mark.parametrize("tag_index", [0, 2, None])
def test_localized_tag_creation(tag_name: str, tag_index: Optional[int]) -> None:
    tag = LocalizedTag(tag_name, tag_index)
    is_local = True if tag_index is not None else False
    # Ensure a duplicate tag is equal.
    tag2 = LocalizedTag(tag_name, tag_index)

    assert tag.is_local == is_local
    assert tag.global_tag == tag_name
    assert tag.local_index == tag_index
    assert tag == tag2
    assert hash(tag) == hash(tag2)
    assert repr(tag) == f"LocalizedTag(global_tag='{tag_name}', local_index={tag_index})"


@pytest.mark.parametrize(
    "tag, exp_tag",
    [
        ("test_tag_1", LocalizedTag("test_tag_1", None)),
        (LocalizedTag("test_tag_1", 3), LocalizedTag("test_tag_1", 3)),
        (LocalizedTag("test_tag", None), LocalizedTag("test_tag", None)),
    ],
)
def test_localized_tag_from_tag(tag: Union[Tag, LocalizedTag], exp_tag: LocalizedTag) -> None:
    ltag = LocalizedTag.from_tag(tag)
    assert ltag.global_tag == exp_tag.global_tag
    assert ltag.local_index == exp_tag.local_index


def test_Timer() -> None:
    sleep_time = 0.1
    with Timer() as timer:
        sleep(sleep_time)
    npt.assert_allclose(timer.time, sleep_time, rtol=0.01)


def test_Timer_with_nesting() -> None:
    sleep_time = 0.1
    with Timer() as timer_1:
        sleep(sleep_time)
        with Timer() as timer_2:
            sleep(sleep_time)
    npt.assert_allclose(timer_1.time, 2.0 * sleep_time, rtol=0.01)
    npt.assert_allclose(timer_2.time, 1.0 * sleep_time, rtol=0.01)


def test_flatten_leading_dims() -> None:
    x_old = tf.random.uniform([2, 3, 4, 5])  # [2, 3, 4, 5]
    flat_x_old, unflatten = flatten_leading_dims(x_old)  # [24, 5]

    npt.assert_array_equal(tf.shape(flat_x_old), [24, 5])

    x_new = unflatten(flat_x_old)  # [2, 3, 4, 5]
    npt.assert_array_equal(x_old, x_new)


def test_unflatten_raises_for_invalid_shape() -> None:
    x_old = tf.random.uniform([2, 3, 4, 5])  # [2, 3, 4, 5]
    flat_x_old, unflatten = flatten_leading_dims(x_old)  # [24, 5]
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        unflatten(x_old)


def test_unflatten_returns_correct_shape() -> None:
    x = tf.random.uniform([2, 3, 4, 5])
    flat_x, unflatten = flatten_leading_dims(x)  # [24, 5]

    y1 = tf.random.uniform([24, 7])
    y2 = tf.random.uniform([24, 7, 11])

    unflat_y1 = unflatten(y1)
    unflat_y2 = unflatten(y2)

    npt.assert_array_equal(tf.shape(unflat_y1), [2, 3, 4, 7])
    npt.assert_array_equal(tf.shape(unflat_y2), [2, 3, 4, 7, 11])


@pytest.mark.parametrize(
    "output_dims,expected_shape",
    [
        (1, [120]),
        (2, [24, 5]),
        (3, [6, 4, 5]),
        (4, [2, 3, 4, 5]),
    ],
)
def test_flatten_leading_dims_output_dims(output_dims: int, expected_shape: list[int]) -> None:
    x_old = tf.random.uniform([2, 3, 4, 5])
    flat_x_old, unflatten = flatten_leading_dims(x_old, output_dims=output_dims)

    npt.assert_array_equal(tf.shape(flat_x_old), expected_shape)

    x_new = unflatten(flat_x_old)
    npt.assert_array_equal(x_old, x_new)


@pytest.mark.parametrize("output_dims", [-1, 0, 5, 100])
def test_flatten_leading_dims_invalid_output_dims(output_dims: int) -> None:
    x_old = tf.random.uniform([2, 3, 4, 5])  # [2, 3, 4, 5]
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        flatten_leading_dims(x_old, output_dims=output_dims)


@pytest.mark.parametrize(
    "t, expected",
    [
        (
            tf.constant(0, dtype=tf.float32),
            tf.constant(1e-15, dtype=tf.float32),
        ),
        (
            tf.constant(0, dtype=tf.float64),
            tf.constant(1e-30, dtype=tf.float64),
        ),
        (
            tf.constant([[-1.0, 0.0], [1e-35, 1.0]], dtype=tf.float32),
            tf.constant([[1e-15, 1e-15], [1e-15, 1.0]], dtype=tf.float32),
        ),
        (
            tf.constant([[-1.0, 0.0], [1e-35, 1.0]], dtype=tf.float64),
            tf.constant([[1e-30, 1e-30], [1e-30, 1.0]], dtype=tf.float64),
        ),
    ],
)
def test_ensure_positive(t: TensorType, expected: TensorType) -> None:
    npt.assert_array_equal(ensure_positive(t), expected)

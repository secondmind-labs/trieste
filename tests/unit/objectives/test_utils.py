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
from typing import Callable, Mapping, Set, Union

import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.data import Dataset
from trieste.objectives.utils import mk_batch_observer, mk_multi_observer, mk_observer
from trieste.observer import Observer
from trieste.types import Tag, TensorType
from trieste.utils.misc import LocalizedTag


def test_mk_observer() -> None:
    def foo(x: tf.Tensor) -> tf.Tensor:
        return x + 1

    x_ = tf.constant([[3.0]])
    ys = mk_observer(foo, "bar")(x_)

    assert ys.keys() == {"bar"}
    npt.assert_array_equal(ys["bar"].query_points, x_)
    npt.assert_array_equal(ys["bar"].observations, x_ + 1)


def test_mk_observer_unlabelled() -> None:
    def foo(x: tf.Tensor) -> tf.Tensor:
        return x + 1

    x_ = tf.constant([[3.0]])
    ys = mk_observer(foo)(x_)

    npt.assert_array_equal(ys.query_points, x_)
    npt.assert_array_equal(ys.observations, x_ + 1)


def test_mk_multi_observer() -> None:
    x_ = tf.constant([[3.0]])
    ys = mk_multi_observer(foo=lambda x: x + 1, bar=lambda x: x - 1)(x_)

    assert ys.keys() == {"foo", "bar"}
    npt.assert_array_equal(ys["foo"].query_points, x_)
    npt.assert_array_equal(ys["foo"].observations, x_ + 1)
    npt.assert_array_equal(ys["bar"].query_points, x_)
    npt.assert_array_equal(ys["bar"].observations, x_ - 1)


@pytest.mark.parametrize(
    "input_objective, exp_o_call",
    [
        (lambda x: x, {"baz": lambda x: x}),
        (lambda x: Dataset(x, x), {"baz": lambda x: x}),
        (
            mk_multi_observer(foo=lambda x: x + 1, bar=lambda x: x - 1),
            {"foo": lambda x: x + 1, "bar": lambda x: x - 1},
        ),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("num_query_points_per_batch", [1, 2])
def test_mk_batch_observer(
    input_objective: Union[Callable[[TensorType], TensorType], Observer],
    exp_o_call: Mapping[Tag, Callable[[TensorType], TensorType]],
    batch_size: int,
    num_query_points_per_batch: int,
) -> None:
    x_ = tf.reshape(
        tf.constant(range(batch_size * num_query_points_per_batch), tf.float64),
        (num_query_points_per_batch, batch_size, 1),
    )
    ys = mk_batch_observer(input_objective, "baz")(x_)

    assert isinstance(ys, dict)

    # Check keys.
    exp_keys: Set[Union[Tag, LocalizedTag]] = set()
    for key in exp_o_call:
        exp_keys.update({LocalizedTag(key, i) for i in range(batch_size)})
        exp_keys.add(key)
    assert ys.keys() == exp_keys

    # Check datasets.
    for key, call in exp_o_call.items():
        # Get expected observations.
        exp_o = call(x_)

        npt.assert_array_equal(ys[key].query_points, tf.reshape(x_, [-1, 1]))
        npt.assert_array_equal(ys[key].observations, tf.reshape(exp_o, [-1, 1]))
        for i in range(batch_size):
            npt.assert_array_equal(ys[LocalizedTag(key, i)].query_points, x_[:, i])
            npt.assert_array_equal(ys[LocalizedTag(key, i)].observations, exp_o[:, i])

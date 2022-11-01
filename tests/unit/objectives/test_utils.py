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
import numpy.testing as npt
import tensorflow as tf

from trieste.objectives.utils import mk_multi_observer, mk_observer


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

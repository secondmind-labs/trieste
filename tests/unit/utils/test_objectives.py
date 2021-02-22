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
import tensorflow as tf

from trieste.utils.objectives import (
    BRANIN_MINIMUM,
    BRANIN_MINIMIZERS,
    GRAMACY_LEE_MINIMIZER,
    GRAMACY_LEE_MINIMUM,
    branin,
    gramacy_lee,
    mk_observer,
)


def test_branin_no_points_are_less_than_global_minimum() -> None:
    search_values_1d = tf.range(1001.0) / 1000
    x0, x1 = (tf.reshape(t, [-1, 1]) for t in tf.meshgrid(search_values_1d, search_values_1d))
    x = tf.squeeze(tf.stack([x0, x1], axis=-1))
    npt.assert_array_less(tf.broadcast_to(BRANIN_MINIMUM, [1001 ** 2, 1]), branin(x))


def test_branin_maps_argmin_values_to_global_minima() -> None:
    expected = tf.broadcast_to(BRANIN_MINIMUM, [len(BRANIN_MINIMIZERS), 1])
    npt.assert_allclose(branin(BRANIN_MINIMIZERS), expected, atol=1e-6)


def test_gramacy_lee_no_points_are_less_than_global_minimum() -> None:
    xs = tf.linspace([0.5], [2.5], 1_000_000)
    npt.assert_array_less(tf.broadcast_to(GRAMACY_LEE_MINIMUM, xs.shape), gramacy_lee(xs))


def test_gramacy_lee_maps_argmin_values_to_global_minima() -> None:
    expected = tf.broadcast_to(GRAMACY_LEE_MINIMUM, [1, 1])
    npt.assert_allclose(gramacy_lee(GRAMACY_LEE_MINIMIZER), expected, atol=1e-6)


def test_mk_observer() -> None:
    def foo(x: tf.Tensor) -> tf.Tensor:
        return x + 1

    x_ = tf.constant([[3.0]])
    ys = mk_observer(foo, "bar")(x_)

    assert ys.keys() == {"bar"}
    npt.assert_array_equal(ys["bar"].query_points, x_)
    npt.assert_array_equal(ys["bar"].observations, x_ + 1)

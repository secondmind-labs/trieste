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

from trieste.utils.objectives import branin, BRANIN_GLOBAL_ARGMIN, BRANIN_GLOBAL_MINIMUM


def test_branin_no_points_are_less_than_global_minimum() -> None:
    search_values_1d = tf.range(1001.) / 1000
    x0, x1 = (tf.reshape(t, [-1, 1]) for t in tf.meshgrid(search_values_1d, search_values_1d))
    x = tf.squeeze(tf.stack([x0, x1], axis=-1))
    assert tf.reduce_all(branin(x) > BRANIN_GLOBAL_MINIMUM)


def test_branin_maps_argmin_values_to_global_minima() -> None:
    npt.assert_array_almost_equal(branin(BRANIN_GLOBAL_ARGMIN), BRANIN_GLOBAL_MINIMUM)

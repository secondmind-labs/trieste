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
from typing import Callable

import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.type import TensorType
from tests.util.misc import TF_DEBUGGING_ERROR_TYPES
from trieste.utils.objectives import (
    BRANIN_MINIMIZERS,
    BRANIN_MINIMUM,
    GRAMACY_LEE_MINIMIZER,
    GRAMACY_LEE_MINIMUM,
    LOGARITHMIC_GOLDSTEIN_PRICE_MINIMIZER,
    LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM,
    branin,
    gramacy_lee,
    logarithmic_goldstein_price,
    mk_observer,
    DTLZ1,
    DTLZ2
)


def _unit_grid_2d() -> TensorType:
    search_values_1d = tf.range(1001.0, dtype=tf.float64) / 1000
    x0, x1 = (tf.reshape(t, [-1, 1]) for t in tf.meshgrid(search_values_1d, search_values_1d))
    return tf.squeeze(tf.stack([x0, x1], axis=-1))


@pytest.mark.parametrize(
    "objective, minimizers, minimum",
    [
        (branin, BRANIN_MINIMIZERS, BRANIN_MINIMUM),
        (gramacy_lee, GRAMACY_LEE_MINIMIZER, GRAMACY_LEE_MINIMUM),
        (
            logarithmic_goldstein_price,
            LOGARITHMIC_GOLDSTEIN_PRICE_MINIMIZER,
            LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM,
        ),
    ],
)
def test_objective_maps_minimizers_to_minimum(
    objective: Callable[[TensorType], TensorType], minimizers: TensorType, minimum: TensorType
) -> None:
    npt.assert_allclose(objective(minimizers), tf.squeeze(minimum), rtol=1e-5)


def test_branin_no_function_values_are_less_than_global_minimum() -> None:
    npt.assert_array_less(tf.squeeze(BRANIN_MINIMUM) - 1e-6, branin(_unit_grid_2d()))


def test_gramacy_lee_no_points_are_less_than_global_minimum() -> None:
    xs = tf.linspace([0.5], [2.5], 1_000_000)
    npt.assert_array_less(tf.squeeze(GRAMACY_LEE_MINIMUM) - 1e-6, gramacy_lee(xs))


def test_logarithmic_goldstein_price_no_function_values_are_less_than_global_minimum() -> None:
    npt.assert_array_less(
        tf.squeeze(LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM) - 1e-6,
        logarithmic_goldstein_price(_unit_grid_2d()),
    )


@pytest.mark.parametrize(
    "obj_inst, input_dim, num_obj, gen_pf_num",
    [
        (DTLZ1(3, 2), 3, 2, 1000),
        (DTLZ1(5, 3), 5, 3, 1000),
        (DTLZ2(3, 2), 3, 2, 1000),
        (DTLZ2(12, 6), 12, 6, 1000)
    ],
)
def test_gen_pareto_front_is_equal_to_math_defined(obj_inst, input_dim, num_obj, gen_pf_num):
    pfs = obj_inst.gen_pareto_optimal_points(gen_pf_num)
    if isinstance(obj_inst, DTLZ1):
        tf.assert_equal(tf.reduce_sum(pfs, axis=1), 1.0)
    elif isinstance(obj_inst, DTLZ2):
        tf.debugging.assert_near(tf.norm(pfs, axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize(
    "obj_inst, actual_x",
    [
        (DTLZ1(3, 2), tf.constant([[0.3, 0.1]])),
        (DTLZ2(5, 2), tf.constant([[0.3, 0.1]])),
    ],
)
def test_func_raises_specified_input_dim_not_alin_with_actual_input_dim(obj_inst, actual_x):
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        obj_inst.prepare_benchmark()(actual_x)


def test_mk_observer() -> None:
    def foo(x: tf.Tensor) -> tf.Tensor:
        return x + 1

    x_ = tf.constant([[3.0]])
    ys = mk_observer(foo, "bar")(x_)

    assert ys.keys() == {"bar"}
    npt.assert_array_equal(ys["bar"].query_points, x_)
    npt.assert_array_equal(ys["bar"].observations, x_ + 1)

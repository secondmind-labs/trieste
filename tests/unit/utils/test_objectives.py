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

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES
from trieste.space import Box
from trieste.type import TensorType
from trieste.utils.multi_objectives import DTLZ1, DTLZ2, VLMOP2, MultiObjectiveTestProblem, vlmop2
from trieste.utils.objectives import (
    BRANIN_MINIMIZERS,
    BRANIN_MINIMUM,
    GRAMACY_LEE_MINIMIZER,
    GRAMACY_LEE_MINIMUM,
    HARTMANN_3_MINIMIZER,
    HARTMANN_3_MINIMUM,
    HARTMANN_6_MINIMIZER,
    HARTMANN_6_MINIMUM,
    LOGARITHMIC_GOLDSTEIN_PRICE_MINIMIZER,
    LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM,
    branin,
    gramacy_lee,
    hartmann_3,
    hartmann_6,
    logarithmic_goldstein_price,
    mk_observer,
)


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
        (hartmann_3, HARTMANN_3_MINIMIZER, HARTMANN_3_MINIMUM),
        (hartmann_6, HARTMANN_6_MINIMIZER, HARTMANN_6_MINIMUM),
    ],
)
def test_objective_maps_minimizers_to_minimum(
    objective: Callable[[TensorType], TensorType], minimizers: TensorType, minimum: TensorType
) -> None:
    objective_values_at_minimizers = objective(minimizers)
    tf.debugging.assert_shapes([(objective_values_at_minimizers, [len(minimizers), 1])])
    npt.assert_allclose(objective_values_at_minimizers, tf.squeeze(minimum), rtol=1e-5)


def test_branin_no_function_values_are_less_than_global_minimum() -> None:
    samples = Box([0.0, 0.0], [1.0, 1.0]).sample(1000)
    npt.assert_array_less(tf.squeeze(BRANIN_MINIMUM) - 1e-6, branin(samples))


def test_gramacy_lee_no_points_are_less_than_global_minimum() -> None:
    samples = Box([0.5], [2.5]).sample(1000)
    npt.assert_array_less(tf.squeeze(GRAMACY_LEE_MINIMUM) - 1e-6, gramacy_lee(samples))


def test_logarithmic_goldstein_price_no_function_values_are_less_than_global_minimum() -> None:
    samples = Box([0.0, 0.0], [1.0, 1.0]).sample(1000)
    npt.assert_array_less(
        tf.squeeze(LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM) - 1e-6,
        logarithmic_goldstein_price(samples),
    )


@pytest.mark.parametrize(
    "test_x, expected",
    [
        (
            tf.constant([[0.0, 0.0]]),
            tf.constant([[0.63212055, 0.63212055]]),
        ),
        (
            tf.constant([[0.5, 1.0]]),
            tf.constant([[0.12074441, 0.9873655]]),
        ),
        (
            tf.constant([[[0.5, 1.0]], [[0.0, 0.0]]]),
            tf.constant([[[0.12074441, 0.9873655]], [[0.63212055, 0.63212055]]]),
        ),
        (
            tf.constant([[[0.5, 1.0], [0.0, 0.0]]]),
            tf.constant([[[0.12074441, 0.9873655], [0.63212055, 0.63212055]]]),
        ),
    ],
)
def test_vlmop2_has_expected_output(test_x: TensorType, expected: TensorType):
    npt.assert_allclose(vlmop2(test_x), expected, rtol=1e-5)


@pytest.mark.parametrize(
    "test_x, input_dim, num_obj, expected",
    [
        (tf.constant([[0.0, 0.2, 0.4]]), 3, 2, tf.constant([[0.0, 5.5]])),
        (
            tf.constant([[[0.0, 0.2, 0.4]], [[0.0, 0.2, 0.4]]]),
            3,
            2,
            tf.constant([[[0.0, 5.5]], [[0.0, 5.5]]]),
        ),
        (tf.constant([[0.8, 0.6, 0.4, 0.2]]), 4, 2, tf.constant([[4.8, 1.2]])),
        (tf.constant([[0.1, 0.2, 0.3, 0.4]]), 4, 3, tf.constant([[0.06, 0.24, 2.7]])),
        (
            tf.constant([[[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]]),
            4,
            3,
            tf.constant([[[0.06, 0.24, 2.7], [0.06, 0.24, 2.7]]]),
        ),
    ],
)
def test_dtlz1_has_expected_output(
    test_x: TensorType, input_dim: int, num_obj: int, expected: TensorType
):
    f = DTLZ1(input_dim, num_obj).objective()
    npt.assert_allclose(f(test_x), expected, rtol=1e-5)


@pytest.mark.parametrize(
    "test_x, input_dim, num_obj, expected",
    [
        (tf.constant([[0.0, 0.2, 0.4]]), 3, 2, tf.constant([[1.1, 0.0]])),
        (
            tf.constant([[[0.0, 0.2, 0.4]], [[0.0, 0.2, 0.4]]]),
            3,
            2,
            tf.constant([[[1.1, 0.0]], [[1.1, 0.0]]]),
        ),
        (tf.constant([[0.8, 0.6, 0.4, 0.2]]), 4, 2, tf.constant([[0.3430008637, 1.055672733]])),
        (
            tf.constant([[[0.8, 0.6, 0.4, 0.2], [0.8, 0.6, 0.4, 0.2]]]),
            4,
            2,
            tf.constant([[[0.3430008637, 1.055672733], [0.3430008637, 1.055672733]]]),
        ),
        (
            tf.constant([[0.1, 0.2, 0.3, 0.4]]),
            4,
            3,
            tf.constant([[0.9863148, 0.3204731, 0.16425618]]),
        ),
    ],
)
def test_dtlz2_has_expected_output(
    test_x: TensorType, input_dim: int, num_obj: int, expected: TensorType
):
    f = DTLZ2(input_dim, num_obj).objective()
    npt.assert_allclose(f(test_x), expected, rtol=1e-4)


@pytest.mark.parametrize(
    "obj_inst, input_dim, num_obj, gen_pf_num",
    [
        (DTLZ1(3, 2), 3, 2, 1000),
        (DTLZ1(5, 3), 5, 3, 1000),
        (DTLZ2(3, 2), 3, 2, 1000),
        (DTLZ2(12, 6), 12, 6, 1000),
    ],
)
def test_gen_pareto_front_is_equal_to_math_defined(
    obj_inst: MultiObjectiveTestProblem, input_dim: int, num_obj: int, gen_pf_num: int
):
    pfs = obj_inst.gen_pareto_optimal_points(gen_pf_num)
    if isinstance(obj_inst, DTLZ1):
        tf.assert_equal(tf.reduce_sum(pfs, axis=1), 0.5)
    elif isinstance(obj_inst, DTLZ2):
        tf.debugging.assert_near(tf.norm(pfs, axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize(
    "obj_inst, actual_x",
    [
        (VLMOP2(), tf.constant([[0.4, 0.2, 0.5]])),
        (DTLZ1(3, 2), tf.constant([[0.3, 0.1]])),
        (DTLZ2(5, 2), tf.constant([[0.3, 0.1]])),
    ],
)
def test_func_raises_specified_input_dim_not_align_with_actual_input_dim(
    obj_inst: MultiObjectiveTestProblem, actual_x: TensorType
):
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        obj_inst.objective()(actual_x)


def test_hartmann_3_no_function_values_are_less_than_global_minimum() -> None:
    samples = Box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]).sample(1000)
    npt.assert_array_less(tf.squeeze(HARTMANN_3_MINIMUM) - 1e-6, hartmann_3(samples))


def test_hartmann_6_no_function_values_are_less_than_global_minimum() -> None:
    samples = Box([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).sample(1000)
    npt.assert_array_less(tf.squeeze(HARTMANN_6_MINIMUM) - 1e-6, hartmann_6(samples))


def test_mk_observer() -> None:
    def foo(x: tf.Tensor) -> tf.Tensor:
        return x + 1

    x_ = tf.constant([[3.0]])
    ys = mk_observer(foo, "bar")(x_)

    assert ys.keys() == {"bar"}
    npt.assert_array_equal(ys["bar"].query_points, x_)
    npt.assert_array_equal(ys["bar"].observations, x_ + 1)

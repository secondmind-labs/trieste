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
import pytest
import tensorflow as tf

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES
from trieste.type import TensorType
from trieste.utils.multi_objectives import DTLZ1, DTLZ2, VLMOP2, MultiObjectiveTestProblem, vlmop2


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

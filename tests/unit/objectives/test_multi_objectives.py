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
from typing import Callable, Union

import numpy.testing as npt
import pytest
import tensorflow as tf

from tests.util.misc import TF_DEBUGGING_ERROR_TYPES
from trieste.objectives.multi_objectives import (
    DTLZ1,
    DTLZ2,
    VLMOP2,
    ConstrainedBraninCurrin,
    ConstrainedMultiObjectiveTestProblem,
    MultiObjectiveTestProblem,
    NoAnalyticalParetoPointsError,
)
from trieste.types import TensorType


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
def test_vlmop2_has_expected_output(test_x: TensorType, expected: TensorType) -> None:
    f = VLMOP2(2).objective
    npt.assert_allclose(f(test_x), expected, rtol=1e-5)


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
) -> None:
    f = DTLZ1(input_dim, num_obj).objective
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
) -> None:
    f = DTLZ2(input_dim, num_obj).objective
    npt.assert_allclose(f(test_x), expected, rtol=1e-4)


@pytest.mark.parametrize(
    "test_x, expected_obj, expected_con, threshold",
    [
        (
            tf.constant([[0.0, 0.0]]),
            tf.constant([[308.12909601160663, 3.0]]),
            tf.constant([[62.5]]),
            0.0,
        ),
        (
            tf.constant([[0.5, 1.0]]),
            tf.constant([[150.45202034083485, 4.609388478538837]]),
            tf.constant([[-3.75]]),
            10.0,
        ),
        (
            tf.constant([[[0.5, 1.0]], [[0.0, 0.0]]]),
            tf.constant([[[150.45202034083485, 4.609388478538837]], [[308.12909601160663, 3.0]]]),
            tf.constant([[[6.25]], [[62.5]]]),
            0.0,
        ),
        (
            tf.constant([[0.5, 1.0], [0.0, 0.0]]),
            tf.constant([[150.45202034083485, 4.609388478538837], [308.12909601160663, 3.0]]),
            tf.constant([[11.25], [67.5]]),
            -5.0,
        ),
    ],
)
def test_constrainedbranincurrin_has_expected_output(
    test_x: TensorType,
    expected_obj: TensorType,
    expected_con: TensorType,
    threshold: Union[TensorType, float],
) -> None:
    f = ConstrainedBraninCurrin().objective
    c = ConstrainedBraninCurrin().constraint
    npt.assert_allclose(f(test_x), expected_obj, rtol=1e-5)
    npt.assert_allclose(c(test_x, threshold), expected_con, rtol=1e-5)


@pytest.mark.parametrize(
    "obj_type, input_dim, num_obj, gen_pf_num",
    [
        (DTLZ1, 3, 2, 1000),
        (DTLZ1, 5, 3, 1000),
        (DTLZ2, 3, 2, 1000),
        (DTLZ2, 12, 6, 1000),
    ],
)
def test_gen_pareto_front_is_equal_to_math_defined(
    obj_type: Callable[[int, int], MultiObjectiveTestProblem],
    input_dim: int,
    num_obj: int,
    gen_pf_num: int,
) -> None:
    obj_inst = obj_type(input_dim, num_obj)
    pfs = obj_inst.gen_pareto_optimal_points(gen_pf_num, None)
    if obj_type == DTLZ1:
        tf.assert_equal(tf.reduce_sum(pfs, axis=1), tf.cast(0.5, pfs.dtype))
    else:
        assert obj_type == DTLZ2
        tf.debugging.assert_near(tf.norm(pfs, axis=1), tf.cast(1.0, pfs.dtype), rtol=1e-6)


@pytest.mark.parametrize(
    "obj_inst, actual_x",
    [
        (VLMOP2(2), tf.constant([[0.4, 0.2, 0.5]])),
        (DTLZ1(3, 2), tf.constant([[0.3, 0.1]])),
        (DTLZ2(5, 2), tf.constant([[0.3, 0.1]])),
        (ConstrainedBraninCurrin(), tf.constant([[0.3, 0.2, 0.1]])),
    ],
)
def test_func_raises_specified_input_dim_not_align_with_actual_input_dim(
    obj_inst: Union[MultiObjectiveTestProblem, ConstrainedMultiObjectiveTestProblem],
    actual_x: TensorType,
) -> None:
    with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
        obj_inst.objective(actual_x)
    if isinstance(obj_inst, ConstrainedMultiObjectiveTestProblem):
        with pytest.raises(TF_DEBUGGING_ERROR_TYPES):
            obj_inst.constraint(actual_x)


@pytest.mark.parametrize(
    "problem, input_dim, num_obj, num_con",
    [
        (VLMOP2(2), 2, 2, 0),
        (VLMOP2(10), 10, 2, 0),
        (DTLZ1(3, 2), 3, 2, 0),
        (DTLZ1(10, 5), 10, 5, 0),
        (DTLZ2(3, 2), 3, 2, 0),
        (DTLZ2(10, 5), 10, 5, 0),
        (ConstrainedBraninCurrin(), 2, 2, 1),
    ],
)
@pytest.mark.parametrize("num_obs", [1, 5, 10])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_objective_and_constraint_has_correct_shape_and_dtype(
    problem: MultiObjectiveTestProblem,
    input_dim: int,
    num_obj: int,
    num_obs: int,
    num_con: int,
    dtype: tf.DType,
) -> None:
    x = problem.search_space.sample(num_obs)
    assert x.dtype == tf.float64  # default dtype

    x = tf.cast(x, dtype)
    y = problem.objective(x)

    assert y.dtype == x.dtype
    tf.debugging.assert_shapes([(x, [num_obs, input_dim])])
    tf.debugging.assert_shapes([(y, [num_obs, num_obj])])

    if isinstance(problem, ConstrainedMultiObjectiveTestProblem):
        c = problem.constraint(x)
        tf.debugging.assert_shapes([(c, [num_obs, num_con])])
        assert x.dtype == c.dtype

    try:  # check if the problem has a valid `gen_pareto_optimal_points` method
        pf = problem.gen_pareto_optimal_points(num_obs * 2)
        assert pf.dtype == tf.float64  # default dtype
        tf.debugging.assert_shapes([(pf, [num_obs * 2, num_obj])])
    except NoAnalyticalParetoPointsError:
        pass

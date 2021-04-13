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

from trieste.space import Box
from trieste.type import TensorType
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

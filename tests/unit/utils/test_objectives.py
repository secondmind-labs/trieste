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
    ACKLEY_5_MINIMIZER,
    ACKLEY_5_MINIMUM,
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
    ROSENBROCK_4_MINIMIZER,
    ROSENBROCK_4_MINIMUM,
    SHEKEL_4_MINIMIZER,
    SHEKEL_4_MINIMUM,
    ackley_5,
    branin,
    gramacy_lee,
    hartmann_3,
    hartmann_6,
    logarithmic_goldstein_price,
    mk_observer,
    rosenbrock_4,
    shekel_4,
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
        (rosenbrock_4, ROSENBROCK_4_MINIMIZER, ROSENBROCK_4_MINIMUM),
        (shekel_4, SHEKEL_4_MINIMIZER, SHEKEL_4_MINIMUM),
        (ackley_5, ACKLEY_5_MINIMIZER, ACKLEY_5_MINIMUM),
        (hartmann_6, HARTMANN_6_MINIMIZER, HARTMANN_6_MINIMUM),
    ],
)
def test_objective_maps_minimizers_to_minimum(
    objective: Callable[[TensorType], TensorType], minimizers: TensorType, minimum: TensorType
) -> None:
    objective_values_at_minimizers = objective(minimizers)
    tf.debugging.assert_shapes([(objective_values_at_minimizers, [len(minimizers), 1])])
    npt.assert_allclose(objective_values_at_minimizers, tf.squeeze(minimum), atol=1e-4)


@pytest.mark.parametrize(
    "objective, space, minimum",
    [
        (branin, Box([0.0, 0.0], [1.0, 1.0]), BRANIN_MINIMUM),
        (gramacy_lee, Box([0.5], [2.5]), GRAMACY_LEE_MINIMUM),
        (
            logarithmic_goldstein_price,
            Box([0.0, 0.0], [1.0, 1.0]),
            LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM,
        ),
        (hartmann_3, Box([0.0] * 3, [1.0] * 3), HARTMANN_3_MINIMUM),
        (rosenbrock_4, Box([0.0] * 4, [1.0] * 4), ROSENBROCK_4_MINIMUM),
        (shekel_4, Box([0.0] * 4, [1.0] * 4), SHEKEL_4_MINIMUM),
        (ackley_5, Box([0.0] * 5, [1.0] * 5), ACKLEY_5_MINIMUM),
        (hartmann_6, Box([0.0] * 6, [1.0] * 6), HARTMANN_6_MINIMUM),
    ],
)
def test_no_function_values_are_less_than_global_minimum(
    objective: Callable[[TensorType], TensorType], space: Box, minimum: TensorType
) -> None:
    samples = space.sample(1000 * len(space.lower))
    npt.assert_array_less(tf.squeeze(minimum) - 1e-6, objective(samples))


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

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

from typing import Callable

import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.objectives import (
    ACKLEY_5_MINIMIZER,
    ACKLEY_5_MINIMUM,
    ACKLEY_5_SEARCH_SPACE,
    BRANIN_MINIMIZERS,
    BRANIN_MINIMUM,
    BRANIN_SEARCH_SPACE,
    GRAMACY_LEE_MINIMIZER,
    GRAMACY_LEE_MINIMUM,
    GRAMACY_LEE_SEARCH_SPACE,
    HARTMANN_3_MINIMIZER,
    HARTMANN_3_MINIMUM,
    HARTMANN_3_SEARCH_SPACE,
    HARTMANN_6_MINIMIZER,
    HARTMANN_6_MINIMUM,
    HARTMANN_6_SEARCH_SPACE,
    LOGARITHMIC_GOLDSTEIN_PRICE_MINIMIZER,
    LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM,
    LOGARITHMIC_GOLDSTEIN_PRICE_SEARCH_SPACE,
    MICHALEWICZ_2_MINIMIZER,
    MICHALEWICZ_2_MINIMUM,
    MICHALEWICZ_2_SEARCH_SPACE,
    MICHALEWICZ_5_MINIMIZER,
    MICHALEWICZ_5_MINIMUM,
    MICHALEWICZ_5_SEARCH_SPACE,
    MICHALEWICZ_10_MINIMIZER,
    MICHALEWICZ_10_MINIMUM,
    MICHALEWICZ_10_SEARCH_SPACE,
    ROSENBROCK_4_MINIMIZER,
    ROSENBROCK_4_MINIMUM,
    ROSENBROCK_4_SEARCH_SPACE,
    SCALED_BRANIN_MINIMUM,
    SHEKEL_4_MINIMIZER,
    SHEKEL_4_MINIMUM,
    SHEKEL_4_SEARCH_SPACE,
    SIMPLE_QUADRATIC_MINIMIZER,
    SIMPLE_QUADRATIC_MINIMUM,
    TRID_10_MINIMIZER,
    TRID_10_MINIMUM,
    TRID_10_SEARCH_SPACE,
    ackley_5,
    branin,
    gramacy_lee,
    hartmann_3,
    hartmann_6,
    logarithmic_goldstein_price,
    michalewicz_2,
    michalewicz_5,
    michalewicz_10,
    rosenbrock_4,
    scaled_branin,
    shekel_4,
    simple_quadratic,
    trid_10,
)
from trieste.space import Box
from trieste.types import TensorType


@pytest.mark.parametrize(
    "objective, minimizers, minimum",
    [
        (branin, BRANIN_MINIMIZERS, BRANIN_MINIMUM),
        (scaled_branin, BRANIN_MINIMIZERS, SCALED_BRANIN_MINIMUM),
        (simple_quadratic, SIMPLE_QUADRATIC_MINIMIZER, SIMPLE_QUADRATIC_MINIMUM),
        (gramacy_lee, GRAMACY_LEE_MINIMIZER, GRAMACY_LEE_MINIMUM),
        (michalewicz_2, MICHALEWICZ_2_MINIMIZER, MICHALEWICZ_2_MINIMUM),
        (michalewicz_5, MICHALEWICZ_5_MINIMIZER, MICHALEWICZ_5_MINIMUM),
        (michalewicz_10, MICHALEWICZ_10_MINIMIZER, MICHALEWICZ_10_MINIMUM),
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
        (trid_10, TRID_10_MINIMIZER, TRID_10_MINIMUM),
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
        (branin, BRANIN_SEARCH_SPACE, BRANIN_MINIMUM),
        (scaled_branin, BRANIN_SEARCH_SPACE, SCALED_BRANIN_MINIMUM),
        (gramacy_lee, GRAMACY_LEE_SEARCH_SPACE, GRAMACY_LEE_MINIMUM),
        (michalewicz_2, MICHALEWICZ_2_SEARCH_SPACE, MICHALEWICZ_2_MINIMUM),
        (michalewicz_5, MICHALEWICZ_5_SEARCH_SPACE, MICHALEWICZ_5_MINIMUM),
        (michalewicz_10, MICHALEWICZ_10_SEARCH_SPACE, MICHALEWICZ_10_MINIMUM),
        (
            logarithmic_goldstein_price,
            LOGARITHMIC_GOLDSTEIN_PRICE_SEARCH_SPACE,
            LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM,
        ),
        (hartmann_3, HARTMANN_3_SEARCH_SPACE, HARTMANN_3_MINIMUM),
        (rosenbrock_4, ROSENBROCK_4_SEARCH_SPACE, ROSENBROCK_4_MINIMUM),
        (shekel_4, SHEKEL_4_SEARCH_SPACE, SHEKEL_4_MINIMUM),
        (ackley_5, ACKLEY_5_SEARCH_SPACE, ACKLEY_5_MINIMUM),
        (hartmann_6, HARTMANN_6_SEARCH_SPACE, HARTMANN_6_MINIMUM),
        (trid_10, TRID_10_SEARCH_SPACE, TRID_10_MINIMUM),
    ],
)
def test_no_function_values_are_less_than_global_minimum(
    objective: Callable[[TensorType], TensorType], space: Box, minimum: TensorType
) -> None:
    samples = space.sample(1000 * len(space.lower))
    npt.assert_array_less(tf.squeeze(minimum) - 1e-6, objective(samples))

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

"""
This package contains examples of popular objective functions used in (Bayesian) optimization.
"""

from collections import namedtuple
from enum import Enum

from . import multi_objectives, utils
from .single_objectives import (
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
    SIMPLE_QUADRATIC_SEARCH_SPACE,
    TRID_10_MINIMIZER,
    TRID_10_MINIMUM,
    TRID_10_SEARCH_SPACE,
    ackley_5,
    branin,
    gramacy_lee,
    hartmann_3,
    hartmann_6,
    logarithmic_goldstein_price,
    michalewicz,
    michalewicz_2,
    michalewicz_5,
    michalewicz_10,
    rosenbrock_4,
    scaled_branin,
    shekel_4,
    simple_quadratic,
    trid,
    trid_10,
)



class Objective(Enum):
    """
    This enumeration lists all the names of the objective functions available in trieste.
    """
    ACKLEY_5 = "ACKLEY_5"
    BRANIN = "BRANIN"
    GRAMACY_LEE = "GRAMACY_LEE"
    HARTMANN_3 = "HARTMANN_3"
    HARTMANN_6 = "HARTMANN_6"
    LOGARITHMIC_GOLDSTEIN_PRICE = "LOGARITHMIC_GOLDSTEIN_PRICE"
    MICHALEWICZ_2 = "MICHALEWICZ_2"
    MICHALEWICZ_5 = "MICHALEWICZ_5"
    MICHALEWICZ_10 = "MICHALEWICZ_10"
    ROSENBROCK_4 = "ROSENBROCK_4"
    SCALED_BRANIN = "SCALED_BRANIN"
    SHEKEL_4 = "SHEKEL_4"
    SIMPLE_QUADRATIC = "SIMPLE_QUADRATIC"
    TRID_10 = "TRID_10"


ObjectiveSpec = namedtuple("ObjectiveSpec", ("fun", "search_space", "minimizers", "minima"))
"""
This named tuple defines all the relevant objects from Trieste related to objectives in
``Objective`` that we need for finding thresholds corresponding to targeted volumes.
"""


OBJECTIVE_FUNCTION_SPECS = {
    Objective.ACKLEY_5: ObjectiveSpec(ackley_5, ACKLEY_5_SEARCH_SPACE),
    Objective.GRAMACY_LEE: ObjectiveSpec(gramacy_lee, GRAMACY_LEE_SEARCH_SPACE),
    Objective.HARTMANN_3: ObjectiveSpec(hartmann_3, HARTMANN_3_SEARCH_SPACE),
    Objective.HARTMANN_6: ObjectiveSpec(hartmann_6, HARTMANN_6_SEARCH_SPACE),
    Objective.LOGARITHMIC_GOLDSTEIN_PRICE: ObjectiveSpec(
        logarithmic_goldstein_price, LOGARITHMIC_GOLDSTEIN_PRICE_SEARCH_SPACE
    ),
    Objective.MICHALEWICZ_2: ObjectiveSpec(michalewicz_2, MICHALEWICZ_2_SEARCH_SPACE),
    Objective.MICHALEWICZ_5: ObjectiveSpec(michalewicz_5, MICHALEWICZ_5_SEARCH_SPACE),
    Objective.MICHALEWICZ_10: ObjectiveSpec(michalewicz_10, MICHALEWICZ_10_SEARCH_SPACE),
    Objective.ROSENBROCK_4: ObjectiveSpec(rosenbrock_4, ROSENBROCK_4_SEARCH_SPACE),
    Objective.SCALED_BRANIN: ObjectiveSpec(scaled_branin, BRANIN_SEARCH_SPACE),
    Objective.SHEKEL_4: ObjectiveSpec(shekel_4, SHEKEL_4_SEARCH_SPACE),
    Objective.SIMPLE_QUADRATIC: ObjectiveSpec(simple_quadratic, SIMPLE_QUADRATIC_SEARCH_SPACE),
    Objective.TRID_10: ObjectiveSpec(trid_10, TRID_10_SEARCH_SPACE),
}

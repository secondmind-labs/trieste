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
    ROSENBROCK_4_MINIMIZER,
    ROSENBROCK_4_MINIMUM,
    ROSENBROCK_4_SEARCH_SPACE,
    SCALED_BRANIN_MINIMUM,
    SHEKEL_4_MINIMIZER,
    SHEKEL_4_MINIMUM,
    SHEKEL_4_SEARCH_SPACE,
    ackley_5,
    branin,
    gramacy_lee,
    hartmann_3,
    hartmann_6,
    logarithmic_goldstein_price,
    rosenbrock_4,
    scaled_branin,
    shekel_4,
)

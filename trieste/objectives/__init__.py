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
from .multi_objectives import DTLZ1, DTLZ2, VLMOP2, MultiObjectiveTestProblem
from .multifidelity_objectives import (
    Linear2Fidelity,
    Linear3Fidelity,
    Linear5Fidelity,
    SingleObjectiveMultifidelityTestProblem,
)
from .single_objectives import (
    Ackley5,
    Branin,
    ConstrainedScaledBranin,
    GramacyLee,
    Hartmann3,
    Hartmann6,
    Levy8,
    LogarithmicGoldsteinPrice,
    Michalewicz2,
    Michalewicz5,
    Michalewicz10,
    ObjectiveTestProblem,
    Rosenbrock4,
    ScaledBranin,
    Shekel4,
    SimpleQuadratic,
    SingleObjectiveTestProblem,
    Trid10,
)
from .utils import mk_multi_observer, mk_observer

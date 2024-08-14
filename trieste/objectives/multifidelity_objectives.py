# Copyright 2023 The Trieste Contributors
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
This module contains synthetic multi-fidelity objective functions, useful for experimentation.
"""
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from ..space import Box, DiscreteSearchSpace, SearchSpace, SearchSpaceType, TaggedProductSearchSpace
from ..types import TensorType
from .single_objectives import SingleObjectiveTestProblem


@dataclass(frozen=True)
class SingleObjectiveMultifidelityTestProblem(SingleObjectiveTestProblem[SearchSpaceType]):
    num_fidelities: int
    """The number of fidelities of test function"""

    fidelity_search_space: TaggedProductSearchSpace
    """The search space including fidelities"""


def linear_multifidelity(x: TensorType) -> TensorType:
    x_input = x[..., :-1]
    x_fidelity = x[..., -1:]

    f = 0.5 * ((6.0 * x_input - 2.0) ** 2) * tf.math.sin(12.0 * x_input - 4.0) + 10.0 * (
        x_input - 1.0
    )
    f = f + x_fidelity * (f - 20.0 * (x_input - 1.0))

    return f


_LINEAR_MULTIFIDELITY_MINIMIZERS = {
    2: tf.constant([[0.75724875]], tf.float64),
    3: tf.constant([[0.76333767]], tf.float64),
    5: tf.constant([[0.76801846]], tf.float64),
}


_LINEAR_MULTIFIDELITY_MINIMA = {
    2: tf.constant([-6.020740055], tf.float64),
    3: tf.constant([-6.634287061], tf.float64),
    5: tf.constant([-7.933019704], tf.float64),
}


def _linear_multifidelity_search_space_builder(
    n_fidelities: int, input_search_space: SearchSpace
) -> TaggedProductSearchSpace:
    fidelity_search_space = DiscreteSearchSpace(np.arange(n_fidelities, dtype=float).reshape(-1, 1))
    search_space = TaggedProductSearchSpace(
        [input_search_space, fidelity_search_space], ["input", "fidelity"]
    )
    return search_space


Linear2Fidelity = SingleObjectiveMultifidelityTestProblem(
    name="Linear 2 Fidelity",
    objective=linear_multifidelity,
    search_space=Box(np.zeros(1), np.ones(1)),
    fidelity_search_space=_linear_multifidelity_search_space_builder(
        2, Box(np.zeros(1), np.ones(1))
    ),
    minimizers=_LINEAR_MULTIFIDELITY_MINIMIZERS[2],
    minimum=_LINEAR_MULTIFIDELITY_MINIMA[2],
    num_fidelities=2,
)

Linear3Fidelity = SingleObjectiveMultifidelityTestProblem(
    name="Linear 3 Fidelity",
    objective=linear_multifidelity,
    search_space=Box(np.zeros(1), np.ones(1)),
    fidelity_search_space=_linear_multifidelity_search_space_builder(
        3, Box(np.zeros(1), np.ones(1))
    ),
    minimizers=_LINEAR_MULTIFIDELITY_MINIMIZERS[3],
    minimum=_LINEAR_MULTIFIDELITY_MINIMA[3],
    num_fidelities=3,
)

Linear5Fidelity = SingleObjectiveMultifidelityTestProblem(
    name="Linear 5 Fidelity",
    objective=linear_multifidelity,
    search_space=Box(np.zeros(1), np.ones(1)),
    fidelity_search_space=_linear_multifidelity_search_space_builder(
        5, Box(np.zeros(1), np.ones(1))
    ),
    minimizers=_LINEAR_MULTIFIDELITY_MINIMIZERS[5],
    minimum=_LINEAR_MULTIFIDELITY_MINIMA[5],
    num_fidelities=5,
)

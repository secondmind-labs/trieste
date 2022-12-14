import tensorflow as tf
import numpy as np
from dataclasses import dataclass

from .single_objectives import SingleObjectiveTestProblem
from ..types import TensorType
from ..space import Box, DiscreteSearchSpace, TaggedProductSearchSpace, SearchSpace


@dataclass(frozen=True)
class SingleObjectiveMultifidelityTestProblem(SingleObjectiveTestProblem):

    num_fidelities: int
    """The number of fidelities of test function"""

    input_search_space: SearchSpace
    """The search space of the inputs, ignoring fidelities"""


def linear_multifidelity(x: TensorType):

    x_input = x[..., :-1]
    x_fidelity = x[..., -1:]

    f = 0.5 * ((6.0 * x_input - 2.0) ** 2) * tf.math.sin(12.0 * x_input - 4.0) + 10.0 * (
        x_input - 1.0
    )
    f = f + x_fidelity * (f - 20.0 * (x_input - 1.0))

    return f


_LINEAR_MULTIFIDELITY_MINIMIZERS = {2: 0.75724875, 3: 0.76333767, 5: 0.76801846}

_LINEAR_MULTIFIDELITY_MINIMA = {
    2: -6.020740055,
    3: -6.634287061,
    5: -7.933019704,
}


def _linear_multifidelity_search_space_builder(
    n_fidelities: int, input_search_space
) -> TaggedProductSearchSpace:

    fidelity_search_space = DiscreteSearchSpace(
        np.array([np.arange(n_fidelities, dtype=float)]).reshape(-1, 1)
    )
    search_space = TaggedProductSearchSpace(
        [input_search_space, fidelity_search_space], ["input", "fidelity"]
    )
    return search_space


Linear2Fidelity = SingleObjectiveMultifidelityTestProblem(
    name="Linear 2 Fidelity",
    objective=linear_multifidelity,
    input_search_space=Box(np.zeros(1), np.ones(1)),
    search_space=_linear_multifidelity_search_space_builder(2, Box(np.zeros(1), np.ones(1))),
    minimizers=_LINEAR_MULTIFIDELITY_MINIMIZERS[2],
    minimum=_LINEAR_MULTIFIDELITY_MINIMA[2],
    num_fidelities=2,
)

Linear3Fidelity = SingleObjectiveMultifidelityTestProblem(
    name="Linear 3 Fidelity",
    objective=linear_multifidelity,
    input_search_space=Box(np.zeros(1), np.ones(1)),
    search_space=_linear_multifidelity_search_space_builder(3, Box(np.zeros(1), np.ones(1))),
    minimizers=_LINEAR_MULTIFIDELITY_MINIMIZERS[3],
    minimum=_LINEAR_MULTIFIDELITY_MINIMA[3],
    num_fidelities=3,
)

Linear5Fidelity = SingleObjectiveMultifidelityTestProblem(
    name="Linear 5 Fidelity",
    objective=linear_multifidelity,
    input_search_space=Box(np.zeros(1), np.ones(1)),
    search_space=_linear_multifidelity_search_space_builder(5, Box(np.zeros(1), np.ones(1))),
    minimizers=_LINEAR_MULTIFIDELITY_MINIMIZERS[5],
    minimum=_LINEAR_MULTIFIDELITY_MINIMA[5],
    num_fidelities=5,
)

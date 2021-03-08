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
r"""
This module contains functionality for optimizing
:data:`~trieste.acquisition.AcquisitionFunction`\ s over :class:`~trieste.space.SearchSpace`\ s.
"""
from functools import singledispatch
from typing import Callable, TypeVar, Union

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

from ..space import Box, DiscreteSearchSpace, SearchSpace
from ..type import TensorType
from .function import AcquisitionFunction

SP = TypeVar("SP", bound=SearchSpace)
""" Type variable bound to :class:`~trieste.space.SearchSpace`. """


AcquisitionOptimizer = Callable[[SP, AcquisitionFunction], TensorType]
"""
Type alias for a function that returns the single point that maximizes an acquisition function over
a search space. For a search space with points of shape S, and acquisition function with shapes
[...] + S to [..., 1], the return shape should be [1] + S.
"""


# todo name this more precisely. This may mean making the dss and box variants their own functions.
@singledispatch
def optimize(
    space: Union[Box, DiscreteSearchSpace], target_func: AcquisitionFunction
) -> TensorType:
    """
    :param space: The space of points over which to search, for points with shape [D].
    :param target_func: The function to maximise, with input shape [..., D] and output shape
        [..., 1].
    :return: The point in ``space`` that maximises ``target_func``, with shape [1, D].
    """


@optimize.register
def _discrete_space(space: DiscreteSearchSpace, target_func: AcquisitionFunction) -> TensorType:
    target_func_values = target_func(space.points)
    tf.debugging.assert_shapes(
        [(target_func_values, ("_", 1))],
        message=(
            f"The result of function target_func has an invalid shape:"
            f" {tf.shape(target_func_values)}."
        ),
    )
    max_value_idx = tf.argmax(target_func_values, axis=0)[0]
    return space.points[max_value_idx : max_value_idx + 1]


@optimize.register
def _box(space: Box, target_func: AcquisitionFunction) -> TensorType:
    trial_search_space = space.discretize(20 * tf.shape(space.lower)[-1])
    initial_point = optimize(trial_search_space, target_func)

    bijector = tfp.bijectors.Sigmoid(low=space.lower, high=space.upper)
    variable = tf.Variable(bijector.inverse(initial_point))

    def _objective() -> TensorType:
        return -target_func(bijector.forward(variable))

    gpflow.optimizers.Scipy().minimize(_objective, (variable,))

    return bijector.forward(variable)

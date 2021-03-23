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
from __future__ import annotations

from functools import singledispatch
from typing import Callable, TypeVar

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
a search space. For a search space with points of shape [D], and acquisition function with input
shape [..., B, D] output shape [..., 1], the :data:`AcquisitionOptimizer` return shape should be
[B, D].
"""























@singledispatch
def optimize(
    space: Box | DiscreteSearchSpace, target_func: AcquisitionFunction, num_query_points: int = 1
) -> TensorType:
    """
    :param space: The space of points over which to search, for points with shape [D].
    :param target_func: The :const:`AcquisitionFunction` to maximise, with input shape [..., B, D]
            and output shape [..., 1].
    :param num_query_points: The number of points to acquire.
    :return: The points in ``space`` that together maximize ``target_func``, with shape [B, D].
    """















@optimize.register
def optimize_discrete(
    space: DiscreteSearchSpace, target_func: AcquisitionFunction, num_query_points: int = 1
) -> TensorType:

    if num_query_points > 1:
        raise ValueError("Batch optimisation not yet supported for discrete search spaces")

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
def optimize_continuous(
    space: Box, target_func: AcquisitionFunction, num_query_points: int = 1
) -> TensorType:
    expanded_space = space ** num_query_points
    
    def vectorized_target_func(at):
        return target_func(tf.reshape(at, at.shape[:-1].as_list() + [num_query_points, -1]))



    trial_search_space = expanded_space.discretize(
        tf.minimum(2000, 500 * tf.shape(expanded_space.lower)[-1])
    )

    initial_point = optimize_discrete(trial_search_space, vectorized_target_func)

    bijector = tfp.bijectors.Sigmoid(low=expanded_space.lower, high=expanded_space.upper)
    variable = tf.Variable(bijector.inverse(initial_point))

    def _objective() -> TensorType:
        return -vectorized_target_func(bijector.forward(variable))

    gpflow.optimizers.Scipy().minimize(_objective, (variable,))
    points = bijector.forward(variable)
    return tf.reshape(points, [num_query_points, -1])

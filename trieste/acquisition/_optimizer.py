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
from functools import singledispatch

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

from ..type import QueryPoints
from ..space import SearchSpace, DiscreteSearchSpace, Box

TensorMapping = Callable[[tf.Tensor], tf.Tensor]


@singledispatch
def optimize(space: SearchSpace, target_func: TensorMapping) -> QueryPoints:
    """
    Return the point in ``space`` (with shape S) that maximises the function ``target_func``, as the
    single entry in a 1 by S tensor.

    ``target_func`` must satisfy the following:

      * indices in the leading dimension for both the argument and the result of ``target_func``
        must run over points in the space. For example, the element at index 0 is the first point,
        and the element at index 1 is the second.
      * the result of ``target_func`` must have exactly one additional dimension of size 1. This is
        needed to unambiguously define a maximum.

    :param space: The space of points over which to search.
    :param target_func: The function to maximise.
    :return: The point in ``space`` that maximises ``target_func``.
    """
    raise TypeError(f"No optimize implementation found for space of type {type(space)}")


@optimize.register
def _discrete_space(space: DiscreteSearchSpace, target_func: TensorMapping) -> QueryPoints:
    target_func_values = target_func(space.points)
    tf.debugging.assert_shapes(
        [(target_func_values, ("_", 1))],
        message=f"The result of function target_func has an invalid shape.",
    )
    max_value_idx = tf.argmax(target_func_values, axis=0)[0]
    return space.points[max_value_idx : max_value_idx + 1]


@optimize.register
def _box(space: Box, target_func: TensorMapping) -> QueryPoints:
    trial_search_space = space.discretize(20 * tf.shape(space.lower)[-1])
    initial_point = optimize(trial_search_space, target_func)

    bijector = tfp.bijectors.Sigmoid(low=space.lower, high=space.upper)
    variable = tf.Variable(bijector.inverse(initial_point))

    def _objective() -> tf.Tensor:
        return -target_func(bijector.forward(variable))

    gpflow.optimizers.Scipy().minimize(_objective, (variable,))

    return bijector.forward(variable)

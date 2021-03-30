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
shape [..., B, D] output shape [..., 1], the :const:`AcquisitionOptimizer` return shape should be
[B, D].
"""


def automatic_optimizer_selector(
    space: SearchSpace, target_func: AcquisitionFunction
) -> TensorType:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class performs
    an :const:`AcquisitionOptimizer` appropriate for the
    problem's :class:`~trieste.space.SearchSpace`.
    :param space: The space of points over which to search, for points with shape [D].
    :param target_func: The function to maximise, with input shape [..., 1, D] and output shape
        [..., 1].
    :return: The batch of points in ``space`` that maximises ``target_func``, with shape [1, D].
    """

    optimizer: AcquisitionOptimizer[Box | DiscreteSearchSpace]

    if isinstance(space, DiscreteSearchSpace):
        return optimize_discrete(space, target_func)

    elif isinstance(space, Box):
        return optimize_continuous(space, target_func)

    else:
        raise NotImplementedError(
            f""" No optimizer currentely supports acquisition function
                    maximisation over search spaces of type {space}"""
        )


def optimize_discrete(space: DiscreteSearchSpace, target_func: AcquisitionFunction) -> TensorType:
    """
    An :const:`AcquisitionOptimizer` for :class:'DiscreteSearchSpace' spaces and
    batches of size of 1.
    :param space: The space of points over which to search, for points with shape [D].
    :param target_func: The function to maximise, with input shape [..., 1, D] and output shape
        [..., 1].
    :return: The **one** point in ``space`` that maximises ``target_func``, with shape [1, D].
    """
    target_func_values = target_func(space.points[:, None, :])
    tf.debugging.assert_shapes(
        [(target_func_values, ("_", 1))],
        message=(
            f"The result of function target_func has an invalid shape:"
            f" {tf.shape(target_func_values)}."
        ),
    )
    max_value_idx = tf.argmax(target_func_values, axis=0)[0]
    return space.points[max_value_idx : max_value_idx + 1]


def optimize_continuous(space: Box, target_func: AcquisitionFunction) -> TensorType:
    """
    An :const:`AcquisitionOptimizer` for :class:'Box' spaces and batches of size of 1.
    :param space: The space of points over which to search, for points with shape [D].
    :param target_func: The function to maximise, with input shape [..., 1, D] and output shape
        [..., 1].
    :return: The **one** point in ``space`` that maximises ``target_func``, with shape [1, D].
    """
    trial_search_space = space.discretize(tf.minimum(2000, 500 * tf.shape(space.lower)[-1]))

    initial_point = optimize_discrete(trial_search_space, target_func)  # [1, D]

    bijector = tfp.bijectors.Sigmoid(low=space.lower, high=space.upper)
    variable = tf.Variable(bijector.inverse(initial_point))  # [1, D]

    def _objective() -> TensorType:
        return -target_func(bijector.forward(variable[:, None, :]))  # [1]

    gpflow.optimizers.Scipy().minimize(_objective, (variable,))

    return bijector.forward(variable)  # [1, D]


def batchify(
    batch_size_one_optimizer: AcquisitionOptimizer[SP],
    batch_size: int,
) -> AcquisitionOptimizer[SP]:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` to allow it to optimize batch acquisition functions.
    :param batch_size_one_optimizer: An optimizer that returns only batch size one, i.e. produces a
        single point with shape [1, D].
    :param batch_size: The number of points in the batch.
    :return: An :const:`AcquisitionOptimizer` that will provide a batch of points with shape [B, D].
    """
    tf.debugging.assert_positive(batch_size)

    if batch_size == 1:
        return batch_size_one_optimizer

    def optimizer(search_space: SP, f: AcquisitionFunction) -> TensorType:
        expanded_search_space = search_space ** batch_size  # points have shape [B * D]

        def target_func_with_vectorized_inputs(
            x: TensorType,
        ) -> TensorType:  # [..., 1, B * D] -> [..., 1]
            return f(tf.reshape(x, x.shape[:-2].as_list() + [batch_size, -1]))

        vectorized_points = batch_size_one_optimizer(  # [1, B * D]
            expanded_search_space, target_func_with_vectorized_inputs
        )
        return tf.reshape(vectorized_points, [batch_size, -1])  # [B, D]

    return optimizer


def batchify_greedy(
    batch_size_one_optimizer: AcquisitionOptimizer[SP],
    batch_size: int,
) -> AcquisitionOptimizer[SP]:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` to allow it to approximately optimize batch acquisition functions
        by greedily choosing each batch element in sequence.
    :param batch_size_one_optimizer: An optimizer that returns only batch size one, i.e. produces a
        single point with shape [1, D].
    :param batch_size: The number of points in the batch.
    :return: An :const:`AcquisitionOptimizer` that will provide a batch of points with shape [B, D].
    """
    tf.debugging.assert_positive(batch_size)

    def optimizer(search_space: SP, f: AcquisitionFunction) -> TensorType:

        current_batch = batch_size_one_optimizer(search_space, f)  # [1, D]

        for i in range(1, batch_size):

            def target_func_with_fixed_partial_batch(
                x: TensorType,
            ) -> TensorType:  # [..., 1, D] -> [..., 1]

                current_batch_tiled = tf.repeat(current_batch[None, :, :], x.shape[-3], axis=-3)
                x_with_current_batch = tf.concat([current_batch_tiled, x], -2)
                return f(x_with_current_batch)

            next_batch_point = batch_size_one_optimizer(
                search_space, target_func_with_fixed_partial_batch
            )  # [1, D]

            current_batch = tf.concat([current_batch, next_batch_point], axis=0)  # [i+1, D]

        return current_batch  # [B, D]

    return optimizer

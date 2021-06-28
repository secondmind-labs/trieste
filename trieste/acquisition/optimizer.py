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
import scipy.optimize as spo
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

    if isinstance(space, DiscreteSearchSpace):
        return optimize_discrete(space, target_func)

    elif isinstance(space, Box):
        num_samples = tf.minimum(2000, 500 * tf.shape(space.lower)[-1])
        return generate_continuous_optimizer(num_samples)(space, target_func)

    else:
        raise NotImplementedError(
            f""" No optimizer currentely supports acquisition function
                    maximisation over search spaces of type {space}.
                    Try specifying the optimize_random optimizer"""
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


def generate_continuous_optimizer(
    num_samples: int = 1000, sigmoid: bool = True, num_restarts: int = 10
) -> AcquisitionOptimizer[Box]:
    """
    Generate a gradient-based acquisition optimizer for :class:'Box' spaces and batches
    of size of 1. We perfom gradient-based optimization starting from the best location
    across a sample of `num_samples` random points.

    This optimizer supports either Tensorflow's L-BFGS or Scipy's L-BFGS-B optimizers. We
    constrain L-BFGS's search with a sigmoid bijector that maps an unconstrained space into
    the search space. In contrast, L-BFGS-B optimizes directly within the bounds of the
    search space.

    If using Tensorflow's L-BFGS, then we can run `num_restarts` optimizations in parallel
    (each starting from one of the top  `num_restarts` initial query points ). Note that
    Scipy's L-BFGS-B does not yet support parallel evaluations.

    The default behaviour of this method is to return a L-BFGS optimizer that perfoms
    10 optimizations in parallel.

    :param num_samples: The size of the random sample used to find the starting point(s) of
        the optimization.
    :param sigmoid: If True then use Tensorflow's L-BFGS optimizer, otherwise use
        Scipy's L-BFGS-B optimizer.
    :param num_restarts: The number of optimizations ran in parallel.
    :return: The acquisition optimizer.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    if num_restarts <= 0:
        raise ValueError(f"num_parallel must be positive, got {num_restarts}")

    if not sigmoid and num_restarts > 1:
        raise NotImplementedError(
            """
            Must have  `sigmoid=True` for `num_restarts>1`, as L-BFGS-B does not
            yet support parallel evaluations."
            """
        )

    def optimize_continuous(space: Box, target_func: AcquisitionFunction) -> TensorType:
        """
        A gradient-based :const:`AcquisitionOptimizer` for :class:'Box' spaces and batches
        of size of 1.

        :param space: The space over which to search.
        :param target_func: The function to maximise, with input shape [..., 1, D] and output shape
                [..., 1].
        :return: The **one** point in ``space`` that maximises ``target_func``, with shape [1, D].
        """

        trial_search_space = space.sample(num_samples)  # [num_samples, D]
        target_func_values = target_func(trial_search_space[:, None, :])  # [num_samples, 1]
        _, top_k_indicies = tf.math.top_k(
            tf.squeeze(target_func_values), k=num_restarts
        )  # [num_restarts]
        initial_points = tf.gather(trial_search_space, top_k_indicies)  # [num_restarts, D]

        if sigmoid:  # use tensorflow's L-BFGS optimizer
            bijector = tfp.bijectors.Sigmoid(low=space.lower, high=space.upper)
            unconstrained_initial_points = bijector.inverse(initial_points)  # [num_restarts, D]
            unconstrained_initial_points += 0  # fix for tensorflow bug #1182

            def _objective_tf(x: TensorType) -> TensorType:  # [N, D] -> [N]
                x_new = tf.expand_dims(x, -2)  # [N, 1, D]
                return tf.squeeze(-target_func(bijector.forward(x_new)), -1)  # [N]

            optim_results = tfp.optimizer.lbfgs_minimize(
                lambda x: tfp.math.value_and_gradient(_objective_tf, x),
                initial_position=unconstrained_initial_points,
                stopping_condition=tfp.optimizer.converged_all,
            )

            best_found_point = tf.gather(
                optim_results.position, tf.argmin(optim_results.objective_value)
            )  # [D]
            return bijector(tf.expand_dims(best_found_point, 0))  # [1, D]

        else:  # use scipy's L-BFGS-B optimizer
            opt_kwargs = {"bounds": spo.Bounds(space.lower, space.upper)}
            variable = tf.Variable(initial_points[0:1])  # [1, D]

            def _objective_scipy() -> TensorType:
                return -target_func(variable[:, None, :])  # [1]

            gpflow.optimizers.Scipy().minimize(_objective_scipy, (variable,), **opt_kwargs)

            return variable  # [1, D]

    return optimize_continuous


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
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

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


def generate_random_search_optimizer(num_samples: int = 1000) -> AcquisitionOptimizer[SP]:
    """
    Generate an acquisition optimizer that samples `num_samples` random points across the space.

    :param num_samples: The number of random points to sample.
    :return: The acquisition optimizer.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    def optimize_random(space: SP, target_func: AcquisitionFunction) -> TensorType:
        """
        A random search :const:`AcquisitionOptimizer` defined for
        any :class:'SearchSpace' with a :meth:`sample` and for batches of size of 1.
        If we have a :class:'DiscreteSearchSpace' with fewer than `num_samples` points,
        then we query all the points in the space.

        :param space: The space over which to search.
        :param target_func: The function to maximise, with input shape [..., 1, D] and output shape
                [..., 1].
        :return: The **one** point in ``space`` that maximises ``target_func``, with shape [1, D].
        """

        if isinstance(space, DiscreteSearchSpace) and (num_samples > len(space.points)):
            _num_samples = len(space.points)
        else:
            _num_samples = num_samples

        samples = space.sample(_num_samples)
        target_func_values = target_func(samples[:, None, :])
        max_value_idx = tf.argmax(target_func_values, axis=0)[0]
        return samples[max_value_idx : max_value_idx + 1]

    return optimize_random

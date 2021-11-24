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

r"""
This module contains functionality for optimizing
:data:`~trieste.acquisition.AcquisitionFunction`\ s over :class:`~trieste.space.SearchSpace`\ s.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, TypeVar

import gpflow
import scipy.optimize as spo
import tensorflow as tf
from scipy.optimize import OptimizeResult

from ..space import Box, DiscreteSearchSpace, SearchSpace, TaggedProductSearchSpace
from ..types import TensorType
from .interface import AcquisitionFunction


import ray 

SP = TypeVar("SP", bound=SearchSpace)
""" Type variable bound to :class:`~trieste.space.SearchSpace`. """


NUM_SAMPLES_MIN: int = 5000
"""
The default minimum number of initial samples for :func:`generate_continuous_optimizer` and
:func:`generate_random_search_optimizer` function, used for determining the number of initial
samples in the multi-start acquisition function optimization.
"""


NUM_SAMPLES_DIM: int = 1000
"""
The default minimum number of initial samples per dimension of the search space for
:func:`generate_continuous_optimizer` function in :func:`automatic_optimizer_selector`, used for
determining the number of initial samples in the multi-start acquisition function optimization.
"""


class FailedOptimizationError(Exception):
    """Raised when an acquisition optimizer fails to optimize"""


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

    elif isinstance(space, (Box, TaggedProductSearchSpace)):
        num_samples = tf.minimum(NUM_SAMPLES_MIN, NUM_SAMPLES_DIM * tf.shape(space.lower)[-1])
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
    num_initial_samples: int = NUM_SAMPLES_MIN,
    num_optimization_runs: int = 1,
    num_recovery_runs: int = 5,
    optimizer_args: dict[str, Any] = dict(),
) -> AcquisitionOptimizer[Box | TaggedProductSearchSpace]:
    """
    Generate a gradient-based optimizer for :class:'Box' and :class:'TaggedProductSearchSpace'
    spaces and batches of size 1. In the case of a :class:'TaggedProductSearchSpace', We perform
    gradient-based optimization across all :class:'Box' subspaces, starting from the best location
    found across a sample of `num_initial_samples` random points.

    We advise the user to either use the default `NUM_SAMPLES_MIN` for `num_initial_samples`, or
    `NUM_SAMPLES_DIM` times the dimensionality of the search space, whichever is smaller.

    This optimizer supports Scipy's L-BFGS-B optimizer (used via GPflow's Scipy optimizer wrapper),
    which optimizes directly within and up to the bounds of the search space.

    For challenging acquisition function optimizations, we run `num_optimization_runs` separate
    optimizations, each starting from one of the top `num_optimization_runs` initial query points.

    If all `num_optimization_runs` optimizations fail to converge then we run up to
    `num_recovery_runs` starting from random locations.

    The default behavior of this method is to return a L-BFGS-B optimizer that performs
    a single optimization from the best of `NUM_SAMPLES_MIN` initial locations. If this
    optimization fails then we run up to `num_recovery_runs` recovery runs starting from additional
    random locations.

    :param num_initial_samples: The size of the random sample used to find the starting point(s) of
        the optimization.
    :param num_optimization_runs: The number of separate optimizations to run.
    :param num_recovery_runs: The maximum number of recovery optimization runs in case of failure.
    :param optimizer_args: The keyword arguments to pass to the GPflow's Scipy optimizer wrapper.
        Check `minimize` method  of :class:`~gpflow.optimizers.Scipy` for details what arguments
        can be passed.
    :return: The acquisition optimizer.
    """
    if num_initial_samples <= 0:
        raise ValueError(f"num_initial_samples must be positive, got {num_initial_samples}")

    if num_optimization_runs <= 0:
        raise ValueError(f"num_optimization_runs must be positive, got {num_optimization_runs}")

    if num_initial_samples < num_optimization_runs:
        raise ValueError(
            f"""
            num_initial_samples {num_initial_samples} must be at
            least num_optimization_runs {num_optimization_runs}
            """
        )

    if num_recovery_runs <= -1:
        raise ValueError(f"num_recovery_runs must be zero or greater, got {num_recovery_runs}")

    def optimize_continuous(
        space: Box | TaggedProductSearchSpace, target_func: AcquisitionFunction
    ) -> TensorType:
        """
        A gradient-based :const:`AcquisitionOptimizer` for :class:'Box'
        and :class:`TaggedProductSearchSpace' spaces and batches of size of 1.

        For :class:'TaggedProductSearchSpace' we only apply gradient updates to
        its class:'Box' subspaces, fixing the discrete elements to the best values
        found across the initial random search. To fix these discrete elements, we
        optimize over a continuous class:'Box' relaxation of the discrete subspaces
        which has equal upper and lower bounds, i.e. we specify an equality constraint
        for this dimension in the scipy optimizer.

        :param space: The space over which to search.
        :param target_func: The function to maximise, with input shape [..., 1, D] and output shape
                [..., 1].
        :return: The **one** point in ``space`` that maximises ``target_func``, with shape [1, D].
        """

        trial_search_space = space.sample(num_initial_samples)  # [num_initial_samples, D]
        target_func_values = target_func(trial_search_space[:, None, :])  # [num_samples, 1]
        _, top_k_indicies = tf.math.top_k(
            target_func_values[:, 0], k=num_optimization_runs
        )  # [num_optimization_runs]
        initial_points = tf.gather(trial_search_space, top_k_indicies)  # [num_optimization_runs, D]

        

        @ray.remote
        def _perform_optimization(starting_point: TensorType, optimizer_args: dict[str, Any]) -> scipy.optimize.OptimizeResult:
            variable = tf.Variable(starting_point)  # [1, D]
            
            if isinstance(space, TaggedProductSearchSpace): # only optimize over continuous variables
                bounds = get_bounds_of_box_relaxation_around_point(space, starting_point)
            else:
                bounds = spo.Bounds(space.lower, space.upper)

            optimizer_args_local = copy.deepcopy(optimizer_args)
            optimizer_args_local["bounds"] = bounds

            def _objective() -> TensorType:
                return -target_func(variable[:, None, :])  # [1]

            return gpflow.optimizers.Scipy().minimize(_objective, (variable,), **optimizer_args) 


        if num_optimization_runs==1:
            opt_result = _perform_optimization(initial_points[i : i + 1], optimizer_args)
            if opt_result.success:
                successful_optimization = True
                chosen_point = tf.convert_to_tensor(opt_result.x, dtype= initial_points.dtype) # [D]
                chosen_point = tf.expand_dims(chosen_point,0) # [1, D]
        
        else: # parallelize using ray
            ray.init()
            try:
                result_ids = []
                for i in tf.range(num_optimization_runs): # start num_optimization_runs optimizations in parallel
                    result_ids.append(_perform_optimization.remote(initial_points[i : i + 1], optimizer_args))
                results = ray.get(result_ids)



            finally: # clean up 
                ray.shutdown()
            if tf.math.reduce_any([result.success for result in results]): # TODO
                successful_optimization = True
                new_point_scores = [-result.fun[0][0] for result in results]
                chosen_point = results[tf.argmax(new_point_scores)].x
                chosen_point = tf.convert_to_tensor(chosen_point, dtype= initial_points.dtype) # [D]
                chosen_point = tf.expand_dims(chosen_point,0) # [1, D]


        if not successful_optimization:  # if all optimizations failed then try from random starts
            for i in tf.range(num_recovery_runs):
                random_start = space.sample(1) # [1, D]
                opt_result= _perform_optimization(random_start, optimizer_args)
                if opt_result.success:
                    successful_optimization = True
                    chosen_point = tf.convert_to_tensor(opt_result.x, dtype= initial_points.dtype) # [D]
                    chosen_point = tf.expand_dims(chosen_point,0) # [1, D]
                    break
            if not successful_optimization:  # return error if still failed
                raise FailedOptimizationError(
                    f"""
                    Acquisition function optimization failed,
                    even after {num_recovery_runs + num_optimization_runs} restarts.
                    """
                )

        return chosen_point

    return optimize_continuous



def get_bounds_of_box_relaxation_around_point(
    space: TaggedProductSearchSpace, current_point: TensorType
) -> spo.Bounds:
    """
    A function to return the bounds of a continuous relaxation of
    a :class:'TaggedProductSearchSpace' space, i.e. replacing discrete
    spaces with continuous spaces. In particular, all :class:'DiscreteSearchSpace'
    subspaces are replaced with a new :class:'DiscreteSearchSpace' fixed at their
    respective component of the specified 'current_point'. Note that
    all :class:'Box' subspaces remain the same.

    :param space: The original search space.
    :param current_point: The point at which to make the continuous relaxation.
    :return: Bounds for the Scipy optimizer.
    """
    tf.debugging.Assert(isinstance(space, TaggedProductSearchSpace), [])

    space_with_fixed_discrete = space
    for tag in space.subspace_tags:
        if isinstance(
            space.get_subspace(tag), DiscreteSearchSpace
        ):  # convert discrete subspaces to box spaces.
            subspace_value = space.get_subspace_component(tag, current_point)
            space_with_fixed_discrete = space_with_fixed_discrete.fix_subspace(tag, subspace_value)
    return spo.Bounds(space_with_fixed_discrete.lower, space_with_fixed_discrete.upper)


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


def generate_random_search_optimizer(
    num_samples: int = NUM_SAMPLES_MIN,
) -> AcquisitionOptimizer[SP]:
    """
    Generate an acquisition optimizer that samples `num_samples` random points across the space.
    The default is to sample at `NUM_SAMPLES_MIN` locations.

    We advise the user to either use the default `NUM_SAMPLES_MIN` for `num_samples`, or
    `NUM_SAMPLES_DIM` times the dimensionality of the search space, whichever is smaller.

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
        samples = space.sample(num_samples)
        target_func_values = target_func(samples[:, None, :])
        max_value_idx = tf.argmax(target_func_values, axis=0)
        return tf.gather(samples, max_value_idx)

    return optimize_random

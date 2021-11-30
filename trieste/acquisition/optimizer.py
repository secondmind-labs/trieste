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

from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

import greenlet as gr
import numpy as np
import scipy.optimize as spo
import tensorflow as tf
import tensorflow_probability as tfp

from ..space import Box, DiscreteSearchSpace, SearchSpace, TaggedProductSearchSpace
from ..types import TensorType
from .interface import AcquisitionFunction

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

NUM_RUNS_DIM: int = 10
"""
The default minimum number of optimization runs per dimension of the search space for
:func:`generate_continuous_optimizer` function in :func:`automatic_optimizer_selector`, used for
determining the number of acquisition function optimizations to be perfomed in parallel.
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
        num_samples = tf.maximum(NUM_SAMPLES_MIN, NUM_SAMPLES_DIM * tf.shape(space.lower)[-1])
        num_runs = NUM_RUNS_DIM * tf.shape(space.lower)[-1]
        return generate_continuous_optimizer(
            num_initial_samples=num_samples,
            num_optimization_runs=num_runs,
        )(space, target_func)

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
    num_optimization_runs: int = 10,
    num_recovery_runs: int = 10,
    optimizer_args: dict[str, Any] = dict(),
) -> AcquisitionOptimizer[Box | TaggedProductSearchSpace]:
    """
    Generate a gradient-based optimizer for :class:'Box' and :class:'TaggedProductSearchSpace'
    spaces and batches of size 1. In the case of a :class:'TaggedProductSearchSpace', We perform
    gradient-based optimization across all :class:'Box' subspaces, starting from the best location
    found across a sample of `num_initial_samples` random points.

    We advise the user to either use the default `NUM_SAMPLES_MIN` for `num_initial_samples`, or
    `NUM_SAMPLES_DIM` times the dimensionality of the search space, whichever is greater.
    Similarly, for `num_optimization_runs`, we recommend using `NUM_RUNS_DIM` times the
    dimensionality of the search space.

    This optimizer uses Scipy's L-BFGS-B optimizer. We run `num_optimization_runs` separate
    optimizations in parallel, each starting from one of the best `num_optimization_runs` initial
    query points.

    If all `num_optimization_runs` optimizations fail to converge then we run
    `num_recovery_runs` additional runs starting from random locations (also ran in parallel).

    :param num_initial_samples: The size of the random sample used to find the starting point(s) of
        the optimization.
    :param num_optimization_runs: The number of separate optimizations to run.
    :param num_recovery_runs: The maximum number of recovery optimization runs in case of failure.
    :param optimizer_args: The keyword arguments to pass to the Scipy L-BFGS-B optimizer.
        Check `minimize` method  of :class:`~scipy.optimize` for details of which arguments
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
        its class:'Box' subspaces.

        :param space: The space over which to search.
        :param target_func: The function to maximise, with input shape [..., 1, D] and output shape
                [..., 1].
        :return: The **one** point in ``space`` that maximises ``target_func``, with shape [1, D].
        """

        trial_search_space = space.sample(num_initial_samples)  # [num_initial_samples, D]
        target_func_values = target_func(trial_search_space[:, None, :])  # [num_samples, 1]
        _, top_k_indices = tf.math.top_k(
            target_func_values[:, 0], k=num_optimization_runs
        )  # [num_optimization_runs]
        initial_points = tf.gather(trial_search_space, top_k_indices)  # [num_optimization_runs, D]

        results = _perform_parallel_continuous_optimization(
            target_func,
            space,
            initial_points,
            optimizer_args,
        )
        successful_optimization = np.any(
            [result.success for result in results]
        )  # Check that at least one optimization was successful

        if not successful_optimization:  # if all optimizations failed then try from random starts
            random_points = space.sample(num_recovery_runs)  # [num_recovery_runs, D]
            results = _perform_parallel_continuous_optimization(
                target_func, space, random_points, optimizer_args
            )
            successful_optimization = np.any([result.success for result in results])

        if not successful_optimization:  # return error if still failed
            raise FailedOptimizationError(
                f"""
                    Acquisition function optimization failed,
                    even after {num_recovery_runs + num_optimization_runs} restarts.
                    """
            )

        best_run_id = np.argmax([-result.fun for result in results])  # identify best solution
        np_chosen_point = results[best_run_id].x.reshape(1, -1)  # [1, D]
        chosen_point = tf.constant(np_chosen_point, dtype=initial_points.dtype)  # convert to TF

        return chosen_point

    return optimize_continuous


def _perform_parallel_continuous_optimization(
    target_func: AcquisitionFunction,
    space: SearchSpace,
    starting_points: TensorType,
    optimizer_args: dict[str, Any],
) -> List[spo.OptimizeResult]:
    """
    A function to perform parallel optimization of our acquisition functions
    using Scipy. We perform L-BFGS-B starting from each of the locations contained
    in `starting_points`, i.e. the number of individual optimization runs is
    given by the leading dimension of `starting_points`.

    To provide a parallel implementation of Scipy's L-BFGS-B that can leverage
    batch calculations with TensorFlow, this function uses the Greenlet package
    to run each individual optimization on micro-threads.

    L-BFGS-B updates for each individual optimization are performed by
    independent greenlets working with Numpy arrays, however, the evaluation
    of our acquisition function (and its gradients) is calculated in parallel
    (for each optimization step) using Tensorflow.

    For :class:'TaggedProductSearchSpace' we only apply gradient updates to
    its :class:'Box' subspaces, fixing the discrete elements to the best values
    found across the initial random search. To fix these discrete elements, we
    optimize over a continuous :class:'Box' relaxation of the discrete subspaces
    which has equal upper and lower bounds, i.e. we specify an equality constraint
    for this dimension in the scipy optimizer.

    :param target_func: The function to maximise, with input shape [..., 1, D] and
        output shape [..., 1].
    :param space: The original search space.
    :param starting_points: The points at which to begin our optimizations. The
        leading dimension of `starting_points` controls the number of individual
        optimization runs.
    :param optimizer_args: Keyword arguments to pass to the Scipy optimizer.
    :return: A list containing a Scipy OptimizeResult object for each optimization.
    """

    tf_dtype = starting_points.dtype  # type for communication with Trieste

    num_optimization_runs = tf.shape(starting_points)[0].numpy()

    def _objective_value(x: TensorType) -> TensorType:
        return -target_func(tf.expand_dims(x, 1))  # [len(x),1]

    def _objective_value_and_gradient(x: TensorType) -> Tuple[TensorType, TensorType]:
        return tfp.math.value_and_gradient(_objective_value, x)  # [len(x), 1], [len(x), D]

    if isinstance(
        space, TaggedProductSearchSpace
    ):  # build continuous relaxation of discrete subspaces
        bounds = [
            get_bounds_of_box_relaxation_around_point(space, starting_points[i : i + 1])
            for i in tf.range(num_optimization_runs)
        ]
    else:
        bounds = [spo.Bounds(space.lower, space.upper)] * num_optimization_runs

    # Initialize the numpy arrays to be passed to the greenlets
    np_batch_x = np.zeros((num_optimization_runs, tf.shape(starting_points)[1]), dtype=np.float64)
    np_batch_y = np.zeros((num_optimization_runs,), dtype=np.float64)
    np_batch_dy_dx = np.zeros(
        (num_optimization_runs, tf.shape(starting_points)[1]), dtype=np.float64
    )

    # Set up child greenlets
    child_greenlets = [ScipyLbfgsBGreenlet() for _ in range(num_optimization_runs)]
    child_results: List[Union[spo.OptimizeResult, np.ndarray]] = [
        gr.switch(starting_points[i].numpy(), bounds[i], optimizer_args)
        for i, gr in enumerate(child_greenlets)
    ]

    while True:
        all_done = True
        for i, result in enumerate(child_results):  # Process results from children.
            if isinstance(result, spo.OptimizeResult):
                continue  # children return a `spo.OptimizeResult` if they are finished
            all_done = False
            assert isinstance(result, np.ndarray)  # or an `np.ndarray` with the query `x` otherwise
            np_batch_x[i, :] = result

        if all_done:
            break

        # Batch evaluate query `x`s from all children.
        batch_x = tf.constant(np_batch_x, dtype=tf_dtype)  # [num_optimization_runs, d]
        batch_y, batch_dy_dx = _objective_value_and_gradient(batch_x)
        np_batch_y = batch_y.numpy().astype("float64")
        np_batch_dy_dx = batch_dy_dx.numpy().astype("float64")

        for i, greenlet in enumerate(child_greenlets):  # Feed `y` and `dy_dx` back to children.
            if greenlet.dead:  # Allow for crashed greenlets
                continue
            child_results[i] = greenlet.switch(np_batch_y[i], np_batch_dy_dx[i, :])

    return child_results


class ScipyLbfgsBGreenlet(gr.greenlet):
    """
    Worker greenlet that runs a single Scipy L-BFGS-B. Each greenlet performs all the L-BFGS-B
    update steps required for an individual optimization. However, the evaluation
    of our acquisition function (and its gradients) is delegated back to the main Tensorflow
    process (the parent greenlet) where evaluations can be made efficiently in parallel.
    """

    def run(
        self, start: np.ndarray, bounds: spo.Bounds, optimizer_args: dict[str, Any] = dict()
    ) -> spo.OptimizeResult:
        cache_x = start + 1  # Any value different from `start`.
        cache_y: Optional[np.ndarray] = None
        cache_dy_dx: Optional[np.ndarray] = None

        def value_and_gradient(
            x: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:  # Collect function evaluations from parent greenlet
            nonlocal cache_x
            nonlocal cache_y
            nonlocal cache_dy_dx

            if not (cache_x == x).all():
                cache_x[:] = x  # Copy the value of `x`. DO NOT copy the reference.
                # Send `x` to parent greenlet, which will evaluate all `x`s in a batch.
                cache_y, cache_dy_dx = self.parent.switch(cache_x)

            return cache_y, cache_dy_dx

        return spo.minimize(
            lambda x: value_and_gradient(x)[0],
            start,
            jac=lambda x: value_and_gradient(x)[1],
            method="l-bfgs-b",
            bounds=bounds,
            **optimizer_args,
        )


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

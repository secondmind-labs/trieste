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

from itertools import chain
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union, cast

import greenlet as gr
import numpy as np
import scipy.optimize as spo
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import check_shapes

from .. import logging
from ..space import (
    Box,
    CollectionSearchSpace,
    Constraint,
    GeneralDiscreteSearchSpace,
    SearchSpace,
    SearchSpaceType,
    TaggedMultiSearchSpace,
    TaggedProductSearchSpace,
)
from ..types import TensorType
from .interface import AcquisitionFunction

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
determining the number of acquisition function optimizations to be performed in parallel.
"""


class FailedOptimizationError(Exception):
    """Raised when an acquisition optimizer fails to optimize"""


AcquisitionOptimizer = Callable[
    [SearchSpaceType, Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]]], TensorType
]
"""
Type alias for a function that returns the single point that maximizes an acquisition function over
a search space or the V points that maximize a vectorized acquisition function (as represented by an
acquisition-int tuple).

If this function receives a search space with points of shape [D] and an acquisition function
with input shape [..., 1, D] output shape [..., 1], the :const:`AcquisitionOptimizer` return shape
should be [1, D].

If instead it receives a search space and a tuple containing the acquisition function and its
vectorization V then the :const:`AcquisitionOptimizer` return shape should be [V, D].
"""


def automatic_optimizer_selector(
    space: SearchSpace, target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]]
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

    if isinstance(space, GeneralDiscreteSearchSpace):
        return optimize_discrete(space, target_func)

    elif isinstance(space, (Box, CollectionSearchSpace)):
        space_dim = space.dimension
        num_samples = tf.maximum(NUM_SAMPLES_MIN, NUM_SAMPLES_DIM * space_dim)
        num_runs = NUM_RUNS_DIM * space_dim
        return generate_continuous_optimizer(
            num_initial_samples=num_samples,
            num_optimization_runs=num_runs,
        )(space, target_func)

    else:
        raise NotImplementedError(
            f""" No optimizer currently supports acquisition function
                    maximisation over search spaces of type {space}.
                    Try specifying the optimize_random optimizer"""
        )


def _get_max_discrete_points(
    points: TensorType, target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]]
) -> TensorType:
    # check if we need a vectorized optimizer
    if isinstance(target_func, tuple):
        target_func, V = target_func
    else:
        V = 1

    if V < 0:
        raise ValueError(f"vectorization must be positive, got {V}")

    tiled_points = tf.tile(points, [1, V, 1])
    target_func_values = target_func(tiled_points)
    tf.debugging.assert_shapes(
        [(target_func_values, ("_", V))],
        message=(
            f"""
            The result of function target_func has shape
            {tf.shape(target_func_values)}, however, expected a trailing
            dimension of size {V}.
            """
        ),
    )

    best_indices = tf.math.argmax(target_func_values, axis=0)  # [V]
    return tf.gather(tf.transpose(tiled_points, [1, 0, 2]), best_indices, batch_dims=1)  # [V, D]


def optimize_discrete(
    space: GeneralDiscreteSearchSpace,
    target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
) -> TensorType:
    """
    An :const:`AcquisitionOptimizer` for :class:'GeneralDiscreteSearchSpace' spaces.

    When this functions receives an acquisition-integer tuple as its `target_func`,it evaluates
    all the points in the search space for each of the individual V functions making
    up `target_func`.

    :param space: The space of points over which to search, for points with shape [D].
    :param target_func: The function to maximise, with input shape [..., V, D] and output shape
            [..., V].
    :return: The V points in ``space`` that maximises ``target_func``, with shape [V, D].
    """
    points = space.points[:, None, :]
    return _get_max_discrete_points(points, target_func)


InitialPointSampler = Callable[[SearchSpace], Iterable[TensorType]]
"""
Type alias for a function that returns initial point candidates for an optimization.
Candidates are returned in one or more batches, and each batch should have the shape [N, D],
even when N=1.

For simplicity and memory usage, it is recommended to define these as generators. For example,
the following initial point sampler returns both a set of pre-optimized points and 50,000
random samples:

    def sampler(space: SearchSpace) -> Iterable[TensorType]:
        yield pre_optimized_points
        yield space.sample(50_000)

While the following does the same but groups the random samples into batches of size 1,000
to conserve memory:

    def sampler(space: SearchSpace) -> Iterable[TensorType]:
        yield pre_optimized_points
        yield from sample_from_space(50_000, batch_size=1_000)(space)
"""


def sample_from_space(
    num_samples: int,
    batch_size: Optional[int] = None,
    vectorization: int = 1,
) -> InitialPointSampler:
    """
    An initial point sampler that just samples from the search pace.

    :param num_samples: Number of samples to return.
    :param batch_size: If specified, points are return in batches of this size,
        to preserve memory usage.
    :param vectorization: Vectorization of the target function.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    if isinstance(batch_size, int) and batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if vectorization <= 0:
        raise ValueError(f"vectorization must be positive, got {vectorization}")

    batch_size_int = batch_size or num_samples

    def sampler(space: SearchSpace) -> Iterable[TensorType]:

        # generate additional points for each vectorization (rather than just replicating them)
        if isinstance(space, TaggedMultiSearchSpace):
            remainder = vectorization % len(space.subspace_tags)
            tf.debugging.assert_equal(
                remainder,
                0,
                message=(
                    f"The vectorization of the target function {vectorization} must be a"
                    f"multiple of the batch shape of initial samples "
                    f"{len(space.subspace_tags)}."
                ),
            )
            multiple = vectorization // len(space.subspace_tags)
        else:
            multiple = vectorization

        for offset in range(0, num_samples, batch_size_int):
            num_batch_samples = min(num_samples - offset, batch_size_int)
            candidates = space.sample(num_batch_samples * multiple)
            candidates = tf.reshape(candidates, [num_batch_samples, vectorization, -1])
            yield candidates

    return sampler


def generate_initial_points(
    num_initial_points: int,
    initial_sampler: InitialPointSampler,
    space: SearchSpace,
    target_func: AcquisitionFunction,
    vectorization: int = 1,
) -> TensorType:
    """
    Return the best starting points for an optimization from those generated by a given sampler.

    :param num_initial_points: Number of best starting points to return.
    :param initial_sampler: Initial point sampler.
    :param space: Search space.
    :param target_func: Target function being optimized.
    :param vectorization: Vectorization of the target function.
    """
    top_fun_values: Optional[TensorType] = None  # [V, num_optimization_runs]
    top_candidates: Optional[TensorType] = None  # [V, num_optimization_runs, D]

    for candidates in initial_sampler(space):
        if tf.rank(candidates) == 3:
            # If samples is a tensor of rank 3, then it is a batch of samples. In this case
            # the vectorization of the target function must be a multiple of the length of the
            # second (batch) dimension.
            remainder = vectorization % tf.shape(candidates)[1]
            tf.debugging.assert_equal(
                remainder,
                tf.constant(0, dtype=remainder.dtype),
                message=(
                    f"""
                    The vectorization of the target function {vectorization} must be a multiple of
                    the batch shape of initial samples {tf.shape(candidates)[1]}.
                    """
                ),
            )
            multiple = vectorization // tf.shape(candidates)[1]
            tiled_candidates = tf.tile(candidates, [1, multiple, 1])  # [samples, V, D]
        else:
            tf.debugging.assert_rank(
                candidates,
                2,
                message=(
                    f"""
                    The initial samples must be a tensor of rank 2, got a tensor of rank
                    {tf.rank(candidates)}.
                    """
                ),
            )
            tiled_candidates = tf.tile(
                candidates[:, None, :], [1, vectorization, 1]
            )  # [samples, V, D]

        target_func_values = target_func(tiled_candidates)  # [samples, V]
        tf.debugging.assert_shapes(
            [(target_func_values, ("_", vectorization))],
            message=(
                f"""
                The result of function target_func has shape
                {tf.shape(target_func_values)}, however, expected a trailing
                dimension of size {vectorization}.
                """
            ),
        )

        # now that we know the output dimension and dtypes, initialize empty top tensors
        if top_candidates is None:
            top_candidates = tf.zeros(
                [vectorization, 0, tf.shape(candidates)[-1]], dtype=candidates.dtype
            )
        if top_fun_values is None:
            top_fun_values = tf.zeros([vectorization, 0], dtype=target_func_values.dtype)

        top_candidates = tf.concat(
            [top_candidates, tf.transpose(tiled_candidates, [1, 0, 2])], 1
        )  # [V, samples+num_initial_points, D]
        top_fun_values = tf.concat(
            [top_fun_values, tf.transpose(target_func_values)], 1
        )  # [V, samples+num_initial_points]

        _, top_k_indices = tf.math.top_k(
            top_fun_values, k=min(num_initial_points, tf.shape(top_fun_values)[-1])
        )  # [V, num_initial_points]

        top_candidates = tf.gather(
            top_candidates, top_k_indices, batch_dims=1
        )  # [V, num_initial_points, D]
        top_fun_values = tf.gather(
            top_fun_values, top_k_indices, batch_dims=1
        )  # [V, num_initial_points]

    if top_candidates is None:
        raise ValueError("No initial point generated!")

    initial_points = tf.transpose(top_candidates, [1, 0, 2])  # [num_initial_points,V,D]
    return initial_points


def generate_continuous_optimizer(
    num_initial_samples: int | InitialPointSampler = NUM_SAMPLES_MIN,
    num_optimization_runs: int = 10,
    num_recovery_runs: int = 10,
    optimizer_args: Optional[dict[str, Any]] = None,
) -> AcquisitionOptimizer[Box | CollectionSearchSpace]:
    """
    Generate a gradient-based optimizer for :class:'Box' and :class:'CollectionSearchSpace'
    spaces and batches of size 1. In the case of a :class:'CollectionSearchSpace', We perform
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

    **Note:** using a large number of `num_initial_samples` and `num_optimization_runs` with a
    high-dimensional search space can consume a large amount of CPU memory (RAM).

    :param num_initial_samples: The starting point(s) of the optimization. This can be either
        the number of random samples to use, or a function that given the search space returns
        the points to use. The latter can be used for example to add pre-optimized starting points
        to the random points, as well as to batch point generation to reduce memory usage for
        high-dimensional problems.
    :param num_optimization_runs: The number of separate optimizations to run.
    :param num_recovery_runs: The maximum number of recovery optimization runs in case of failure.
    :param optimizer_args: The keyword arguments to pass to the Scipy L-BFGS-B optimizer.
        Check `minimize` method  of :class:`~scipy.optimize` for details of which arguments
        can be passed. Note that method, jac and bounds cannot/should not be changed.
    :return: The acquisition optimizer.
    """
    if num_optimization_runs <= 0:
        raise ValueError(f"num_optimization_runs must be positive, got {num_optimization_runs}")

    if not callable(num_initial_samples) and num_initial_samples < num_optimization_runs:
        raise ValueError(
            f"""
            num_initial_samples {num_initial_samples} must be at
            least num_optimization_runs {num_optimization_runs}
            """
        )

    if num_recovery_runs < 0:
        raise ValueError(f"num_recovery_runs must be zero or greater, got {num_recovery_runs}")

    def optimize_continuous(
        space: Box | CollectionSearchSpace,
        target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    ) -> TensorType:
        """
        A gradient-based :const:`AcquisitionOptimizer` for :class:'Box'
        and :class:`CollectionSearchSpace' spaces.

        For :class:'CollectionSearchSpace' we only apply gradient updates to
        its class:'Box' subspaces.

        When this function receives an acquisition-integer tuple as its `target_func`,it
        optimizes each of the individual V functions making up `target_func`, i.e.
        evaluating `num_initial_samples` samples, running `num_optimization_runs` runs, and
        (if necessary) running `num_recovery_runs` recovery run for each of the individual
        V functions.

        :param space: The space over which to search.
        :param target_func: The function to maximise, with input shape [..., V, D] and output shape
                [..., V].
        :return: The V points in ``space`` that maximises``target_func``, with shape [V, D].
        """

        if isinstance(target_func, tuple):  # check if we need a vectorized optimizer
            target_func, V = target_func
        else:
            V = 1

        if V <= 0:
            raise ValueError(f"vectorization must be positive, got {V}")

        initial_sampler = (
            sample_from_space(num_initial_samples, vectorization=V)
            if not callable(num_initial_samples)
            else num_initial_samples
        )

        initial_points = generate_initial_points(
            num_optimization_runs, initial_sampler, space, target_func, V
        )  # [num_optimization_runs,V,D]

        if len(initial_points) < num_optimization_runs:
            raise ValueError(
                f"Not enough initial points generated ({len(initial_points)} "
                f"for {num_optimization_runs} optimization runs)"
            )

        (
            successes,
            fun_values,
            chosen_x,
            nfev,
        ) = _perform_parallel_continuous_optimization(  # [num_optimization_runs, V]
            target_func,
            space,
            initial_points,
            optimizer_args or {},
        )

        successful_optimization = tf.reduce_all(
            tf.reduce_any(successes, axis=0)
        )  # Check that at least one optimization was successful for each function
        total_nfev = tf.reduce_max(nfev)  # acquisition function is evaluated in parallel

        recovery_run = False
        if num_recovery_runs and not successful_optimization:
            # if all optimizations failed for a function then try again from random starts
            random_points = space.sample(num_recovery_runs)
            if tf.rank(random_points) == 3:
                # If samples is a tensor of rank 3, then it is a batch of samples. In this case
                # the vectorization of the target function must be a multiple of the length of the
                # second (batch) dimension.
                remainder = V % tf.shape(random_points)[1]
                tf.debugging.assert_equal(
                    remainder,
                    tf.constant(0, dtype=remainder.dtype),
                    message=(
                        f"""
                        The vectorization of the target function {V} must be a multiple of the batch
                        shape of random samples {tf.shape(random_points)[1]}.
                        """
                    ),
                )
                multiple = V // tf.shape(random_points)[1]
                tiled_random_points = tf.tile(
                    random_points, [1, multiple, 1]  # [num_recovery_runs, V, D]
                )
            else:
                tf.debugging.assert_rank(
                    random_points,
                    2,
                    message=(
                        f"""
                        The random samples must be a tensor of rank 2, got a tensor of rank
                        {tf.rank(random_points)}.
                        """
                    ),
                )
                tiled_random_points = tf.tile(
                    random_points[:, None, :], [1, V, 1]
                )  # [num_recovery_runs, V, D]

            (
                recovery_successes,
                recovery_fun_values,
                recovery_chosen_x,
                recovery_nfev,
            ) = _perform_parallel_continuous_optimization(
                target_func, space, tiled_random_points, optimizer_args or {}
            )

            successes = tf.concat(
                [successes, recovery_successes], axis=0
            )  # [num_optimization_runs + num_recovery_runs, V]
            fun_values = tf.concat(
                [fun_values, recovery_fun_values], axis=0
            )  # [num_optimization_runs + num_recovery_runs, V]
            chosen_x = tf.concat(
                [chosen_x, recovery_chosen_x], axis=0
            )  # [num_optimization_runs + num_recovery_runs, V, D]

            successful_optimization = tf.reduce_all(
                tf.reduce_any(successes, axis=0)
            )  # Check that at least one optimization was successful for each function
            total_nfev += tf.reduce_max(recovery_nfev)
            recovery_run = True

        if not successful_optimization:  # return error if still failed
            raise FailedOptimizationError(
                f"""
                    Acquisition function optimization failed,
                    even after {num_recovery_runs + num_optimization_runs} restarts.
                    """
            )

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                logging.scalar("spo_af_evaluations", total_nfev)
                if recovery_run:
                    logging.text(
                        "spo_recovery_run",
                        f"Acquisition function optimization failed after {num_optimization_runs} "
                        f"optimization runs, requiring recovery runs",
                    )

                _target_func: AcquisitionFunction = target_func  # make mypy happy

                def improvements() -> tf.Tensor:
                    best_initial_values = tf.math.reduce_max(_target_func(initial_points), axis=0)
                    best_values = tf.math.reduce_max(fun_values, axis=0)
                    improve = best_values - tf.cast(best_initial_values, best_values.dtype)
                    return improve[0] if V == 1 else improve

                if V == 1:
                    logging.scalar("spo_improvement_on_initial_samples", improvements)
                else:
                    logging.histogram("spo_improvements_on_initial_samples", improvements)

        best_run_ids = tf.math.argmax(fun_values, axis=0)  # [V]
        chosen_points = tf.gather(
            tf.transpose(chosen_x, [1, 0, 2]), best_run_ids, batch_dims=1
        )  # [V, D]

        return chosen_points

    return optimize_continuous


def _perform_parallel_continuous_optimization(
    target_func: AcquisitionFunction,
    space: SearchSpace,
    starting_points: TensorType,
    optimizer_args: dict[str, Any],
) -> Tuple[TensorType, TensorType, TensorType, TensorType]:
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

    For :class:'CollectionSearchSpace' we only apply gradient updates to
    its :class:'Box' subspaces, fixing the discrete elements to the best values
    found across the initial random search. To fix these discrete elements, we
    optimize over a continuous :class:'Box' relaxation of the discrete subspaces
    which has equal upper and lower bounds, i.e. we specify an equality constraint
    for this dimension in the scipy optimizer.

    This function also support the maximization of vectorized target functions (with
    vectorization V).

    :param target_func: The function(s) to maximise, with input shape [..., V, D] and
        output shape [..., V].
    :param space: The original search space.
    :param starting_points: The points at which to begin our optimizations of shape
        [num_optimization_runs, V, D]. The leading dimension of
        `starting_points` controls the number of individual optimization runs
        for each of the V target functions.
    :param optimizer_args: Keyword arguments to pass to the Scipy optimizer.
    :return: A tuple containing the failure statuses, maximum values, maximisers and
        number of evaluations for each of our optimizations.
    """

    tf_dtype = starting_points.dtype  # type for communication with Trieste

    num_optimization_runs_per_function = tf.shape(starting_points)[0].numpy()

    V = tf.shape(starting_points)[-2].numpy()  # vectorized batch size
    D = tf.shape(starting_points)[-1].numpy()  # search space dimension
    num_optimization_runs = num_optimization_runs_per_function * V

    vectorized_starting_points = tf.reshape(
        starting_points, [-1, D]
    )  # [num_optimization_runs*V, D]

    def _objective_value(vectorized_x: TensorType) -> TensorType:  # [N, D] -> [N, 1]
        vectorized_x = vectorized_x[:, None, :]  # [N, 1, D]
        x = tf.reshape(vectorized_x, [-1, V, D])  # [N/V, V, D]
        evals = -target_func(x)  # [N/V, V]
        vectorized_evals = tf.reshape(evals, [-1, 1])  # [N, 1]
        return vectorized_evals

    def _objective_value_and_gradient(x: TensorType) -> Tuple[TensorType, TensorType]:
        return tfp.math.value_and_gradient(_objective_value, x)  # [len(x), 1], [len(x), D]

    # Get the bounds for the optimization
    bounds = get_bounds_of_optimization(space, starting_points)

    # Initialize the numpy arrays to be passed to the greenlets
    np_batch_x = np.zeros((num_optimization_runs, tf.shape(starting_points)[-1]), dtype=np.float64)
    np_batch_y = np.zeros((num_optimization_runs,), dtype=np.float64)
    np_batch_dy_dx = np.zeros(
        (num_optimization_runs, tf.shape(starting_points)[-1]), dtype=np.float64
    )

    # Set up child greenlets
    child_greenlets = [ScipyOptimizerGreenlet() for _ in range(num_optimization_runs)]
    vectorized_child_results: List[Union[spo.OptimizeResult, "np.ndarray[Any, Any]"]] = [
        gr.switch(
            vectorized_starting_points[i].numpy(), bounds[i], space.constraints, optimizer_args
        )
        for i, gr in enumerate(child_greenlets)
    ]

    while True:
        all_done = True
        for i, result in enumerate(vectorized_child_results):  # Process results from children.
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
            vectorized_child_results[i] = greenlet.switch(np_batch_y[i], np_batch_dy_dx[i, :])

    final_vectorized_child_results: List[spo.OptimizeResult] = vectorized_child_results
    vectorized_successes = tf.constant(
        [result.success for result in final_vectorized_child_results]
    )  # [num_optimization_runs]
    vectorized_fun_values = tf.constant(
        [-result.fun for result in final_vectorized_child_results], dtype=tf_dtype
    )  # [num_optimization_runs]
    vectorized_chosen_x = tf.constant(
        [result.x for result in final_vectorized_child_results], dtype=tf_dtype
    )  # [num_optimization_runs, D]
    vectorized_nfev = tf.constant(
        [result.nfev for result in final_vectorized_child_results], dtype=tf_dtype
    )

    # Ensure chosen points satisfy any constraints in the search-space.
    if space.has_constraints:
        is_feasible = space.is_feasible(vectorized_chosen_x)
        vectorized_successes = tf.logical_and(vectorized_successes, is_feasible)

    successes = tf.reshape(vectorized_successes, [-1, V])  # [num_optimization_runs, V]
    fun_values = tf.reshape(vectorized_fun_values, [-1, V])  # [num_optimization_runs, V]
    chosen_x = tf.reshape(vectorized_chosen_x, [-1, V, D])  # [num_optimization_runs, V, D]
    nfev = tf.reshape(vectorized_nfev, [-1, V])  # [num_optimization_runs, V]

    return (successes, fun_values, chosen_x, nfev)


class ScipyOptimizerGreenlet(gr.greenlet):  # type: ignore[misc]
    """
    Worker greenlet that runs a single Scipy L-BFGS-B (by default). Each greenlet performs all the
    optimizer update steps required for an individual optimization. However, the evaluation
    of our acquisition function (and its gradients) is delegated back to the main Tensorflow
    process (the parent greenlet) where evaluations can be made efficiently in parallel.
    """

    def run(
        self,
        start: "np.ndarray[Any, Any]",
        bounds: spo.Bounds,
        constraints: Sequence[Constraint],
        optimizer_args: Optional[dict[str, Any]] = None,
    ) -> spo.OptimizeResult:
        """Greenlet run method."""
        cache_x = start + 1  # Any value different from `start`.
        cache_y: Optional["np.ndarray[Any, Any]"] = None
        cache_dy_dx: Optional["np.ndarray[Any, Any]"] = None

        def value_and_gradient(
            x: "np.ndarray[Any, Any]",
        ) -> Tuple["np.ndarray[Any, Any]", "np.ndarray[Any, Any]"]:
            # Collect function evaluations from parent greenlet
            nonlocal cache_x
            nonlocal cache_y
            nonlocal cache_dy_dx

            if not (cache_x == x).all():
                cache_x[:] = x  # Copy the value of `x`. DO NOT copy the reference.
                # Send `x` to parent greenlet, which will evaluate all `x`s in a batch.
                cache_y, cache_dy_dx = self.parent.switch(cache_x)

            return cast("np.ndarray[Any, Any]", cache_y), cast("np.ndarray[Any, Any]", cache_dy_dx)

        method = "trust-constr" if len(constraints) else "l-bfgs-b"
        optimizer_args = dict(
            {"method": method, "constraints": constraints}, **(optimizer_args or {})
        )
        return spo.minimize(
            lambda x: value_and_gradient(x)[0],
            start,
            jac=lambda x: value_and_gradient(x)[1],
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
    tf.debugging.Assert(isinstance(space, TaggedProductSearchSpace), [tf.constant([])])

    space_with_fixed_discrete = space
    for tag in space.subspace_tags:
        if isinstance(
            space.get_subspace(tag), GeneralDiscreteSearchSpace
        ):  # convert discrete subspaces to box spaces.
            subspace_value = space.get_subspace_component(tag, current_point)
            space_with_fixed_discrete = space_with_fixed_discrete.fix_subspace(tag, subspace_value)
    return spo.Bounds(space_with_fixed_discrete.lower, space_with_fixed_discrete.upper)


@check_shapes(
    "starting_points: [num_optimization_runs, V, D]",
)
def get_bounds_of_optimization(space: SearchSpace, starting_points: TensorType) -> List[spo.Bounds]:
    """
    Returns a list of bounds for all the optimization runs, to be used in the Scipy optimizer.
    The bounds are based on the provided search space and the starting points. The length of the
    list is equal to the total number of individual optimization runs,
    i.e. the number of starting points: `num_optimization_runs` x `V`.

    For each starting point, the bounds are obtained as follows depending on the type of search
    space:
    - For a `TaggedProductSearchSpace`, a "continuous relaxation" of the discrete subspaces is
    built by creating bounds around the point. The discrete components of the created
    search space are fixed at the respective component of the point, and the
    remaining continuous components are set to the original bounds of the search space.
    - For a `TaggedMultiSearchSpace`, the bounds for each point are obtained in a similar
    manner as above, but based potentially on a different subspace for each point;
    instead of sharing a single set of bounds from the common search space.
    The subspaces are assigned to the optimization runs in a round-robin fashion along
    dimension 1 (of size `V`) of the starting points. An error is raised if size `V` is not a
    multiple of the number of subspaces.
    - For other types of search spaces, the original bounds are used for each optimization run.

    :param space: The original search space.
    :param starting_points: The points at which to begin the optimizations, with shape
        `[num_optimization_runs, V, D]`. The leading dimension of `starting_points` controls the
        number of individual optimization runs for each of the `V` starting points or target
        functions.
    :return: A list of bounds for the Scipy optimizer. The length of the list
        is equal to the number of individual optimization runs, i.e. `num_optimization_runs` x `V`.

    For example, for a 2D `TaggedMultiSearchSpace` with two subspaces, each with a continuous
    and a discrete component::

        space = TaggedMultiSearchSpace(
            [
                TaggedProductSearchSpace(
                    [Box([0], [1]), DiscreteSearchSpace([[11], [15], [21], [25], [31], [35]])]
                ),
                TaggedProductSearchSpace(
                    [Box([2], [3]), DiscreteSearchSpace([[13], [17], [23], [27], [33], [37]])]
                )
            ]
        )

    Consider 2 optimization runs per point and a vectorization `V` of 4, for a total of 8
    optimization runs. Given the following starting points::

        starting_points = tf.constant(
            [
                [[10, 11], [12, 13], [14, 15], [16, 17]],
                [[20, 21], [22, 23], [24, 25], [26, 27]],
            ],
            dtype=tf.float64
        )

    The returned list of bounds for the optimization would be::

        [
            spo.Bounds([0, 11], [1, 11]),  # For point at index [0, 0], using subspace 0
            spo.Bounds([2, 13], [3, 13]),  #         //         [0, 1],       //       1
            spo.Bounds([0, 15], [1, 15]),  #         //         [0, 2],       //       0
            spo.Bounds([2, 17], [3, 17]),  #         //         [0, 3],       //       1
            spo.Bounds([0, 21], [1, 21]),  #         //         [1, 0],       //       0
            spo.Bounds([2, 23], [3, 23]),  #         //         [1, 1],       //       1
            spo.Bounds([0, 25], [1, 25]),  #         //         [1, 2],       //       0
            spo.Bounds([2, 27], [3, 27]),  #         //         [1, 3],       //       1
        ]
    """

    num_optimization_runs_per_function, V, D = starting_points.shape
    num_total_optimization_runs = num_optimization_runs_per_function * V

    if isinstance(
        space, TaggedProductSearchSpace
    ):  # build continuous relaxation of discrete subspaces
        vectorized_starting_points = tf.reshape(
            starting_points, [-1, D]
        )  # [num_total_optimization_runs, D]
        bounds = [
            get_bounds_of_box_relaxation_around_point(space, vectorized_starting_points[i : i + 1])
            for i in tf.range(num_total_optimization_runs)
        ]
    elif isinstance(space, TaggedMultiSearchSpace):
        subspaces = [space.get_subspace(tag) for tag in space.subspace_tags]
        # The vectorization `V` of the target function must be a multple of the number of subspaces.
        remainder = V % len(subspaces)
        tf.debugging.assert_equal(
            remainder,
            0,
            message=(
                f"""
                The vectorization of the target function {V} must be a multiple of the number of
                subspaces {len(subspaces)}.
                """
            ),
        )
        multiple = V // len(subspaces)
        subspaces = subspaces * multiple

        # Create a sequence of bounds for each optimization run, per subspace.
        # For product subspaces, we build a continuous relaxation of the discrete subspaces.
        # Otherwise, we use the original bounds.
        bounds = [
            [
                (
                    get_bounds_of_box_relaxation_around_point(ss, starting_points[i : i + 1, j])
                    if isinstance(ss, TaggedProductSearchSpace)
                    else spo.Bounds(ss.lower, ss.upper)
                )
                for j, ss in enumerate(subspaces)
            ]
            for i in tf.range(num_optimization_runs_per_function)
        ]
        bounds = list(chain.from_iterable(bounds))
    else:
        bounds = [spo.Bounds(space.lower, space.upper)] * num_total_optimization_runs

    return bounds


def batchify_joint(
    batch_size_one_optimizer: AcquisitionOptimizer[SearchSpaceType],
    batch_size: int,
) -> AcquisitionOptimizer[SearchSpaceType]:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` to allow it to jointly optimize the batch elements considered
    by a batch acquisition function.

    :param batch_size_one_optimizer: An optimizer that returns only batch size one, i.e. produces a
            single point with shape [1, D].
    :param batch_size: The number of points in the batch.
    :return: An :const:`AcquisitionOptimizer` that will provide a batch of points with shape [B, D].
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    def optimizer(
        search_space: SearchSpaceType,
        f: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    ) -> TensorType:
        expanded_search_space = search_space**batch_size  # points have shape [B * D]

        if isinstance(f, tuple):
            raise ValueError(
                "batchify_joint cannot be applied to a vectorized acquisition function"
            )
        af: AcquisitionFunction = f  # type checking can get confused by closure of f

        def target_func_with_vectorized_inputs(
            x: TensorType,
        ) -> TensorType:  # [..., 1, B * D] -> [..., 1]
            return af(tf.reshape(x, x.shape[:-2].as_list() + [batch_size, -1]))

        vectorized_points = batch_size_one_optimizer(  # [1, B * D]
            expanded_search_space, target_func_with_vectorized_inputs
        )
        return tf.reshape(vectorized_points, [batch_size, -1])  # [B, D]

    return optimizer


def batchify_vectorize(
    batch_size_one_optimizer: AcquisitionOptimizer[SearchSpaceType],
    batch_size: int,
) -> AcquisitionOptimizer[SearchSpaceType]:
    """
    A wrapper around our :const:`AcquisitionOptimizer`s. This class wraps a
    :const:`AcquisitionOptimizer` to allow it to optimize batch acquisition functions.

    Unlike :func:`batchify_joint`, :func:`batchify_vectorize` is suitable
    for a :class:`AcquisitionFunction` whose individual batch element can be
    optimized independently (i.e. they can be vectorized).

    :param batch_size_one_optimizer: An optimizer that returns only batch size one, i.e. produces a
            single point with shape [1, D].
    :param batch_size: The number of points in the batch.
    :return: An :const:`AcquisitionOptimizer` that will provide a batch of points with shape [V, D].
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    def optimizer(
        search_space: SearchSpaceType,
        f: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    ) -> TensorType:
        if isinstance(f, tuple):
            raise ValueError(
                "batchify_vectorize cannot be applied to an already vectorized acquisition function"
            )

        return batch_size_one_optimizer(search_space, (f, batch_size))

    return optimizer


def generate_random_search_optimizer(
    num_samples: int = NUM_SAMPLES_MIN,
) -> AcquisitionOptimizer[SearchSpace]:
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

    def optimize_random(
        space: SearchSpace,
        target_func: Union[AcquisitionFunction, Tuple[AcquisitionFunction, int]],
    ) -> TensorType:
        """
        A random search :const:`AcquisitionOptimizer` defined for
        any :class:'SearchSpace' with a :meth:`sample`. If we have a :class:'DiscreteSearchSpace'
        with fewer than `num_samples` points, then we query all the points in the space.

        When this functions receives an acquisition-integer tuple as its `target_func`,it
        optimizes each of the individual V functions making up `target_func`, i.e.
        evaluating `num_samples` samples for each of the individual V functions making up
        target_func.

        :param space: The space over which to search.
        :param target_func: The function to maximise, with input shape [..., V, D] and output shape
                [..., V].
        :return: The V points in ``space`` that maximises ``target_func``, with shape [V, D].
        """
        points = space.sample(num_samples)[:, None, :]
        return _get_max_discrete_points(points, target_func)

    return optimize_random

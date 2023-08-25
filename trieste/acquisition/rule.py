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
"""
This module contains acquisition rules, which choose the optimal point(s) to query on each step of
the Bayesian optimization process.
"""
from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Sequence, Tuple, TypeVar, Union, cast, overload

import numpy as np
from check_shapes import check_shapes, inherit_check_shapes

try:
    import pymoo
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.optimize import minimize
except ImportError:  # pragma: no cover (tested but not by coverage)
    pymoo = None
    PymooProblem = object

import tensorflow as tf

from .. import logging, types
from ..data import Dataset
from ..models import ProbabilisticModel
from ..models.interfaces import (
    HasReparamSampler,
    ModelStack,
    ProbabilisticModelType,
    TrainableSupportsGetKernel,
)
from ..observer import OBJECTIVE
from ..space import Box, SearchSpace, TaggedMultiSearchSpace
from ..types import State, Tag, TensorType
from ..utils.misc import get_value_for_tag
from .function import (
    BatchMonteCarloExpectedImprovement,
    ExpectedImprovement,
    ProbabilityOfImprovement,
)
from .interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    GreedyAcquisitionFunctionBuilder,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
    SingleModelVectorizedAcquisitionBuilder,
    VectorizedAcquisitionFunctionBuilder,
)
from .multi_objective import Pareto
from .optimizer import (
    AcquisitionOptimizer,
    automatic_optimizer_selector,
    batchify_joint,
    batchify_vectorize,
)
from .sampler import ExactThompsonSampler, ThompsonSampler
from .utils import get_local_dataset, get_unique_points_mask, select_nth_output

ResultType = TypeVar("ResultType", covariant=True)
""" Unbound covariant type variable. """

SearchSpaceType = TypeVar("SearchSpaceType", bound=SearchSpace, contravariant=True)
""" Contravariant type variable bound to :class:`~trieste.space.SearchSpace`. """


class AcquisitionRule(ABC, Generic[ResultType, SearchSpaceType, ProbabilisticModelType]):
    """
    The central component of the acquisition API.

    An :class:`AcquisitionRule` can produce any value from the search space for this step, and the
    historic data and models. This value is typically a set of query points, either on its own as
    a `TensorType` (see e.g. :class:`EfficientGlobalOptimization`), or within some context
    (see e.g. :class:`TrustRegion`). Indeed, to use an :class:`AcquisitionRule` in the main
    :class:`~trieste.bayesian_optimizer.BayesianOptimizer` Bayesian optimization loop, the rule
    must return either a `TensorType` or `State`-ful `TensorType`.

    Note that an :class:`AcquisitionRule` might only support models with specific features (for
    example, if it uses an acquisition function that relies on those features). The type of
    models supported by a rule is indicated by the generic type variable
    class:`ProbabilisticModelType`.
    """

    @abstractmethod
    def acquire(
        self,
        search_space: SearchSpaceType,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> ResultType:
        """
        Return a value of type `T_co`. Typically this will be a set of query points, either on its
        own as a `TensorType` (see e.g. :class:`EfficientGlobalOptimization`), or within some
        context (see e.g. :class:`TrustRegion`). We assume that this requires at least models, but
        it may sometimes also need data.

        **Type hints:**
          - The search space must be a :class:`~trieste.space.SearchSpace`. The exact type of
            :class:`~trieste.space.SearchSpace` depends on the specific :class:`AcquisitionRule`.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations for each tag (optional).
        :return: A value of type `T_co`.
        """

    def acquire_single(
        self,
        search_space: SearchSpaceType,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
    ) -> ResultType:
        """
        A convenience wrapper for :meth:`acquire` that uses only one model, dataset pair.

        :param search_space: The global search space over which the optimization problem
            is defined.
        :param model: The model to use.
        :param dataset: The known observer query points and observations (optional).
        :return: A value of type `T_co`.
        """
        if isinstance(dataset, dict) or isinstance(model, dict):
            raise ValueError(
                "AcquisitionRule.acquire_single method does not support multiple datasets "
                "or models: use acquire instead"
            )
        return self.acquire(
            search_space,
            {OBJECTIVE: model},
            datasets=None if dataset is None else {OBJECTIVE: dataset},
        )


class EfficientGlobalOptimization(
    AcquisitionRule[TensorType, SearchSpaceType, ProbabilisticModelType]
):
    """Implements the Efficient Global Optimization, or EGO, algorithm."""

    @overload
    def __init__(
        self: "EfficientGlobalOptimization[SearchSpaceType, ProbabilisticModel]",
        builder: None = None,
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
        initial_acquisition_function: Optional[AcquisitionFunction] = None,
    ):
        ...

    @overload
    def __init__(
        self: "EfficientGlobalOptimization[SearchSpaceType, ProbabilisticModelType]",
        builder: (
            AcquisitionFunctionBuilder[ProbabilisticModelType]
            | GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelAcquisitionBuilder[ProbabilisticModelType]
            | SingleModelGreedyAcquisitionBuilder[ProbabilisticModelType]
        ),
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
        initial_acquisition_function: Optional[AcquisitionFunction] = None,
    ):
        ...

    def __init__(
        self,
        builder: Optional[
            AcquisitionFunctionBuilder[ProbabilisticModelType]
            | GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]
            | VectorizedAcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelAcquisitionBuilder[ProbabilisticModelType]
            | SingleModelGreedyAcquisitionBuilder[ProbabilisticModelType]
            | SingleModelVectorizedAcquisitionBuilder[ProbabilisticModelType]
        ] = None,
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
        initial_acquisition_function: Optional[AcquisitionFunction] = None,
    ):
        """
        :param builder: The acquisition function builder to use. Defaults to
            :class:`~trieste.acquisition.ExpectedImprovement`.
        :param optimizer: The optimizer with which to optimize the acquisition function built by
            ``builder``. This should *maximize* the acquisition function, and must be compatible
            with the global search space. Defaults to
            :func:`~trieste.acquisition.optimizer.automatic_optimizer_selector`.
        :param num_query_points: The number of points to acquire.
        :param initial_acquisition_function: The initial acquisition function to use. Defaults
            to using the builder to construct one, but passing in a previously constructed
            function can occasionally be useful (e.g. to preserve random seeds).
        """

        if num_query_points <= 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )

        if builder is None:
            if num_query_points == 1:
                builder = ExpectedImprovement()
            else:
                raise ValueError(
                    """Need to specify a batch acquisition function when number of query points
                    is greater than 1"""
                )

        if optimizer is None:
            optimizer = automatic_optimizer_selector

        if isinstance(
            builder,
            (
                SingleModelAcquisitionBuilder,
                SingleModelGreedyAcquisitionBuilder,
                SingleModelVectorizedAcquisitionBuilder,
            ),
        ):
            builder = builder.using(OBJECTIVE)

        if num_query_points > 1:  # need to build batches of points
            if isinstance(builder, VectorizedAcquisitionFunctionBuilder):
                # optimize batch elements independently
                optimizer = batchify_vectorize(optimizer, num_query_points)
            elif isinstance(builder, AcquisitionFunctionBuilder):
                # optimize batch elements jointly
                optimizer = batchify_joint(optimizer, num_query_points)
            elif isinstance(builder, GreedyAcquisitionFunctionBuilder):
                # optimize batch elements sequentially using the logic in acquire.
                pass

        self._builder: Union[
            AcquisitionFunctionBuilder[ProbabilisticModelType],
            GreedyAcquisitionFunctionBuilder[ProbabilisticModelType],
            VectorizedAcquisitionFunctionBuilder[ProbabilisticModelType],
        ] = builder
        self._optimizer = optimizer
        self._num_query_points = num_query_points
        self._acquisition_function: Optional[AcquisitionFunction] = initial_acquisition_function

    def __repr__(self) -> str:
        """"""
        return f"""EfficientGlobalOptimization(
        {self._builder!r},
        {self._optimizer!r},
        {self._num_query_points!r})"""

    @property
    def acquisition_function(self) -> Optional[AcquisitionFunction]:
        """The current acquisition function, updated last time :meth:`acquire` was called."""
        return self._acquisition_function

    def acquire(
        self,
        search_space: SearchSpaceType,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        """
        Return the query point(s) that optimizes the acquisition function produced by ``builder``
        (see :meth:`__init__`).

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations. Whether this is required
            depends on the acquisition function used.
        :return: The single (or batch of) points to query.
        """
        if self._acquisition_function is None:
            self._acquisition_function = self._builder.prepare_acquisition_function(
                models,
                datasets=datasets,
            )
        else:
            self._acquisition_function = self._builder.update_acquisition_function(
                self._acquisition_function,
                models,
                datasets=datasets,
            )

        summary_writer = logging.get_tensorboard_writer()
        step_number = logging.get_step_number()
        greedy = isinstance(self._builder, GreedyAcquisitionFunctionBuilder)

        with tf.name_scope("EGO.optimizer" + "[0]" * greedy):
            points = self._optimizer(search_space, self._acquisition_function)

        if summary_writer:
            with summary_writer.as_default(step=step_number):
                batched_points = tf.expand_dims(points, axis=0)
                values = self._acquisition_function(batched_points)[0]
                if len(values) == 1:
                    logging.scalar(
                        "EGO.acquisition_function/maximum_found" + "[0]" * greedy, values[0]
                    )
                else:  # vectorized acquisition function
                    logging.histogram(
                        "EGO.acquisition_function/maximums_found" + "[0]" * greedy, values
                    )

        if isinstance(self._builder, GreedyAcquisitionFunctionBuilder):
            for i in range(
                self._num_query_points - 1
            ):  # greedily allocate remaining batch elements
                self._acquisition_function = self._builder.update_acquisition_function(
                    self._acquisition_function,
                    models,
                    datasets=datasets,
                    pending_points=points,
                    new_optimization_step=False,
                )
                with tf.name_scope(f"EGO.optimizer[{i+1}]"):
                    chosen_point = self._optimizer(search_space, self._acquisition_function)
                points = tf.concat([points, chosen_point], axis=0)

                if summary_writer:
                    with summary_writer.as_default(step=step_number):
                        batched_points = tf.expand_dims(chosen_point, axis=0)
                        values = self._acquisition_function(batched_points)[0]
                        if len(values) == 1:
                            logging.scalar(
                                f"EGO.acquisition_function/maximum_found[{i + 1}]", values[0]
                            )
                        else:  # vectorized acquisition function
                            logging.histogram(
                                f"EGO.acquisition_function/maximums_found[{i+1}]", values
                            )

        return points


@dataclass(frozen=True)
class AsynchronousRuleState:
    """Stores pending points for asynchronous rules.
    These are points which were requested but are not observed yet.
    """

    pending_points: Optional[TensorType] = None

    def __post_init__(self) -> None:
        if self.pending_points is None:
            # that's fine, no validation needed
            return

        tf.debugging.assert_shapes(
            [(self.pending_points, ["N", "D"])],
            message=f"""Pending points are expected to be a 2D tensor,
                        instead received tensor of shape {tf.shape(self.pending_points)}""",
        )

    @property
    def has_pending_points(self) -> bool:
        """Returns `True` if there is at least one pending point, and `False` otherwise."""
        return (self.pending_points is not None) and tf.size(self.pending_points) > 0

    def remove_points(self, points_to_remove: TensorType) -> AsynchronousRuleState:
        """Removes all rows from current `pending_points` that are present in `points_to_remove`.
        If a point to remove occurs multiple times in the list of pending points,
        only first occurrence of it will be removed.

        :param points_to_remove: Points to remove.
        :return: New instance of `AsynchronousRuleState` with updated pending points.
        """

        @tf.function
        def _remove_point(pending_points: TensorType, point_to_remove: TensorType) -> TensorType:
            # find all points equal to the one we need to remove
            are_points_equal = tf.reduce_all(tf.equal(pending_points, point_to_remove), axis=1)
            if not tf.reduce_any(are_points_equal):
                # point to remove isn't there, nothing to do
                return pending_points

            # since we're compiling, we still need to handle pending_points = [] here
            top = tf.cond(tf.math.greater(1, tf.shape(are_points_equal)[0]), lambda: 0, lambda: 1)

            # this line converts all bool values to 0 and 1
            # then finds first 1 and returns its index as 1x1 tensor
            _, first_index_tensor = tf.math.top_k(tf.cast(are_points_equal, tf.int8), k=top)
            # to use it as index for slicing, we need to convert 1x1 tensor to a TF scalar
            first_index = tf.reshape(first_index_tensor, [])
            return tf.concat(
                [pending_points[:first_index, :], pending_points[first_index + 1 :, :]], axis=0
            )

        if not self.has_pending_points:
            # nothing to do if there are no pending points
            return self

        tf.debugging.assert_shapes(
            [(self.pending_points, [None, "D"]), (points_to_remove, [None, "D"])],
            message=f"""Point to remove shall be 1xD where D is
                        the last dimension of pending points.
                        Got {tf.shape(self.pending_points)} for pending points
                        and {tf.shape(points_to_remove)} for other points.""",
        )

        new_pending_points = tf.foldl(
            _remove_point, points_to_remove, initializer=self.pending_points
        )
        return AsynchronousRuleState(new_pending_points)

    def add_pending_points(self, new_points: TensorType) -> AsynchronousRuleState:
        """Adds `new_points` to the already known pending points.

        :param new_points: Points to add.
        :return: New instance of `AsynchronousRuleState` with updated pending points.
        """
        if not self.has_pending_points:
            return AsynchronousRuleState(new_points)

        tf.debugging.assert_shapes(
            [(self.pending_points, [None, "D"]), (new_points, [None, "D"])],
            message=f"""New points shall be 2D and have same last dimension as pending points.
                        Got {tf.shape(self.pending_points)} for pending points
                        and {tf.shape(new_points)} for new points.""",
        )

        new_pending_points = tf.concat([self.pending_points, new_points], axis=0)
        return AsynchronousRuleState(new_pending_points)


class AsynchronousOptimization(
    AcquisitionRule[
        State[Optional["AsynchronousRuleState"], TensorType],
        SearchSpaceType,
        ProbabilisticModelType,
    ]
):
    """AsynchronousOptimization rule is designed for asynchronous BO scenarios.
    By asynchronous BO we understand a use case when multiple objective function
    can be launched in parallel and are expected to arrive at different times.
    Instead of waiting for the rest of observations to return, we want to immediately
    use acquisition function to launch a new observation and avoid wasting computational resources.
    See :cite:`Alvi:2019` or :cite:`kandasamy18a` for more details.

    To make the best decision about next point to observe, acquisition function
    needs to be aware of currently running observations.
    We call such points "pending", and consider them a part of acquisition state.
    We use :class:`AsynchronousRuleState` to store these points.

    `AsynchronousOptimization` works with non-greedy batch acquisition functions.
    For example, it would work with
    :class:`~trieste.acquisition.BatchMonteCarloExpectedImprovement`,
    but cannot be used with :class:`~trieste.acquisition.ExpectedImprovement`.
    If there are P pending points and the batch of size B is requested,
    the acquisition function is used with batch size P+B.
    During optimization first P points are fixed to pending,
    and thus we optimize and return the last B points only.
    """

    @overload
    def __init__(
        self: "AsynchronousOptimization[SearchSpaceType, HasReparamSampler]",
        builder: None = None,
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        ...

    @overload
    def __init__(
        self: "AsynchronousOptimization[SearchSpaceType, ProbabilisticModelType]",
        builder: (
            AcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelAcquisitionBuilder[ProbabilisticModelType]
        ),
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        ...

    def __init__(
        self,
        builder: Optional[
            AcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelAcquisitionBuilder[ProbabilisticModelType]
        ] = None,
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        """
        :param builder: Batch acquisition function builder. Defaults to
            :class:`~trieste.acquisition.BatchMonteCarloExpectedImprovement` with 10 000 samples.
        :param optimizer: The optimizer with which to optimize the acquisition function built by
            ``builder``. This should *maximize* the acquisition function, and must be compatible
            with the global search space. Defaults to
            :func:`~trieste.acquisition.optimizer.automatic_optimizer_selector`.
        :param num_query_points: The number of points to acquire.
        """
        if num_query_points <= 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )

        if builder is None:
            builder = cast(
                SingleModelAcquisitionBuilder[ProbabilisticModelType],
                BatchMonteCarloExpectedImprovement(10_000),
            )

        if optimizer is None:
            optimizer = automatic_optimizer_selector

        if isinstance(builder, SingleModelAcquisitionBuilder):
            builder = builder.using(OBJECTIVE)

        # even though we are only using batch acquisition functions
        # there is no need to batchify_joint the optimizer if our batch size is 1
        if num_query_points > 1:
            optimizer = batchify_joint(optimizer, num_query_points)

        self._builder: AcquisitionFunctionBuilder[ProbabilisticModelType] = builder
        self._optimizer = optimizer
        self._acquisition_function: Optional[AcquisitionFunction] = None

    def __repr__(self) -> str:
        """"""
        return f"""AsynchronousOptimization(
        {self._builder!r},
        {self._optimizer!r})"""

    def acquire(
        self,
        search_space: SearchSpaceType,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> types.State[AsynchronousRuleState | None, TensorType]:
        """
        Constructs a function that, given ``AsynchronousRuleState``,
        returns a new state object and points to evaluate.
        The state object contains currently known pending points,
        that is points that were requested for evaluation,
        but observation for which was not received yet.
        To keep them up to date, pending points are compared against the given dataset,
        and whatever points are in the dataset are deleted.

        Let's suppose we have P pending points. To optimize the acquisition function
        we call it with batches of size P+1, where first P points are fixed to pending points.
        Optimization therefore happens over the last point only, which is returned.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model of the known data. Uses the single key `OBJECTIVE`.
        :param datasets: The known observer query points and observations.
        :return: A function that constructs the next acquisition state and the recommended query
            points from the previous acquisition state.
        """
        if models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )
        if datasets is None or datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""datasets must be provided and contain the single key {OBJECTIVE}"""
            )

        if self._acquisition_function is None:
            self._acquisition_function = self._builder.prepare_acquisition_function(
                models,
                datasets=datasets,
            )
        else:
            self._acquisition_function = self._builder.update_acquisition_function(
                self._acquisition_function,
                models,
                datasets=datasets,
            )

        def state_func(
            state: AsynchronousRuleState | None,
        ) -> tuple[AsynchronousRuleState | None, TensorType]:
            tf.debugging.Assert(self._acquisition_function is not None, [tf.constant([])])

            if state is None:
                state = AsynchronousRuleState(None)

            assert datasets is not None
            state = state.remove_points(datasets[OBJECTIVE].query_points)

            if state.has_pending_points:
                pending_points: TensorType = state.pending_points

                def function_with_pending_points(x: TensorType) -> TensorType:
                    # stuff below is quite tricky, and thus deserves an elaborate comment
                    # we receive unknown number N of batches to evaluate
                    # and need to collect batch of B new points
                    # so the shape of `x` is [N, B, D]
                    # we want to add P pending points to each batch
                    # so that acquisition actually receives N batches of shape [P+B, D] each
                    # therefore here we prepend each batch with all pending points
                    # resulting a shape [N, P+B, D]
                    # we do that by repeating pending points N times and concatenating with x

                    # pending points are 2D, we need 3D and repeat along first axis
                    expanded = tf.expand_dims(pending_points, axis=0)
                    pending_points_repeated = tf.repeat(expanded, [tf.shape(x)[0]], axis=0)
                    all_points = tf.concat([pending_points_repeated, x], axis=1)
                    return cast(AcquisitionFunction, self._acquisition_function)(all_points)

                acquisition_function = cast(AcquisitionFunction, function_with_pending_points)
            else:
                acquisition_function = cast(AcquisitionFunction, self._acquisition_function)

            with tf.name_scope("AsynchronousOptimization.optimizer"):
                new_points = self._optimizer(search_space, acquisition_function)
            state = state.add_pending_points(new_points)

            return state, new_points

        return state_func


class AsynchronousGreedy(
    AcquisitionRule[
        State[Optional["AsynchronousRuleState"], TensorType],
        SearchSpaceType,
        ProbabilisticModelType,
    ]
):
    """AsynchronousGreedy rule, as name suggests,
    is designed for asynchronous BO scenarios. To see what we understand by
    asynchronous BO, see documentation for :class:`~trieste.acquisition.AsynchronousOptimization`.

    AsynchronousGreedy rule works with greedy batch acquisition functions
    and performs B steps of a greedy batch collection process,
    where B is the requested batch size.
    """

    def __init__(
        self,
        builder: GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]
        | SingleModelGreedyAcquisitionBuilder[ProbabilisticModelType],
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        """
        :param builder: Acquisition function builder. Only greedy batch approaches are supported,
            because they can be told what points are pending.
        :param optimizer: The optimizer with which to optimize the acquisition function built by
            ``builder``. This should *maximize* the acquisition function, and must be compatible
            with the global search space. Defaults to
            :func:`~trieste.acquisition.optimizer.automatic_optimizer_selector`.
        :param num_query_points: The number of points to acquire.
        """
        if num_query_points <= 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )

        if builder is None:
            raise ValueError("Please specify an acquisition builder")

        if not isinstance(
            builder, (GreedyAcquisitionFunctionBuilder, SingleModelGreedyAcquisitionBuilder)
        ):
            raise NotImplementedError(
                f"""Only greedy acquisition strategies are supported,
                    got {type(builder)}"""
            )

        if optimizer is None:
            optimizer = automatic_optimizer_selector

        if isinstance(builder, SingleModelGreedyAcquisitionBuilder):
            builder = builder.using(OBJECTIVE)

        self._builder: GreedyAcquisitionFunctionBuilder[ProbabilisticModelType] = builder
        self._optimizer = optimizer
        self._acquisition_function: Optional[AcquisitionFunction] = None
        self._num_query_points = num_query_points

    def __repr__(self) -> str:
        """"""
        return f"""AsynchronousGreedy(
        {self._builder!r},
        {self._optimizer!r},
        {self._num_query_points!r})"""

    def acquire(
        self,
        search_space: SearchSpaceType,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> types.State[AsynchronousRuleState | None, TensorType]:
        """
        Constructs a function that, given ``AsynchronousRuleState``,
        returns a new state object and points to evaluate.
        The state object contains currently known pending points,
        that is points that were requested for evaluation,
        but observation for which was not received yet.
        To keep them up to date, pending points are compared against the given dataset,
        and whatever points are in the dataset are deleted.
        Then the current batch is generated by calling the acquisition function,
        and all points in the batch are added to the known pending points.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model of the known data. Uses the single key `OBJECTIVE`.
        :param datasets: The known observer query points and observations.
        :return: A function that constructs the next acquisition state and the recommended query
            points from the previous acquisition state.
        """
        if models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )
        if datasets is None or datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""datasets must be provided and contain the single key {OBJECTIVE}"""
            )

        def state_func(
            state: AsynchronousRuleState | None,
        ) -> tuple[AsynchronousRuleState | None, TensorType]:
            if state is None:
                state = AsynchronousRuleState(None)

            assert datasets is not None
            state = state.remove_points(datasets[OBJECTIVE].query_points)

            if self._acquisition_function is None:
                self._acquisition_function = self._builder.prepare_acquisition_function(
                    models,
                    datasets=datasets,
                    pending_points=state.pending_points,
                )
            else:
                self._acquisition_function = self._builder.update_acquisition_function(
                    self._acquisition_function,
                    models,
                    datasets=datasets,
                    pending_points=state.pending_points,
                )

            with tf.name_scope("AsynchronousOptimization.optimizer[0]"):
                new_points_batch = self._optimizer(search_space, self._acquisition_function)
            state = state.add_pending_points(new_points_batch)

            summary_writer = logging.get_tensorboard_writer()
            step_number = logging.get_step_number()

            for i in range(self._num_query_points - 1):
                # greedily allocate additional batch elements
                self._acquisition_function = self._builder.update_acquisition_function(
                    self._acquisition_function,
                    models,
                    datasets=datasets,
                    pending_points=state.pending_points,
                    new_optimization_step=False,
                )
                with tf.name_scope(f"AsynchronousOptimization.optimizer[{i+1}]"):
                    new_point = self._optimizer(search_space, self._acquisition_function)
                if summary_writer:
                    with summary_writer.as_default(step=step_number):
                        batched_point = tf.expand_dims(new_point, axis=0)
                        value = self._acquisition_function(batched_point)[0][0]
                        logging.scalar(
                            f"AsyncGreedy.acquisition_function/maximum_found[{i}]", value
                        )
                state = state.add_pending_points(new_point)
                new_points_batch = tf.concat([new_points_batch, new_point], axis=0)

            return state, new_points_batch

        return state_func


class RandomSampling(AcquisitionRule[TensorType, SearchSpace, ProbabilisticModel]):
    """
    This class performs random search for choosing optimal points. It uses ``sample`` method
    from :class:`~trieste.space.SearchSpace` to take random samples from the search space that
    are used as optimal points. Hence, it does not use any acquisition function. This
    acquisition rule can be useful as a baseline for other acquisition functions of interest.
    """

    def __init__(self, num_query_points: int = 1):
        """
        :param num_query_points: The number of points to acquire. By default set to 1 point.
        :raise ValueError: If ``num_query_points`` is less or equal to 0.
        """
        if num_query_points <= 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )
        self._num_query_points = num_query_points

    def __repr__(self) -> str:
        """"""
        return f"""RandomSampling({self._num_query_points!r})"""

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[Tag, ProbabilisticModel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        """
        Sample ``num_query_points`` (see :meth:`__init__`) points from the
        ``search_space``.

        :param search_space: The acquisition search space.
        :param models: Unused.
        :param datasets: Unused.
        :return: The ``num_query_points`` points to query.
        """
        samples = search_space.sample(self._num_query_points)

        return samples


class DiscreteThompsonSampling(AcquisitionRule[TensorType, SearchSpace, ProbabilisticModelType]):
    r"""
    Implements Thompson sampling for choosing optimal points.

    This rule returns the minimizers of functions sampled from our model and evaluated across
    a discretization of the search space (containing `N` candidate points).

    The model is sampled either exactly (with an :math:`O(N^3)` complexity), or sampled
    approximately through a random Fourier `M` feature decompisition
    (with an :math:`O(\min(n^3,M^3))` complexity for a model trained on `n` points). The number
    `M` of Fourier features is specified when building the model.

    """

    @overload
    def __init__(
        self: "DiscreteThompsonSampling[ProbabilisticModel]",
        num_search_space_samples: int,
        num_query_points: int,
        thompson_sampler: None = None,
        select_output: Callable[[TensorType], TensorType] = select_nth_output,
    ):
        ...

    @overload
    def __init__(
        self: "DiscreteThompsonSampling[ProbabilisticModelType]",
        num_search_space_samples: int,
        num_query_points: int,
        thompson_sampler: Optional[ThompsonSampler[ProbabilisticModelType]] = None,
        select_output: Callable[[TensorType], TensorType] = select_nth_output,
    ):
        ...

    def __init__(
        self,
        num_search_space_samples: int,
        num_query_points: int,
        thompson_sampler: Optional[ThompsonSampler[ProbabilisticModelType]] = None,
        select_output: Callable[[TensorType], TensorType] = select_nth_output,
    ):
        """
        :param num_search_space_samples: The number of points at which to sample the posterior.
        :param num_query_points: The number of points to acquire.
        :param thompson_sampler: Sampler to sample maximisers from the underlying model.
        :param select_output: A method that returns the desired trajectory from a trajectory
            sampler with shape [..., B], where B is a batch dimension. Defaults to the
            :func:~`trieste.acquisition.utils.select_nth_output` function with output dimension 0.
        """
        if not num_search_space_samples > 0:
            raise ValueError(f"Search space must be greater than 0, got {num_search_space_samples}")

        if not num_query_points > 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )

        if thompson_sampler is not None:
            if thompson_sampler.sample_min_value:
                raise ValueError(
                    """
                    Thompson sampling requires a thompson_sampler that samples minimizers,
                    not just minimum values. However the passed sampler has sample_min_value=True.
                    """
                )
        else:
            thompson_sampler = ExactThompsonSampler(sample_min_value=False)

        self._thompson_sampler = thompson_sampler
        self._num_search_space_samples = num_search_space_samples
        self._num_query_points = num_query_points
        self._select_output = select_output

    def __repr__(self) -> str:
        """"""
        return f"""DiscreteThompsonSampling(
        {self._num_search_space_samples!r},
        {self._num_query_points!r},
        {self._thompson_sampler!r},
        {self._select_output!r})"""

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        """
        Sample `num_search_space_samples` (see :meth:`__init__`) points from the
        ``search_space``. Of those points, return the `num_query_points` points at which
        random samples yield the **minima** of the model posterior.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model of the known data. Uses the single key `OBJECTIVE`.
        :param datasets: The known observer query points and observations.
        :return: The ``num_query_points`` points to query.
        :raise ValueError: If ``models`` do not contain the key `OBJECTIVE`, or it contains any
            other key.
        """
        if models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )

        if datasets is None or datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""datasets must be provided and contain the single key {OBJECTIVE}"""
            )

        query_points = search_space.sample(self._num_search_space_samples)
        thompson_samples = self._thompson_sampler.sample(
            models[OBJECTIVE],
            self._num_query_points,
            query_points,
            select_output=self._select_output,
        )

        return thompson_samples


class TrustRegion(
    AcquisitionRule[
        types.State[Optional["TrustRegion.State"], TensorType], Box, ProbabilisticModelType
    ]
):
    """Implements the *trust region* acquisition algorithm."""

    @dataclass(frozen=True)
    class State:
        """The acquisition state for the :class:`TrustRegion` acquisition rule."""

        acquisition_space: Box
        """ The search space. """

        eps: TensorType
        """
        The (maximum) vector from the current best point to each bound of the acquisition space.
        """

        y_min: TensorType
        """ The minimum observed value. """

        is_global: bool | TensorType
        """
        `True` if the search space was global, else `False` if it was local. May be a scalar boolean
        `TensorType` instead of a `bool`.
        """

        def __deepcopy__(self, memo: dict[int, object]) -> TrustRegion.State:
            box_copy = copy.deepcopy(self.acquisition_space, memo)
            return TrustRegion.State(box_copy, self.eps, self.y_min, self.is_global)

    @overload
    def __init__(
        self: "TrustRegion[ProbabilisticModel]",
        rule: None = None,
        beta: float = 0.7,
        kappa: float = 1e-4,
    ):
        ...

    @overload
    def __init__(
        self: "TrustRegion[ProbabilisticModelType]",
        rule: AcquisitionRule[TensorType, Box, ProbabilisticModelType],
        beta: float = 0.7,
        kappa: float = 1e-4,
    ):
        ...

    def __init__(
        self,
        rule: AcquisitionRule[TensorType, Box, ProbabilisticModelType] | None = None,
        beta: float = 0.7,
        kappa: float = 1e-4,
    ):
        """
        :param rule: The acquisition rule that defines how to search for a new query point in a
            given search space. Defaults to :class:`EfficientGlobalOptimization` with default
            arguments.
        :param beta: The inverse of the trust region contraction factor.
        :param kappa: Scales the threshold for the minimal improvement required for a step to be
            considered a success.
        """
        if rule is None:
            rule = EfficientGlobalOptimization()

        self._rule = rule
        self._beta = beta
        self._kappa = kappa

    def __repr__(self) -> str:
        """"""
        return f"TrustRegion({self._rule!r}, {self._beta!r}, {self._kappa!r})"

    def acquire(
        self,
        search_space: Box,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> types.State[State | None, TensorType]:
        """
        Construct a local search space from ``search_space`` according the trust region algorithm,
        and use that with the ``rule`` specified at :meth:`~TrustRegion.__init__` to find new
        query points. Return a function that constructs these points given a previous trust region
        state.

        If no ``state`` is specified (it is `None`), ``search_space`` is used as the search space
        for this step.

        If a ``state`` is specified, and the new optimum improves over the previous optimum
        by some threshold (that scales linearly with ``kappa``), the previous acquisition is
        considered successful.

        If the previous acquisition was successful, ``search_space`` is used as the new
        search space. If the previous step was unsuccessful, the search space is changed to the
        trust region if it was global, and vice versa.

        If the previous acquisition was over the trust region, the size of the trust region is
        modified. If the previous acquisition was successful, the size is increased by a factor
        ``1 / beta``. Conversely, if it was unsuccessful, the size is reduced by the factor
        ``beta``.

        **Note:** The acquisition search space will never extend beyond the boundary of the
        ``search_space``. For a local search, the actual search space will be the
        intersection of the trust region and ``search_space``.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations. Uses the data for key
            `OBJECTIVE` to calculate the new trust region.
        :return: A function that constructs the next acquisition state and the recommended query
            points from the previous acquisition state.
        :raise KeyError: If ``datasets`` does not contain the key `OBJECTIVE`.
        """
        if datasets is None or OBJECTIVE not in datasets.keys():
            raise ValueError(f"""datasets must be provided and contain the key {OBJECTIVE}""")

        dataset = datasets[OBJECTIVE]

        global_lower = search_space.lower
        global_upper = search_space.upper

        y_min = tf.reduce_min(dataset.observations, axis=0)

        def state_func(
            state: TrustRegion.State | None,
        ) -> tuple[TrustRegion.State | None, TensorType]:
            if state is None:
                eps = 0.5 * (global_upper - global_lower) / (5.0 ** (1.0 / global_lower.shape[-1]))
                is_global = True
            else:
                tr_volume = tf.reduce_prod(
                    state.acquisition_space.upper - state.acquisition_space.lower
                )
                step_is_success = y_min < state.y_min - self._kappa * tr_volume

                eps = (
                    state.eps
                    if state.is_global
                    else state.eps / self._beta
                    if step_is_success
                    else state.eps * self._beta
                )

                is_global = step_is_success or not state.is_global

            if is_global:
                acquisition_space = search_space
            else:
                xmin = dataset.query_points[tf.argmin(dataset.observations)[0], :]
                acquisition_space = Box(
                    tf.reduce_max([global_lower, xmin - eps], axis=0),
                    tf.reduce_min([global_upper, xmin + eps], axis=0),
                )

            points = self._rule.acquire(acquisition_space, models, datasets=datasets)
            state_ = TrustRegion.State(acquisition_space, eps, y_min, is_global)

            return state_, points

        return state_func


class UpdatableTrustRegion(SearchSpace):
    """A search space that can be updated."""

    @abstractmethod
    def initialize(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        """
        Initialize the search space using the given models and datasets.

        :param models: The model for each tag.
        :param datasets: The dataset for each tag.
        """
        ...

    @abstractmethod
    def update(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        """
        Update the search space using the given models and datasets.

        :param models: The model for each tag.
        :param datasets: The dataset for each tag.
        """
        ...


UpdatableTrustRegionType = TypeVar("UpdatableTrustRegionType", bound=UpdatableTrustRegion)
""" A type variable bound to :class:`UpdatableTrustRegion`. """


class BatchTrustRegion(
    AcquisitionRule[
        types.State[Optional["BatchTrustRegion.State"], TensorType],
        SearchSpace,
        ProbabilisticModelType,
    ],
    Generic[ProbabilisticModelType, UpdatableTrustRegionType],
):
    """Abstract class for multi trust region acquisition rules. These are batch algorithms where
    each query point is optimized in parallel, with its own separate trust region.
    """

    @dataclass(frozen=True)
    class State:
        """The acquisition state for the :class:`BatchTrustRegion` acquisition rule."""

        acquisition_space: TaggedMultiSearchSpace
        """ The search space. """

        def __deepcopy__(self, memo: dict[int, object]) -> BatchTrustRegion.State:
            acquisition_space_copy = copy.deepcopy(self.acquisition_space, memo)
            return BatchTrustRegion.State(acquisition_space_copy)

    def __init__(
        self: "BatchTrustRegion[ProbabilisticModelType, UpdatableTrustRegionType]",
        init_subspaces: Sequence[UpdatableTrustRegionType],
        rule: AcquisitionRule[TensorType, SearchSpace, ProbabilisticModelType] | None = None,
    ):
        """
        :param init_subspaces: The initial search spaces for each trust region.
        :param rule: The acquisition rule that defines how to search for a new query point in each
            subspace. Defaults to :class:`EfficientGlobalOptimization` with default arguments.
        """
        if rule is None:
            rule = EfficientGlobalOptimization()

        self._init_subspaces = tuple(init_subspaces)
        self._tags = tuple([str(index) for index in range(len(init_subspaces))])
        self._rule = rule

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}({self._init_subspaces!r}, {self._rule!r})"""

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> types.State[State | None, TensorType]:
        def state_func(
            state: BatchTrustRegion.State | None,
        ) -> Tuple[BatchTrustRegion.State | None, TensorType]:
            """
            If state is None, initialize the subspaces by picking new locations. Otherwise,
            update the existing subspaces.

            Re-initialize the subspaces if necessary, potentially looking at the entire group.

            Use the rule to acquire points from the acquisition space.
            """
            # If state is set, the tags should be the same as the tags of the acquisition space
            # in the state.
            if state is not None:
                assert (
                    self._tags == state.acquisition_space.subspace_tags
                ), f"""The tags of the state acquisition space
                    {state.acquisition_space.subspace_tags} should be the same as the tags of the
                    BatchTrustRegion acquisition rule {self._tags}"""

            subspaces = []
            for tag, init_subspace in zip(self._tags, self._init_subspaces):
                if state is None:
                    subspace = init_subspace
                    subspace.initialize(models, datasets)
                else:
                    _subspace = state.acquisition_space.get_subspace(tag)
                    assert isinstance(_subspace, type(init_subspace))
                    subspace = _subspace
                    subspace.update(models, datasets)

                subspaces.append(subspace)

            self.maybe_initialize_subspaces(subspaces, models, datasets)

            if state is None:
                acquisition_space = TaggedMultiSearchSpace(subspaces, self._tags)
            else:
                acquisition_space = state.acquisition_space

            state_ = BatchTrustRegion.State(acquisition_space)
            points = self._rule.acquire(acquisition_space, models, datasets=datasets)

            return state_, points

        return state_func

    def maybe_initialize_subspaces(
        self,
        subspaces: Sequence[UpdatableTrustRegionType],
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        """
        Initialize subspaces if necessary.
        Get a mask of subspaces that need to be initialized using an abstract method.
        Initialize individual subpaces by calling the method of the UpdatableTrustRegionType class.

        This method can be overridden by subclasses to change this behaviour.
        """
        mask = self.get_initialize_subspaces_mask(subspaces, models, datasets)
        tf.debugging.assert_equal(
            tf.shape(mask),
            (len(subspaces),),
            message="The mask for initializing subspaces should be of the same length as the "
            "number of subspaces",
        )
        for ix, subspace in enumerate(subspaces):
            if mask[ix]:
                subspace.initialize(models, datasets)

    @abstractmethod
    @check_shapes("return: [V]")
    def get_initialize_subspaces_mask(
        self,
        subspaces: Sequence[UpdatableTrustRegionType],
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        """
        Return a boolean mask for subspaces that should be initialized.
        This method is called during the acquisition step to determine which subspaces should be
        initialized and which should be updated. The subspaces corresponding to True values in the
        mask will be re-initialized.

        :param subspaces: The sequence of subspaces.
        :param models: The model for each tag.
        :param datasets: The dataset for each tag.
        :return: A boolean mask of length V, where V is the number of subspaces.
        """
        ...


class SingleObjectiveTrustRegionBox(Box, UpdatableTrustRegion):
    """An updatable box search space for use with trust region acquisition rules."""

    def __init__(
        self,
        global_search_space: SearchSpace,
        beta: float = 0.7,
        kappa: float = 1e-4,
        min_eps: float = 1e-2,
    ):
        """
        Calculates the bounds of the box from the location/centre and global bounds.

        :param global_search_space: The global search space this search space lives in.
        :param beta: The inverse of the trust region contraction factor.
        :param kappa: Scales the threshold for the minimal improvement required for a step to be
            considered a success.
        :param min_eps: The minimal size of the search space. If the size of the search space is
            smaller than this, the search space is reinitialized.
        """

        self._global_search_space = global_search_space
        self._beta = beta
        self._kappa = kappa
        self._min_eps = min_eps

        super().__init__(global_search_space.lower, global_search_space.upper)

    @property
    def global_search_space(self) -> SearchSpace:
        """The global search space this search space lives in."""
        return self._global_search_space

    def _init_eps(self) -> None:
        global_lower = self.global_search_space.lower
        global_upper = self.global_search_space.upper
        self.eps = 0.5 * (global_upper - global_lower) / (5.0 ** (1.0 / global_lower.shape[-1]))

    def _update_bounds(self) -> None:
        self._lower = tf.reduce_max(
            [self.global_search_space.lower, self.location - self.eps], axis=0
        )
        self._upper = tf.reduce_min(
            [self.global_search_space.upper, self.location + self.eps], axis=0
        )

    def initialize(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        """
        Initialize the box by sampling a location from the global search space and setting the
        bounds.
        """
        dataset = get_value_for_tag(datasets)

        self.location = tf.squeeze(self.global_search_space.sample(1), axis=0)
        self._step_is_success = False
        self._init_eps()
        self._update_bounds()
        _, self._y_min = self.get_local_min(dataset)

    def update(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        """
        Update this box, including centre/location, using the given dataset. If the size of the
        box is less than the minimum size, re-initialize the box.
        """
        dataset = get_value_for_tag(datasets)

        if tf.reduce_any(self.eps < self._min_eps):
            self.initialize(models, datasets)
            return

        x_min, y_min = self.get_local_min(dataset)
        self.location = x_min

        tr_volume = tf.reduce_prod(self.upper - self.lower)
        self._step_is_success = y_min < self._y_min - self._kappa * tr_volume
        self.eps = self.eps / self._beta if self._step_is_success else self.eps * self._beta
        self._update_bounds()
        self._y_min = y_min

    @check_shapes(
        "return[0]: [D]",
        "return[1]: []",
    )
    def get_local_min(self, dataset: Optional[Dataset]) -> Tuple[TensorType, TensorType]:
        """Calculate the local minimum of the box using the given dataset."""
        if dataset is None:
            raise ValueError("""dataset must be provided""")

        in_tr = self.contains(dataset.query_points)
        in_tr_obs = tf.where(
            tf.expand_dims(in_tr, axis=-1),
            dataset.observations,
            tf.constant(np.inf, dtype=dataset.observations.dtype),
        )
        ix = tf.argmin(in_tr_obs)
        x_min = tf.gather(dataset.query_points, ix)
        y_min = tf.gather(in_tr_obs, ix)

        return tf.squeeze(x_min, axis=0), tf.squeeze(y_min)


class BatchTrustRegionBox(BatchTrustRegion[ProbabilisticModelType, SingleObjectiveTrustRegionBox]):
    """
    Implements the :class:`BatchTrustRegion` *trust region* acquisition algorithm for box regions.
    This is intended to be used for single-objective optimization with batching.
    """

    @inherit_check_shapes
    def get_initialize_subspaces_mask(
        self,
        subspaces: Sequence[SingleObjectiveTrustRegionBox],
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        # Initialize the subspaces that have non-unique locations.
        centres = tf.stack([subspace.location for subspace in subspaces])
        return tf.logical_not(get_unique_points_mask(centres, tolerance=1e-6))


class TURBO(
    AcquisitionRule[
        types.State[Optional["TURBO.State"], TensorType], Box, TrainableSupportsGetKernel
    ]
):
    """Implements the TURBO algorithm as detailed in :cite:`eriksson2019scalable`."""

    @dataclass(frozen=True)
    class State:
        """The acquisition state for the :class:`TURBO` acquisition rule."""

        acquisition_space: Box
        """ The search space. """

        L: float
        """ Length of the trust region (before standardizing by model lengthscales) """

        failure_counter: int
        """ Number of consecutive failures (reset if we see a success). """

        success_counter: int
        """ Number of consecutive successes (reset if we see a failure).  """

        y_min: TensorType
        """ The minimum observed value. """

        def __deepcopy__(self, memo: dict[int, object]) -> TURBO.State:
            box_copy = copy.deepcopy(self.acquisition_space, memo)
            return TURBO.State(
                box_copy, self.L, self.failure_counter, self.success_counter, self.y_min
            )

    def __init__(
        self,
        search_space: SearchSpace,
        num_trust_regions: int = 1,
        rule: Optional[AcquisitionRule[ResultType, Box, TrainableSupportsGetKernel]] = None,
        L_min: Optional[float] = None,
        L_init: Optional[float] = None,
        L_max: Optional[float] = None,
        success_tolerance: int = 3,
        failure_tolerance: Optional[int] = None,
        local_models: Optional[Mapping[Tag, TrainableSupportsGetKernel]] = None,
    ):
        """
        Note that the optional parameters are set by a heuristic if not given by the user.

        :param search_space: The search space.
        :param num_trust_regions: Number of trust regions controlled by TURBO
        :param rule: rule used to select points from within the trust region, using the local model.
        :param L_min: Minimum allowed length of the trust region.
        :param L_init: Initial length of the trust region.
        :param L_max: Maximum allowed length of the trust region.
        :param success_tolerance: Number of consecutive successes before changing region size.
        :param failure tolerance: Number of consecutive failures before changing region size.
        :param local_models: Optional model to act as the local model. This will be refit using
            the data from each trust region. If no local_models are provided then we just
            copy the global model.
        """

        if not num_trust_regions > 0:
            raise ValueError(f"Num trust regions must be greater than 0, got {num_trust_regions}")

        if num_trust_regions > 1:
            raise NotImplementedError(
                f"TURBO does not yet support multiple trust regions, but got {num_trust_regions}"
            )

        # implement heuristic defaults for TURBO if not specified by user
        if rule is None:  # default to Thompson sampling with batches of size 1
            rule = DiscreteThompsonSampling(tf.minimum(100 * search_space.dimension, 5_000), 1)

        if failure_tolerance is None:
            if isinstance(
                rule,
                (
                    EfficientGlobalOptimization,
                    DiscreteThompsonSampling,
                    RandomSampling,
                    AsynchronousOptimization,
                ),
            ):
                failure_tolerance = math.ceil(search_space.dimension / rule._num_query_points)
            else:
                failure_tolerance == search_space.dimension
            assert isinstance(failure_tolerance, int)
        search_space_max_width = tf.reduce_max(search_space.upper - search_space.lower)
        if L_min is None:
            L_min = (0.5**7) * search_space_max_width
        if L_init is None:
            L_init = 0.8 * search_space_max_width
        if L_max is None:
            L_max = 1.6 * search_space_max_width

        if not success_tolerance > 0:
            raise ValueError(
                f"success tolerance must be an integer greater than 0, got {success_tolerance}"
            )
        if not failure_tolerance > 0:
            raise ValueError(
                f"success tolerance must be an integer greater than 0, got {failure_tolerance}"
            )

        if L_min <= 0:
            raise ValueError(f"L_min must be postive, got {L_min}")
        if L_init <= 0:
            raise ValueError(f"L_min must be postive, got {L_init}")
        if L_max <= 0:
            raise ValueError(f"L_min must be postive, got {L_max}")

        self._num_trust_regions = num_trust_regions
        self._L_min = L_min
        self._L_init = L_init
        self._L_max = L_max
        self._success_tolerance = success_tolerance
        self._failure_tolerance = failure_tolerance
        self._rule = rule
        self._local_models = local_models

    def __repr__(self) -> str:
        """"""
        return f"TURBO({self._num_trust_regions!r}, {self._rule})"

    def acquire(
        self,
        search_space: Box,
        models: Mapping[Tag, TrainableSupportsGetKernel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> types.State[State | None, TensorType]:
        """
        Construct a local search space from ``search_space`` according the TURBO algorithm,
        and use that with the ``rule`` specified at :meth:`~TURBO.__init__` to find new
        query points. Return a function that constructs these points given a previous trust region
        state.

        If no ``state`` is specified (it is `None`), then we build the initial trust region.

        If a ``state`` is specified, and the new optimum improves over the previous optimum,
        the previous acquisition is considered successful.

        If ``success_tolerance`` previous consecutive acquisitions were successful then the search
        space is made larger. If  ``failure_tolerance`` consecutive acquisitions were unsuccessful
        then the search space is shrunk. If neither condition is triggered then the search space
        remains the same.

        **Note:** The acquisition search space will never extend beyond the boundary of the
        ``search_space``. For a local search, the actual search space will be the
        intersection of the trust region and ``search_space``.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations. Uses the data for key
            `OBJECTIVE` to calculate the new trust region.
        :return: A function that constructs the next acquisition state and the recommended query
            points from the previous acquisition state.
        :raise KeyError: If ``datasets`` does not contain the key `OBJECTIVE`.
        """
        if self._local_models is None:  # if user doesnt specifiy a local model
            self._local_models = copy.copy(models)  # copy global model (will be fit locally later)

        if self._local_models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )

        if datasets is None or datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""datasets must be provided and contain the single key {OBJECTIVE}"""
            )

        dataset = datasets[OBJECTIVE]
        local_model = self._local_models[OBJECTIVE]
        global_lower = search_space.lower
        global_upper = search_space.upper

        y_min = tf.reduce_min(dataset.observations, axis=0)

        def state_func(
            state: TURBO.State | None,
        ) -> tuple[TURBO.State | None, TensorType]:
            if state is None:  # initialise first TR
                L, failure_counter, success_counter = self._L_init, 0, 0
            else:  # update TR
                step_is_success = y_min < state.y_min - 1e-10  # maybe make this stronger?
                failure_counter = (
                    0 if step_is_success else state.failure_counter + 1
                )  # update or reset counter
                success_counter = (
                    state.success_counter + 1 if step_is_success else 0
                )  # update or reset counter
                L = state.L
                if success_counter == self._success_tolerance:
                    L *= 2.0  # make region bigger
                    success_counter = 0
                elif failure_counter == self._failure_tolerance:
                    L *= 0.5  # make region smaller
                    failure_counter = 0

                L = tf.minimum(L, self._L_max)
                if L < self._L_min:  # if gets too small then start again
                    L, failure_counter, success_counter = self._L_init, 0, 0

            # build region with volume according to length L but stretched according to lengthscales
            xmin = dataset.query_points[tf.argmin(dataset.observations)[0], :]  # centre of region
            lengthscales = (
                local_model.get_kernel().lengthscales
            )  # stretch region according to model lengthscales
            tr_width = (
                lengthscales * L / tf.reduce_prod(lengthscales) ** (1.0 / global_lower.shape[-1])
            )  # keep volume fixed
            acquisition_space = Box(
                tf.reduce_max([global_lower, xmin - tr_width / 2.0], axis=0),
                tf.reduce_min([global_upper, xmin + tr_width / 2.0], axis=0),
            )

            # fit the local model using just data from the trust region
            local_dataset = get_local_dataset(acquisition_space, dataset)
            local_model.update(local_dataset)
            local_model.optimize(local_dataset)

            # use local model and local dataset to choose next query point(s)
            points = self._rule.acquire_single(acquisition_space, local_model, local_dataset)
            state_ = TURBO.State(acquisition_space, L, failure_counter, success_counter, y_min)

            return state_, points

        return state_func


class BatchHypervolumeSharpeRatioIndicator(
    AcquisitionRule[TensorType, SearchSpace, ProbabilisticModel]
):
    """Implements the Batch Hypervolume Sharpe-ratio indicator acquisition
    rule, designed for large batches, introduced by Binois et al, 2021.
    See :cite:`binois2021portfolio` for details.
    """

    def __init__(
        self,
        num_query_points: int = 1,
        ga_population_size: int = 500,
        ga_n_generations: int = 200,
        filter_threshold: float = 0.1,
        noisy_observations: bool = True,
    ):
        """
        :param num_query_points: The number of points in a batch. Defaults to 5.
        :param ga_population_size: The population size used in the genetic algorithm
             that finds points on the Pareto front. Defaults to 500.
        :param ga_n_generations: The number of genenrations to run in the genetic
             algorithm. Defaults to 200.
        :param filter_threshold: The probability of improvement below which to exlude
             points from the Sharpe ratio optimisation. Defaults to 0.1.
        :param noisy_observations: Whether the observations have noise. Defaults to True.
        """
        if not num_query_points > 0:
            raise ValueError(f"Num query points must be greater than 0, got {num_query_points}")
        if not ga_population_size >= num_query_points:
            raise ValueError(
                "Population size must be greater or equal to num query points size, got num"
                f" query points as {num_query_points} and population size as {ga_population_size}"
            )
        if not ga_n_generations > 0:
            raise ValueError(f"Number of generation must be greater than 0, got {ga_n_generations}")
        if not 0.0 <= filter_threshold < 1.0:
            raise ValueError(f"Filter threshold must be in range [0.0,1.0), got {filter_threshold}")
        if pymoo is None:
            raise ImportError(
                "BatchHypervolumeSharpeRatioIndicator requires pymoo, "
                "which can be installed via `pip install trieste[qhsri]`"
            )
        builder = ProbabilityOfImprovement().using(OBJECTIVE)

        self._builder: AcquisitionFunctionBuilder[ProbabilisticModel] = builder
        self._num_query_points: int = num_query_points
        self._population_size: int = ga_population_size
        self._n_generations: int = ga_n_generations
        self._filter_threshold: float = filter_threshold
        self._noisy_observations: bool = noisy_observations
        self._acquisition_function: Optional[AcquisitionFunction] = None

    def __repr__(self) -> str:
        """"""
        return f"""BatchHypervolumeSharpeRatioIndicator(
        num_query_points={self._num_query_points}, ga_population_size={self._population_size},
        ga_n_generations={self._n_generations}, filter_threshold={self._filter_threshold},
        noisy_observations={self._noisy_observations}
        )
        """

    def _find_non_dominated_points(
        self, model: ProbabilisticModel, search_space: SearchSpaceType
    ) -> tuple[TensorType, TensorType]:
        """Uses NSGA-II to find high-quality non-dominated points"""

        problem = _MeanStdTradeoff(model, search_space)
        algorithm = NSGA2(pop_size=self._population_size)
        res = minimize(problem, algorithm, ("n_gen", self._n_generations), seed=1, verbose=False)

        return res.X, res.F

    def _filter_points(
        self, nd_points: TensorType, nd_mean_std: TensorType
    ) -> tuple[TensorType, TensorType]:
        if self._acquisition_function is None:
            raise ValueError("Acquisition function has not been defined yet")

        probs_of_improvement = np.array(
            self._acquisition_function(np.expand_dims(nd_points, axis=-2))
        )

        above_threshold = probs_of_improvement > self._filter_threshold

        if np.sum(above_threshold) >= self._num_query_points and nd_mean_std.shape[1] == 2:
            # There are enough points above the threshold to get a batch
            out_points, out_mean_std = (
                nd_points[above_threshold.squeeze(), :],
                nd_mean_std[above_threshold.squeeze(), :],
            )
        else:
            # We don't filter
            out_points, out_mean_std = nd_points, nd_mean_std

        return out_points, out_mean_std

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[Tag, ProbabilisticModel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        """Acquire a batch of points to observe based on the batch hypervolume
        Sharpe ratio indicator method.
        This method uses NSGA-II to create a Pareto set of the mean and standard
        deviation of the posterior of the probabilistic model, and then selects
        points to observe based on maximising the Sharpe ratio.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations.
        :return: The batch of points to query.
        """
        if models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )

        if datasets is None or datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""datasets must be provided and contain the single key {OBJECTIVE}"""
            )

        if self._acquisition_function is None:
            self._acquisition_function = self._builder.prepare_acquisition_function(
                models, datasets=datasets
            )
        else:
            self._acquisition_function = self._builder.update_acquisition_function(
                self._acquisition_function,
                models,
                datasets=datasets,
            )

        # Find non-dominated points
        nd_points, nd_mean_std = self._find_non_dominated_points(models[OBJECTIVE], search_space)

        # Filter out points below a threshold probability of improvement
        filtered_points, filtered_mean_std = self._filter_points(nd_points, nd_mean_std)

        # Set up a Pareto set of the filtered points
        pareto_set = Pareto(filtered_mean_std, already_non_dominated=True)

        # Sample points from set using qHSRI
        _, batch_ids = pareto_set.sample_diverse_subset(
            self._num_query_points, allow_repeats=self._noisy_observations
        )

        batch = filtered_points[batch_ids]

        return batch


class _MeanStdTradeoff(PymooProblem):  # type: ignore[misc]
    """Inner class that formulates the mean/std optimisation problem as a
    pymoo problem"""

    def __init__(self, probabilistic_model: ProbabilisticModel, search_space: SearchSpaceType):
        """
        :param probabilistic_model: The probabilistic model to find optimal mean/stds from
        :param search_space: The search space for the optimisation
        """
        # If we have a stack of models we have mean and std for each
        if isinstance(probabilistic_model, ModelStack):
            n_obj = 2 * len(probabilistic_model._models)
        else:
            n_obj = 2
        super().__init__(
            n_var=int(search_space.dimension),
            n_obj=n_obj,
            n_constr=0,
            xl=np.array(search_space.lower),
            xu=np.array(search_space.upper),
        )
        self.probabilistic_model = probabilistic_model

    def _evaluate(
        self, x: TensorType, out: dict[str, TensorType], *args: Any, **kwargs: Any
    ) -> None:
        mean, var = self.probabilistic_model.predict(x)
        # Flip sign on std so that minimising is increasing std
        std = -1 * np.sqrt(np.array(var))
        out["F"] = np.concatenate([np.array(mean), std], axis=1)
